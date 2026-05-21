import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .nets.graph_model.attention_model import *
# , Decoder


class Problem:
    def __init__(self, name: str):
        self.NAME = name


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization for linear layers."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class StateWrapper:
    """
    Convert raw env observations into model-friendly tensors and
    expose commonly used state accessors.
    """

    def __init__(self, states: Dict[str, Any], device: str, problem: str = "evrptw"):
        self.device = device
        self.problem = problem

        self.states: Dict[str, torch.Tensor] = {
            k: torch.as_tensor(v, device=self.device) for k, v in states.items()
        }

        if problem == "evrptw":
            self._build_evrptw_state()

    def _build_evrptw_state(self) -> None:
        observations = {
            "depot_loc": self.states["depot_loc"],
            "cus_loc": self.states["cus_loc"],
            "rs_loc": self.states["rs_loc"],
            "time_window": self.states["time_window"],
            "demand": self.states["demand"].unsqueeze(-1),
            "service_time": self.states["service_time"].unsqueeze(-1),
        }

        self.states["observations"] = observations

        self.VEHICLE_CAPACITY = self.states["loading_capacity"]
        self.VEHICLE_BATTERY = self.states["battery_capacity"]

        self.used_capacity = self.states["current_load"]
        self.used_battery = self.states["current_battery"]
        self.current_time = self.states["current_time"]

        # keep env key compatibility
        self.visited_customers_ratio = self.states["visited_customers_ratio"]
        self.remain_feasible_customers_ratio = self.states["remain_feasible_customers_ratio"]

    def get_current_node(self) -> torch.Tensor:
        return self.states["last_node_idx"].to(dtype=torch.long)

    def get_mask(self) -> torch.Tensor:
        action_mask = self.states["action_mask"]

        # action_mask: True/1 means feasible action
        action_mask = action_mask.to(torch.bool)

        # returned mask: True means infeasible / masked
        return ~action_mask

    @property
    def observations(self) -> Dict[str, torch.Tensor]:
        return self.states["observations"]


class Backbone(nn.Module):
    """
    Shared backbone:
        raw obs -> graph embedding -> graph encoder -> decoder

    Notes:
    - Graph token / virtual node handling is fully internal to graph_model.
    - Wrapper only passes raw node mask of shape [B, N].
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        problem_name: str = "evrptw",
        n_encode_layers: int = 2,
        tanh_clipping: float = 15.0,
        n_heads: int = 16,
        device: str = "cpu",
        use_candidate_dynamic_embedding: bool = True,
    ):
        super().__init__()

        self.device = device
        self.problem = Problem(problem_name)
        self.embedding_dim = embedding_dim
        self.use_candidate_dynamic_embedding = bool(use_candidate_dynamic_embedding)

        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )

        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            step_context_dim=embedding_dim + 5,
            n_heads=n_heads,
            problem=self.problem,
            tanh_clipping=tanh_clipping,
            use_candidate_dynamic_embedding=self.use_candidate_dynamic_embedding,
        )

         # graph bias parameters
        self.dist_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.type_pair_bias = nn.Embedding(3 * 3, 1)   # 3 node types: depot / RS / customer
        nn.init.zeros_(self.type_pair_bias.weight)

    def _build_node_type(self, node_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Node order: [depot, customers, RS]
        Returns:
            node_type: [B, N]
                depot=0, rs=1, customer=2
        """
        depot_loc = node_inputs["depot_loc"]
        if depot_loc.dim() == 2:
            B = depot_loc.size(0)
        else:
            B = depot_loc.size(0)

        n_cus = node_inputs["cus_loc"].size(1)
        n_rs = node_inputs["rs_loc"].size(1)
        device = node_inputs["cus_loc"].device

        depot_type = torch.zeros(B, 1, dtype=torch.long, device=device)          # 0
        cus_type = torch.full((B, n_cus), 2, dtype=torch.long, device=device)    # 2
        rs_type = torch.ones(B, n_rs, dtype=torch.long, device=device)           # 1

        node_type = torch.cat([depot_type, cus_type, rs_type], dim=1)            # [B,N]
        return node_type

    def _build_distance_matrix(self, node_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build pairwise Euclidean distance matrix from normalized coordinates.
        Node order: [depot, customers, RS]
        Returns:
            dist_mat: [B, N, N]
        """
        depot_loc = node_inputs["depot_loc"]
        if depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)   # [B,1,2]

        cus_loc = node_inputs["cus_loc"]         # [B,n_cus,2]
        rs_loc = node_inputs["rs_loc"]           # [B,n_rs,2]

        nodes = torch.cat([depot_loc, cus_loc, rs_loc], dim=1)   # [B,N,2]
        dist_mat = torch.cdist(nodes, nodes, p=2)                # [B,N,N]
        return dist_mat

    def _build_attn_bias(self, state: StateWrapper) -> torch.Tensor:
        """
        Returns:
            attn_bias: [B, N, N]
        """
        node_inputs = state.observations
        dist_mat = self._build_distance_matrix(node_inputs)   # [B,N,N]
        node_type = self._build_node_type(node_inputs)        # [B,N]

        # distance bias: farther nodes get smaller bias
        dist_bias = -self.dist_bias_scale * dist_mat          # [B,N,N]

        # type-pair bias
        type_i = node_type.unsqueeze(2)                       # [B,N,1]
        type_j = node_type.unsqueeze(1)                       # [B,1,N]
        pair_id = type_i * 3 + type_j                         # [B,N,N], values in [0,8]

        type_bias = self.type_pair_bias(pair_id).squeeze(-1)  # [B,N,N]

        attn_bias = dist_bias + type_bias

        edge_energy = state.states.get("edge_energy", None)
        battery_capacity = state.states.get("battery_capacity", None)
        if edge_energy is not None and battery_capacity is not None:
            edge_energy = edge_energy.to(
                device=attn_bias.device,
                dtype=attn_bias.dtype,
            )
            if edge_energy.dim() == 2:
                edge_energy = edge_energy.unsqueeze(0)
            if edge_energy.size(0) == 1 and attn_bias.size(0) != 1:
                edge_energy = edge_energy.expand(attn_bias.size(0), -1, -1)

            battery_capacity = battery_capacity.to(
                device=attn_bias.device,
                dtype=attn_bias.dtype,
            )
            if battery_capacity.dim() == 0:
                battery_capacity = battery_capacity.view(1, 1, 1)
            elif battery_capacity.dim() == 1:
                battery_capacity = battery_capacity.view(-1, 1, 1)
            else:
                battery_capacity = battery_capacity.reshape(battery_capacity.size(0), -1)
                battery_capacity = battery_capacity[:, :1].view(-1, 1, 1)
            if battery_capacity.size(0) == 1 and attn_bias.size(0) != 1:
                battery_capacity = battery_capacity.expand(attn_bias.size(0), -1, -1)

            unreachable = edge_energy > (battery_capacity + 1e-6)
            eye = torch.eye(
                attn_bias.size(-1),
                dtype=torch.bool,
                device=attn_bias.device,
            ).unsqueeze(0)
            unreachable = unreachable & ~eye
            attn_bias = attn_bias.masked_fill(unreachable, -1e9)
        return attn_bias

    def _build_state(self, obs: Dict[str, Any]) -> StateWrapper:
        return StateWrapper(obs, device=self.device, problem=self.problem.NAME)

    def _encode_from_state(
        self,
        state: StateWrapper,
        use_mask: bool = False,
    ):
        node_mask = state.states["instance_mask"] if use_mask else None

        node_inputs = state.observations
        node_embeddings = self.embedding(node_inputs)         # [B, N, D]

        attn_bias = self._build_attn_bias(state)              # [B, N, N]
        encoded_nodes = self.encoder(
            node_embeddings,
            mask=None,                # no encoder mask for current fixed-size setting
            attn_bias=attn_bias,
        )
        cached_embeddings = self.decoder._precompute(encoded_nodes, mask=node_mask)

        return cached_embeddings, node_mask

    def forward(
        self,
        obs: Dict[str, Any],
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._build_state(obs)
        cached_embeddings, node_mask = self._encode_from_state(state, use_mask=use_mask)

        logits, glimpse = self.decoder.advance(
            cached_embeddings,
            state,
            node_mask=node_mask,
        )
        return logits, glimpse

    def encode(
        self,
        obs: Dict[str, Any],
        use_mask: bool = False,
    ):
        state = self._build_state(obs)
        cached_embeddings, _ = self._encode_from_state(state, use_mask=use_mask)
        return cached_embeddings

    def decode(
        self,
        obs: Dict[str, Any],
        cached_embeddings,
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._build_state(obs)
        node_mask = state.states["instance_mask"] if use_mask else None

        logits, glimpse = self.decoder.advance(
            cached_embeddings,
            state,
            node_mask=node_mask,
        )
        return logits, glimpse


class Actor(nn.Module):
    """Backbone already produces logits; actor is a thin wrapper."""

    def __init__(self):
        super().__init__()

    def forward(self, backbone_output: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        logits, _ = backbone_output
        return logits


class Critic(nn.Module):
    """Critic reads backbone glimpse/context representation."""

    def __init__(self, hidden_size: int, value_heads: int = 1):
        super().__init__()
        self.value_heads = int(value_heads)
        if self.value_heads == 1:
            self.mlp = self._make_head(hidden_size)
        else:
            self.heads = nn.ModuleList(
                [self._make_head(hidden_size) for _ in range(self.value_heads)]
            )

    @staticmethod
    def _make_head(hidden_size: int) -> nn.Sequential:
        mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        for layer in mlp:
            orthogonal_init(layer, gain=0.01)

        return mlp

    def forward(self, backbone_output: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, glimpse = backbone_output
        if self.value_heads == 1:
            return self.mlp(glimpse)

        return torch.cat([head(glimpse) for head in self.heads], dim=-1)


class Agent(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        tanh_clipping: float = 15.0,
        n_encode_layers: int = 2,
        device: str = "cpu",
        name: str = "evrptw",
        value_heads: int = 1,
        use_candidate_dynamic_embedding: bool = True,
    ):
        super().__init__()

        self.backbone = Backbone(
            embedding_dim=embedding_dim,
            device=device,
            tanh_clipping=tanh_clipping,
            n_encode_layers=n_encode_layers,
            problem_name=name,
            use_candidate_dynamic_embedding=use_candidate_dynamic_embedding,
        )
        self.actor = Actor()
        self.value_heads = int(value_heads)
        self.critic = Critic(hidden_size=embedding_dim, value_heads=self.value_heads)

    def forward(
        self,
        obs: Dict[str, Any],
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone_output = self.backbone(obs, use_mask=use_mask)
        logits = self.actor(backbone_output)
        action = logits.max(dim=2)[1]
        return action, logits

    def get_value(
        self,
        obs: Dict[str, Any],
        use_mask: bool = False,
    ) -> torch.Tensor:
        backbone_output = self.backbone(obs, use_mask=use_mask)
        return self.critic(backbone_output)

    def get_action_and_value(
        self,
        obs: Dict[str, Any],
        action: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ):
        backbone_output = self.backbone(obs, use_mask=use_mask)
        logits = self.actor(backbone_output)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(backbone_output)
        return action, probs.log_prob(action), probs.entropy(), value

    def get_acction_and_value(
        self,
        obs: Dict[str, Any],
        action: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ):
        return self.get_action_and_value(obs, action=action, use_mask=use_mask)

    def get_value_cached(
        self,
        obs: Dict[str, Any],
        cached_embeddings,
        use_mask: bool = False,
    ) -> torch.Tensor:
        backbone_output = self.backbone.decode(
            obs,
            cached_embeddings,
            use_mask=use_mask,
        )
        return self.critic(backbone_output)

    def get_action_and_value_cached(
        self,
        obs: Dict[str, Any],
        action: Optional[torch.Tensor] = None,
        state=None,                  # keep old name for compatibility
        cached_embeddings=None,      # optional new name
        use_mask: bool = False,
        print_probs: bool = False,
    ):
        # backward compatibility:
        # if old caller passes state=..., treat it as cached embeddings
        if cached_embeddings is None:
            cached_embeddings = state

        if cached_embeddings is None:
            cached_embeddings = self.backbone.encode(obs, use_mask=use_mask)

        backbone_output = self.backbone.decode(
            obs,
            cached_embeddings,
            use_mask=use_mask,
        )

        logits = self.actor(backbone_output)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(backbone_output)
        return action, probs.log_prob(action), probs.entropy(), value, cached_embeddings
