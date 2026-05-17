# EVRP-TW Online-Offline 实验总结

更新时间：2026-05-13

这份文档总结我们在 EVRP-TW 的 online PPO 与 offline/ALNS expert 结合方向上做过的尝试，包括：用了哪些离线信号、哪些实验看起来没效果、哪些诊断最有价值、当前模型完整结构、当前训练方法，以及下一步最有希望的研究路线。

核心结论先放在前面：

> 目前的结果不支持“直接把 ALNS/offline 数据加到 PPO actor 上”这一简单路线。更合理的方向是：先训练一个强 PPO base policy，再用同一批 instance 比较 PPO 与 ALNS，判断 ALNS 只在哪些区域真的优于 PPO，然后只在这些区域通过 residual adapter 做条件式修正，而不是让 expert loss 直接污染主 actor。

也就是说，ALNS 不应该被当成 universal teacher，而应该被当成一个 conditional complementarity oracle：它告诉我们 RL 在哪里弱、哪些结构可能值得补充、什么时候需要更强 decoding。

---

## 1. 当前最重要的判断

我们尝试过很多 offline trick：

- expert delta PBRS；
- expert route PBRS；
- critic reward-to-go pretraining；
- AWBC；
- safe filter；
- focus bucket sampling；
- all offline vs R/RC-wide only；
- route-level preference / DPO-like update；
- online ALNS DPO；
- single-customer PBRS 与 ALNS-route-distance PBRS 混合。

整体观察是：

- **纯 PPO / routeevent repair progress objective-progress 这条线本身已经很强。**
- **offline 信号没有稳定超过 baseline。**
- **很多情况下 offline 曲线和 baseline 几乎一样。**
- **有时 random seed 的影响比 offline 组件更明显。**
- **step-level expert imitation 经常和 PPO 梯度方向冲突。**
- **ALNS 不是所有 instance 上都比 RL 好，所以盲目学习 ALNS 会负迁移。**

因此，继续调 `lambda_bc`、`AWBC coef` 或 `DPO coef` 不是最核心问题。根因是：

```text
PPO policy 和 ALNS heuristic 在不同 instance 子分布上互补，但不是一致。
```

所以我们下一步应该从“加一个 offline loss”转向：

```text
regret-aware + selective + residual adapter
```

---

## 2. 当前模型架构

主要代码：

```text
evrptw_gen/benchmarks/DRL_Solver/models/graph_attention_model_wrapper.py
evrptw_gen/benchmarks/DRL_Solver/train.py
evrptw_gen/benchmarks/DRL_Solver/DRL_train.py
evrptw_gen/benchmarks/DRL_Solver/envs/evrp_vector_env.py
```

### 2.1 输入状态

当前 EVRP-TW 环境包含：

- depot；
- customers；
- charging stations；
- customer demand；
- service time；
- time window；
- vehicle load；
- battery；
- current time；
- current node；
- visited mask；
- feasible action mask；
- edge distance / edge energy。

节点顺序是：

```text
[depot, customers, charging stations]
```

动作就是选择下一个 node。环境会用 action mask 屏蔽不可行动作。

### 2.2 Backbone

当前 backbone 是：

```text
raw observation
  -> StateWrapper
  -> AutoEmbedding
  -> GraphAttentionEncoder
  -> Decoder
  -> logits
```

Backbone 里有几类重要 inductive bias：

1. 距离 bias：

```text
dist_bias = - dist_bias_scale * dist_matrix
```

2. node type pair bias：

```text
depot / customer / charging station
```

3. battery infeasible edge mask：

```text
edge_energy > battery_capacity -> attention bias = -1e9
```

所以现在的模型不是一个完全通用 Transformer，而是已经嵌入了 EVRP-TW 的几何与可行性结构。

### 2.3 Actor

Actor 很薄，本质上就是：

```text
Backbone(obs) -> logits -> Categorical(logits)
```

训练时 sample action。  
eval 时当前主要看：

```text
eval_decode_mode = sampling
test_agent = 8
aggregation = best_of_8
```

也就是我们说的 POMO best-of-8。

### 2.4 Critic

Critic 读 decoder 的 glimpse/context 表示。

它支持：

1. 单头 value：

```text
V(s)
```

2. 多头 value：

```text
V_objective(s), V_progress(s)
```

旧版本里也有：

```text
V_task, V_terminal, V_teacher
```

当前强 baseline/base-ref 线主要用：

```text
use_decomposed_reward_adv = True
decomposed_reward_mode = objective_progress
adv_objective_weight = 0.5
adv_progress_weight = 0.5
```

这部分是有效的，因为它让 objective 和 progress 的 advantage 分开计算/记录，避免所有 reward shaping 信号都挤进一个 scalar critic。

### 2.5 Residual Adapter / Regret Gate

我们已经在模型里加了后续 regret-aware distillation 需要的结构，但默认关闭：

```text
LogitResidualAdapter
RegretGate
```

形式是：

```text
logits_final = logits_actor + beta(x, s) * adapter(logits_actor)
```

其中：

- `logits_actor` 是 PPO 主 actor；
- `adapter(logits_actor)` 是一个小 residual correction；
- `beta(x, s)` 可以由 regret gate 输出；
- 当前 base-ref 训练里：

```text
use_residual_adapter = False
adapter_beta_max = 0.0
use_regret_gate = False
```

所以 base-ref 训练没有被 adapter 改变。这个结构只是为了后续做“ALNS 只作为条件修正器”。

---

## 3. 当前 PPO 训练方法

当前 base-reference 训练目标是先得到一个干净的 PPO reference policy。

当前主要配置：

```text
num_updates = 800
num_envs = 128
num_steps = 75
n_traj = 50
num_minibatches = 64
update_epochs = 5
accum_steps = 8
learning_rate = 4e-05
critic_lr = 3e-05
target_kl = 0.01
eval_freq = 20
eval_batch_size = 1024
checkpoint_milestones = 300,500,800
```

训练 batch 使用 config cycle：

```text
R_narrow
R_wide
C_narrow
C_wide
RC_narrow
RC_wide
```

固定项：

```text
service_time_policy = cargoweight
cluster_number_policy = random
```

PPO loop 包括：

1. 从 schedule 采样训练 instance；
2. rollout `num_steps`；
3. 记录 reward components；
4. 计算 GAE；
5. PPO clipped policy loss；
6. value loss；
7. entropy bonus；
8. backbone / critic 分开 grad norm clip；
9. 记录 `GradSplitCos` 等诊断；
10. 每 20 update eval 一次；
11. 300 / 500 / 800 保存 checkpoint。

当前 base-ref 是纯 PPO：

```text
use_offline_final = False
use_offline_awbc = False
use_offline_critic_pretrain = False
use_alns_teacher = False
use_online_alns_dpo = False
use_residual_adapter = False
```

这一步的目的不是直接拿最高结果，而是得到后续 regret label 的 reference policy。

---

## 4. Eval 数据与 plot 注意事项

当前 base-ref eval 用的是：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
```

也就是 `buffer_1k`。

当前 plot：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/logs_result_base_ref800/base_ref800_curves.png
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/logs_result_base_ref800/base_ref800_curves_zoom.png
```

plot 里现在有：

- 4 条 base-ref seed 曲线；
- 1 条之前 baseline 曲线；
- ALNS 横线。

但需要注意：之前 baseline 的 eval path 是老的：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl
```

不是 `buffer_1k`。所以这条 baseline 可以作为视觉参考，但不能作为严格同数据集对比。

---

## 5. 尝试过的 offline / ALNS 方法

### 5.1 Expert Delta PBRS

最早的想法是从专家路线里计算每个 customer 的边际贡献：

```text
expert_delta_j
```

然后构造：

```text
h_delta(s) = current_to_depot + sum_{unvisited j} expert_delta_j
```

环境里支持：

```text
use_expert_delta_potential
expert_delta_alpha
```

混合方式：

```text
h_mix(s) = (1 - alpha) * h_single(s) + alpha * h_delta(s)
```

我们跑过：

```text
Codex_Res_offline_delta1A_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

包括不同 `expert_delta_alpha` 的版本。

结果：

- 没有稳定超过 baseline；
- 曲线和 baseline 很像；
- alpha 改动没有带来明确收益；
- 很多差异看起来小于 seed variance。

判断：

```text
expert_delta PBRS 单独不够。
```

### 5.2 Delta 的定义问题

我们讨论过一个关键问题：

如果 `delta_j` 只是：

```text
dist(prev, j) + dist(j, next) - dist(prev, next)
```

它只能表示 expert route 上的局部边贡献，不等价于“服务 customer j 的真实全局开销”。

它的问题是：

- 太依赖当前 expert route 的局部邻居；
- 不能反映完整 route feasibility；
- 不能保证和 total distance objective 一致；
- 当 expert route 本身弱时，delta 也弱。

因此后面我们更倾向用 route-level remaining distance，而不是只看局部 edge delta。

### 5.3 Expert Route PBRS

更合理的 expert potential 是保存完整 expert routes，然后在当前状态下：

1. 删除已经访问过的 customers；
2. 保留未访问 customer；
3. 可选保留 charging stations；
4. collapse route；
5. 计算剩余 route distance。

即：

```text
h_route(s) = projected remaining distance on expert reference routes
```

环境里支持：

```text
use_expert_route_potential
expert_route_alpha
expert_route_keep_stations
```

这个比 local delta 更有学术解释性：

> 给定一个 expert feasible solution structure，访问 customer 的奖励来自“剩余 expert solution cost 的减少”。

但实验上，用它和单车 PBRS 混合：

```text
0.7 * single_customer_repair + 0.3 * ALNS_route_distance
```

没有看到明显优势，因此中途停掉。

### 5.4 Critic Pretraining / Reward-to-Go Regression

我们实现/讨论过：

1. 用 expert trajectory replay；
2. 得到 `(s, a, r, s')`；
3. 计算 reward-to-go；
4. 先训练 critic；
5. 再正常 PPO / AWBC。

优点：

- 不直接强迫 actor 模仿 ALNS；
- 理论上能改善 AWBC 的 advantage 估计。

问题：

- 当前 critic 主要是 state value `V(s)`；
- 没有显式 `Q(s,a)`；
- 因此很难准确评价某个 expert action 是否真的优于当前 policy action。

结果：

- 作为稳定 critic 的想法合理；
- 但在目前实验里没有单独体现出强收益。

### 5.5 AWBC

AWBC 的基本形式：

```text
L_awbc = E[- w(s,a_expert) log pi(a_expert | s)]
```

其中：

```text
w = exp(A / tau), clipped
```

我们尝试过：

- positive-only advantage；
- weight cap；
- safe filter；
- focus buckets；
- all offline；
- R/RC-wide 或 C/RC-wide；
- cosine gate；
- conflict mode。

结果：

- 大多数曲线和 baseline 很接近；
- 有时 offline 梯度和 PPO 梯度方向冲突；
- `lambda` 大会拉坏 actor；
- `lambda` 小则几乎没有效果。

结论：

```text
直接 step-level imitation 不可靠。
```

关键原因：

```text
good final ALNS solution != every intermediate ALNS action is a good autoregressive label
```

### 5.6 Offline Final

我们试过把能想到的 offline trick 放在一起：

- expert progress reward；
- critic pretraining；
- AWBC；
- safe filter；
- focus sampling；
- all vs focused buckets；
- conflict diagnostics。

目录：

```text
Codex_Res_offline_final_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

观察：

- 没有明显超过 baseline；
- 到 500 update 左右趋势也很像；
- focus-only 也没有比 all 明显好；
- seed 影响仍然很大。

结论：

```text
把所有 offline trick 堆起来，但不解决 complementarity / bad teacher 问题，仍然不够。
```

### 5.7 Online ALNS DPO / Preference

我们也尝试过 route-level preference：

```text
ALNS route vs PPO route
```

目录：

```text
Codex_Res_online_alns_dpo_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

这比 step-level BC 更合理，因为 ALNS 本质输出完整 solution，而不是一步步 policy。

但目前问题是：

- online ALNS 代价高；
- route-level preference 仍可能把 actor 往 ALNS 分布拉；
- 如果不做 regret selection，仍会学到 bad teacher；
- 目前没有形成强证据。

结论：

```text
route-level preference 是值得保留的方向，但必须结合 regret-aware selection 和 adapter 隔离。
```

### 5.8 Dynamic Expert Pool

我们设计过：

```text
ALNS 3k -> 5k -> 8k -> 25k
```

每个 instance 保存 best-so-far expert solution：

```text
ExpertBuffer[i] = best ALNS route so far
```

当 PPO 超过当前 expert，则升级 expert 或移除该 instance。

这部分目前还没有完整异步集成进训练 loop。

原因：

- 工程复杂度更高；
- 需要先证明 static regret-aware selector 有价值；
- 否则动态专家池也可能只是更贵的 bad teacher。

---

## 6. 哪些东西似乎有用

### 6.1 纯 PPO 强 baseline

目前最可靠的仍是：

```text
routeevent + repair progress + objective/progress decomposed advantage + POMO eval
```

它稳定、可解释，而且经常接近或超过旧 baseline。

### 6.2 Decomposed Advantage

把 reward 拆成：

```text
objective
progress
```

再用：

```text
adv_objective_weight = 0.5
adv_progress_weight = 0.5
```

是有价值的，因为它让我们看到两个信号是否冲突。

### 6.3 Gradient Cosine Diagnostics

`GradSplitCos` 很有用：

```text
[GradSplitCos] objective_progress_cos=...
[ALNSBC] ppo_cos=...
```

它告诉我们：

- objective/progress 是否方向一致；
- offline imitation 是否和 PPO 冲突。

这直接支持了“不能盲目加 expert loss”的判断。

### 6.4 Base-Reference 多 seed 训练

当前 base-ref 800 update / 4 seed 是必要的。

它的作用是：

- 得到 `pi_ref`；
- 用于生成 regret label；
- 避免只用一个 seed 误判 RL vs ALNS 的胜负区域。

### 6.5 Residual Adapter 结构

已经实现但默认关闭的 adapter 是后续最有价值的结构基础。

它让我们可以做：

```text
主 actor 不动
expert 只学 residual correction
regret gate 控制 correction 强度
```

这比直接 finetune actor 安全。

---

## 7. 哪些基本失败或不够有证据

### 7.1 直接 BC / dense teacher

失败原因：

- ALNS 不是 sequential policy；
- step label 不一定是最优动作；
- actor distribution mismatch；
- bad teacher 会伤害 PPO 已学到的行为。

### 7.2 静态 expert delta PBRS

失败或不明显原因：

- expert solution 质量有限；
- local delta 不等于全局任务价值；
- scale 很敏感；
- PPO 后期可能已经超过该 expert。

### 7.3 AWBC

问题：

- 当前 `V(s)` 难以评价 `a_expert`；
- advantage proxy 噪声大；
- bad teacher filtering 不充分；
- 梯度可能和 PPO actor 负相关。

### 7.4 只手工 focus C-wide / RC-wide

这个思路作为分析有用，但作为训练策略不够。

原因：

```text
C-wide / RC-wide 只是粗粒度分布标签，不等于 ALNS 在这个具体 instance 上真的优于 PPO。
```

更好的方式是 instance-level regret predictor。

### 7.5 不同 eval set 的对比

之前一度混过：

```text
dataset/unanchored/Cus_50/pickle/...
dataset/unanchored/Cus_50/buffer_1k/pickle/...
```

这个会让曲线对比不严格。后续必须锁定同一 eval set。

---

## 8. 当前最推荐的方法

推荐方向：

```text
Regret-Aware Residual Distillation
```

或者论文里可以叫：

```text
Complementarity-Aware Offline-to-Online Policy Distillation
```

核心思想：

> ALNS 不是 teacher policy，而是 conditional residual teacher。它只在 PPO 弱、ALNS 强的 instance/state 区域里提供修正信号。

### 8.1 Stage 0：训练 base policy

先训练：

```text
pi_ref
```

当前就是：

```text
800 updates
300/500/800 checkpoint
4 seeds
buffer_1k eval
```

### 8.2 Stage 1：生成 regret label

在同一批 offline instances 上比较：

```text
J_ref(x_i)
J_ALNS(x_i)
```

EVRP-TW 是 minimization，所以：

```text
Delta_i = J_ref(x_i) - J_ALNS(x_i)
```

解释：

```text
Delta_i > 0  => ALNS 更好
Delta_i < 0  => PPO 更好
```

需要设置 margin：

```text
epsilon = 1% ~ 3% objective
```

避免 tie/noise。

### 8.3 Stage 2：训练 regret predictor

训练：

```text
p(ALNS wins | instance/state)
```

输入可以包括：

- encoder embedding；
- instance type；
- TW width / tightness；
- customer clustering；
- nearest charging station distance；
- demand/capacity pressure；
- battery pressure；
- early rollout entropy；
- early feasibility margin。

第一步先看：

```text
AUC > 0.65
```

如果 AUC 接近 0.5，说明 ALNS vs PPO 胜负不可预测，继续做复杂 adapter 意义不大。

### 8.4 Stage 3：训练 residual adapter

冻结或大部分冻结：

```text
PPO backbone
PPO actor
```

只训练：

```text
residual adapter
regret gate
small last-layer adapter
```

loss 只在：

```text
ALNS wins by margin
```

的样本上启用。

这样 expert 不会在 PPO 已经强的区域拉坏 actor。

### 8.5 Expert signal 优先级

建议优先级：

1. route-level preference；
2. selective AWBC；
3. step-level BC 只作为 ablation。

原因：

```text
ALNS 是完整 solution solver，不是逐步 autoregressive policy。
```

### 8.6 推理阶段

推理时不跑 ALNS，只用神经网络：

```text
logits_final = logits_actor + beta(x) * logits_adapter
```

其中：

```text
beta(x) = beta_max * p(ALNS wins | x)
```

如果 regret predictor 判断是 hard-for-RL instance，可以：

- 增大 POMO；
- 使用 beam；
- 增加 sampling budget；
- 打开 adapter bias。

这条路线工业上可实现，因为不会在 inference 时调用慢速 ALNS。

---

## 9. Versioned Expert Buffer

ALNS-3k 不是绝对 expert，只是一个 teacher version。

后续应该维护：

```text
ExpertBuffer[i] = best ALNS solution found so far for instance i
```

版本：

```text
ALNS-3k
ALNS-5k
ALNS-8k
ALNS-25k
```

更新规则：

```text
if J_ALNS_new[i] < J_best[i]:
    replace route
    recompute expert fields
    recompute regret labels
```

regret label 是 policy-dependent：

```text
Delta_i^k = J_{pi_k}(x_i) - J_{ExpertBuffer}(x_i)
```

所以 expert adapter 也应该是 relative to current PPO，而不是永久相信某个 ALNS-3k。

---

## 10. 为什么 adapter 需要 refresh

如果直接让 ALNS-3k 更新 encoder/actor，会出现：

```text
representation bias toward outdated expert
```

也就是：

- encoder 学了旧 expert 的模式；
- 后续 PPO 或更强 ALNS 出现时，representation 已经偏了；
- 继续 fine-tune 可能 residual drift。

更安全：

```text
冻结 PPO 主干
训练当前 adapter
后续 ALNS/PPO 更新后 refresh adapter
必要时把稳定知识 distill 回 backbone
再 reset adapter
```

核心句子：

> heuristic guidance should be modeled as dynamic residual correction, not static policy imitation.

---

## 11. 下一步具体实验顺序

### Experiment A：完成 base-ref

当前正在做：

```text
800 updates
4 seeds
300/500/800 checkpoints
buffer_1k eval
```

选 checkpoint：

- 如果 800 还在提升，用 800；
- 如果 500 更稳，用 500；
- 如果 300 已经足够且后面过拟合，用 300；
- 最好以同一 eval set 的 best checkpoint 为准。

### Experiment B：生成 RL vs ALNS regret CSV

对每个 `buffer_1k` instance 保存：

```text
instance_id
instance_type
time_window_policy
J_ref
J_ALNS
Delta = J_ref - J_ALNS
winner
margin
```

### Experiment C：regret predictor 可预测性

训练一个轻量 predictor，先只回答：

```text
ALNS 是否在这个 instance 上优于 pi_ref？
```

报告：

- AUC；
- accuracy；
- precision/recall；
- by R/C/RC + narrow/wide 的 confusion。

这是 gate 是否值得做的前置验证。

### Experiment D：selective residual adapter

比较：

1. PPO base；
2. PPO + all ALNS BC；
3. PPO + selective ALNS route preference；
4. PPO + selective residual adapter；
5. PPO + selective residual adapter + regret gate。

成功标准：

```text
selective adapter > PPO base
selective adapter > all ALNS
PPO strong buckets 不退化
ALNS strong buckets 有提升
```

### Experiment E：adaptive decoding

用 regret predictor 控制 inference budget：

```text
easy instance: greedy / small POMO
hard instance: larger POMO / beam
```

报告：

```text
same average inference time 下 objective 更低
```

这可能比训练时强行 distill 更容易出工业结果。

---

## 12. 当前重要路径

### 当前 base-ref 结果

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/Codex_Res_base_ref800_repair_progress_objprog_enc512_s75_ntraj50_nm64
```

### Stage A Reference Checkpoint

Stage A 已完成。后续 regret label、residual adapter、regret gate 实验默认以 GPU1 / seed2030 的 800-update base-ref run 中的 best checkpoint 作为 `pi_ref`：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth
```

同目录下的 `model_update0800.pth` 是严格第 800 次 update 的快照；若要做 “best vs final snapshot” ablation，再单独使用它。主线默认用 `best_model.pth`。

### 当前 base-ref eval

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
```

### 当前 plot

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/logs_result_base_ref800/base_ref800_curves.png
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/logs_result_base_ref800/base_ref800_curves_zoom.png
```

### ALNS buffer_1k 对比文件

```text
/data/Maojie/Github2/EVRP-TW-D-B/evrptw_gen/benchmarks/ALNS_Solver_MULTI/logs_Cus50_3000_32mul_1k_instance/alns_summary.csv
```

### Stage B Versioned Heuristic Buffer

Stage B 已经为 `buffer_1k` 构造了当前 ALNS-3k teacher buffer：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000.pkl
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000_summary.json
```

兼容旧 offline loader 的 progress 版本：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000_progress.pkl
```

构建脚本：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/build_stage_b_expert_buffer.py
```

当前统计：

```text
teacher_version = alns_3000
valid_records = 1024
invalid_or_unmatched = 0
C_narrow = 190
C_wide = 214
R_narrow = 193
R_wide = 216
RC_narrow = 114
RC_wide = 97
```

### Stage C Regret Labels

Stage C 使用 GPU1 / seed2030 的 800-update best checkpoint 作为 `pi_ref`，在同一个 `buffer_1k` 上跑 POMO-8 eval：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu1_seed2030_best_pomo8_rl_results.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu1_seed2030_best_pomo8_objective_results.csv
```

然后与 Stage B 的 ALNS-3k teacher buffer 合并，得到 regret/complementarity labels：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.pkl
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000_summary.json
```

构建脚本：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/build_stage_c_regret_labels.py
```

`pi_ref` eval：

```text
instances = 1024
solved_rate = 100%
mean J_ref = 957.023249
```

Label 定义：

```text
normalized_regret = (J_ref - J_ALNS) / J_ref
teacher_win if normalized_regret > 0.01
rl_win      if normalized_regret < -0.01
tie         otherwise
soft_usefulness = sigmoid((normalized_regret - 0.01) / 0.02)
teacher_gate = 0.30 * soft_usefulness
```

整体结果：

```text
teacher_win = 598 / 1024 = 58.40%
tie         = 101 / 1024 = 9.86%
rl_win      = 325 / 1024 = 31.74%
mean normalized regret = 0.01998
```

按 bucket：

```text
C_wide    teacher_win_rate = 89.72%, mean_regret =  0.0873
RC_wide   teacher_win_rate = 78.35%, mean_regret =  0.0509
R_wide    teacher_win_rate = 69.44%, mean_regret =  0.0297
C_narrow  teacher_win_rate = 56.84%, mean_regret =  0.0108
RC_narrow teacher_win_rate = 25.44%, mean_regret = -0.0292
R_narrow  teacher_win_rate = 22.28%, mean_regret = -0.0431
```

结论：ALNS-3k 确实不是全局 teacher，但它在 wide bucket，尤其是 `C_wide / RC_wide` 上明显更有指导价值；在 `R_narrow / RC_narrow` 上则经常输给 `pi_ref`，后续 adapter 必须 selective / gated。

### 之前生成的 RL vs ALNS 分析文件

```text
rl_vs_alns_3000_32mul_buffer1k.csv
rl_vs_alns_3000_32mul_buffer1k_by_dist.csv
rl_vs_alns_3000_32mul_buffer1k_by_tw.csv
rl_vs_alns_3000_32mul_buffer1k_by_dist_tw.csv
```

### 相关实验目录

```text
Codex_Res_offline_delta1A_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_offline_final_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_online_alns_dpo_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_offline_all_vs_nooffline_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_routeevent_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

---

## 13. 最终论文故事线

可以这样讲：

> Existing offline-to-online RL methods often assume that offline expert data provides broadly useful supervision.  In EVRP-TW neural combinatorial optimization, however, a learned PPO policy and an ALNS heuristic solver show complementary but conflicting strengths.  Direct imitation causes destructive interference because ALNS is not a universal autoregressive teacher.  We therefore propose a regret-aware residual distillation framework that identifies when heuristic solutions are beneficial, isolates heuristic supervision from the main actor, and transfers only conditional residual corrections.

中文版本：

> 现有 offline-to-online RL 往往默认离线 expert 数据整体有益。但在 EVRP-TW 里，PPO policy 和 ALNS heuristic 是互补但冲突的：PPO 在部分分布上更强，ALNS 在部分分布上更强。直接模仿 ALNS 会导致负迁移。我们提出 regret-aware residual distillation：先判断 ALNS 在哪里真的优于 PPO，再只用一个小 residual adapter 在这些区域修正 PPO，而不是让 expert loss 直接更新主 actor。

这一故事比：

```text
PPO + lambda * BC
```

更符合我们的实验观察，也更有理论和工业实现价值。

---

## Appendix: English Draft

下面保留此前英文草稿，作为写论文时的英文表达参考。

# EVRP-TW Online-Offline Experiments Summary

Last updated: 2026-05-13

This note summarizes what we tried in the online/offline EVRP-TW training line, what looked useful, what failed or was inconclusive, and what the current model/training stack looks like.  The goal is to keep the research story honest: the current evidence does not support "just add ALNS data to PPO"; it points toward a more conditional, regret-aware way of using heuristic solutions.

## 1. High-Level Conclusion

The main empirical lesson is:

> Static ALNS/offline data is not a universal teacher for the current PPO policy.  It helps on some instance regions, hurts or conflicts on others, and direct actor imitation often produces gradient conflict.

The best current direction is therefore not more tuning of `lambda * BC`, but a staged framework:

1. Train a strong PPO base/reference policy.
2. Evaluate that policy and ALNS on the same offline instance set.
3. Use the difference as a regret label: where does ALNS actually beat the policy?
4. Train a small residual/expert adapter only for those weak regions.
5. Keep the main PPO actor protected from direct expert gradients.

We have already added default-off model compatibility for this direction:

- `LogitResidualAdapter`
- `RegretGate`
- checkpoint milestones at 300/500/800
- base-reference 800-update training/plotting support

The current base-reference runs are in:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/Codex_Res_base_ref800_repair_progress_objprog_enc512_s75_ntraj50_nm64
```

The current base-ref eval set is:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
```

The current base-ref plot is:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/logs_result_base_ref800/base_ref800_curves.png
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/logs_result_base_ref800/base_ref800_curves_zoom.png
```

Important caveat: the old baseline curve currently added to the plot was evaluated on the older unanchored eval pickle:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl
```

So it is useful visually, but not a perfectly matched baseline for the current `buffer_1k` base-ref runs.

## 2. Current Model Architecture

### 2.1 Environment and State

The task is EVRP-TW with:

- 50 customers
- 12 charging stations
- depot + customers + charging stations as action nodes
- time windows
- service time
- demand/capacity
- battery/charging feasibility
- action masks for infeasible nodes

The environment is vectorized and POMO-style:

- `num_envs` parallel instances
- `n_traj` trajectories per instance
- evaluation commonly uses `test_agent=8`, i.e. POMO best-of-8.

Current base-ref training uses dynamic train config cycling:

```text
instance_type: R, C, RC
time_window_policy: narrow, wide
service_time_policy: cargoweight
cluster_number_policy: random
```

### 2.2 Policy Backbone

The model is implemented in:

```text
evrptw_gen/benchmarks/DRL_Solver/models/graph_attention_model_wrapper.py
```

The main stack is:

```text
raw env observation
  -> StateWrapper
  -> AutoEmbedding
  -> GraphAttentionEncoder
  -> Decoder
  -> action logits
```

The graph token order is:

```text
[depot, customers, charging stations]
```

The backbone builds an attention bias from:

- Euclidean distance matrix
- node type pair bias: depot / charging station / customer
- battery infeasibility masking through edge energy vs battery capacity

This is important because the network is not a plain Transformer over arbitrary tokens; it already receives routing-specific geometric and feasibility bias.

### 2.3 Actor

The actor is thin:

```text
Backbone(obs) -> logits
Categorical(logits) -> action
```

At eval time:

- `eval_decode_mode=sampling`
- `test_agent=8`
- aggregation is `best_of_8`

This is the "POMO best-of-8" result shown in most current plots.

### 2.4 Critic

The critic reads the decoder glimpse/context representation.

It can be:

1. single-head value:

```text
V(s)
```

2. multi-head value:

```text
V_objective(s), V_progress(s)
```

or in older modes:

```text
V_task(s), V_terminal(s), V_teacher(s)
```

The current strong routeevent/base-ref configuration uses decomposed reward advantage with:

```text
decomposed_reward_mode = objective_progress
adv_objective_weight = 0.5
adv_progress_weight = 0.5
```

This was useful because it lets us inspect and combine objective and progress signals separately, instead of forcing all reward shaping into a single scalar critic.

### 2.5 Residual Adapter and Regret Gate

We added default-off support for future regret-aware distillation:

```text
LogitResidualAdapter
RegretGate
```

Conceptually:

```text
logits_final = logits_actor + beta(x, s) * adapter(logits_actor)
```

where:

- `adapter(logits_actor)` is a small residual correction.
- `beta(x, s)` is either fixed or predicted by the regret gate.
- by default:

```text
use_residual_adapter = False
adapter_beta_max = 0.0
use_regret_gate = False
```

So current base-ref PPO is behaviorally unchanged.  The adapter exists so that later ALNS/offline knowledge can be added as a conditional correction instead of rewriting the main actor.

## 3. Current PPO Training Method

The current PPO loop is in:

```text
evrptw_gen/benchmarks/DRL_Solver/train.py
```

Core settings in the current base-ref runs:

```text
num_updates = 800
num_envs = 128
num_steps = 75
n_traj = 50
num_minibatches = 64
update_epochs = 5
accum_steps = 8
learning_rate = 4e-05
critic_lr = 3e-05
target_kl = 0.01
eval_freq = 20
eval_batch_size = 1024
checkpoint_milestones = 300,500,800
```

The training loop:

1. samples a batch from the current train config schedule;
2. runs vectorized rollouts;
3. computes reward components;
4. computes GAE;
5. applies PPO clipped policy loss;
6. applies value loss;
7. applies entropy regularization;
8. clips backbone and critic gradients separately;
9. logs gradient split diagnostics and objective/progress gradient cosine;
10. evaluates every `eval_freq` updates.

The base-ref line is intentionally pure PPO:

```text
use_offline_final = False
use_offline_awbc = False
use_offline_critic_pretrain = False
use_alns_teacher = False
use_online_alns_dpo = False
use_residual_adapter = False
```

This gives us a clean reference policy for the next regret-label stage.

## 4. Reward and PBRS Design

### 4.1 Original Single-Customer Repair Potential

The original conservative progress/repair idea was:

```text
h_single(s) = current_to_depot + sum_{unvisited customer j} single_customer_repair_dist_j
```

where `single_customer_repair_dist_j` is a single-customer serving cost based on depot-customer-depot style distance.

The failure penalty can be normalized by the remaining repair ratio:

```text
remaining_ratio = h_single(s_fail) / h_single(s_start)
```

This is academically interpretable as an upper-bound-like "serve each remaining customer independently" baseline.  It is crude, but stable and easy to explain.

### 4.2 Expert Delta Potential

We then tried to replace or mix this with expert-derived marginal costs:

```text
h_delta(s) = current_to_depot + sum_{unvisited customer j} expert_delta_j
```

and:

```text
h_mix(s) = (1 - alpha) * h_single(s) + alpha * h_delta(s)
```

In code this is controlled by:

```text
use_expert_delta_potential
expert_delta_alpha
```

The environment reads fields such as:

```text
expert_customer_delta_norm
expert_delta_j_norm
customer_marginal_delta_norm
expert_customer_delta
expert_delta_j
customer_marginal_delta
```

If no valid expert delta is found, it falls back to the single-customer repair potential.

### 4.3 Expert Route Potential

We later added a route-level expert potential:

```text
h_route(s) = distance of remaining customers projected onto expert reference routes
```

Implementation idea:

1. keep expert/reference routes;
2. remove already served customers;
3. optionally keep charging stations;
4. collapse route and compute remaining route distance;
5. normalize by total expert route distance.

Controlled by:

```text
use_expert_route_potential
expert_route_alpha
expert_route_keep_stations
```

This was the more academically defensible route-based PBRS than treating each local edge marginal as a universal customer cost.  It says: "given an expert feasible route structure, progress is the reduction in remaining projected route cost."

### 4.4 What We Learned from PBRS Experiments

The single-customer PBRS remains the most stable baseline.

Expert delta/route potentials did not show robust improvement in the runs we inspected.  In particular:

- simple local edge delta was too local and not a reliable route objective proxy;
- expert route/delta shaping could be swamped by scaling differences;
- normalization and alpha mixing were necessary;
- even with normalization, gains were not clearly above seed variance;
- static expert potentials still depend heavily on expert solution quality.

## 5. Offline / ALNS Components Tried

### 5.1 ALNS Buffer and Delta Data

Offline data lives mainly under:

```text
dataset/unanchored/Cus_50/buffer
dataset/unanchored/Cus_50/buffer_1k
```

The multi-process ALNS reference we used came from:

```text
/data/Maojie/Github2/EVRP-TW-D-B/evrptw_gen/benchmarks/ALNS_Solver_MULTI
```

ALNS comparison for `buffer_1k` is associated with:

```text
/data/Maojie/Github2/EVRP-TW-D-B/evrptw_gen/benchmarks/ALNS_Solver_MULTI/logs_Cus50_3000_32mul_1k_instance/alns_summary.csv
```

We added/used offline fields such as:

- expert/customer delta values;
- expert route structures;
- progress metadata.

Important lesson: delta values must be treated as versioned and source-dependent.  A delta computed from a weak 3k-iteration ALNS route is not ground truth.

### 5.2 Offline Delta 1A

Experiment directory:

```text
Codex_Res_offline_delta1A_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

Main idea:

- use expert delta as PBRS/progress potential;
- keep actor loss mostly PPO;
- test different `expert_delta_alpha` values.

Observed result:

- no clear improvement over baseline;
- curves were very similar to baseline;
- seed effects were often larger than the offline signal;
- this made us suspicious that static expert progress shaping was too weak or misaligned.

Conclusion:

> Expert delta PBRS alone is not enough.

### 5.3 Critic Pretraining and Reward-to-Go Regression

Idea:

- replay expert routes;
- compute reward-to-go on expert states;
- pretrain critic before normal PPO/AWBC;
- avoid directly forcing the actor to imitate expert actions early.

Why it made sense:

- AWBC depends on a reasonable critic;
- if `V(s)` is inaccurate, advantage filtering can accept bad expert actions or reject good ones.

Observed result:

- useful as a diagnostic and stabilizing concept;
- did not by itself produce a visible performance lift in the runs we watched.

Limitation:

- current critic is state-value-based, not action-value-based, so it cannot directly judge `Q(s, a_expert)`.
- This weakens action-level advantage filtering.

### 5.4 AWBC

Idea:

```text
L_awbc = E[- w(s,a_expert) * log pi(a_expert | s)]
```

with weights based on estimated advantage:

```text
w = exp(A / tau), clipped
```

and often positive-only filtering.

Tried variants:

- all offline samples;
- focused buckets such as wide/hard cases;
- safe filter by comparing current policy vs expert;
- cosine gate / conflict logic;
- smaller coefficients.

Observed result:

- no robust improvement;
- when coefficient was large, it could pull the policy toward inferior or mismatched trajectories;
- when coefficient was small, the effect was mostly invisible;
- gradient cosine diagnostics often showed conflict between PPO and offline imitation gradients.

Conclusion:

> Step-level ALNS imitation is a poor default because ALNS is a solution-level heuristic, not necessarily a good autoregressive teacher.

### 5.5 Offline Final Runs

Experiment directory:

```text
Codex_Res_offline_final_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

Main idea:

- combine the best available offline tricks before dynamic expert:
  - critic pretraining / reward-to-go;
  - AWBC;
  - safe filter;
  - focus bucket sampling;
  - expert-delta progress reward.

Variants included:

- all offline instances;
- focused hard buckets;
- fixed hyperparameters vs different seeds.

Observed result:

- early curves looked very close to baseline;
- at around several hundred updates, no convincing advantage was visible;
- "all" sometimes looked slightly more active than focused subsets, but not enough to claim success;
- user stopped runs once the trend looked baseline-like.

Conclusion:

> Adding many offline losses without solving policy/expert complementarity did not reliably improve PPO.

### 5.6 Single PBRS vs Expert Route PBRS Mix

Experiment idea:

- GPU0/1: single-customer PBRS baseline.
- GPU2/3: offline route mix:

```text
0.7 * single_customer_repair + 0.3 * ALNS/expert_route_distance
```

Purpose:

- isolate reward shaping from AWBC;
- test whether expert route potential helps even without direct actor imitation.

Observed result:

- no clear advantage from the offline/expert mix;
- trend looked similar enough to stop.

Conclusion:

> Expert route PBRS is academically more interpretable than local delta, but the current static version still did not show clear empirical lift.

### 5.7 Online ALNS / DPO / Preference Attempts

We also explored route-level preference ideas:

```text
ALNS route vs PPO route
```

and online ALNS-DPO style updates:

```text
Codex_Res_online_alns_dpo_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

Why this was more sensible than step-level BC:

- ALNS produces full route solutions.
- Route-level objective comparisons are more faithful than treating every ALNS step as a correct label.

Observed state:

- implementation/logging exists;
- computationally heavier because ALNS improvements are slow;
- no strong final evidence yet that it improves the main PPO line.

Conclusion:

> Route-level preference is conceptually better than action BC, but it still needs regret-aware selection and careful isolation from the main actor.

### 5.8 Dynamic Expert Pool

We discussed a dynamic expert pool:

```text
ALNS-3k -> ALNS-5k -> ALNS-8k -> ALNS-25k
```

with:

- best-so-far route per instance;
- policy comparison every K epochs;
- upgrade expert when PPO catches current expert;
- remove expert if it is consistently worse than PPO.

This is not fully integrated as an asynchronous lifecycle in the current training loop.

Conclusion:

> This is still a recommended direction, but it should be paired with regret labels and residual adapters, not with global actor imitation.

## 6. Diagnostics and What They Told Us

### 6.1 Gradient Cosine

We added/used diagnostics such as:

```text
[GradSplitCos] objective_progress_cos=...
[ALNSBC] ... ppo_cos=...
```

These were useful because they showed whether two signals were cooperating or fighting.

Key observation:

> PPO actor gradients and offline/ALNS imitation gradients were often not aligned; sometimes cosine was negative.

This supports the interpretation that direct loss addition is structurally wrong:

```text
L = L_PPO + lambda * L_expert
```

If gradients conflict, tuning `lambda` only changes the damage size.  It does not solve the conflict.

### 6.2 Eval Consistency

A key practical lesson:

> Do not compare curves evaluated on different pickles as if they were the same benchmark.

Current base-ref runs use:

```text
dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
```

Some older baselines used:

```text
dataset/unanchored/Cus_50/pickle/evrptw_50C_12R.pkl
```

The plot can include older baseline visually, but strict claims need same eval data.

### 6.3 ALNS vs RL Complementarity

Comparisons by distribution suggested:

- RL can be strong on some distributions, e.g. certain R/narrow cases.
- ALNS can remain competitive or better on some harder/wider distributions.
- C-wide / RC-wide were plausible weak regions to inspect.

But focusing only on these buckets did not automatically improve training.  This suggests:

1. weak-region detection should be learned from actual regret labels;
2. fixed human bucket rules are too crude;
3. ALNS quality also matters.

## 7. What Seems Useful So Far

### 7.1 Strong Base PPO

The strongest reliable component is still the online PPO solver with:

- routeevent repair progress/objective decomposition;
- direct progress PBRS;
- decomposed objective/progress advantage;
- POMO best-of-8 eval.

This should remain the base policy.

### 7.2 Multi-Seed Base Reference

The current 800-update base-ref run with four seeds is useful because:

- it gives a cleaner reference policy;
- it saves checkpoints at 300/500/800;
- it supports later regret-label generation.

Current base-ref logs:

```text
Codex_Res_base_ref800_repair_progress_objprog_enc512_s75_ntraj50_nm64/
  result_gpu0_...seed1234_u800.txt
  result_gpu1_...seed2030_u800.txt
  result_gpu2_...seed2025_u800.txt
  result_gpu3_...seed3407_u800.txt
```

### 7.3 Diagnostics

Useful diagnostics:

- objective/progress gradient cosine;
- PPO/offline gradient cosine;
- per-distribution eval;
- RL vs ALNS CSV comparisons;
- POMO vs greedy distinction;
- full vs zoomed plot.

These helped prevent us from overclaiming small or seed-driven changes.

### 7.4 Residual Adapter Compatibility

The residual adapter / regret gate addition is useful because it lets us test the next idea without changing the base actor by default.

This is the right architectural direction:

```text
main PPO actor = protected capability
expert adapter = conditional correction
regret gate = when to apply correction
```

Current Stage-D code audit:

```text
Status: not ready to train as-is.
```

Already present:

- `Agent` supports `use_residual_adapter`, `adapter_beta_max`, and `use_regret_gate`.
- `LogitResidualAdapter` and `RegretGate` are default-off, so base PPO behavior is unchanged.
- Compatible checkpoint loading can initialize from `pi_ref` while skipping new adapter/gate tensors.
- Stage C labels provide `soft_usefulness`, `teacher_gate`, and `teacher_win / rl_win / tie` supervision.

Missing or insufficient:

- The main optimizer currently optimizes `agent.backbone` and `agent.critic`, but not `residual_adapter` / `regret_gate`.
- Existing ALNS preference code still calls `optim_backbone.step()`, so it updates the backbone directly and is not a decoupled Stage-D residual method.
- Current `LogitResidualAdapter` only receives scalar logits. It is useful as calibration, but too weak for real customer-level correction because it cannot condition on candidate/customer embeddings.
- `eval.py` currently has no adapter/gate CLI args, so an adapter checkpoint would not be evaluated with the correct model shape.
- Stage C labels are not yet consumed by `train.py`; a Stage-D label loader is needed.

Minimal Stage-D implementation:

```text
1. Add --use-stage-d-residual and --stage-c-label-path.
2. Load Stage C label records.
3. Initialize Agent with residual_adapter + regret_gate from pi_ref using compatible load.
4. Freeze PPO backbone / actor / critic.
5. Create a separate optimizer over adapter + gate only.
6. Train route-level preference on ALNS-win or soft-gated records.
7. Train regret gate toward soft_usefulness.
8. Add eval.py adapter/gate args.
```

Better second version:

```text
Replace scalar-logit adapter with contextual residual adapter:
residual = f(glimpse, candidate/action embedding, base logits)
```

This is the version likely needed for meaningful customer-level correction.

## 8. What Failed or Was Inconclusive

### 8.1 Blind Expert Imitation

Plain BC / dense teacher / action imitation is not reliable.

Reason:

```text
good final ALNS route != every intermediate ALNS action is the right autoregressive target
```

### 8.2 Static Expert Delta PBRS

Static expert delta did not clearly improve over single-customer repair PBRS.

Likely reasons:

- delta quality depends on expert route quality;
- local marginal cost is not necessarily a global objective proxy;
- scale/normalization is delicate;
- static expert routes become stale as PPO improves.

### 8.3 AWBC Without a Strong Action-Value Estimate

AWBC with a state-value critic is weak because:

```text
A(s, a_expert) is hard to estimate from V(s) alone
```

We can use one-step or Monte Carlo proxies, but they are noisy and policy-dependent.

### 8.4 Human-Defined Focus Buckets

Focusing on wide/hard buckets made sense diagnostically, but did not reliably improve results.

Reason:

```text
bucket label != ALNS actually better than PPO on this instance
```

The selector should be learned from regret labels.

### 8.5 Eval Mismatch

Some previous comparisons mixed:

- old unanchored eval pickle;
- buffer_1k eval pickle.

This can mislead interpretation.  Future claims should use one locked eval set.

## 9. Why Direct Offline-to-Online Failed Conceptually

The core issue is policy complementarity:

```text
PPO wins on some instances
ALNS wins on some instances
```

Therefore ALNS is not a universal teacher.

Direct actor loss assumes:

```text
ALNS action is good for all sampled states
```

but the actual situation is:

```text
ALNS action is useful only when ALNS is better than current policy
and only if the step-level action corresponds to useful policy behavior
```

This explains:

- negative gradient cosine;
- baseline-like curves;
- seed variance dominating offline improvements;
- no robust lift from larger offline trick bundles.

## 10. Recommended Next Method: Regret-Aware Residual Distillation

The most promising framework now is:

```text
Regret-Aware Residual Distillation
```

or:

```text
Complementarity-Aware Offline-to-Online Policy Distillation
```

### 10.1 Stage 0: Train Base Policy

Train a strong PPO reference policy:

```text
pi_ref
```

Current plan:

```text
800 updates
checkpoints at 300, 500, 800
eval every 20
4 seeds
```

This is what the current base-ref run is for.

### 10.2 Stage 1: Generate Regret Labels

On the same offline instance set:

```text
J_ref(x_i)  = objective from pi_ref
J_ALNS(x_i) = objective from best ALNS route
```

For minimization:

```text
Delta_i = J_ref(x_i) - J_ALNS(x_i)
```

Interpretation:

```text
Delta_i > 0  => ALNS better
Delta_i < 0  => PPO better
```

Use a margin:

```text
epsilon = 1% ~ 3% objective
```

to avoid noisy ties.

### 10.3 Stage 2: Train Regret Predictor

Train:

```text
p_psi(ALNS wins | x_i)
```

Possible inputs:

- encoder pooled embedding;
- instance statistics;
- time-window tightness;
- cluster/dispersion features;
- RS/customer ratio;
- demand/capacity pressure;
- battery pressure;
- early rollout entropy or feasibility margins.

First sanity check:

```text
AUC > 0.65
```

If regret labels are not predictable, complicated gating is probably not worth it.

### 10.4 Stage 3: Train Residual Expert Adapter

Freeze or mostly freeze the base PPO actor/backbone.

Train only:

- residual adapter;
- regret gate;
- optionally small last-layer adapter.

Expert loss applies only on:

```text
ALNS-win-by-margin instances
```

This avoids letting bad-teacher examples damage the base actor.

### 10.5 Expert Signal Choice

Preferred order:

1. route-level preference;
2. selective AWBC;
3. step-level BC only as a diagnostic baseline.

Route-level preference is preferred because ALNS is a solution-level solver.

### 10.6 Inference

At inference:

```text
logits_final = logits_actor + beta(x) * logits_adapter
```

where:

```text
beta(x) = beta_max * p_psi(ALNS wins | x)
```

For hard-for-RL instances, also consider:

- larger POMO;
- beam search;
- more sampling budget.

This is industrially implementable because ALNS is not run at inference time.

## 11. Versioned Expert Buffer

ALNS-3k is not a final expert.

Future expert data should be versioned:

```text
ALNS-3k
ALNS-5k
ALNS-8k
ALNS-25k
```

Maintain:

```text
ExpertBuffer[i] = best route found so far for instance i
```

When stronger ALNS becomes available:

```text
if J_ALNS_new[i] < J_best[i]:
    replace best route
    recompute expert fields
    recompute regret labels
```

Regret labels are policy-dependent:

```text
Delta_i^k = J_{pi_k}(x_i) - J_{ExpertBuffer}(x_i)
```

Therefore the expert head/adapter should be refreshed relative to the current policy, not treated as permanent ground truth.

## 12. Why Adapter Should Be Versioned or Refreshed

If ALNS-3k updates the shared encoder too strongly, it can bias representation toward outdated heuristic behavior.

This is dangerous:

```text
old expert bias -> representation lock-in -> later PPO/ALNS improvements harder to absorb
```

Safer approach:

```text
base PPO backbone mostly frozen
current residual adapter trained for current regret labels
periodically refresh adapter when teacher/policy changes
```

Long-term:

```text
PPO backbone
  -> adapter_k for current expert version
  -> optional consolidation
  -> reset/refresh adapter
```

The key idea:

> ALNS should be a conditional residual correction, not a replacement policy.

## 13. Concrete Next Experiments

### Experiment A: Finish Base Reference

Use the current base-ref runs:

```text
300 / 500 / 800 checkpoints
buffer_1k eval
POMO best-of-8
```

Pick reference checkpoint:

- 500 if 800 overfits or plateaus;
- 800 if still improving;
- best by matched buffer_1k eval if using fixed eval selection.

### Experiment B: RL vs ALNS Regret Dataset

For each `buffer_1k` instance:

1. evaluate `pi_ref`;
2. read ALNS best route/objective;
3. compute:

```text
Delta_i = J_ref - J_ALNS
```

Save:

```text
instance_id
instance_type
time_window_policy
J_ref
J_ALNS
Delta
winner
margin
```

### Experiment C: Regret Predictability

Train a lightweight predictor:

```text
features -> ALNS wins
```

Report:

- AUC;
- accuracy with margin;
- per bucket confusion matrix.

If AUC is low, stop and revisit data/expert quality.

### Experiment D: Selective Adapter

Compare:

1. PPO base only;
2. PPO + all ALNS BC;
3. PPO + selective ALNS route preference;
4. PPO + selective residual adapter;
5. PPO + selective residual adapter + regret gate.

Success criterion:

```text
selective residual > PPO base
selective residual > all ALNS
no degradation on PPO-strong buckets
```

### Experiment E: Adaptive Decoding

Use regret predictor to allocate more inference budget to hard-for-RL instances:

```text
easy: greedy or small POMO
hard: larger POMO / beam
```

Report objective at equal average inference budget.

This could be a strong industrial result even if training-time distillation gives modest gains.

## 14. Current Files and Artifacts

### Key Code

```text
evrptw_gen/benchmarks/DRL_Solver/train.py
evrptw_gen/benchmarks/DRL_Solver/DRL_train.py
evrptw_gen/benchmarks/DRL_Solver/envs/evrp_vector_env.py
evrptw_gen/benchmarks/DRL_Solver/models/graph_attention_model_wrapper.py
plot.py
```

### Relevant Result Directories

```text
Codex_Res_base_ref800_repair_progress_objprog_enc512_s75_ntraj50_nm64
Codex_Res_offline_delta1A_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_offline_final_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_online_alns_dpo_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_offline_all_vs_nooffline_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
Codex_Res_routeevent_repair_progress_objprog_enc512_s75_ntraj50_nm64_u1000
```

### Plot Artifacts

```text
logs_result_base_ref800/base_ref800_curves.png
logs_result_base_ref800/base_ref800_curves_zoom.png
```

### ALNS Comparison Artifacts

```text
rl_vs_alns_3000_32mul_buffer1k.csv
rl_vs_alns_3000_32mul_buffer1k_by_dist.csv
rl_vs_alns_3000_32mul_buffer1k_by_tw.csv
rl_vs_alns_3000_32mul_buffer1k_by_dist_tw.csv
```

## 15. Final Research Story

The most coherent story is:

> In EVRP-TW neural combinatorial optimization, a learned PPO policy and a heuristic ALNS solver are complementary but conflicting.  Direct imitation of heuristic solutions causes negative transfer because ALNS is not a universal sequential teacher.  We therefore train a strong PPO reference policy, identify where ALNS actually improves over it, and use heuristic information only as a conditional residual correction through a regret-aware adapter.  This preserves PPO strengths while allowing expert structure to help in weak regions.

This is much stronger than:

```text
PPO + lambda * BC
```

because it explains the failures we observed and gives a concrete route to avoid them.
