# RGAE-PPO / POMO50 + ALNS5k 实验说明

更新时间：2026-05-17

本文档解释当前这组实验里的 `Exp1 / Exp2 / Exp3` 到底做了什么。目标是从“小白视角”说清楚：它们为什么设计成这样、公式是什么、代码里实际用了哪些参数，以及目前 `seed=3000/3001` 的结果如何。

---

## 1. 先说一句话版本

我们现在不把 ALNS 当成“每一步都该模仿的老师”，而是把 ALNS5k 当成一个高质量解库，也就是 archive / incumbent buffer。

PPO 仍然自己 rollout，自己产生 action，PPO loss 仍然只用当前 policy 采样出来的轨迹。ALNS 只做三件事：

1. 影响哪些 offline instance 更常被抽到训练；
2. 在同一个 instance 的 50 条 PPO trajectory 里，强化更好的那几条；
3. 当 PPO-POMO50 仍然明显弱于 ALNS5k 时，把 ALNS5k objective 当作 reference，给 PPO 一个质量比较信号。

当前三个实验可以理解成逐层加功能：

```text
Exp1 = regret-aware archive sampler
Exp2 = Exp1 + group-relative advantage
Exp3 = Exp2 + reference-conditioned advantage
```

---

## 2. 共同训练设定

四组实验分别是：

| GPU | 实验 | 说明 |
|---|---|---|
| GPU0 | Vanilla PPO 2-head | 没有 offline archive，只跑普通 online PPO |
| GPU1 | Exp1 sampler | 加 ALNS5k archive sampling |
| GPU2 | Exp2 group | Exp1 + 同 instance 内 trajectory ranking |
| GPU3 | Exp3 group+ref | Exp2 + reference-conditioned advantage |

共同配置：

| 参数 | 值 |
|---|---:|
| `num_updates` | `300` |
| `num_envs` | `128` |
| `num_steps` | `75` |
| `n_traj` | `50` |
| `eval_freq` | `10` |
| `eval_batch_size` | `1000` |
| `test_agent` | `8` |
| `num_minibatches` | `128` |
| `accum_steps` | `16` |
| train seed | `3000 / 3001 / 3002 ...` |
| eval decode | `sampling` |
| train config schedule | `batch_cycle` |
| stratify keys | `instance_type,time_window_policy` |
| fixed overrides | `service_time_policy=cargoweight,cluster_number_policy=random` |

注意：训练时是 `n_traj=50`，所以我们用 POMO50 的搜索结果估计 PPO 当前潜力；但 eval 目前还是 `test_agent=8`，也就是日志里的主 eval 是 `best_of_8`，不是 eval POMO50。

---

## 3. 当前 reward 和 PPO 基础 loss

### 3.1 Reward 组成

当前 reward 沿用原来的两头方案：

```text
objective head: 距离 / objective 相关 reward
progress head: 服务进度 / repair progress / feasibility scaffold 相关 reward
```

主要参数：

| 参数 | 值 |
|---|---:|
| `use_direct_progress_pbrs` | `true` |
| `progress_pbrs_coef` | `2.0` |
| `progress_pbrs_beta` | `0.5` |
| `use_repair_fail_reward` | `true` |
| `repair_fail_coef` | `1.0` |
| `repair_progress_coef` | `1.0` |
| `repair_success_bonus` | `1.0` |
| `use_decomposed_reward_adv` | `true` |
| `decomposed_reward_mode` | `objective_progress` |
| `adv_objective_weight` | `0.5` |
| `adv_progress_weight` | `0.5` |

所以基础 actor advantage 是：

```text
A_base = 0.5 * A_obj_norm + 0.5 * A_progress_norm
```

这里 `A_obj_norm` 和 `A_progress_norm` 都是通过 GAE 算出来后再归一化。

### 3.2 GAE 公式

对某个 reward head，代码里按标准 GAE 递推：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)

A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}

Return_t = A_t + V(s_t)
```

这里：

```text
gamma = args.gamma
gae_lambda = args.gae_lambda
```

注意：这些额外的 group/ref advantage 只用于 actor；critic 仍然拟合 environment/decomposed reward return。也就是说 group/ref 不直接污染 critic target。

### 3.3 PPO clipped objective

PPO 的 actor loss 仍然是 clipped surrogate：

```text
ratio_t = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))

L_actor = - mean(
    min(
        ratio_t * A_final_t,
        clip(ratio_t, 1 - eps, 1 + eps) * A_final_t
    )
)
```

关键点：`A_final_t` 会根据 Exp1/2/3 不同而变化，但 action 本身还是 PPO 当前 policy 采样出来的，不是 ALNS trajectory。

---

## 4. Offline archive 是什么

当前 archive 来自：

```text
dataset/unanchored/Cus_50/buffer
```

它已经从 ALNS 3k 更新到了 ALNS 5k。每条 offline record 至少包含：

```text
instance data
ALNS / incumbent route
incumbent_obj
incumbent_action_sequence
```

代码里叫 `incumbent`，可以理解为“当前这个 instance 已知的最好参考解”。初始化时它来自 ALNS5k。

如果训练中 PPO 在某个 offline instance 上找到更好的完整解，会在内存里更新当前 record：

```text
if J_ppo_best + incumbent_update_eps < J_incumbent:
    J_incumbent = J_ppo_best
    incumbent_action_sequence = PPO_best_sequence
    incumbent_source = "policy"
```

当前参数：

```text
incumbent_update_eps = 1e-6
use_self_generated_buffer = false
```

所以：offline record 可以被 PPO 改写成更好的 incumbent；但 fresh online instance 不会加入 self-generated buffer。

---

## 5. Exp1: regret-aware archive sampler

### 5.1 Exp1 想解决什么

Vanilla PPO 每轮只看 fresh online generated instances。Exp1 加入 20% archive instance：

```text
每个 update 有 128 个 env slot
20% 来自 ALNS5k archive，大约 26 个
80% fresh online，大约 102 个
```

参数：

| 参数 | 值 |
|---|---:|
| `incumbent_ratio` | `0.20` |
| `buffer_normal_frac` | `1.0` |
| `buffer_temp_frac` | `0.0` |
| `buffer_prefix_frac` | `0.0` |
| `online_normal_frac` | `1.0` |
| `online_temp_frac` | `0.0` |
| `temperature_sampling` | `1.0` |

也就是说当前 Exp1 没有高温采样、没有 expert prefix，只是 normal sampling。

### 5.2 为什么不是随机抽 archive

不是每个 ALNS instance 都同样有价值。我们更关心当前 PPO 仍然弱于 archive 的 instance。

对一个 archive instance，PPO 用 `n_traj=50` 跑 50 条 trajectory，得到：

```text
J_1, J_2, ..., J_50
```

objective 越小越好。定义：

```text
J_best = min_k J_k
J_mean = mean_k J_k
J_ref  = incumbent_obj
```

当前 margin 是：

```text
margin = pomo50_regret_margin_abs + pomo50_regret_margin_rel * |J_ref|
       = 0 + 0.01 * |J_ref|
```

也就是 1% gap。

再定义 beat rate：

```text
beat_rate = mean_k [ J_k < J_ref - margin ]
```

意思是：50 条 PPO trajectory 里面，有多少比例真的明显赢了 archive。

### 5.3 POMO50 regret-state 分类

代码里按下面规则给每个 archive instance 标状态：

```text
gap = J_best - J_ref
```

因为 objective 越小越好：

```text
gap < 0: PPO best 比 archive 更好
gap > 0: archive 比 PPO best 更好
```

分类规则：

```text
if gap < -margin:
    if beat_rate >= 0.20:
        state = stable_ppo_win
    else:
        state = lucky_ppo_win

elif gap > margin:
    if beat_rate <= 0.05:
        state = stable_alns_win
    else:
        state = uncertain

else:
    state = uncertain
```

参数：

| 参数 | 值 |
|---|---:|
| `use_pomo50_regret_state` | `true` |
| `pomo50_stable_beat_rate` | `0.20` |
| `pomo50_low_beat_rate` | `0.05` |
| `pomo50_regret_margin_rel` | `0.01` |
| `pomo50_regret_margin_abs` | `0.0` |

直觉解释：

- `stable_ppo_win`: PPO 不只是偶然一条好，而是至少 20% 的 trajectory 都能赢 archive。
- `lucky_ppo_win`: PPO best 赢了，但赢得不稳定，可能只是 50 条里一条撞上了。
- `stable_alns_win`: PPO-POMO50 也基本赢不了 archive，说明这个 instance 对 PPO 还有教学价值。
- `uncertain`: 两边差距不够明确。

### 5.4 采样权重公式

archive records 的采样概率不是均匀的，而是：

```text
p_i = w_i / sum_j w_j
```

其中权重 `w_i` 按状态设置。

当前运行参数：

| 参数 | 值 |
|---|---:|
| `use_regret_aware_sampler` | `true` |
| `buffer_regret_relative` | `true` |
| `buffer_regret_kappa` | `25.0` |
| `buffer_regret_margin` | `0.0` |
| `buffer_unknown_weight` | `1.0` |
| `buffer_uncertain_weight` | `0.5` |
| `buffer_ppo_win_weight` | `0.05` |
| `buffer_lucky_ppo_weight` | `0.35` |
| `buffer_alns_win_base_weight` | `1.0` |
| `regret_recompute_freq` | `10` |
| `regret_probe_size` | `256` |

具体规则：

```text
if state == stable_ppo_win:
    w_i = 0.05

elif state == lucky_ppo_win:
    w_i = 0.35

elif state == uncertain:
    w_i = 0.5

elif state == stable_alns_win:
    regret_rel = max(0, (J_best - J_ref) / |J_ref|)
    w_i = 1.0 + 25.0 * regret_rel

elif state unknown:
    w_i = 1.0
```

意思是：

- PPO 已经稳定赢的 archive instance，少抽；
- PPO 只是 lucky win 的，保留一点；
- 不确定的中等频率；
- ALNS 稳定赢 PPO 的，多抽，而且输得越多越常抽。

### 5.5 Exp1 的 final advantage

Exp1 不加 group/ref，所以 actor 用：

```text
A_final = A_base
```

但它和 Vanilla 的区别是训练数据分布变了：20% 来自 regret-aware archive。

---

## 6. Exp2: Exp1 + group-relative advantage

### 6.1 Exp2 想解决什么

同一个 instance 上 PPO 跑 50 条 trajectory。它们面对的是同一道题，所以 objective 可以公平比较。

如果某条 trajectory 比同 instance 的 50 条平均更短，就应该被更强 reinforce；如果比平均差，就应该被弱化。

这就是 group-relative advantage。

### 6.2 公式

对第 `i` 个 instance，第 `k` 条 trajectory 的 objective：

```text
J_{i,k}
```

对同一个 instance 的 50 条 trajectory 计算：

```text
mean_i = mean_k J_{i,k}
std_i  = std_k  J_{i,k}
```

因为 objective 越小越好，trajectory-level group advantage 定义为：

```text
A_group(i,k) = (mean_i - J_{i,k}) / (std_i + eps)
```

然后 clip：

```text
A_group(i,k) = clip(A_group(i,k), -3, 3)
```

如果某条 trajectory 更短：

```text
J_{i,k} < mean_i => A_group > 0
```

如果某条 trajectory 更长：

```text
J_{i,k} > mean_i => A_group < 0
```

这个值会加到这条 trajectory 的每一个 action step 上。

### 6.3 Exp2 的 final advantage

当前参数：

| 参数 | 值 |
|---|---:|
| `group_adv_coef` | `0.30` |
| `group_adv_clip` | `3.0` |
| `reference_adv_coef` | `0.0` |

所以：

```text
A_final = A_base + 0.30 * A_group
```

直觉解释：

- `A_base` 负责 step-level credit assignment。
- `A_group` 负责告诉 PPO：同一道题里，哪条完整路线更值得强化。

Exp2 没有直接使用 ALNS reference advantage；它只用了 archive sampler 和 PPO 自己 50 条 trajectory 的相对排名。

---

## 7. Exp3: Exp2 + reference-conditioned advantage

### 7.1 Exp3 想解决什么

Exp2 只比较 PPO 自己的 50 条 trajectory。Exp3 进一步问：

```text
这条 PPO trajectory 和当前 archive reference 比，质量如何？
```

这里 reference 是 `incumbent_obj`，初始化来自 ALNS5k；如果训练中 PPO 找到更好的 archive 解，也可能被更新成 PPO incumbent。

### 7.2 reference 什么时候启用

当前参数：

```text
reference_adv_alns_win_only = true
```

所以 reference advantage 只在：

```text
regret_state == stable_alns_win
```

时启用。

也就是说，如果 POMO50 已经说明 PPO 可以稳定赢 archive，就不让 archive 继续指导这个 instance，避免反向污染 PPO。

### 7.3 公式

对某个 offline instance：

```text
J_ref = incumbent objective
J_{i,k} = PPO 第 k 条 trajectory objective
```

reference advantage raw：

```text
A_ref_raw(i,k) = (J_ref - J_{i,k}) / (rho * |J_ref| + eps)
```

当前：

```text
rho = 0.05
```

然后 clip：

```text
A_ref_used(i,k) = clip(A_ref_raw(i,k), -2, 2)
```

解释符号：

```text
J_{i,k} < J_ref  => PPO 比 reference 更好 => A_ref > 0
J_{i,k} > J_ref  => PPO 比 reference 更差 => A_ref < 0
```

如果这个 instance 不是 offline archive，或者不是 `stable_alns_win`，则：

```text
A_ref = 0
```

### 7.4 Exp3 的 final advantage

当前参数：

| 参数 | 值 |
|---|---:|
| `group_adv_coef` | `0.30` |
| `group_adv_clip` | `3.0` |
| `reference_adv_coef` | `0.10` |
| `reference_adv_rho` | `0.05` |
| `reference_adv_clip` | `2.0` |
| `reference_adv_alns_win_only` | `true` |

所以：

```text
A_final = A_base + 0.30 * A_group + 0.10 * A_ref_used
```

更完整地写：

```text
A_base = 0.5 * A_obj_norm + 0.5 * A_progress_norm

A_group = clip((mean_i - J_{i,k}) / (std_i + eps), -3, 3)

A_ref_raw = (J_ref - J_{i,k}) / (0.05 * |J_ref| + eps)
A_ref_used = clip(A_ref_raw, -2, 2)

A_final = A_base + 0.30 * A_group + 0.10 * A_ref_used
```

---

## 8. Exp1/2/3 的区别总结

| 实验 | archive sampler | POMO50 regret-state | group advantage | reference advantage | final advantage |
|---|---|---|---|---|---|
| Vanilla | no | no | no | no | `A_base` |
| Exp1 | yes | yes | no | no | `A_base` |
| Exp2 | yes | yes | yes | no | `A_base + 0.30 A_group` |
| Exp3 | yes | yes | yes | yes | `A_base + 0.30 A_group + 0.10 A_ref` |

---

## 9. 当前没有启用的东西

为了避免误解，当前这三组实验没有启用下面这些：

| 功能 | 当前值 |
|---|---|
| route-level preference aux loss | off |
| selective BC | off |
| ALNS step-level imitation | off |
| DPO adapter | off |
| expert-prefix rollout | off |
| high-temperature sampling | off |
| suffix comparative advantage | off, `cmp_adv_coef=0.0` |
| self-generated online buffer | off |

对应参数：

```text
use_route_preference = false
use_selective_bc = false
use_self_generated_buffer = false
cmp_adv_coef = 0.0
buffer_temp_frac = 0.0
buffer_prefix_frac = 0.0
online_temp_frac = 0.0
temperature_sampling = 1.0
```

所以这组实验的结论比较干净：提升主要来自 sampler、group advantage、reference advantage。

---

## 10. 当前 seed3000 / seed3001 结果

### 10.1 Seed 3000 final @300

| 实验 | best_of_8 obj | trajectory_avg obj |
|---|---:|---:|
| Vanilla | 994.643 | 1068.348 |
| Exp1 sampler | 985.747 | 1050.623 |
| Exp2 group | 980.717 | 1040.194 |
| Exp3 group+ref | 972.500 | 1036.705 |

相对 Vanilla：

| 实验 | best_of_8 gain | trajectory_avg gain |
|---|---:|---:|
| Exp1 | +8.896 | +17.724 |
| Exp2 | +13.925 | +28.154 |
| Exp3 | +22.142 | +31.643 |

### 10.2 Seed 3001 final @300

| 实验 | best_of_8 obj | trajectory_avg obj |
|---|---:|---:|
| Vanilla | 984.280 | 1051.452 |
| Exp1 sampler | 978.525 | 1042.717 |
| Exp2 group | 973.467 | 1028.731 |
| Exp3 group+ref | 971.736 | 1030.534 |

相对 Vanilla：

| 实验 | best_of_8 gain | trajectory_avg gain |
|---|---:|---:|
| Exp1 | +5.755 | +8.735 |
| Exp2 | +10.813 | +22.721 |
| Exp3 | +12.544 | +20.918 |

### 10.3 两个 seed 的 final 平均

| 实验 | best_of_8 mean | trajectory_avg mean |
|---|---:|---:|
| Vanilla | 989.461 | 1059.900 |
| Exp1 sampler | 982.136 | 1046.670 |
| Exp2 group | 977.092 | 1034.463 |
| Exp3 group+ref | 972.118 | 1033.620 |

两 seed 平均相对 Vanilla：

| 实验 | best_of_8 mean gain | trajectory_avg mean gain |
|---|---:|---:|
| Exp1 | +7.325, 0.74% | +13.230, 1.25% |
| Exp2 | +12.369, 1.25% | +25.437, 2.40% |
| Exp3 | +17.343, 1.75% | +26.280, 2.48% |

---

## 11. 怎么解读这些结果

### 11.1 Exp1 有用说明什么

Exp1 只改变训练 instance 的采样分布，不改 actor advantage。

它提升说明：

```text
ALNS5k archive 确实包含对 PPO 有价值的 hard cases。
regret-aware sampler 能把 PPO 拉到更有训练价值的 instance 上。
```

### 11.2 Exp2 有用说明什么

Exp2 加了同 instance 内的 trajectory ranking。

它提升说明：

```text
n_traj=50 里确实存在质量差异。
同一道题内部比较，比跨 instance 混合比较更干净。
```

这也说明我们不需要模仿 ALNS action；只要强化 PPO 自己采样出的好 trajectory，就能提升。

### 11.3 Exp3 有用说明什么

Exp3 加了 reference-conditioned advantage，而且只在 stable ALNS-win 上启用。

它提升说明：

```text
ALNS5k 对 PPO-POMO50 仍然弱的 instance 有参考价值。
reference term 在 POMO50 gate 之后不会明显污染 PPO。
```

尤其 seed3000 和 seed3001 都显示 Exp3 的 best_of_8 很强，说明 reference signal 对搜索质量有帮助。

### 11.4 Exp2 vs Exp3

当前观察：

```text
Exp3 通常 best_of_8 最好。
Exp2 有时 trajectory_avg 接近甚至略强。
```

解释：

- `A_ref` 更偏向强化能接近/超过 archive 的高质量 trajectory；
- 这可能提升 best-of-8；
- 但如果 ref 太强，也可能增加 trajectory 分布的方差；
- 所以后续需要 sweep `reference_adv_coef`。

---

## 12. 当前推荐的下一步

当前主线已经比较清楚：

```text
reference-guided PPO exploration > ALNS step-level imitation
```

下一步建议：

1. 跑完 `seed3002`，形成 3 seed 统计。
2. 做 reference coefficient sweep：

```text
reference_adv_coef in {0.05, 0.10, 0.15}
group_adv_coef fixed at 0.30
```

3. 把 reference gate 从二值改成 adaptive：

```text
margin_i = (J_ppo_pomo50_best - J_ref) / J_ref

lambda_ref(i) = base_lambda_ref * clip(margin_i / 0.05, 0, 1)
```

4. 把 reference 来源从 ALNS-only 扩展成 best-of-archive：

```text
J_ref = min(J_ALNS5k, J_PPO_POMO50_archive)
```

5. 暂时不要做 BC / AWBC / DPO adapter，避免把已经有效的 PPO signal 搞混。

---

## 13. 最短总结

当前三组实验的逻辑是：

```text
Exp1: 让 PPO 多看自己仍然弱的 ALNS5k hard cases。
Exp2: 在同一道题的 50 条 PPO trajectory 里强化更短的路线。
Exp3: 当 PPO-POMO50 仍然明显弱于 archive 时，用 archive objective 做参考质量信号。
```

当前结果说明：

```text
sampler 有用；
group advantage 有用；
reference advantage 在 POMO50 gate 之后也有用。
```

这是一条比 ALNS step-level imitation 更稳的 offline 注入路线。
