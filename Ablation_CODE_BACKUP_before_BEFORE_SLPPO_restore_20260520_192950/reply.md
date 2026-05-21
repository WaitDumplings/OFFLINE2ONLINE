According to a document from 2026-05-13，你们目前的实验记录已经足够支撑一个很明确的判断：“直接把 ALNS/offline signal 加到 PPO actor 上”不是主线，真正值得做的是“以 ALNS 为离线互补性 oracle 的 residual online-offline architecture”。你们记录里反复出现的现象包括：纯 PPO strong baseline 已经很强，offline 信号没有稳定超过 baseline，seed variance 有时比 offline component 更显著，step-level imitation 和 PPO 梯度冲突，ALNS 也不是所有 instance 都优于 RL。 ￼

我最终建议的顶会主线是：

CARD: Complementarity-Aware Residual Distillation

一句话定义：

Train a strong online PPO solver first, use slow heuristic solvers offline only to identify complementary weak regions, and distill heuristic advantages into a conservative residual adapter plus adaptive neural decoding, without allowing heuristic gradients to directly overwrite the PPO actor.

这不是：

PPO + λ BC

而是：

PPO reference policy
+ versioned heuristic buffer
+ regret / complementarity predictor
+ residual preference adapter
+ adaptive inference budget
+ periodic expert refresh

⸻

1. 先定性：你们的问题不是 offline data 不够，而是 teacher identity 错了

你们之前尝试过 expert delta PBRS、expert route PBRS、critic reward-to-go pretraining、AWBC、safe filter、focus bucket、route-level DPO-like update、online ALNS DPO 等，但整体没有稳定超过 PPO baseline。记录里已经指出：继续调 lambda_bc、AWBC coef、DPO coef 不是核心，因为根因是 PPO policy 和 ALNS heuristic 在不同 instance 子分布上互补，但不是一致。 ￼

这意味着 ALNS 不是：

universal expert

而是：

conditional complementarity oracle

这个 framing 很重要。你们不能再讲：

We use ALNS as expert demonstrations.

应该改成：

We use ALNS to reveal where the learned policy is weak and transfer only conditional residual corrections.

这个故事和 algorithm portfolio 文献是一致的。SATzilla 这类工作证明了一个核心事实：在复杂组合问题中，通常不存在单一 dominant solver，不同 solver 在不同 instance 上占优，因此 per-instance algorithm selection 是合理的。区别是你们不能在线跑 ALNS，所以要把这个 portfolio oracle amortize 到 neural policy 里。 ￼

⸻

2. 为什么已有路线失败：不是实现问题，是结构问题

2.1 Step-level BC 的假设是错的

ALNS 输出的是完整 route solution，不是 autoregressive policy。PPO actor 学的是：

π(a_t | s_t)

但 ALNS 的局部 action 顺序不一定对应一个好的 sequential decision trajectory。你们记录里已经明确：good final ALNS route != every intermediate ALNS action is the right autoregressive target。 ￼

这和 imitation learning 的经典问题一致。DAgger 的动机就是普通 expert-state imitation 在 learner-induced distribution 下会发生 covariate shift；它通过在 learner 访问的状态上重新聚合 expert labels 来缓解问题。但你们的 ALNS 太慢，不能在线大量 query，所以不能简单套 DAgger。 ￼

2.2 AWBC 在你们当前 critic 结构下不稳

AWBC 的思想本身合理：只模仿 advantage 高的 offline actions。AWAC 也确实展示了 offline data 可以加速 online RL fine-tuning。 ￼

但你们当前 critic 主要是 V(s)，没有可靠的 Q(s,a)，所以很难判断 a_ALNS 是否真的优于当前 policy action。你们记录里也指出：AWBC with state-value critic 很弱，因为 A(s, a_expert) 很难从 V(s) 单独估计，proxy 噪声大且 policy-dependent。 ￼

因此，继续在 AWBC 上调温度、clip、positive-only、safe filter，很可能只是继续做 noisy action-level supervision。

2.3 Offline RL 不是直接答案

CQL/IQL 这类 offline RL 方法强调 distribution shift 和 OOD action 估值问题：CQL 用 conservative Q 来避免 offline RL 中的过估计，IQL 避免显式评估 dataset 外动作。 ￼

但你们的问题不是标准 offline RL。你们有一个强 online PPO solver，也有一个慢 heuristic solver。ALNS 数据不是普通 behavior dataset，而是 solution-level, versioned, sometimes superior, sometimes inferior 的外部 heuristic output。直接把它当 offline RL dataset 会错过最关键的 complementarity 结构。

⸻

3. 最终架构：CARD

我建议方法叫：

CARD: Complementarity-Aware Residual Distillation

完整架构：

Stage A: Strong PPO reference policy π_ref
Stage B: Versioned heuristic buffer H
Stage C: Complementarity / regret predictor gψ
Stage D: Residual preference adapter rφ
Stage E: Adaptive neural decoder
Stage F: Periodic teacher refresh

推理时不跑 ALNS：

logits_final
=
logits_ref
+
β_i · q_t · Δlogits_adapter

其中：

β_i = instance-level complementarity gate
q_t = state-level risk gate, optional but recommended
Δlogits_adapter = residual correction over feasible actions

核心原则：

PPO actor 是主 solver
ALNS 只提供 conditional residual correction
adapter 不能覆盖 actor
gate 必须 conservative
inference 不调用 ALNS

⸻

4. Stage A：训练强 PPO reference policy

先保留你们当前最稳的 base-ref PPO 线：

routeevent
+ repair progress
+ objective/progress decomposed advantage
+ POMO best-of-8 eval

你们当前 base-reference 配置是 800 updates、4 seeds、300/500/800 checkpoints，并且关闭所有 offline / ALNS / adapter 组件，目的就是得到干净的 reference policy，用来后续生成 regret labels。 ￼

这里我建议：

π_ref = best checkpoint selected on locked buffer_1k eval set

当前 Stage A 已经完成。后续实验默认使用 GPU1 / seed2030 的 800-update base-ref run 中的 best checkpoint 作为 π_ref，而不是重新从头训练：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth
```

同目录下的 `model_update0800.pth` 是严格第 800 次 update 的快照，可以用于 ablation 或 sanity check；但若文中写 “800 epoch best / 800-update best”，默认指上面的 `best_model.pth`。

不要混用旧 eval pickle 和新 buffer_1k，否则顶会 reviewer 会抓这个。你们记录里已经提醒：旧 baseline 曲线来自不同 eval path，只能做视觉参考，不能作为严格对比。 ￼

⸻

5. Stage B：构造 versioned heuristic buffer

ALNS-3k 不能被叫作 ground-truth expert。它只是：

H^3k = current heuristic teacher version

所以维护：

ExpertBuffer[i] = best heuristic solution found so far for instance i

当前 Stage B 已经为 `buffer_1k` 生成了 ALNS-3k 的 versioned heuristic buffer：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000.pkl
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000_summary.json
```

同时保留了一个兼容旧 ALNS loader 的 progress 格式：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000_progress.pkl
```

生成脚本为：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/build_stage_b_expert_buffer.py
```

当前版本统计：`valid_records=1024`，`invalid_or_unmatched=0`，teacher version 为 `alns_3000`。bucket 分布为 `C_narrow=190, C_wide=214, R_narrow=193, R_wide=216, RC_narrow=114, RC_wide=97`。

每次 ALNS 从 3k 到 5k、8k、25k：

if J_ALNS_new[i] < J_best[i]:
    replace route
    update objective
    recompute regret label

你们记录里已经明确：ALNS-3k 不是 final expert，delta / route values 必须 versioned，weak 3k ALNS route 不是 ground truth。 ￼

这个设计解决两个问题：

1. ALNS 以后变强，数据不会过时
2. adapter 不会永远绑定旧 teacher

工业上也合理：ALNS 可以离线异步跑，线上永远只部署 neural solver。

⸻

6. Stage C：生成 regret / complementarity label

在同一批 offline instances 上评估：

J_ref^K(x_i)
J_H(x_i)

其中 J_ref^K 应该用你真实部署时的 neural budget，例如：

POMO best-of-8

因为你们 eval 本身用的是 sampling + best-of-8。 ￼

定义 normalized regret：

Δ_i = (J_ref^K(x_i) - J_H(x_i)) / J_ref^K(x_i)

EVRP-TW 是 minimization：

Δ_i > 0  => heuristic better
Δ_i < 0  => PPO better

设置 margin：

ε = 1% ~ 3%

构造 soft usefulness target：

u_i = sigmoid((Δ_i - ε) / τ)

不要直接 hard 0/1。原因是：

ALNS-3k 有噪声
PPO sampling 有方差
tie region 不应该强行监督

当前 Stage C 已经完成第一版。使用 GPU1 / seed2030 的 `best_model.pth` 作为 `π_ref`，在同一个 `buffer_1k` 上跑 POMO-8 eval：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu1_seed2030_best_pomo8_rl_results.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu1_seed2030_best_pomo8_objective_results.csv
```

然后与 Stage B 的 `expert_buffer_alns_3000.pkl` 合成 regret labels：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.pkl
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000_summary.json
```

当前 `π_ref` eval 统计：

```text
instances = 1024
solved_rate = 100%
mean J_ref = 957.023249
```

Stage C label 统计，margin = 1%，tau = 0.02：

```text
teacher_win = 598 / 1024 = 58.40%
tie         = 101 / 1024 = 9.86%
rl_win      = 325 / 1024 = 31.74%
mean normalized regret = 0.01998
```

按 bucket 看：

```text
C_wide    teacher_win_rate = 89.72%, mean_regret =  0.0873
RC_wide   teacher_win_rate = 78.35%, mean_regret =  0.0509
R_wide    teacher_win_rate = 69.44%, mean_regret =  0.0297
C_narrow  teacher_win_rate = 56.84%, mean_regret =  0.0108
RC_narrow teacher_win_rate = 25.44%, mean_regret = -0.0292
R_narrow  teacher_win_rate = 22.28%, mean_regret = -0.0431
```

这个结果支持后续 selective residual 思路：ALNS-3k 在 wide bucket 尤其是 `C_wide / RC_wide` 上确实更像 useful teacher；但在 `R_narrow / RC_narrow` 上更像 bad teacher，不能直接做全局 imitation。

⸻

7. Stage D：Complementarity predictor，不是普通 classifier

训练：

gψ(x_i) ≈ u_i

它输出的是：

β_i ∈ [0, β_max]

而不是 hard decision。

第一版建议做 instance-level gate：

β_i = β_max · gψ(x_i)

输入：

encoder pooled embedding
TW tightness
average TW width
customer clustering / dispersion
RS/customer ratio
nearest-RS distance statistics
battery pressure
demand/capacity pressure
depot-customer distance scale

你们记录里也建议先训练轻量 predictor，看 AUC、accuracy、precision/recall、按 R/C/RC + narrow/wide 的 confusion，并把 AUC > 0.65 作为 gate 是否值得做的前置验证。 ￼

我会更严格一点：

AUC > 0.65 是最低线
precision@top-30% > 0.75 更重要
ECE 要可接受

为什么 precision@top 更重要？因为 adapter 应该 conservative。False positive 会伤 PPO 强区；false negative 只是错过提升。

⸻

8. 解决你之前提出的 β 固定问题：双层 gate

你之前指出得很对：如果 regret label 是 route-level，那么直接训练 step-level β(x,s) 会导致每一步 label 都一样。这是不严谨的。

所以我建议最终版本用：

β_total(i,t) = β_instance(i) · q_risk(s_t)

8.1 β_instance：来自 ALNS-vs-PPO route-level regret

这个回答：

这个 instance 是否属于 ALNS-helpful / PPO-weak region？

训练信号来自：

Δ_i = J_ref - J_H

8.2 q_risk(s_t)：来自 online trajectory risk，不来自 ALNS route-level label

这个回答：

当前 state 是否处于容易失败 / 可行性紧张 / 后期崩溃的区域？

训练信号可以来自 PPO rollout 中已有的环境信息：

future failure
remaining repair potential ratio
battery slack
TW slack
mask sparsity
near-infeasible transition count
terminal penalty
progress/objective decomposed returns

这样就避免了用 route-level ALNS label 硬监督每个 state。

最终：

logits_final_t
=
logits_ref_t
+
β_instance(x_i) · q_risk(s_t) · Δlogits_adapter_t

这比单一 instance-level β 更强，也更合理：

instance-level gate 决定是否该相信 heuristic
state-level risk gate 决定什么时候需要 correction

第一篇实现可以先只做 β_instance，但如果你想冲顶会，我建议把 q_risk(s_t) 加进去。它不是很难，因为你们环境里已经有 battery、time window、mask、progress、failure 等信号。 ￼

⸻

9. Stage E：Residual adapter 训练目标

这是最关键部分。

9.1 不要训练主 actor

冻结：

PPO actor head
early encoder
most decoder parameters

训练：

small residual adapter
gate head
optional LoRA / last-layer adapter
optional risk head

你们现有模型已经有 LogitResidualAdapter 和 RegretGate，默认关闭时不改变 base-ref PPO；形式上已经支持 logits_final = logits_actor + beta(x,s) * adapter(logits_actor)。 ￼

Stage D 代码检查结果：当前还不能直接开始训练 Stage D，只是有部分“壳子”。

已经具备的部分：

```text
1. Agent 支持 --use-residual-adapter / --use-regret-gate。
2. base checkpoint 在开启 residual adapter 时可以 compatible load；adapter/gate 新参数随机初始化。
3. forward 里已经有 logits_final = logits + beta * residual(logits) 的路径。
4. Stage C 已经提供 teacher_gate / soft_usefulness / teacher_win labels。
```

主要缺口：

```text
1. 当前 optimizer 只包含 agent.backbone 和 agent.critic，不包含 residual_adapter / regret_gate。
2. 现有 ALNS preference update 仍然调用 optim_backbone.step()，会直接改 backbone，不符合 Stage D 的 decoupled residual 原则。
3. 当前 LogitResidualAdapter 只看单个 scalar logit，表达力很弱；它更像 logit calibration，不能真正根据 state/customer feature 学“该提升哪个 customer”。
4. eval.py 目前没有 adapter/gate CLI 参数；如果训练出 Stage D checkpoint，当前 eval.py 默认 Agent 结构无法严格加载含 adapter 的 checkpoint。
5. Stage C labels 目前还没有被 train.py 读取；需要一个 stage_d label loader，把 regret label record 转成 teacher/student route pair + gate 权重。
```

所以 Stage D 的最小可行实现应该是：

```text
1. 新增 --use-stage-d-residual / --stage-c-label-path。
2. 从 Stage C pkl/csv 读入 teacher_win 或 soft_usefulness 高的 records。
3. 初始化 Agent 时开启 residual_adapter + regret_gate，并从 pi_ref compatible load。
4. 冻结 PPO backbone / actor / critic；只训练 adapter + gate。
5. 使用 route-level preference loss：
      y+ = ALNS route, y- = pi_ref route
   权重用 teacher_gate 或 soft_usefulness。
6. gate 额外做 BCE/MSE：
      regret_gate(x) -> soft_usefulness
7. eval.py 增加 adapter/gate 参数，保证 Stage D checkpoint 可评估。
```

更稳的第二版实现则应该把 adapter 从 `scalar-logit MLP` 升级为 `contextual residual adapter`，输入至少包含 decoder glimpse 和 candidate/action embedding，否则 adapter 很难学到真正的 customer-level correction。

9.2 Adapter 输出什么？

输出：

Δlogits_t ∈ R^{num_nodes}

但必须满足：

1. infeasible action 永远保持 -inf
2. residual norm 被限制
3. residual 只改变 feasible action 的相对偏好

推荐实现：

Δlogits = Adapter(h_t, logits_ref)
Δlogits = mask_infeasible(Δlogits)
Δlogits = Δlogits - mean_feasible(Δlogits)
Δlogits = clip(Δlogits, -ρ, ρ)

最终：

logits_final = logits_ref + β_instance · q_risk · Δlogits

推荐：

β_max = 0.1 ~ 0.3
ρ = 1 ~ 3

一开始一定 conservative。

⸻

10. Preference loss：用 route-level residual DPO，不用 BC

ALNS 是完整 solution solver，所以训练目标应该是 route-level preference，而不是 step-level BC。

对每个 instance 构造 pair：

y+ = better solution
y- = worse solution

如果 ALNS wins by margin：

y+ = y_H
y- = y_ref

如果 PPO wins：

不做反向 DPO
只训练 gate -> 0, residual -> 0

这是保守策略，避免把 adapter 训练成“反 PPO”。

DPO 的核心优点是把 preference optimization 写成一个简单的分类式目标，不需要显式 reward model；BOPO/Preference Optimization for combinatorial optimization 也说明了 objective-value-induced preference 可以用于 NCO 训练。 ￼

我建议你的 residual preference loss 写成：

L_pref =
- w_i · log σ(
    α [
      (log π_final(y+ | x_i) - log π_ref(y+ | x_i))
      -
      (log π_final(y- | x_i) - log π_ref(y- | x_i))
    ]
)

重点是这两项：

log π_final(y | x) - log π_ref(y | x)

这不是让整个 policy 模仿 ALNS，而是让 residual adapter 对 better solution 提供更高 incremental support。

这和普通 DPO 的区别是：

普通 DPO：更新整个 policy
CARD residual DPO：只更新 residual adapter / gate，reference actor 冻结

这就是你们的创新点之一。

⸻

11. 总训练目标

最终 Stage 2 loss：

L_CARD
=
λ_gate L_gate
+
λ_risk L_risk
+
λ_pref L_residual_pref
+
λ_KL L_KL
+
λ_res L_residual_norm
+
λ_zero L_zero_on_PPO_win

逐项解释：

11.1 Gate loss

L_gate = BCE(gψ(x_i), u_i)

其中：

u_i = sigmoid((Δ_i - ε) / τ)

11.2 Risk loss

L_risk = BCE(q_risk(s_t), future_failure_or_high_slack_risk)

或者用 regression：

q_risk(s_t) ≈ normalized future repair difficulty

这部分不依赖 ALNS step label。

11.3 Residual preference loss

只在：

Δ_i > ε

启用。

11.4 KL anchor

L_KL = KL(π_final(.|s) || π_ref(.|s))

作用是保护 PPO policy manifold。DPO / RLHF 里 reference model 的作用之一也是限制 policy drift；你们这里更需要，因为实验已经看到 offline loss 会和 PPO actor 冲突。 ￼

11.5 Residual norm penalty

L_res = ||β · q · Δlogits||²

11.6 PPO-win zero residual

在：

Δ_i < -ε

的样本上训练：

β_instance -> 0
β_instance · q_risk · Δlogits -> 0

这比“anti-imitation ALNS”更稳。

⸻

12. 必须使用的 training tricks

Trick 1：Oracle portfolio upper bound 先算

在训练 CARD 前，必须先算：

J_oracle(x) = min(J_ref^K(x), J_H(x))

报告：

oracle gap = J_ref^K - J_oracle
ALNS-win ratio
PPO-win ratio
tie ratio

如果 oracle gap 很小，说明没有足够 complementarity，adapter 没有空间。这个实验是顶会 reviewer 会认可的 sanity check。

Trick 2：只在 high-confidence ALNS-win 上训 residual

不要用所有 ALNS-win。先用：

Δ_i > ε
and ALNS solution stable across versions/seeds

不稳定或 tie 的样本只训 gate calibration，不训 adapter。

Trick 3：trajectory replay 必须严格验证 feasible

训练 route-level logprob 时，要 replay ALNS route：

for a_t in y_H:
    step env with a_t
    check feasible mask
    accumulate log π(a_t | s_t)

如果 ALNS route 和当前环境状态编码、charging station 表示、service-time 规则不完全一致，直接 skip 或修复。否则你会引入隐形 label corruption。

Trick 4：credit mask，不让所有 step 平等反传

虽然是 route-level DPO，梯度仍然会分解到每一步 logprob。为了避免又变成“隐式 BC”，可以加 step mask：

m_t = 1 if entropy_ref(s_t) high
       or rank_ref(a_H) poor
       or risk(s_t) high
       else 0

也就是只在 PPO 不确定、风险高、或 ALNS action 明显不同的地方让 residual 起作用。

Trick 5：PCGrad 只用于 adapter，不用于 actor

如果你同时训练 gate、risk、preference、KL，梯度仍然可能冲突。PCGrad 的做法是在两个任务梯度内积为负时，投影掉冲突分量。 ￼

但我建议：

PCGrad only on adapter / gate parameters
actor frozen

不要再让 heuristic loss 影响主 actor。

Trick 6：adaptive decoding 比训练 adapter 更容易稳定出效果

推理时：

if gψ(x) low:
    greedy / POMO-4
if gψ(x) medium:
    POMO-8
if gψ(x) high:
    POMO-16 / beam / more samples

POMO 本身就是 NCO 中利用 multiple optima / multi-start 的强方法；EAS 也说明只更新少量参数或使用轻量 test-time search 能提升 NCO 求解质量。你们这里更工业：不做 per-instance gradient update，只做 amortized gate + budget allocation。 ￼

Trick 7：adapter refresh，不要无限累计训练

每次 ALNS buffer 或 PPO reference 更新后：

freeze current π_ref
reset or partially reset adapter
retrain adapter relative to current π_ref and H_best

不要：

adapter_3k -> adapter_5k -> adapter_25k 无限累积

否则会 residual drift。你们记录里已经明确：旧 expert 会造成 representation bias，安全做法是冻结 PPO 主干、训练当前 adapter、后续 refresh adapter，必要时再把稳定知识 distill 回 backbone。 ￼

⸻

13. 推理阶段：工业友好版本

线上完全不跑 ALNS。

Input instance x
    ↓
PPO backbone forward
    ↓
gψ(x) predicts complementarity score
    ↓
q_risk(s_t) predicts state risk
    ↓
logits_final = logits_ref + β_i q_t Δlogits
    ↓
adaptive POMO / beam sampling
    ↓
exact objective evaluation of generated candidates
    ↓
choose best feasible solution

工业优点：

1. ALNS 不参与 inference
2. latency 可控
3. 可以 fallback 到 β=0，恢复纯 PPO
4. 可以按 gate 动态分配 sampling budget
5. offline teacher 可以持续异步变强

这比在线 ALNS-DPO 或 runtime portfolio 更可部署。

⸻

14. 自我挑战：这个方案可能失败在哪里？

14.1 如果 regret predictor AUC 低怎么办？

如果：

AUC ≈ 0.5

说明 ALNS-vs-PPO 胜负无法从 instance features 预测。此时不要硬做 adapter。

fallback：

1. 加 early rollout features
2. 只做 adaptive decoding
3. 扩大 / 提升 ALNS expert quality
4. 重新检查 eval noise 和 PPO sampling variance

14.2 如果 adapter 提升 ALNS-win bucket，但伤 PPO-win bucket 怎么办？

说明 gate false positive 或 residual 太强。

解决：

β_max 降低
KL coef 增大
PPO-win zero residual loss 增大
precision@top gate threshold 提高
adapter 只在 high-confidence region 开启

目标不是全局 aggressive，而是 conservative improvement。

14.3 如果 route-level DPO 还是像 BC 一样拉坏 actor 怎么办？

三个保护：

actor frozen
reference-subtracted residual DPO
credit mask only on uncertain/risky steps

如果仍然不稳，就把 adapter 从 action residual 改成 decoding prior only：

只影响 sampling budget / beam ranking
不直接改 logits

这是保底版本。

14.4 如果 ALNS-3k 太弱怎么办？

那不是方法错，而是 teacher version 太弱。需要：

ExpertBuffer best-so-far
teacher confidence
only use stable ALNS-win samples
refresh with ALNS-5k/8k/25k

千万不要把 3k 当 ground truth。

CARD:
1. 训练强 PPO reference policy π_ref
2. 用同一 offline set 生成 PPO-vs-ALNS regret labels
3. 维护 versioned ExpertBuffer
4. 训练 calibrated complementarity gate gψ
5. 训练 risk gate q_risk(s_t)
6. 冻结 PPO actor，用 residual adapter 学 heuristic correction
7. 用 reference-subtracted route-level preference loss，而不是 step-level BC
8. 用 KL / residual norm / PPO-win zero residual 保护 base policy
9. inference 不跑 ALNS，只做 gate-controlled residual logits + adaptive POMO
10. ALNS 更新后 refresh adapter，而不是无限累计 fine-tune

这个方案最靠谱的地方在于：

它解释了你们已有失败结果；
它不假设 ALNS 永远正确；
它避免 actor 梯度打架；
它能利用 ALNS 但不在线运行 ALNS；
它有清晰的 oracle upper bound 和 gate AUC sanity check；
它能自然连接 algorithm portfolio、offline-to-online RL、DPO/preference optimization、NCO adaptive decoding 文献；
它可以工业部署。

我会把它作为你们现在的主线，而不是继续在 offline loss 上调参。

⸻

15. Stage D implementation status

Current code now includes the full first-version CARD Stage D path:

```text
--use-stage-d-residual True
--stage-c-label-path <regret_labels_*.pkl>
--stage-d-buffer-dir <buffer_1k>
--init-ckpt-path <pi_ref best_model.pth>
--use-residual-adapter True
--use-regret-gate True
--adapter-contextual True
--adapter-beta-max 0.3
```

Implemented pieces:

```text
1. Stage C label loader:
   reads teacher_win / rl_win / tie rows, reconstructs teacher route and pi_ref route,
   and matches each row back to the original buffer_1k instance.

2. Frozen-reference residual training:
   freezes PPO backbone / actor / critic and only trains residual_adapter + regret_gate.

3. Contextual residual adapter:
   the adapter now conditions on decoder glimpse/state context plus each feasible action logit.
   This is no longer a scalar-only logit calibration module.

4. Explicit complementarity gate:
   every Stage D batch also trains the gate at the initial state with Stage C soft_usefulness,
   and logs accuracy plus top-30% precision as a conservative selector sanity check.

5. Reference-subtracted route preference:
   y+ = ALNS route, y- = pi_ref route for teacher_win rows.
   Loss compares:
      [log pi_final(y+) - log pi_ref(y+)]
    - [log pi_final(y-) - log pi_ref(y-)]
   so the adapter learns incremental support instead of rewriting the actor.

6. State-route gate calibration:
   teacher_win rows train gate toward Stage C soft_usefulness;
   rl_win/tie rows train gate toward zero.

7. Conservative regularization:
   KL(pi_final || pi_ref), residual norm, and zero-residual loss on non-teacher-win rows.

8. Safe masked-logit adapter:
   expert and reference route logprobs are computed over the current state's feasible mask.
   masked/infinite logits are not fed through the adapter, avoiding NaNs during route replay.

9. Eval compatibility:
   eval.py can now instantiate residual_adapter/regret_gate and load Stage D checkpoints.
```

Smoke test passed with one full Stage D update on CPU:

```text
records=1024
pref=598
zero=426
miss=0
skip_seq=0
grad_norm finite
contextual=True
instance gate metrics printed
```

Recommended first real Stage D run:

```bash
cd /data/Maojie/Github2/EVRP-TW-D-B_Weekend

/home/npg/miniconda3/envs/maojie/bin/python -m evrptw_gen.benchmarks.DRL_Solver.DRL_train \
  --use-stage-d-residual True \
  --adapter-contextual True \
  --stage-c-label-path /data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.pkl \
  --stage-d-buffer-dir /data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k \
  --init-ckpt-path /data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth \
  --save-dir /data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_residual_alns3k_pi_ref_gpu1_seed2030 \
  --stage-d-updates 200 \
  --stage-d-batch-size 256 \
  --stage-d-zero-batch-size 256 \
  --stage-d-max-steps 100 \
  --adapter-beta-max 0.3 \
  --stage-d-lr 1e-4 \
  --cuda True \
  --cuda-id 1 \
  --debug True
```

⸻

16. Stage E implementation status

Stage E is now implemented in `eval.py` as adaptive neural decoding.  It does not call ALNS at inference time.

What Stage E does:

```text
1. Load a Stage D checkpoint with residual_adapter + regret_gate.
2. Probe each eval instance at the initial state with n_traj=1.
3. Compute the regret/complementarity gate score.
4. Assign each instance to low / mid / high tier by gate thresholds.
5. Allocate POMO budget per tier.
6. Scale residual strength per tier:
   low  -> beta can be 0, recovering frozen PPO
   mid  -> partial residual
   high -> full residual
7. Evaluate each tier separately and merge rows back into one CSV.
```

The new eval arguments are:

```text
--adaptive-decode True
--adaptive-budgets 4,8,16
--adaptive-thresholds 0.35,0.65
--adaptive-residual-scales 0.0,0.5,1.0
--adaptive-probe-batch-size 256
--adaptive-gate-csv <optional path>
```

The output CSV now includes:

```text
adaptive_gate
adaptive_tier
adaptive_budget
adaptive_residual_beta
```

Smoke test passed on a 2-instance temporary eval set:

```text
[StageE] adaptive decode enabled
[StageE] tier=mid | n=2 | budget=2 | residual_beta=0.1500 | gate_mean=0.4988
CSV columns include adaptive_gate / adaptive_tier / adaptive_budget / adaptive_residual_beta
```

Recommended Stage E eval command after Stage D training:

```bash
cd /data/Maojie/Github2/EVRP-TW-D-B_Weekend

/home/npg/miniconda3/envs/maojie/bin/python eval.py \
  --checkpoint_path /data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_residual_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/best_model.pth \
  --eval_data_path /data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl \
  --output_csv /data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_e_rl_results.csv \
  --objective_csv /data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_e_objective_results.csv \
  --adaptive-gate-csv /data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_e_gate_assignments.csv \
  --adaptive-decode True \
  --adaptive-budgets 4,8,16 \
  --adaptive-thresholds 0.35,0.65 \
  --adaptive-residual-scales 0.0,0.5,1.0 \
  --adaptive-probe-batch-size 256 \
  --eval_batch_size 256 \
  --eval-steps 100 \
  --decode_mode sampling \
  --n_encode_layers 2 \
  --value_heads 1 \
  --use-residual-adapter True \
  --use-regret-gate True \
  --adapter-contextual True \
  --adapter-beta-max 0.3 \
  --cuda True \
  --cuda-id 1
```
