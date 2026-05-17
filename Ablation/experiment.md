# EVRP-TW Online-Offline Experiment Log

更新时间：2026-05-13

本文档记录当前 online PPO 与 offline ALNS/CARD 方向的关键实验。重点放在最近的 CARD Stage-D residual adapter 实验：做了什么改动、训练配置、loss 走势、POMO-8 前后 objective 对比，以及当前判断。

---

## 1. 当前主线

我们已经不再把 ALNS 当作 universal expert 来直接模仿，而是采用更保守的框架：

```text
CARD = Complementarity-Aware Residual Distillation
```

核心思想：

```text
先训练强 PPO reference policy
再用 ALNS 和 PPO 的逐实例对比判断互补区域
只在 ALNS 明确优于 PPO 的区域训练 residual adapter
推理时用 gate 控制 residual logits 对 PPO logits 的小幅修正
```

当前主线对应：

```text
logits_final = logits_base + gate * adapter_beta_max * residual_logits
```

主 actor / PPO backbone 不直接被 heuristic loss 覆盖，ALNS 只提供条件式 residual correction。

---

## 2. Stage A: PPO Base Reference

Stage A 已完成。当前后续实验使用 GPU1 / seed2030 的 base-ref800 best checkpoint 作为 reference policy：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth
```

评估数据使用：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
```

这里的 reference 结果用于生成 Stage-C regret label。注意：旧 baseline 曲线如果来自不同 eval set，只能作为视觉参考，不能做严格对比。

---

## 3. Stage B: ALNS Expert Buffer

当前 offline teacher 使用 ALNS-3000 版本，不把它视为 ground truth expert，而是 versioned heuristic teacher：

```text
teacher version = alns_3000
```

主要文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000.pkl
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_b/expert_buffer_alns_3000_progress.pkl
```

当前统计：

```text
valid_records = 1024
invalid_or_unmatched = 0
```

后续如果 ALNS 从 3k 更新到 5k / 8k / 25k，应该维护 best-so-far expert buffer：

```text
if J_ALNS_new[i] < J_best[i]:
    replace route and objective
    recompute regret label
```

---

## 4. Stage C: Regret / Complementarity Labels

当前使用的 Stage-C label 文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_pi_ref_gpu1_seed2030_best_pomo8_vs_alns_3000.pkl
```

定义：

```text
regret_norm = (J_ref - J_ALNS) / J_ref
```

EVRP-TW 是 minimization，因此：

```text
regret_norm > 0  => ALNS better
regret_norm < 0  => PPO better
```

当前 label 分布：

```text
rl_win      = 325
teacher_win = 598
tie         = 101
total       = 1024
```

当前 high-confidence 条件：

```text
teacher_win and regret_norm >= 0.02
```

由此得到：

```text
pref_records = 555
zero_records = 469
low_conf_teacher_zero = 43
```

---

## 5. Stage D v3: Low-Confidence Zero Protection

### 5.1 修改内容

本轮改动目标：低置信度 teacher-win 不能完全不用，否则 gate 对这些区域没有保护训练。修改为：

```text
pref set:
    stage_d_use_for_pref == True

zero/protection set:
    stage_d_use_for_pref == False
```

这意味着：

- high-confidence ALNS-win 用于 residual preference training；
- RL-win / tie 用于 zero gate / zero residual protection；
- low-confidence ALNS-win 也进入 zero/protection set，让 gate 尽量靠近 0，避免不确定区域残差污染 policy。

对应代码位置：

```text
evrptw_gen/benchmarks/DRL_Solver/train.py
```

关键日志：

```text
[StageD] labels loaded |
rows=1024 | records=1024 | pref=555 | zero=469 | miss=0 |
low_conf_teacher_zero=43 |
skip_seq=0 | skip_soft=0 | skip_regret=43 | skip_bucket=0 |
min_regret_pct=0.0200 | focus_buckets=ALL |
classes: rl_win=325 teacher_win=598 tie=101
```

`py_compile` 已通过。

### 5.2 训练配置

本轮训练 200 updates：

```text
save_dir:
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_v3_hiconf2pct_lowconfzero_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12

stage_d_updates        = 200
stage_d_batch_size     = 128
stage_d_zero_batch_size = 128
stage_d_max_steps      = 100
stage_d_min_regret_pct = 0.02
adapter_beta_max       = 0.3
stage_d_lr             = 1e-4
cuda_id                = 1
```

输出：

```text
best_model:
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_v3_hiconf2pct_lowconfzero_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/best_model.pth

loss_csv:
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_v3_hiconf2pct_lowconfzero_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/stage_d_loss_log.csv

loss_plot:
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_v3_hiconf2pct_lowconfzero_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/stage_d_loss_plot.png
```

### 5.3 训练末尾 loss 状态

第 200 update：

```text
loss               = 0.838318
pref_loss          = 0.681416
gate_loss          = 0.645044
instance_gate_loss = 0.526672
kl_loss            = 0.000007
residual_loss      = 0.002884
zero_gate_loss     = 0.397179
zero_residual_loss = 0.000935
delta_mean         = 0.011569
gate_mean          = 0.572801
zero_gate_mean     = 0.290538
infeasible         = 0
grad_norm          = 0.465721
```

整体均值：

```text
pref_loss mean          = 0.689482
gate_mean mean          = 0.568562
zero_gate_mean mean     = 0.366271
kl_loss mean            = 0.000001
residual_loss mean      = 0.000587
delta_mean mean         = 0.003640
```

解释：

- gate 对 high-conf 样本能给到约 0.57；
- zero/protection 样本的 gate 被压到约 0.29；
- KL 非常小，说明 final policy 与 base policy 分布几乎重合；
- residual logits 对 policy 的实际扰动偏小。

---

## 6. Stage D v3 POMO-8 Eval

使用新 Stage-D v3 best model 在 `buffer_1k` 上做 POMO-8 evaluation：

```text
checkpoint:
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/checkpoint/card_stage_d_v3_hiconf2pct_lowconfzero_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/best_model.pth

eval data:
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl

decode_mode = pomo
test_agent  = 8
n_traj      = 8
eval_batch_size = 1024
```

输出：

```text
Average Objective = 957.043337
Solved Rate       = 100%
Eval Time         = 31.9756s
```

CSV：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v3_hiconf2pct_lowconfzero_pomo8_rl_results.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v3_hiconf2pct_lowconfzero_pomo8_objective_results.csv
```

---

## 7. Before / After Objective Comparison

对比：

```text
before = Stage-C 中的 J_ref
after  = Stage-D v3 best model POMO-8 objective
```

结果文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v3_hiconf2pct_lowconfzero_before_after_obj_delta.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v3_hiconf2pct_lowconfzero_before_after_summary.csv
```

### 7.1 总体结果

| group | n | before mean | after mean | after - before |
|---|---:|---:|---:|---:|
| ALL | 1024 | 957.023249 | 957.043337 | +0.020088 |

整体几乎没有变化，略微变差。

### 7.2 按 regret class

| group | n | before mean | after mean | after - before |
|---|---:|---:|---:|---:|
| rl_win | 325 | 1101.851245 | 1110.654895 | +8.803650 |
| teacher_win | 598 | 866.654546 | 861.926284 | -4.728262 |
| tie | 101 | 1026.046871 | 1025.917020 | -0.129851 |

结论：

- teacher_win 区域改善明显；
- rl_win 区域被伤害明显；
- tie 区域基本不变。

### 7.3 按 high-confidence pref 训练与否

| group | n | before mean | after mean | after - before |
|---|---:|---:|---:|---:|
| high-conf trained | 555 | 857.998839 | 852.958190 | -5.040649 |
| not high-conf | 469 | 1074.205654 | 1080.214461 | +6.008807 |

这是当前最重要的发现：

```text
high-confidence ALNS-win 子集确实被 residual adapter 改善了。
但非目标区域仍然被污染，抵消了整体收益。
```

### 7.4 Low-confidence teacher-win zero protection

| group | n | before mean | after mean | after - before |
|---|---:|---:|---:|---:|
| low-conf teacher zero | 43 | 978.373562 | 977.677265 | -0.696296 |
| others | 981 | 956.087405 | 956.138894 | +0.051490 |

low-confidence teacher-win 被加入 zero/protection 后没有明显坏掉，甚至小幅改善。但样本只有 43 个，不能作为强结论。

### 7.5 按 bucket

| bucket | n | before mean | after mean | after - before |
|---|---:|---:|---:|---:|
| C_narrow | 190 | 798.082935 | 798.730654 | +0.647720 |
| C_wide | 214 | 641.509819 | 639.857637 | -1.652182 |
| RC_narrow | 114 | 1085.766485 | 1087.502004 | +1.735518 |
| RC_wide | 97 | 882.448733 | 883.570505 | +1.121772 |
| R_narrow | 193 | 1331.932266 | 1334.292382 | +2.360117 |
| R_wide | 216 | 1039.977382 | 1037.611212 | -2.366171 |

当前改善主要出现在：

```text
C_wide
R_wide
teacher_win / high-conf subset
```

但在：

```text
R_narrow
RC_narrow
RC_wide
rl_win
```

仍然存在负迁移。

---

## 8. Logit / Gate 幅度诊断

当前 residual 分支公式：

```text
logits_final = logits_base + gate * adapter_beta_max * residual_logits
```

本轮训练后：

```text
pref gate_mean ≈ 0.57
zero gate_mean ≈ 0.29
adapter_beta_max = 0.3
```

因此实际 residual 系数大约是：

```text
high-conf: 0.57 * 0.3 ≈ 0.17
zero/protect: 0.29 * 0.3 ≈ 0.087
```

从 loss log 看：

```text
KL(final || base) ≈ 0.000007
residual_loss ≈ 0.0029
sqrt(residual_loss) ≈ 0.054
delta_mean ≈ 0.0116
```

判断：

1. residual/expert branch 确实生效，不是没走；
2. 但 logits 改动幅度很小，final policy 与 base policy 基本贴在一起；
3. 这种小幅改动足够在 high-conf teacher-win 区域产生改善；
4. 但它也足够在 rl_win 区域造成轻微污染；
5. 当前最大问题不是 residual 完全学不到，而是 gate/protection 不够硬，无法只在该开的地方开。

---

## 9. 当前结论

这轮 Stage-D v3 证明了一个重要点：

```text
Selective residual adapter 在选对的 high-confidence ALNS-win 区域是有效的。
```

但同时也说明：

```text
当前 gate / zero protection 还不够强，RL-win 区域仍会被 residual adapter 污染。
```

所以整体 objective 没有提升：

```text
ALL: 957.023249 -> 957.043337, +0.020088
```

但局部有效：

```text
high-conf trained: 857.998839 -> 852.958190, -5.040649
teacher_win:       866.654546 -> 861.926284, -4.728262
```

这说明 CARD 的方向有信号，但还不能直接作为最终提升方案。

---

## 10. 下一步建议

不要马上继续增加 Stage-D updates。更应该先做诊断实验：

```text
1. base vs residual top-1 flip rate
2. mean / max |delta_logits|
3. teacher action rank before / after residual
4. KL by regret_class
5. gate by regret_class
6. objective under beta = 0, 0.3, 0.6, 1.0
```

如果：

```text
beta 增大后 high-conf 明显更好，但 rl_win 更坏
```

说明 residual 有效，gate/protection 不够硬。

如果：

```text
beta 增大后仍没有明显变化
```

说明 residual head 本身没有学到足够强的 action-ranking correction。

后续优先改进方向：

1. 对 rl_win / tie / low-conf 设置更强 zero gate loss；
2. 推理时 hard threshold gate，而不是连续小 gate；
3. 单独报告 oracle portfolio gap，确认 complementarity 上限；
4. 增加 top-action flip / teacher-rank diagnostics；
5. 尝试 larger beta only on high-confidence predicted region；
6. 不让 residual 在 rl_win 区域参与 POMO sampling。

---

## 11. Check 1-3: 从最新讨论中提取并完成的关键 sanity checks

根据最新两组分析意见，最有价值的部分不是继续调 PPO/offline loss 参数，而是先验证三个问题：

```text
1. RL + ALNS 的 oracle portfolio 空间是否真实存在？
2. 这种 complementarity 是否能被 instance 特征预测？
3. ALNS-win 是否跨 PPO random seeds 稳定，而不是 seed2030 的偶然弱点？
```

这三项已经完成初步测试。

### 11.1 Check 1: Bucket selector baseline vs oracle

结果文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_bucket_selector_and_bucket_predictor.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_bucket_selector_and_bucket_predictor.md
```

Selector 结果：

| selector | mean objective | gain vs PPO | choose ALNS | choose PPO |
|---|---:|---:|---:|---:|
| Always PPO | 957.023249 | 0.000000 | 0 | 1024 |
| Always ALNS | 947.564667 | 9.458582 | 1024 | 0 |
| Bucket wide -> ALNS | 934.449246 | 22.574003 | 527 | 497 |
| Bucket C_wide+RC_wide -> ALNS | 941.098017 | 15.925232 | 311 | 713 |
| Oracle per-instance min(PPO, ALNS) | 923.034191 | 33.989058 | 658 | 366 |

关键判断：

```text
bucket rule 已经很强，但离 oracle 仍有明显差距。
```

这说明：

- complementarity 不是虚的；
- wide bucket 是强 prior；
- 但不能只做 hard bucket rule；
- 需要 instance-level regret predictor / gate。

Per-bucket oracle 细节：

| bucket | n | J_ref | J_ALNS | J_oracle | oracle gap % | ALNS win | PPO win |
|---|---:|---:|---:|---:|---:|---:|---:|
| C_narrow | 190 | 798.082935 | 790.676911 | 769.095082 | 3.6322 | 0.6053 | 0.3947 |
| C_wide | 214 | 641.509819 | 585.905714 | 583.121176 | 9.1018 | 0.9299 | 0.0701 |
| RC_narrow | 114 | 1085.766485 | 1118.478919 | 1075.035106 | 0.9884 | 0.3860 | 0.6140 |
| RC_wide | 97 | 882.448733 | 837.003788 | 831.572601 | 5.7653 | 0.8041 | 0.1959 |
| R_narrow | 193 | 1331.932266 | 1389.487283 | 1320.688956 | 0.8441 | 0.2798 | 0.7202 |
| R_wide | 216 | 1039.977382 | 1008.457284 | 1000.747692 | 3.7722 | 0.7778 | 0.2222 |

这进一步支持：

```text
bucket 可以做 prior / sampling weight，但不应该做硬规则。
```

### 11.2 Check 2: Bucket-only predictor

同一文件中还计算了 bucket-only predictor 的 AUC 和 top-k precision。它只使用粗粒度 bucket 信息：

```text
R / C / RC + narrow / wide
```

结果：

| ALNS-win margin | positive rate | AUC | P@10% | P@20% | P@30% |
|---:|---:|---:|---:|---:|---:|
| >0% | 0.6426 | 0.7209 | 0.8529 | 0.9268 | 0.8893 |
| >1% | 0.5840 | 0.7221 | 0.7843 | 0.8927 | 0.8599 |
| >2% | 0.5420 | 0.7197 | 0.7353 | 0.8683 | 0.8306 |
| >3% | 0.4775 | 0.7239 | 0.6863 | 0.8439 | 0.7915 |
| >5% | 0.3447 | 0.7531 | 0.5392 | 0.7707 | 0.6906 |

判断：

```text
bucket-only 已经是一个不弱的 baseline。
```

因此后续 learned gate 必须显著超过 bucket-only，才值得引入更复杂模型。

### 11.3 Check 3: Static regret predictor

结果文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_regret_predictor_static_features.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_regret_predictor_static_features.md
```

训练方式：

```text
5-fold CV logistic predictor
label = regret_norm > margin
```

特征：

```text
bucket:
    R/C/RC + narrow/wide one-hot

static:
    customer geometry
    TW width/start/end statistics
    demand/service statistics
    depot-customer distance
    nearest customer distance
    nearest charging-station distance
    battery/load pressure proxies

static_bucket:
    static + bucket one-hot
```

核心结果：

| margin | mode | pos rate | AUC | PR-AUC | P@10% | P@20% | P@30% |
|---:|---|---:|---:|---:|---:|---:|---:|
| >2% | bucket | 0.5420 | 0.7753 | 0.7687 | 0.8235 | 0.8732 | 0.8339 |
| >2% | static | 0.5420 | 0.8154 | 0.8203 | 0.9216 | 0.8732 | 0.8502 |
| >2% | static_bucket | 0.5420 | 0.8142 | 0.8195 | 0.9216 | 0.8732 | 0.8567 |
| >3% | bucket | 0.4775 | 0.7789 | 0.7346 | 0.8039 | 0.8488 | 0.7980 |
| >3% | static | 0.4775 | 0.8032 | 0.7846 | 0.9020 | 0.8585 | 0.7980 |
| >3% | static_bucket | 0.4775 | 0.8019 | 0.7823 | 0.9020 | 0.8537 | 0.8046 |
| >5% | bucket | 0.3447 | 0.7955 | 0.6458 | 0.7157 | 0.7756 | 0.6938 |
| >5% | static | 0.3447 | 0.8299 | 0.7241 | 0.8333 | 0.7805 | 0.6808 |
| >5% | static_bucket | 0.3447 | 0.8286 | 0.7216 | 0.8333 | 0.7756 | 0.6840 |

判断：

```text
ALNS-helpfulness 是可预测的，不是随机噪声。
```

尤其对 `regret_norm > 3%`：

```text
static AUC = 0.8032
P@10%      = 0.9020
```

这已经超过我们之前设定的 `AUC > 0.65` 继续做 CARD 的门槛。

### 11.4 Check 4: Multi-seed regret stability

为了避免 seed2030 的结果只是单个 PPO checkpoint 的偶然弱点，我们补评估了另外 3 个 base-ref seed，在同一 `buffer_1k` 上做 POMO-8：

```text
gpu0_seed1234 mean = 949.080311
gpu1_seed2030 mean = 957.023249
gpu2_seed2025 mean = 958.418001
gpu3_seed3407 mean = 957.948673
ALNS-3000 mean      = 947.564667
```

新增文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu0_seed1234_best_pomo8_objective_results.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu2_seed2025_best_pomo8_objective_results.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/pi_ref_gpu3_seed3407_best_pomo8_objective_results.csv
```

Stability 结果：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_multiseed_regret_stability_detail.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_multiseed_regret_stability_summary.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_check_multiseed_regret_stability.md
```

总体稳定性：

| margin | ALNS >=3/4 | ALNS 4/4 | PPO >=3/4 | PPO 4/4 | unstable |
|---:|---:|---:|---:|---:|---:|
| 0% | 602 | 477 | 334 | 223 | 88 |
| 1% | 543 | 415 | 275 | 189 | 206 |
| 2% | 488 | 362 | 229 | 164 | 307 |
| 3% | 421 | 316 | 203 | 144 | 400 |
| 5% | 303 | 207 | 149 | 104 | 572 |

Per-bucket, margin `>3%`：

| bucket | n | ALNS >=3/4 | ALNS 4/4 | PPO >=3/4 | PPO 4/4 | unstable |
|---|---:|---:|---:|---:|---:|---:|
| C_narrow | 190 | 73 | 55 | 41 | 29 | 76 |
| C_wide | 214 | 183 | 156 | 7 | 5 | 24 |
| RC_narrow | 114 | 13 | 6 | 39 | 27 | 62 |
| RC_wide | 97 | 54 | 40 | 5 | 4 | 38 |
| R_narrow | 193 | 13 | 6 | 93 | 75 | 87 |
| R_wide | 216 | 85 | 53 | 18 | 4 | 113 |

判断：

```text
high-margin ALNS-win 在多 seed 下相当稳定。
```

例如 margin `>3%` 时：

```text
stable ALNS >=3/4 seeds: 421 / 1024
stable ALNS 4/4 seeds:   316 / 1024
stable PPO >=3/4 seeds:  203 / 1024
```

这说明 Stage-D 第一版不应该继续用单 seed 的 555 个 `>2%` high-conf 样本，而应改成更稳的：

```text
positive adapter set:
    ALNS beats at least 3/4 PPO seeds by >3%
    count = 421

strong zero/protection set:
    PPO beats ALNS in at least 3/4 PPO seeds by >3%
    count = 203

uncertain / unstable:
    neither side wins in >=3/4 seeds
    count = 400
    use for gate calibration only, not residual preference
```

### 11.5 从两组意见中提取出的最有价值结论

最值得保留并继续测试的是：

```text
1. ALNS 不是 global teacher，而是 conditional complementarity oracle。
2. oracle portfolio gap 很大，combine 空间真实存在。
3. bucket 是强 prior，但不是足够好的 hard rule。
4. ALNS-helpfulness 可以被静态特征预测，AUC 已经超过 0.80。
5. high-margin ALNS-win 在多 PPO seeds 下稳定存在。
6. 下一版 Stage-D 应该用 stable multi-seed labels，而不是单 seed label。
7. RL-win / unstable samples 应该更强 zero/protection，避免 residual 污染 PPO 强区。
```

暂时不建议继续主攻：

```text
1. 全量 ALNS BC / DPO；
2. step-level AWBC；
3. 继续调 lambda_bc / tau / alpha；
4. hard-code wide bucket 使用 ALNS；
5. 直接让 heuristic loss 更新主 actor。
```

### 11.6 下一轮最小实验建议

基于以上检查，下一轮 Stage-D 应改成：

```text
Positive residual preference:
    stable_alns_win_3of4_margin3pct

Zero / protection:
    stable_ppo_win_3of4_margin3pct
    all unstable samples with weak zero/gate calibration

Gate target:
    multi-seed win frequency or averaged regret_norm

Adapter:
    actor frozen
    backbone frozen
    residual adapter only
    beta sweep: 0.1 / 0.2 / 0.3 / 0.6
```

同时，先做 inference-only adaptive decoding baseline：

```text
Use static regret predictor score:
    low score  -> POMO-4
    mid score  -> POMO-8
    high score -> POMO-16

Control same average budget as POMO-8.
```

这个实验风险最低，因为它不改 actor、不训练 adapter，但能先验证 gate 的工业价值。

## 12. Stage-D v4: multi-seed stable label residual adapter

### 12.1 实验目的

v3 使用的是单 seed `pi_ref_gpu1_seed2030` 的 high-confidence label。它在 teacher-win 子集上有明显改善，但在非 high-conf / RL-win 区域产生负迁移，整体 objective 没有提升。

v4 的目的，是把 positive residual preference 改成更稳的 multi-seed label：

```text
positive / preference:
    ALNS beats at least 3/4 PPO seeds by >3%

zero / protection:
    all remaining samples
    = stable PPO-win + unstable samples
```

### 12.2 Label 文件

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_multiseed_stable3of4_margin3pct_vs_alns_3000.pkl
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/dataset/unanchored/Cus_50/buffer_1k/stage_c/regret_labels_multiseed_stable3of4_margin3pct_vs_alns_3000.csv
```

样本数：

| class | count | usage |
|---|---:|---|
| teacher_win | 421 | residual preference |
| rl_win | 203 | zero/protection |
| unstable | 400 | zero/protection |
| total | 1024 | - |

### 12.3 训练设置

```text
init checkpoint:
    checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth

save dir:
    checkpoint/card_stage_d_v4_stable3of4_m3_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12

stage_d_updates = 200
stage_d_batch_size = 128
stage_d_zero_batch_size = 128
stage_d_lr = 1e-4
adapter_beta_max = 0.3
adapter_contextual = True
actor frozen
main backbone frozen
```

训练日志：

```text
checkpoint/card_stage_d_v4_stable3of4_m3_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/stage_d_loss_log.csv
checkpoint/card_stage_d_v4_stable3of4_m3_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/stage_d_loss_plot.png
```

训练末尾：

```text
best_loss = 0.815441
last_loss = 0.832272
pref_loss: 0.693 -> 0.681 左右
delta_mean: ~0.0001 -> ~0.0124
KL: still very small, around 1e-5
infeasible replay = 0
```

解释：

```text
residual preference 确实学到了方向，但整体 logit/route probability 改变量仍然很保守。
```

### 12.4 POMO-8 eval

Eval 文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v4_stable3of4_m3_pomo8_rl_results.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v4_stable3of4_m3_pomo8_objective_results.csv
```

总体结果：

| model | mean objective |
|---|---:|
| pi_ref gpu1 seed2030 POMO-8 | 957.023249 |
| Stage-D v4 POMO-8 | 957.066953 |
| ALNS-3000 | 947.564667 |
| Oracle min(ref, ALNS) | 923.034191 |

整体：

```text
Stage-D v4 after - ref = +0.043704
```

也就是基本持平，但没有超过 base reference。

### 12.5 分组前后变化

对比文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v4_stable3of4_m3_before_after_summary.csv
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v4_stable3of4_m3_before_after_obj_delta.csv
```

按 regret class：

| group | n | ref mean | after mean | after - ref |
|---|---:|---:|---:|---:|
| ALL | 1024 | 957.023249 | 957.066953 | +0.043704 |
| teacher_win | 421 | 811.188942 | 809.346821 | -1.842122 |
| rl_win | 203 | 1141.927111 | 1142.399586 | +0.472475 |
| unstable | 400 | 1016.675147 | 1018.486081 | +1.810934 |

按 bucket：

| bucket | n | ref mean | after mean | after - ref |
|---|---:|---:|---:|---:|
| C_narrow | 190 | 798.082935 | 798.743967 | +0.661032 |
| C_wide | 214 | 641.509819 | 639.867925 | -1.641894 |
| RC_narrow | 114 | 1085.766485 | 1087.502004 | +1.735518 |
| RC_wide | 97 | 882.448733 | 883.570505 | +1.121772 |
| R_narrow | 193 | 1331.932266 | 1334.292382 | +2.360117 |
| R_wide | 216 | 1039.977382 | 1037.701265 | -2.276118 |

### 12.6 判断

v4 比 v3 更符合 CARD 的理论设定：

```text
1. Positive set 使用 multi-seed stable ALNS-win，label 更干净。
2. teacher_win 子集确实改善了，说明 residual preference 是有效信号。
3. RL-win 子集的负迁移比 v3 小很多。
```

但 v4 仍然没有整体提升，主要问题是：

```text
1. unstable set 仍然明显变差，是整体被抵消的主因；
2. KL 和 delta 都很小，说明 residual 对 policy 的实际干预幅度偏保守；
3. zero/protection 目前主要压 residual magnitude，但没有充分保证 unstable 的 final policy 不变；
4. fixed beta=0.3 eval 仍然会在不该启用 residual 的样本上施加偏置。
```

当前结论：

```text
multi-seed stable label 是对的；
route-level residual preference 对 teacher_win 区域有用；
下一步关键不是继续全局加大 beta，而是让 beta/gate 在 unstable 和 PPO-win 区域更可靠地接近 0。
```

### 12.7 下一步建议

优先测试：

```text
1. inference 时使用 learned gate/adaptive beta，而不是 fixed beta=0.3；
2. 对 unstable samples 加 stronger zero-final-policy loss：
       KL(pi_final || pi_ref)
       beta * residual logits -> 0
       top-action preservation
3. beta sweep:
       beta_max = 0.05 / 0.1 / 0.2 / 0.3
   并按 teacher_win / rl_win / unstable 分别报告变化；
4. 先做 inference-only adaptive decoding：
       high predicted ALNS-helpfulness -> larger POMO budget
       low predicted ALNS-helpfulness -> normal/smaller budget
```

暂时不建议直接进入 Stage-E 大规模 PPO fine-tune，因为 Stage-D residual 本身还没有在整体 objective 上带来正收益。

### 12.8 Eval-time beta sweep

为了确认是不是 `adapter_beta_max=0.3` 太强，固定 v4 checkpoint，只在 eval 时改变 residual beta：

```text
checkpoint:
    checkpoint/card_stage_d_v4_stable3of4_m3_b128_alns3k_pi_ref_gpu1_seed2030/Cus_50_CS_12/best_model.pth

eval:
    buffer_1k
    POMO-8
    eval_steps = 100
```

结果文件：

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/card_stage_d_v4_beta_sweep_summary.csv
```

总体结果：

| beta | mean objective | after - ref |
|---:|---:|---:|
| 0.00 | 957.141056 | +0.117807 |
| 0.05 | 957.089430 | +0.066181 |
| 0.10 | 957.047076 | +0.023827 |
| 0.20 | 957.118883 | +0.095634 |
| 0.30 | 957.066953 | +0.043704 |

按 regret class，`beta=0.10`：

| group | n | after - ref |
|---|---:|---:|
| teacher_win | 421 | -1.905941 |
| rl_win | 203 | +0.525214 |
| unstable | 400 | +1.800453 |

判断：

```text
1. beta=0.10 是这组 sweep 中最好的，但仍然没超过 ref。
2. beta 变小并不能消除 unstable 的负迁移。
3. teacher_win 的改善稳定存在，大约 -1.8 到 -1.9。
4. 主要瓶颈仍然是 unstable / non-positive samples 的保护，而不是单纯 beta 太大。
```

这说明下一步比单纯调 beta 更重要的是：

```text
1. 使用 learned gate/adaptive beta，而不是所有 instance 固定 beta；
2. 对 unstable samples 增加 top-action preservation / final-policy KL；
3. 或者先做 inference-only adaptive decoding，避免 residual adapter 直接改 policy。
```

## 13. Stage-D v5: stronger protection + hard-gated residual

### 13.1 Motivation

上一轮 v4 的主要问题是：

```text
teacher_win 上 residual 有局部改善；
但是 rl_win / unstable 上仍然会被 residual 污染；
固定 beta sweep 不能解决这个问题。
```

因此 v5 做的不是继续简单调大 beta，而是把 Stage-D 训练改成更接近
“teacher 区域修正、非 teacher 区域保护”的形式。

### 13.2 Code changes

新增 protected-set 约束，只作用于 `zero_records`
（`rl_win`、`unstable`、以及低置信 teacher-win）：

```text
zero_kl_loss       = KL(pi_final || pi_ref)
zero_rev_kl_loss   = KL(pi_ref || pi_final)
zero_top_loss      = penalize decreasing pi_final probability of pi_ref top action
zero_top_change    = diagnostic, whether pi_final top action differs from pi_ref top action
```

新增训练参数：

```text
--stage-d-zero-kl-coef
--stage-d-zero-rev-kl-coef
--stage-d-zero-top-coef
```

新增 eval-only hard gate：

```text
--adapter-gate-threshold
```

当 threshold >= 0 时：

```text
if gate(s) >= threshold:
    logits_final = logits_ref + beta * residual
else:
    logits_final = logits_ref
```

这样 eval 时仍然使用同一个 POMO-8 协议，只是在 gate 认为可靠的状态上打开 residual。

### 13.3 Four optimized Stage-D runs

统一设置：

```text
base checkpoint:
    GPU1 seed2030 base-ref 800 epoch best

label:
    regret_labels_multiseed_stable3of4_margin3pct_vs_alns_3000.pkl

training:
    Stage-D updates = 200
    batch = 256
    zero_batch = 256
    max_steps = 100
    actor/backbone frozen
    trainable = regret_gate + residual_adapter
```

四组：

| run | idea |
|---|---|
| v5a_safe_b08 | beta=0.8, moderate preference, moderate protection |
| v5b_prefstrong_b10 | beta=1.0, stronger route-level residual preference |
| v5c_ultrasafe_b08 | beta=0.8, very strong zero/KL/top protection |
| v5d_strict4of4_b10 | beta=1.0, only 4/4 stable teacher-win used as positive residual data |

训练中观察：

```text
1. zero_top_change 基本保持 0，说明 protected set 的 top-action 没有被翻掉。
2. v5b/v5d 的 pref_loss 明显下降，route-level residual DPO 确实学到了 teacher/student route 区分。
3. v5a/v5c 更保守，pref_loss 下降较慢，但 zero gate 更稳定。
```

### 13.4 POMO-8 eval on buffer_1k

严格 eval 协议：

```text
eval_data:
    dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl

decode:
    POMO-8
    eval_steps = 100
    eval_batch_size = 1024
```

Base:

| model | mean objective |
|---|---:|
| base_ref_gpu1_seed2030 | 957.141056 |

Final/best Stage-D checkpoints with hard gate:

| run | gate threshold | mean objective | delta vs base |
|---|---:|---:|---:|
| v5c_ultrasafe_b08 | 0.45 | 957.167431 | +0.026375 |
| v5a_safe_b08 | 0.45 | 957.180543 | +0.039487 |
| v5b_prefstrong_b10 | 0.45 | 957.183390 | +0.042334 |
| v5b_prefstrong_b10 | 0.55 | 957.231691 | +0.090635 |
| v5d_strict4of4_b10 | 0.45 | 957.310724 | +0.169669 |
| v5b_prefstrong_b10 | 0.65 | 957.331352 | +0.190296 |
| v5d_strict4of4_b10 | 0.55 | 957.383654 | +0.242598 |

Intermediate checkpoints:

| run | update | gate threshold | mean objective | delta vs base |
|---|---:|---:|---:|---:|
| v5b_prefstrong_b10 | 50 | 0.45 | 957.108660 | -0.032396 |
| v5a_safe_b08 | 100 | 0.45 | 957.109799 | -0.031257 |
| v5b_prefstrong_b10 | 150 | 0.45 | 957.188207 | +0.047151 |
| v5b_prefstrong_b10 | 100 | 0.45 | 957.205890 | +0.064834 |

Full summary:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/Codex_Res_card_stage_d_v5_eval/stage_d_v5_eval_summary.csv
```

### 13.5 Interpretation

This is the first Stage-D version that produced a small positive result:

```text
v5b update50 hard-gate0.45:
    957.108660 vs base 957.141056
    improvement = -0.032396

v5a update100 hard-gate0.45:
    957.109799 vs base 957.141056
    improvement = -0.031257
```

但是提升很小，且不是最终 checkpoint 最好。后续继续训练会变差，说明：

```text
1. residual DPO 确实能改变 policy；
2. 但 route-level teacher imitation 很快会过拟合 teacher route distribution；
3. loss 下降不等价于 objective 下降；
4. hard gate 保护能减少灾难性污染，但仍不能保证 teacher_win 区域 objective 改善；
5. 需要 early stopping / validation objective，而不能用 Stage-D training loss 选 best checkpoint。
```

按 regret class 的现象也值得注意：

```text
final strong residual 对 unstable 有时会略有改善，
但 teacher_win 反而变差。
```

这说明当前 residual 学到的并不是“teacher-win 的可泛化结构”，更像是把某些 route-level logprob 拉高；在 POMO 解码下，这个 logit 改动可能打乱原策略已经稳定的 search distribution。

### 13.6 Updated conclusion

Stage-D v5 证明：

```text
1. 参数太保守确实会导致策略几乎不变；
2. 放大 residual 后，policy 会变化；
3. 但只靠 residual DPO + hard gate 还不够稳定；
4. 最好的 checkpoint 出现在 early stage，而不是 DPO loss 最低处；
5. 下一步需要把“融合”表述为 constrained / projected policy fusion，而不是继续无约束训练 residual。
```

更推荐的下一步：

```text
1. Stage-D 使用 validation objective early stopping；
2. residual update 不直接在原 logit 空间加，而是投影到 pi_ref 的 trust-region/tangent space；
3. 对 residual 使用 action-rank preserving constraint：
       只允许提高 teacher action，
       不允许降低 pi_ref top-k feasible actions 太多；
4. 训练 gate 时不仅预测 ALNS-win，还预测 residual 是否能 improve POMO objective；
5. 先做 inference-only adaptive decoding / budget allocation，因为它不改 actor，风险更低。
```

## 14. Stage-D v8: projected/protected fusion attempt

### 14.1 Motivation

v5/v6 说明 residual adapter 可以改变策略，但容易在 PPO-win 或 narrow bucket 上造成负迁移。v8 的目标是更接近“融合 / 投影”的思路：

```text
1. 主 actor 仍然冻结，保留 pi_ref 的主体能力。
2. residual 只在 high-confidence wide teacher-win 上学习。
3. narrow bucket 和 PPO-win / low-confidence 样本强制 residual/gate 接近 0。
4. 加强 top-k preservation，尽量不翻掉 pi_ref 原本高置信动作。
5. 推理时优先测试 hard gate，避免 soft gate 在低置信区域产生小但有害的 logit drift。
```

### 14.2 v8 training setup

四组配置：

| run | positive set / protection idea |
|---|---|
| v8a | wide buckets, regret > 3%, margin 0.007, top3 preserve, beta=1.6 |
| v8b | wide buckets, regret > 3%, margin 0.010, top3 preserve, beta=1.8 |
| v8c | wide buckets, regret > 5%, margin 0.010, top5 preserve, beta=2.0 |
| v8d | only C_wide/RC_wide, regret > 3%, margin 0.008, top5 preserve, beta=1.6 |

训练样本统计：

| run | pref samples | zero/protection samples | low-conf teacher moved to zero |
|---|---:|---:|---:|
| v8a | 319 | 705 | 102 |
| v8b | 319 | 705 | 102 |
| v8c | 266 | 758 | 155 |
| v8d | 237 | 787 | 184 |

训练过程中 zero gate 被明显压低，pref gate 保持更高，说明 gate/residual 至少学到了区分 high-confidence teacher region 与 protected region。

### 14.3 POMO-8 eval summary

Base reference:

| model | mean objective |
|---|---:|
| base_ref_gpu1_seed2030 | 957.141056 |

v8 结果：

| run | checkpoint | gate | mean objective | delta vs base | wins | losses | ties |
|---|---|---:|---:|---:|---:|---:|---:|
| v8a | update60 | hard 0.32 | 957.027197 | -0.113859 | 6 | 7 | 1011 |
| v8a | update60 | hard 0.35 | 957.029926 | -0.111130 | 4 | 5 | 1015 |
| v8a | update60 | soft | 957.041620 | -0.099436 | 6 | 6 | 1012 |
| v8b | update60 | soft | 957.041620 | -0.099436 | 6 | 6 | 1012 |
| v8c | update60 | soft | 957.041620 | -0.099436 | 6 | 6 | 1012 |
| v8a | update60 | hard 0.37 | 957.042269 | -0.098787 | 3 | 4 | 1017 |
| v8a | update60 | hard 0.33 | 957.049228 | -0.091828 | 4 | 6 | 1014 |
| v8b | update60 | hard 0.35 | 957.060730 | -0.080326 | 4 | 6 | 1014 |
| v8d | update60 | soft | 957.073562 | -0.067494 | 5 | 5 | 1014 |
| v8a | update60 | hard 0.30 | 957.123156 | -0.017900 | 8 | 15 | 1001 |

Full summary:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/Codex_Res_card_stage_d_v8_eval/stage_d_v8_eval_summary.csv
```

### 14.4 Instance-level diagnosis

Best v8 so far:

```text
v8a update60 hard-gate0.32:
    mean objective = 957.027197
    improvement vs base = -0.113859
    changed instances = 13 / 1024
    wins = 6
    losses = 7
```

Changed-region summary:

```text
Improved:
    teacher_win = 4
    unstable = 2
    buckets = R_wide 4, C_wide 1, RC_wide 1

Worsened:
    teacher_win = 5
    rl_win = 1
    unstable = 1
    buckets = R_wide 3, C_wide 3, RC_wide 1
```

Mean delta by bucket:

```text
C_narrow  = 0.000000
RC_narrow = 0.000000
R_narrow  = 0.000000
C_wide    = -0.151158
RC_wide   = -0.407083
R_wide    = -0.207206
```

Mean delta by regret class:

```text
teacher_win = -0.215325
unstable    = -0.114262
rl_win      = +0.097367
```

The key difference from earlier soft-gated variants is that narrow buckets are fully protected. The remaining failure is not broad policy collapse; it is mis-triggering inside wide buckets, including one `R_wide` RL-win instance with a large positive delta.

### 14.5 Interpretation

v8 supports the projected/protected fusion direction:

```text
1. hard gate is better than soft gate for this stage;
2. stronger protection can preserve narrow/PPO-strong regions;
3. the residual can still produce objective improvements in teacher-win wide buckets;
4. however, ALNS-win label alone is not enough to know whether the residual adapter will improve POMO objective.
```

The main bottleneck has shifted:

```text
Before v8:
    residual could damage broad protected regions.

After v8:
    broad protected regions are mostly safe,
    but wide-bucket residual activation is still noisy.
```

So the next useful target is not simply larger beta or more preference training. The next useful target is a second-stage selector:

```text
g_help(x) = predict whether enabling residual actually improves POMO objective
```

This differs from the current regret gate:

```text
current gate:
    predict whether ALNS beats pi_ref

needed gate:
    predict whether the learned residual helps pi_ref decoding
```

### 14.6 Next experiment candidates

Most promising next tests:

```text
1. Train a residual-helpfulness classifier using changed-instance labels from v5/v6/v8:
       label = objective_after_residual < objective_base

2. Add a validation objective early-stopping rule:
       choose checkpoint/threshold by held-out buffer split, not by training loss.

3. Inference-only adaptive decoding:
       use regret gate only to allocate POMO budget,
       without changing actor logits.

4. Use stricter hard gate plus allow larger beta only on top predicted-helpful cases:
       beta(x) = beta_max * I[g_regret high] * I[g_help high]

5. Keep zero/protection loss active for PPO-win and narrow buckets;
       this part worked and should not be removed.
```

### 14.7 Eval-only beta / hard-gate probes

After v8a hard-gate0.32 beta=1.6 became the best protected setting, we tested whether the residual was simply too weak.

| run | gate | beta | mean objective | delta vs base | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| v8a update60 | hard 0.32 | 1.6 | 957.027197 | -0.113859 | 6 | 7 | 1011 |
| v8a update60 | hard 0.32 | 2.0 | 957.055669 | -0.085386 | 6 | 8 | 1010 |
| v8a update60 | hard 0.32 | 2.4 | 957.033994 | -0.107062 | 9 | 9 | 1006 |
| v8d update60 | hard 0.32 | 1.6 | 957.090137 | -0.050919 | 2 | 4 | 1018 |
| v8d update60 | hard 0.35 | 1.6 | 957.101308 | -0.039748 | - | - | - |

Conclusion:

```text
Increasing beta does not monotonically improve objective.
beta=2.4 creates more wins, but also more losses.
C/RC-only training is safer but loses too many useful R_wide improvements.
```

Residual-vs-base oracle portfolio:

| candidate | candidate mean | oracle mean | oracle delta vs base | missed loss from imperfect gate |
|---|---:|---:|---:|---:|
| v8a h0.32 b1.6 | 957.027197 | 956.974330 | -0.166726 | +0.052867 |
| v8a h0.32 b2.4 | 957.033994 | 956.952619 | -0.188437 | +0.081375 |
| v8d h0.32 b1.6 | 957.090137 | 957.056893 | -0.084163 | +0.033244 |

This is the most important diagnostic from this round:

```text
1. Stronger beta does create more useful residual opportunities.
2. Stronger beta also creates larger/more harmful false positives.
3. Therefore the current bottleneck is not residual capacity.
4. The current bottleneck is selector quality: whether enabling residual improves POMO objective.
```

Next Stage-D direction should separate two predictors:

```text
g_regret(x):
    predicts whether ALNS beats pi_ref

g_help(x):
    predicts whether the learned residual will improve pi_ref decoding
```

The residual can be stronger only when both gates are high:

```text
beta(x) = beta_max * I[g_regret(x) high] * I[g_help(x) high]
```

This matches the projected-fusion interpretation: residual correction is allowed only in a low-risk subspace where it is likely to improve objective, while all other instances remain close to pi_ref.

### 14.8 Strong-beta threshold sweep

Because beta=2.4 had a better oracle upper bound but worse realized performance at hard-gate0.32, we tested stricter thresholds:

| run | gate | beta | mean objective | delta vs base | wins | losses | ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| v8a update60 | hard 0.37 | 2.4 | 957.020594 | -0.120462 | 6 | 5 | 1013 |
| v8a update60 | hard 0.37 | 2.8 | 957.026799 | -0.114257 | 6 | 7 | 1011 |
| v8a update60 | hard 0.37 | 3.0 | 957.026799 | -0.114257 | 6 | 7 | 1011 |
| v8a update60 | hard 0.35 | 2.4 | 957.036723 | -0.104333 | 7 | 7 | 1010 |
| v8a update60 | hard 0.40 | 2.4 | 957.054205 | -0.086851 | 4 | 3 | 1017 |
| v8a update60 | hard 0.40 | 2.8 | 957.060410 | -0.080646 | - | - | - |
| v8a update60 | hard 0.40 | 3.0 | 957.060410 | -0.080646 | - | - | - |

Current best v8 setting:

```text
v8a update60, hard gate = 0.37, beta = 2.4
mean objective = 957.020594
delta vs base = -0.120462
wins/losses/ties = 6 / 5 / 1013
```

Instance-level diagnosis of the current best:

```text
Improved:
    n = 6
    teacher_win = 3
    unstable = 3
    buckets = C_wide 3, R_wide 2, RC_wide 1
    avg delta = -24.605378

Worsened:
    n = 5
    teacher_win = 3
    unstable = 2
    buckets = C_wide 2, R_wide 2, RC_wide 1
    avg delta = +4.855833

Mean delta by protected buckets:
    C_narrow = 0
    R_narrow = 0
    RC_narrow = 0

Mean delta by regret class:
    rl_win = 0
    teacher_win = -0.172103
    unstable = -0.127244
```

Residual-vs-base oracle for the current best:

```text
candidate_mean = 957.020594
oracle_mean    = 956.996884
oracle_delta   = -0.144172
missed_loss_if_perfect_gate = +0.023710
```

Interpretation:

```text
1. beta=2.4 + stricter hard gate is better than beta=1.6.
2. beta above 2.4 does not help.
3. hard gate around 0.37 is the useful narrow window.
4. protected regions are now actually protected: no RL-win/narrow damage in the best setting.
5. Remaining losses happen inside wide teacher/unstable buckets, so the issue is local action/route compatibility rather than broad bad-teacher transfer.
```

This is the cleanest evidence so far that the projected residual-fusion idea is viable:

```text
safe regions stay unchanged,
teacher/unstable wide regions receive small targeted changes,
average objective improves modestly.
```

But the improvement is still modest and below the earlier less-safe v6 best. The next meaningful improvement likely requires training a selector on actual residual helpfulness, not only on ALNS-vs-PPO regret.

## 15. Q-head / Q-filtered offline neural-candidate BC

New clean workspace:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/Codex_Exp_qhead_offline5k
```

I did not delete previous experiment folders because they contain useful historical evidence. The new Q-head/QBC line is isolated in the clean folder above.

Latest strategy:

```text
1. Keep the strong PPO base as the deployable fallback.
2. Do not run ALNS at inference.
3. Add a standalone state-action Q-head as an offline evaluator / credit model.
4. Use Q to weight neural-candidate behavior cloning.
5. Compare the resulting new policy against the base checkpoint on buffer_1k.
```

Base checkpoint:

```text
checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth
```

Offline labels:

```text
Codex_Exp_goal5_offline5k/stage_c/neural_candidate_labels/base_vs_scale_residual_b15_selected_pomo8.pkl
```

Important implementation detail:

```text
Q-head is a shared per-node scorer, not Linear(hidden, num_nodes).
It outputs Q(s, a_j) for each feasible candidate node.
The base actor is not controlled directly by Q.
Q is only used as a credit signal for offline neural-candidate BC.
```

Four configs:

| run | filter | trainable scope | policy lr |
|---|---|---|---:|
| `qbc_v1_reswin_decoder_lr3e5_q120_bc120` | teacher-win only | decoder | 3e-5 |
| `qbc_v1_reswin_decoder_lr6e5_q120_bc120` | teacher-win only | decoder | 6e-5 |
| `qbc_v1_all_decoder_lr3e5_q120_bc120` | all records | decoder | 3e-5 |
| `qbc_v1_reswin_all_lr1e5_q120_bc120` | teacher-win only | broader trainable scope | 1e-5 |

Training observation:

```text
Q MC loss became small with objective-aligned targets.
Q route-rank was still weak: rank loss around 0.67-0.70, rank_acc mostly 0.1-0.25.
So this Q-head currently works more as local action credit than as a strong branch/routing selector.
```

Buffer-1k eval summary:

```text
base mean objective = 956.853545
best new objective  = 954.840173
best improvement    = -2.013372
```

Best checkpoint:

```text
Codex_Exp_qhead_offline5k/checkpoint/qbc_v1_reswin_all_lr1e5_q120_bc120/Cus_50_CS_12/update0080.pth
```

Top results:

| model | mean objective | delta vs base |
|---|---:|---:|
| `qbc_v1_reswin_all_lr1e5_q120_bc120_update0080` | 954.840173 | -2.013372 |
| `qbc_v1_reswin_decoder_lr6e5_q120_bc120_update0120` | 955.155879 | -1.697666 |
| `qbc_v1_reswin_decoder_lr3e5_q120_bc120_update0080` | 955.395015 | -1.458530 |
| `qbc_v1_reswin_all_lr1e5_q120_bc120_best_qbc` | 955.524908 | -1.328637 |
| `base_seed2030_pomo8_buffer1k` | 956.853545 | 0.000000 |

Conclusion:

```text
This is a real positive new-policy result, not an ALNS/RL oracle selector.
However, the gain is about 2.0 objective points, still below the target 5-10.
The best signal comes from teacher-win filtered offline neural-candidate BC with Q weighting.
The next bottleneck is better route/candidate-level Q supervision and validation-objective checkpoint selection.
```

## 16. Clean baseline + critic-conditioned Q-only

Concern:

```text
The model had become too cluttered with historical adapter/gate/expert modules.
To make the method interpretable, we returned to the baseline PPO Agent structure and added only an offline Q evaluator.
```

Clean workspace:

```text
/data/Maojie/Github2/EVRP-TW-D-B_Weekend/Codex_Exp_baseline_q_only
```

Architecture:

```text
Deployable model = ordinary baseline Agent checkpoint.
No residual adapter.
No regret gate.
No expert head.
Q-head is external during offline training only.
```

Q input:

```text
actor logits
critic value output
decoder glimpse context
candidate node embeddings
state-action features
```

Training:

```text
1. Load base checkpoint.
2. Train Q-head on offline neural-candidate/base route replay.
3. Use Q credit to do weighted BC on teacher-win neural-candidate actions.
4. Save normal Agent checkpoint.
```

Best buffer_1k result:

```text
base mean objective = 956.853545
new mean objective  = 954.839615
improvement         = -2.013930
```

Best checkpoint:

```text
Codex_Exp_baseline_q_only/checkpoint/baseq_v1_value_decoder_lr6e5_q120_bc080_credit15_s15/Cus_50_CS_12/update0060.pth
```

Top results:

| model | mean objective | delta vs base |
|---|---:|---:|
| `baseq_v1_value_decoder_lr6e5_q120_bc080_credit15_s15_update0060` | 954.839615 | -2.013930 |
| `baseq_v1_value_decoder_margin3_lr6e5_q120_bc080_best_qbc` | 955.453538 | -1.400007 |
| `baseq_v1_value_decoder_lr3e5_q120_bc080_credit15_s10_best_qbc` | 955.485535 | -1.368010 |
| `baseq_v1_value_decoder_lr3e5_q120_bc080_credit15_s10_update0060` | 955.531781 | -1.321764 |
| `baseq_v1_value_decoder_lr1e4_q120_bc060_credit20_s20_anchor08_best_qbc` | 955.565562 | -1.287983 |
| `base_seed2030_pomo8_buffer1k` | 956.853545 | 0.000000 |

Interpretation:

```text
The clean baseline+Q-only result reaches essentially the same best improvement as the more complex QBC line.
This is a strong signal that the extra historical modules are not currently needed for the observed gain.
The useful ingredient is Q-weighted offline neural-candidate BC.
```

Remaining issue:

```text
The result is positive but still about 2 points, not 5-10.
Q learns local MC credit, but route-level ranking is weak.
Next improvements should target Q supervision quality and high-confidence action filtering, while keeping the deployable architecture simple.
```

### 16.1 Attribution check: Q vs no-Q route-weighted BC

Question:

```text
Is the +2 improvement really from Q, or just from offline BC?
Is the mean improvement caused by only one or two lucky instances?
```

Per-instance check for best Q checkpoint:

```text
Q checkpoint:
Codex_Exp_baseline_q_only/checkpoint/baseq_v1_value_decoder_lr6e5_q120_bc080_credit15_s15/Cus_50_CS_12/update0060.pth

n = 1024
route changed     = 1024 / 1024
objective changed = 975 / 1024
improved          = 532
worsened          = 443
unchanged obj     = 49

base mean         = 956.853545
Q mean            = 954.839615
mean delta        = -2.013930

sum improvements  = -11929.095888
sum worsenings    = +9866.831422
```

Interpretation:

```text
The Q-trained policy is not improving because of one or two accidental samples.
It changes almost every route and changes objective on most instances.
But it also creates many negative shifts; the net gain is broad but noisy.
```

No-Q ablation:

```text
same teacher-win data
same decoder-only training
same lr = 6e-5
same 80 BC updates
Q credit disabled by setting:
    q_credit_floor = 1.0
    q_credit_scale = 0.0
    q_credit_cap   = 1.0
```

Best no-Q checkpoint:

```text
Codex_Exp_baseline_q_only/checkpoint/noq_v1_decoder_lr6e5_bc080_routecredit/Cus_50_CS_12/update0060.pth
```

No-Q result:

```text
base mean         = 956.853545
no-Q mean         = 953.727651
improvement       = -3.125894

route changed     = 1020 / 1024
improved          = 541
worsened          = 421
unchanged obj     = 62
```

Direct Q vs no-Q:

```text
Q mean            = 954.839615
no-Q mean         = 953.727651
no-Q minus Q      = -1.111964

no-Q better than Q = 471 instances
Q better than no-Q = 409 instances
tie                = 144 instances
```

Conclusion:

```text
The current positive result should NOT be attributed to Q.
The useful signal is offline teacher-win neural-candidate BC / route-level weighting.
Current Q credit is noisy and weaker than the simpler no-Q route-weighted BC.
```

Updated best clean offline result:

```text
base mean objective = 956.853545
best clean offline  = 953.727651
improvement         = -3.125894
```

Current recommendation:

```text
Use no-Q route-weighted BC as the current clean baseline for offline improvement.
Do not claim Q as useful yet.
If we revisit Q, first prove action-level Q ranking quality with a direct same-state action/ranking sanity check.
```

### 16.2 Move main diagnostics from 1k to 5k

Decision:

```text
Future tests should use the 5k buffer as the main diagnostic/eval set.
The 1k buffer is useful for smoke tests, but 5k gives more stable conclusions.
```

5k eval data:

```text
dataset/unanchored/Cus_50/buffer/pickle/evrptw_50C_12R.pkl
```

Core comparison on 5k:

| model | mean objective | delta vs base |
|---|---:|---:|
| no-Q route-weighted BC | 825.099520 | -2.285805 |
| Q-weighted BC | 825.954662 | -1.430664 |
| base | 827.385326 | 0.000000 |

5k per-instance reliability:

```text
Q:
    route_changed      = 5110 / 5120
    objective_changed  = 5051 / 5120
    improved           = 2709
    worsened           = 2342
    median_delta       = -1.111197
    bootstrap95 mean   = [-2.248918, -0.593851]

no-Q:
    route_changed      = 5103 / 5120
    objective_changed  = 5040 / 5120
    improved           = 2772
    worsened           = 2268
    median_delta       = -1.540446
    bootstrap95 mean   = [-3.082205, -1.494331]
```

No-Q vs Q on 5k:

```text
no-Q mean - Q mean = -0.855141
no-Q better than Q = 2552
Q better than no-Q = 2411
tie                = 157
```

Top-k contribution for no-Q:

```text
top1  contribution = 1.50% of net gain
top5  contribution = 6.89%
top10 contribution = 12.80%
top20 contribution = 23.06%
```

Conclusion:

```text
The no-Q route-weighted BC improvement is statistically and distributionally real on 5k.
It is not a few-instance outlier effect.
Q remains weaker than no-Q on the larger diagnostic set, so Q remains a side diagnostic, not the main method.
```

## 17. Clean PBRS B/C/D from no expert/adapter/Q snapshot

Date: 2026-05-14.

User decision:

```text
Return to the latest version snapshot without expert head, residual adapter, or Q-head.
Run only B/C/D PBRS variants on GPU1/2/3. Do not rerun A.
```

Restored code snapshot:

```text
version_snapshots/pre_offline_all_vs_base_fix_20260511_191602_codex/DRL_Solver
-> evrptw_gen/benchmarks/DRL_Solver
```

Sanity:

```text
QHead/q_head/use_q_head: absent
LogitResidualAdapter/ResidualAdapter/regret_gate/expert_head/adapter: absent
```

Small code addition for PBRS ablation:

```text
--pbrs-mode {served,progress,none}
--repair-progress-include-current {true,false}
```

Variant definitions:

```text
B customer-only heuristic PBRS:
    pbrs_mode = none
    use_direct_progress_pbrs = False
    use_repair_fail_reward = True
    repair_progress_coef = 1.0
    repair_progress_include_current = False

C full repair heuristic PBRS:
    pbrs_mode = none
    use_direct_progress_pbrs = False
    use_repair_fail_reward = True
    repair_progress_coef = 1.0
    repair_progress_include_current = True

D no PBRS:
    pbrs_mode = none
    use_direct_progress_pbrs = False
    use_repair_fail_reward = False
    repair_progress_coef = 0.0
```

Launch script:

```text
run_pbrs_bcd_clean.sh
```

Eval protocol:

```text
eval data   = dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
updates     = 800
eval_freq   = 20
POMO/test   = sampling, test_agent=8
seed        = 2025 for B/C/D
GPUs        = B->1, C->2, D->3
```

Logs:

```text
LOGS/Codex_Res_pbrs_bcd_clean_snapshot_repair_progress_objprog_enc512_s75_ntraj50_nm64_u800/
```

Launch status:

```text
Script was updated to use nohup setsid so jobs survive the launching shell.

B GPU1 pid=709179
C GPU2 pid=709180
D GPU3 pid=709181

All three logs printed Training Info and the key PBRS flags matched the intended variants.
```

Follow-up status after checking logs:

```text
The first launch did not actually keep training. B/C/D all reached update 0 and then
failed during the first PPO update with CUDA OOM in decoder multi-head attention.

Failure point:
    train.py::_ppo_update -> agent.get_action_and_value_cached -> decoder.advance
    torch.cuda.OutOfMemoryError: tried to allocate 462 MiB

Original setting:
    num_envs=128, n_traj=50, num_steps=75, num_minibatches=64
```

Memsafe relaunch:

```text
Script:
    run_pbrs_bcd_clean.sh

Changed only the memory shape:
    num_envs=64
    num_minibatches=64
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Experiment semantics are unchanged:
    B = customer-only heuristic repair PBRS
    C = full repair heuristic PBRS
    D = no PBRS

New logs:
    LOGS/Codex_Res_pbrs_bcd_clean_snapshot_memsafe_ne64_repair_progress_objprog_enc512_s75_ntraj50_nm64_u800/

New pids:
    B GPU1 pid=711592
    C GPU2 pid=711593
    D GPU3 pid=711594

Sanity after relaunch:
    All three passed the previous OOM point and printed [Loss] at update 0.
```

Important correction after checking reward code:

```text
The B/C/D memsafe run is not a clean no-served-PBRS ablation.

Reason:
    In envs/evrp_vector_env.py, reward_mode == "vanilla" overrides pbrs_mode:

        if self.reward_mode == "vanilla":
            self.phi_reward = True
            self.lag_reward = True
            if self.use_direct_progress_pbrs:
                self.pbrs_mode = "direct_progress"
            else:
                self.pbrs_mode = "served"

Therefore the command-line flag:
    --pbrs-mode none

is ignored under reward_mode=vanilla when use_direct_progress_pbrs=False.

Actual current runs:
    GPU1 B = served-customer PBRS + repair-progress PBRS without current_to_depot
    GPU2 C = served-customer PBRS + full repair-progress PBRS
    GPU3 D = served-customer PBRS only, not true no-PBRS

Implication:
    The current B/C/D run cannot answer the intended question:
        customer heuristic PBRS vs full repair heuristic PBRS vs no PBRS

Needed fix:
    Make vanilla reward respect pbrs_mode=none, then restart B/C/D.
```

## 18. Fixed PBRS H/S/N 300-Epoch Ablation

User requested the actual three-way PBRS comparison:

```text
H heuristic-only:
    only repair heuristic PBRS, same repair heuristic as prior C

S served-only:
    only served-customer PBRS

N none:
    no PBRS at all
```

Bug fix applied:

```text
File:
    evrptw_gen/benchmarks/DRL_Solver/envs/evrp_vector_env.py

Before:
    reward_mode == "vanilla" always overwrote pbrs_mode to "served"
    whenever use_direct_progress_pbrs=False.

After:
    reward_mode == "vanilla" only falls back to "served" for unknown pbrs_mode.
    It now respects:
        pbrs_mode=served
        pbrs_mode=progress
        pbrs_mode=none
```

Stopped the invalid run:

```text
kill 711592 711593 711594
```

New launch script:

```text
run_pbrs_fixed_300.sh
```

Common protocol:

```text
eval data   = dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
updates     = 300
eval_freq   = 20
num_envs    = 64
n_traj      = 50
POMO/test   = sampling, test_agent=8
seed        = 2025 for all three
```

Variants:

```text
GPU1 H heuristic-only:
    --pbrs-mode none
    --use-direct-progress-pbrs False
    --use-repair-fail-reward True
    --repair-progress-coef 1.0
    --repair-progress-include-current True

GPU2 S served-only:
    --pbrs-mode served
    --use-direct-progress-pbrs False
    --use-repair-fail-reward False
    --repair-progress-coef 0.0

GPU3 N none:
    --pbrs-mode none
    --use-direct-progress-pbrs False
    --use-repair-fail-reward False
    --repair-progress-coef 0.0
```

Logs:

```text
LOGS/Codex_Res_pbrs_fixed300_clean_snapshot_ne64_repair_progress_objprog_enc512_s75_ntraj50_nm64_u300/
```

PIDs:

```text
H GPU1 pid=728046
S GPU2 pid=728047
N GPU3 pid=728048
```

Plot script:

```text
plot.py

Outputs:
    imgs/pbrs_fixed300_objective.png
    imgs/pbrs_fixed300_objective_zoom.png
```

Important correction for H:

```text
The first H run was still not clean. It enabled:
    use_repair_fail_reward=True

This did two things at once:
    1. enabled repair-progress PBRS
    2. changed terminal success/failure rewards

That explains why no-PBRS could learn finish rate faster than H:
    N kept the original strong terminal reward:
        success -> success_bonus
        failure -> -lambda_fail * unserved_ratio

    old H replaced it with:
        success -> repair_success_bonus = 1.0
        failure -> -repair_fail_coef * repair_remaining_ratio

Fix:
    Added a separate flag:
        --use-repair-progress-pbrs

    Repair-progress PBRS now uses:
        use_repair_progress_pbrs or use_repair_fail_reward

    Terminal repair reward is still controlled only by:
        use_repair_fail_reward

Stopped invalid H:
    kill 728046

Clean H restarted:
    pid=775419

Clean H args:
    --pbrs-mode none
    --use-repair-progress-pbrs True
    --use-repair-fail-reward False
    --repair-progress-coef 1.0
    --repair-progress-include-current True

Clean H log:
    LOGS/Codex_Res_pbrs_fixed300_clean_snapshot_ne64_repair_progress_objprog_enc512_s75_ntraj50_nm64_u300/result_gpu1_ablation5_routeevent_pbrsH_heuristic_only_cleanterminal_fixed300_ne64_repair_progress_objprog_enc512_s75_ntraj50_nm64_gpu1_seed2025_u300.txt

plot.py was updated to read the cleanterminal H log rather than the invalid H log.
```

## 2026-05-14: PBRS late-heads 100-epoch sweep

Stopped the previous fixed300 H/S/N runs:

```text
kill 728047 728048 775419
```

Small code change:

```text
Added decomposed_reward_mode=objective_progress_terminal

Reason:
    Existing modes supported 2 heads:
        objective_progress
    and 4 heads:
        objective_progress_terminal_teacher
    but not the requested 3-head split:
        objective + progress + terminal
```

Compile check passed:

```text
python -m py_compile DRL_train.py train.py
```

Run script:

```text
run_pbrs_late_heads_100.sh
```

Common setup:

```text
num_updates       = 100
num_envs          = 128
n_traj            = 50
num_steps         = 75
num_minibatches   = 128
update_epochs     = 5
accum_steps       = 8
eval_freq         = 20
eval_batch_size   = 1024
eval_data         = dataset/unanchored/Cus_50/buffer_1k/pickle/evrptw_50C_12R.pkl
seed              = 2025
```

Variants:

```text
GPU0: No PBRS, single critic
    pbrs_mode=none
    use_repair_progress_pbrs=False
    use_decomposed_reward_adv=False

GPU1: Customer-ratio PBRS, 2 heads
    pbrs_mode=served
    use_repair_progress_pbrs=False
    use_decomposed_reward_adv=True
    decomposed_reward_mode=objective_progress
    adv_objective_weight=0.65
    adv_progress_weight=0.20

GPU2: Heuristic repair PBRS, 2 heads
    pbrs_mode=none
    use_repair_progress_pbrs=True
    repair_progress_coef=1.0
    repair_progress_include_current=True
    use_decomposed_reward_adv=True
    decomposed_reward_mode=objective_progress
    adv_objective_weight=0.65
    adv_progress_weight=0.20

GPU3: All PBRS, 3 heads
    pbrs_mode=served
    use_repair_progress_pbrs=True
    repair_progress_coef=1.0
    repair_progress_include_current=True
    use_decomposed_reward_adv=True
    decomposed_reward_mode=objective_progress_terminal
    adv_objective_weight=0.65
    adv_progress_weight=0.20
    adv_terminal_weight=0.10
```

Logs:

```text
LOGS/Codex_Res_pbrs_lateheads100_clean_snapshot_ne128_repair_progress_objprog_enc512_s75_ntraj50_nm128_u100/
```

PIDs at launch:

```text
GPU0 pid=797477
GPU1 pid=797478
GPU2 pid=797479
GPU3 pid=797480
```

Plot script updated:

```text
plot.py

Outputs:
    imgs/pbrs_lateheads100_objective.png
    imgs/pbrs_lateheads100_objective_zoom.png
```

Stopped by user after clear negative trend:

```text
kill 797477 797478 797480
GPU2 pid=797479 had already exited at epoch 26 with CUDA unspecified launch failure.
```

Last observed status before stopping:

```text
GPU0 No PBRS:
    latest train epoch around 43
    Avg Customer Visits around 43.77
    Finish Rate around 938/6400 = 0.147
    eval@40 best_of_8 solved_rate=0.1865
    objective proxy = 929.97

GPU1 customer-ratio PBRS + 2 heads:
    train collapsed around 10-12 avg customers
    finish rate near 0
    eval@40 solved_rate=0.0010
    objective proxy not meaningful because only 1 solved episode

GPU2 heuristic PBRS + 2 heads:
    early peak near epoch 12:
        Finish Rate 1809/6400 = 0.283
    then degraded:
        epoch 26 Finish Rate 332/6400 = 0.052
    eval@20 best_of_8 solved_rate=0.2910
    exited during PPO update with CUDA unspecified launch failure

GPU3 all PBRS + 3 heads:
    customer-ratio PBRS appears to dominate/damage learning
    train avg customers near 12-15
    finish rate near 0
    eval@40 solved_rate=0.0020
```

Interpretation:

```text
1. Customer-ratio PBRS in the current scale/form is harmful under decomposed 2-head training.
2. All PBRS inherits the customer-ratio failure and is also bad.
3. Heuristic repair PBRS has useful early signal but is unstable and degrades after early epochs.
4. No-PBRS with normal terminal completion penalty remains the clean fallback.
5. The next stable direction should revert to the previous clean PPO/reward setup, or isolate heuristic PBRS with safer normalization/early stopping before combining it with customer PBRS.
```

## 2026-05-14: PBRS late-heads 50-epoch retry with larger effective optimizer batch

Rationale:

```text
The previous late-heads run used:
    num_envs=128
    num_minibatches=128
    accum_steps=8

That means each PPO minibatch had 1 env, and each optimizer step used only 8 envs.
This is likely too noisy in the early low-finish-rate regime.
```

New setting:

```text
num_updates       = 50
num_envs          = 128
num_minibatches   = 64
envs_per_minibatch= 2
accum_steps       = 8
effective envs per optimizer step = 16
eval_freq         = 10
```

Run script:

```text
run_pbrs_late_heads_50_mb64.sh
```

Logs:

```text
LOGS/Codex_Res_pbrs_lateheads50_clean_snapshot_ne128_mb64_repair_progress_objprog_enc512_s75_ntraj50_u50/
```

Variants:

```text
GPU0: No PBRS, single critic
GPU1: Customer-ratio PBRS, 2 heads objective+progress
GPU2: Heuristic repair PBRS, 2 heads objective+progress
GPU3: All PBRS, 3 heads objective+progress+terminal
```

PIDs at launch:

```text
GPU0 pid=904744
GPU1 pid=904745
GPU2 pid=904746
GPU3 pid=904747
```

Initial memory check:

```text
GPU0 ~4540 MiB
GPU1 ~4498 MiB
GPU2 ~4498 MiB
GPU3 ~4498 MiB
```

## 2026-05-14: Gamma-corrected old 2-head PBRS sweep

Motivation:

```text
The old effective baseline used a 2-head objective_progress critic:
    objective head = distance/objective
    progress head  = direct served-customer progress
                   + repair progress
                   + repair success/fail signal

The old implementation did not apply PPO gamma inside the direct progress and repair
progress deltas. This sweep adds gamma consistency to those two progress-shaping
terms while preserving the old 2-head training recipe.
```

Code change:

```text
direct customer progress:
    old: coef * (Phi_customer(s') - Phi_customer(s))
    new: coef * (gamma * Phi_customer(s') - Phi_customer(s))

repair progress, using Phi_repair(s) = -remaining_repair_ratio(s):
    old: repair_coef * (repair_s - repair_s')
    new: repair_coef * (repair_s - gamma * repair_s')
```

Common training setup:

```text
num_updates       = 200
num_envs          = 128
num_minibatches   = 64
accum_steps       = 8
gamma             = 0.99
decomposed mode   = objective_progress
adv weights       = objective 0.5, progress 0.5
use_direct_progress_pbrs = True
use_repair_fail_reward   = True
repair_progress_coef     = 1.0
repair_fail_coef         = 1.0
repair_success_bonus     = 1.0
```

Beta sweep:

```text
The old beta=0.5 was intentionally late-customer heavy.
This sweep uses smoother beta values:
    beta=0.75
    beta=1.00
```

Run script:

```text
run_pbrs_old2head_gamma_200.sh
```

Logs:

```text
LOGS/Codex_Res_pbrs_old2head_gamma200_ne128_mb64_repair_progress_objprog_enc512_s75_ntraj50_u200/
```

Variants:

```text
GPU0: progress_pbrs_coef=2.0, beta=0.75
GPU1: progress_pbrs_coef=2.0, beta=1.00
GPU2: progress_pbrs_coef=1.0, beta=0.75
GPU3: progress_pbrs_coef=1.0, beta=1.00
```

PIDs at launch:

```text
GPU0 pid=950303
GPU1 pid=950304
GPU2 pid=950305
GPU3 pid=950306
```

## 2026-05-15: Mode-Conditioned Offline Proposal Decoding v1-v5

Goal:

```text
Use offline ALNS buffer as a missing-mode proposal source, not as direct
actor imitation.

Eval data:
    dataset/unanchored/Cus_50/solomon
    converted to mode_proposal_v1/data/solomon_eval_1000.pkl

Offline data:
    dataset/unanchored/Cus_50/buffer
    5120 ALNS-3k records
```

Isolated code/logs:

```text
mode_proposal_v1/
LOGS/Codex_Res_mode_proposal_v1/
checkpoint/mode_proposal_v1/
```

Base checkpoint:

```text
checkpoint/ablation5_routeevent_baseref800_repair_progress_objprog_enc512_s75_ntraj50_nm64_era0p0_rf1p0_rp1p0_sb1p0_gpu1_seed2030_u800/Cus_50_CS_12/best_model.pth
```

Step 1: base frontier on 5k buffer

```text
script:
    mode_proposal_v1/policy_eval.py

output:
    LOGS/Codex_Res_mode_proposal_v1/buffer_base_pomo50.csv

PPO POMO-50 on 5120 buffer instances:
    avg obj     = 805.570993
    solved rate = 1.0
```

Step 2: missing-mode selection

```text
script:
    mode_proposal_v1/build_missing_mode_dataset.py

selection:
    ALNS obj improves PPO-50 frontier by >3%
    ALNS route replay feasible
    base mean NLL above positive-set median novelty threshold

output:
    LOGS/Codex_Res_mode_proposal_v1/missing_mode_v1.csv
    mode_proposal_v1/data/missing_mode_v1_selected.pkl

records loaded = 5120
valid replay   = 5120
selected       = 1104
mean selected improvement_rel = 0.078072
mean selected NLL             = 2.561557
novelty threshold             = 2.040416
```

Solomon base and diversity baselines:

```text
base POMO-50:
    avg obj = 935.824993

base POMO-45:
    avg obj = 937.121157

base POMO-40:
    avg obj = 937.968468

base POMO-32:
    avg obj = 940.840249

high-temp PPO, temp=1.5, POMO-10:
    avg obj = 962.424731

high-temp PPO, temp=1.5, POMO-8:
    avg obj = 968.143849
```

v1: independent full-policy offline proposal

```text
script:
    mode_proposal_v1/train_offline_proposal.py

training:
    init from base checkpoint
    train all parameters on selected ALNS missing-mode routes
    epochs=8, lr=3e-5, KL coef=0.02

offline proposal POMO-10:
    avg obj = 1373.086446

portfolio:
    base40 + off10 = 937.968468
    off selected   = 0 / 1000

diagnosis:
    full-model imitation destroys deployable sampling quality.
```

v2: conservative full-policy proposal

```text
training:
    epochs=2, lr=1e-5, KL coef=0.5
    stricter credit mask

offline proposal POMO-10:
    avg obj = 2088.198259

portfolio:
    base40 + off10 = 937.968468
    off selected   = 0 / 1000

diagnosis:
    stronger replay-state KL does not prevent deployment-state distribution drift.
```

v3: decoder-only proposal

```text
training:
    freeze embedding + encoder
    train decoder only
    epochs=2, lr=3e-5, KL coef=0.5

offline proposal POMO-10:
    avg obj = 2348.951904

diagnosis:
    decoder-only fine-tuning is still too disruptive when trained as a full policy.
```

v4b: bounded residual proposal head

```text
script:
    mode_proposal_v1/train_bounded_residual_proposal.py
    mode_proposal_v1/eval_bounded_residual_proposal.py

architecture:
    frozen base PPO
    small shared per-action residual head
    input features = base logit z-score, logprob, prob, rank, node type
    residual bounded by tanh, max_residual=0.75
    branch is used only as separate candidate proposal mode

training:
    epochs=4
    lr=5e-4
    KL coef=0.2
    residual norm coef=0.02

bounded residual POMO-10:
    avg obj = 957.878721

portfolio, total budget 50:
    base40 + bounded-off10 = 935.199637
    delta vs base50        = -0.625355
    off selected           = 193 / 1000

portfolio, safer split:
    base45 + bounded-off5  = 935.543954
    delta vs base50        = -0.281039
    off selected           = 126 / 1000

diagnosis:
    This is the first offline proposal branch that contributes candidates.
    The gain is real but small; high-temp diversity is still stronger.
```

v5: stronger bounded residual proposal

```text
training:
    max_residual=1.5
    epochs=6
    lr=5e-4
    KL coef=0.15

bounded residual POMO-10:
    avg obj = 973.559229

portfolio:
    base40 + bounded-off10 = 935.867716
    delta vs base50        = +0.042723

diagnosis:
    Larger residual increases harmful candidates and removes the portfolio gain.
```

Current best from this block:

```text
Best overall portfolio:
    base45 + high-temp10:
        avg obj = 934.701557
        delta vs base50 = -1.123436
        selected high-temp = 156 / 1000

Best offline-data proposal:
    base40 + bounded-residual-off10 v4b:
        avg obj = 935.199637
        delta vs base50 = -0.625355
        selected offline = 193 / 1000
```

Conclusion:

```text
Mode-level portfolio is useful, but the offline branch is still weak.
Directly training a full proposal policy from ALNS routes is not stable.
The safe direction is a bounded, separate proposal mode with exact objective
selection. Next improvement should focus on making the bounded offline mode
stronger than generic high-temperature diversity, likely by using a real
mode-specific decoder head and validation-selected residual strength.
```

## 2026-05-15 Mode-Conditioned Offline Proposal Iteration

Goal:

```text
Keep the deployed base PPO safe.
Use offline data only to create separate proposal modes.
Total inference budget fixed at K=50 on Solomon eval 1000.
Final candidate is selected by true objective among generated candidates.
```

Reference:

```text
base PPO POMO-50 on Solomon eval 1000:
    avg obj = 935.824993
```

### Temperature + v4b budget sweep

Base high-temp branch:

```text
temp10 @ 1.1  avg obj = 958.103541
temp10 @ 1.2  avg obj = 956.337208
temp10 @ 1.25 avg obj = 957.101074
temp10 @ 1.3  avg obj = 957.619059
temp10 @ 1.5  avg obj = 962.424731
temp10 @ 2.0  avg obj = 998.253182
```

Best v4b portfolio after temperature sweep:

```text
base25 + v4b-off15(step10) + temp10@1.2:
    avg obj = 934.024036
    delta   = -1.800957
    source counts = base25 498, off15 306, temp10 196
```

Diagnosis:

```text
High-temp helps as a diversity mode, but too much temperature is harmful.
temp=1.2 is the best base diversity point tested.
```

### Multi offline head portfolio

Tried splitting offline budget across v4b/v6/v7/v8/v9/v10/v11/v12.

Best multi-head result:

```text
base20 + off10(v4b) + off10(v8) + off10(v11):
    avg obj = 934.096140
    delta   = -1.728853
```

Diagnosis:

```text
Different offline heads are complementary, but reducing base budget to 20
hurts more than the extra offline diversity helps.
```

### v13 strict missing-mode proposal

Change:

```text
records: original missing_mode_v1_selected, 1104 records
credit mask:
    base_prob_threshold = 0.05
    rank_threshold      = 20
max_residual = 1.0
epochs       = 6
lr           = 5e-4
KL coef      = 0.2
res coef     = 0.02
```

Training signal:

```text
credit mask mean ~= 0.264
final train summary:
    output = checkpoint/mode_proposal_v1/bounded_residual_head_v13_strict05_rank20_r100.pth
```

POMO-15 proposal branch:

```text
v13 step10 temp1.0 avg obj = 949.717470
v13 step10 temp1.2 avg obj = 951.126408
v13 step8  temp1.2 avg obj = 950.459373
```

Although proposal-only objective is not best at step8/temp1.2, portfolio
selection shows this setting is most complementary to base and high-temp modes.

Best v13 portfolio:

```text
base25 + v13-off15(step8, residual temp=1.2) + base-temp10@1.2:
    avg obj = 933.562066
    delta   = -2.262927
    median delta = -0.135851
    trimmed mean delta = -1.775629
    wins/losses/ties = 507 / 476 / 17
    source counts = base25 493, off15 312, temp10 195
```

Important sweep:

```text
step5  temp1.2 portfolio delta = -1.779513
step7  temp1.2 portfolio delta = -1.880497
step8  temp1.2 portfolio delta = -2.262927  <-- best
step9  temp1.2 portfolio delta = -1.813122
step10 temp1.2 portfolio delta = -1.936703
step12 temp1.2 portfolio delta = -1.948543
step20 temp1.2 portfolio delta = -1.915692

step8 temp1.1 portfolio delta = -1.389307
step8 temp1.2 portfolio delta = -2.262927  <-- best
step8 temp1.3 portfolio delta = -1.329988
```

Diagnosis:

```text
Offline proposal is useful mostly as an early-route mode perturbation.
The useful window is around the first 8 decisions.
Too short misses useful route-mode changes; too long adds noise.
```

### Wider data variants

Generated from the existing 5k replay table:

```text
v16 data: margin >= 1%, novelty top50, records = 1395
v17 data: margin >= 2%, novelty top50, records = 1255
v18 data: margin >= 1%, novelty top60, records = 1673
```

Same strict training config as v13.

Portfolio results:

```text
v16 step8 temp1.2:
    avg obj = 934.111375
    delta   = -1.713618

v17 step8 temp1.2:
    avg obj = 934.115040
    delta   = -1.709953

v18 step8 temp1.2:
    avg obj = 933.765571
    delta   = -2.059422
```

Diagnosis:

```text
Wider labels improve sample count but soften the missing-mode signal.
v18 is useful but still weaker than v13.
The original margin>=3%, novelty-top50 set remains the best training set so far.
```

### Protection gate test

Tried inference-only max-prob residual gates on v13 step8/temp1.2:

```text
maxprob <= 0.95 proposal avg = 951.643351
maxprob <= 0.90 proposal avg = 951.449074
maxprob <= 0.85 proposal avg = 951.847998
```

Diagnosis:

```text
The gate over-protects and removes useful diversity.
For this proposal branch, fixed early-step activation is better.
```

Current best:

```text
Method:
    Mode-conditioned proposal portfolio
    K_total = 50
    K_base  = 25
    K_off   = 15
    K_temp  = 10

Components:
    base25: PPO base sampling
    off15: v13 bounded residual proposal, active first 8 steps, residual temp=1.2
    temp10: base PPO high-temp branch, temp=1.2

Solomon eval 1000:
    base50 avg = 935.824993
    new avg    = 933.562066
    gain       = -2.262927
```

Interpretation:

```text
This is the strongest deployable offline-data result so far.
It does not use ALNS online and does not modify the base policy.
However, the gain is still below the target 5-10 objective points.
Next iteration should try stronger proposal capacity without logit fusion:
    1. validation-selected separate mode-specific decoder head,
    2. per-instance budget allocator only if it beats fixed allocation,
    3. multi-seed eval to verify that the -2.26 gain is not sampling luck.
```

### 2026-05-15 continued: proposal-mode iterations after v13

Goal:

```text
Keep the base PPO actor frozen.
Continue improving deployable offline proposal decoding on Solomon eval 1000.
Main reference remains base POMO-50 avg = 935.824993.
Current best before this block: v13 portfolio avg = 933.562066, gain = -2.262927.
```

#### z-residual scale-normalized proposal

Implemented `final_logit_mode=z_residual`:

```text
logits_off = normalize_feasible(logits_base) + residual
```

This directly tests the logit-scale concern while still keeping the branch candidate-level only.

Results:

```text
v19 z_residual r=1.0:
    off15 step8 temp1.0 avg = 1437.907871
    off15 step8 temp1.2 avg = 1854.740566

v20 z_residual r=2.0:
    off15 step8 temp1.0 avg = 1411.578784
    off15 step8 temp1.2 avg = 1810.668766

portfolio base25 + off15 + temp10:
    off branch selected = 0 cases
    portfolio avg = 938.328173
    delta = +2.503180
```

Diagnosis:

```text
Normalizing the base logits makes BC loss easier, but it destroys deployed sampling quality.
The issue is not only actor/delta scale mismatch; the proposal branch also needs to preserve the base decoder's calibrated ordering.
z-residual is rejected.
```

#### Raw residual training sweeps around v13

Same data as v13: margin>=3%, novelty top50, 1104 records.

```text
v21: epoch=4, r=1.0
    branch avg = 950.811867
    portfolio avg = 933.952857
    delta = -1.872136

v22: epoch=8, r=1.0
    branch avg = 950.959806
    portfolio avg = 934.034775
    delta = -1.790217

v23: lr=3e-4, epoch=6, r=1.0
    branch avg = 950.166483
    portfolio avg = 933.827336
    delta = -1.997657

v24: r=1.25, kl=0.25, res=0.03
    branch avg = 952.574708
    portfolio avg = 934.616029
    delta = -1.208964
```

Diagnosis:

```text
The original v13 hyperparameters remain best.
More capacity or more epochs do not improve the candidate branch.
```

#### v13 budget allocation

Evaluated v13 with different offline proposal budgets at step8/temp1.2:

```text
off10 branch avg = 958.035881
off20 branch avg = 945.404488
off25 branch avg = 942.410635
off30 branch avg = 940.608106
```

Portfolio allocation:

```text
base30 + off10 + temp10:
    avg = 934.336132
    delta = -1.488861

base20 + off20 + temp10:
    avg = 933.839632
    delta = -1.985361

base25 + off25:
    avg = 934.002521
    delta = -1.822472

base20 + off30:
    avg = 934.079455
    delta = -1.745537

base30 + off20:
    avg = 933.879000
    delta = -1.945992
```

Diagnosis:

```text
The best allocation remains base25 + off15 + temp10.
More offline budget improves the branch avg but gives up too much base/temp coverage.
```

#### rich_norm state-action features

Added `feature_mode=rich_norm`, which appends state-action geometry/time-window features and standardizes them over feasible actions per state.

```text
v25 rich_norm r=1.0:
    branch avg = 950.982819
    portfolio avg = 934.082684
    delta = -1.742309

v26 rich_norm r=0.75:
    branch avg = 950.915859
    portfolio avg = 934.132112
    delta = -1.692881
```

Diagnosis:

```text
Normalized state-action features train stably, unlike the old unnormalized rich variant.
But they still do not improve objective selection beyond v13.
The simple logit/rank proposal remains stronger.
```

#### stricter missing-mode labels

Filtered v13 data by low-probability ALNS branch ratio:

```text
v27 data: improvement>=5%, lowprob_ratio>=0.30, records=349
v28 data: improvement>=3%, lowprob_ratio>=0.35, records=160
```

Results:

```text
v27:
    branch avg = 950.852969
    portfolio avg = 933.887546
    delta = -1.937447

v28:
    branch avg = 950.193254
    portfolio avg = 933.744055
    delta = -2.080938
    median delta = -0.494146
    wins/losses/ties = 516 / 465 / 19

v28 off20:
    branch avg = 946.497567
    portfolio base20+off20+temp10 avg = 934.524111
    delta = -1.300881
```

Diagnosis:

```text
Lowprob filtering improves median paired behavior, especially v28.
But it does not improve mean objective; extra offline budget hurts.
This looks more stable but less high-upside than v13.
```

#### train-only-early-step variants

Aligned training window with inference residual window:

```text
v29: train_step_limit=8
    branch avg = 952.847559
    portfolio avg = 934.277966
    delta = -1.547027

v30: train_step_limit=12
    branch avg = 951.416898
    portfolio avg = 934.193934
    delta = -1.631059
```

Diagnosis:

```text
Training only early steps increases credit density but over-specializes.
The original v13, trained over the full replay and activated only early at inference, remains best.
```

Current best remains:

```text
v13 bounded residual proposal
base25 + off15(step8,temp1.2) + temp10(temp1.2)
avg = 933.562066
delta = -2.262927
```

Next hypothesis:

```text
The small residual scorer is reaching its ceiling.
Further improvement likely needs a proposal mode with more expressive but still isolated parameters:
    A. mode-specific lightweight decoder/head trained with missing-mode BC,
    B. validation-selected checkpoints instead of final-epoch selection,
    C. offline route replay with objective-level candidate selection during validation.
```

### 2026-05-15: Mode-conditioned offline proposal iteration

Goal:

```text
Use offline data as a missing-route-mode proposal source, not as a direct actor teacher.
Keep the PPO base actor unchanged.
Generate candidates from multiple deployable neural modes under the same total K=50 budget.
Select the best candidate by the true objective.
```

Important baseline:

```text
Solomon eval 1k, PPO base POMO-50:
    avg obj = 935.824993
```

#### Single offline proposal recap

The strongest single deployable offline proposal head remains v13:

```text
v13:
    feature_mode = logit
    final_logit_mode = raw_residual
    max_residual = 1.0
    training data = missing_mode_v1_selected, 1104 records
    filter = base_prob < 0.05 or base_rank >= 20
    inference = residual active for first 8 steps, temperature 1.2

base25 + v13-off15 + temp10:
    avg = 933.562066
    delta vs base50 = -2.262927
    wins/losses/ties = 507 / 476 / 17
```

#### New embedding proposal head

Implemented `feature_mode=embed`:

```text
features = [
    base logit/prob/rank/type features,
    current decoder glimpse,
    candidate node embedding
]
in_dim = 7 + 2 * embedding_dim = 519
per-vector layer normalization
shared per-node residual scorer
```

Best embedding variant:

```text
v31:
    feature_mode = embed
    head_hidden = 128
    max_residual = 1.0
    lr = 3e-4
    epochs = 6
    same missing-mode data/filter as v13

v31 off15, step8, temp1.2:
    branch avg = 950.068350

base25 + v31-off15 + temp10:
    avg = 933.640900
    delta vs base50 = -2.184093
```

Diagnosis:

```text
v31 alone is not better than v13.
But it is complementary: it selects different good routes than v13.
```

#### Multi-offline-mode fair K=50 portfolios

The key improvement comes from combining independent proposal modes, not from fusing logits.

```text
Portfolio:
    base15 + v13-off20(step8,temp1.2) + v31-off15(step8,temp1.2)
    total K = 15 + 20 + 15 = 50

Result:
    avg = 932.840875
    delta vs base50 = -2.984118
    median delta = -1.081014
    trimmed mean delta = -2.187638
    wins/losses/ties = 537 / 446 / 17
    selected source counts:
        base15    = 335
        v13-off20 = 377
        v31-off15 = 288

Files:
    LOGS/Codex_Res_mode_proposal_v1/portfolio_base15_off20v13_off15v31_step8_t1p2.csv
    LOGS/Codex_Res_mode_proposal_v1/portfolio_base15_off20v13_off15v31_step8_t1p2.summary.json
```

This is the current best deployable offline-data result on Solomon eval 1k.
It uses no online ALNS and keeps the PPO base actor unchanged.

Other strong fair K=50 combinations:

```text
base15 + v13-off15 + v31-off15 + temp5:
    avg = 933.000566
    delta = -2.824427
    wins/losses/ties = 520 / 458 / 22

base20 + v13-off15 + v31-off15:
    avg = 933.515156
    delta = -2.309837
    wins/losses/ties = 514 / 467 / 19
```

Current interpretation:

```text
The useful offline signal is not a universal teacher signal.
It is a missing-mode proposal signal.
Two isolated offline proposal heads cover different route modes and improve best-of-K coverage.
The next step should search for a third complementary proposal mode or improve the embedding proposal head.
```

#### Iteration after v31: more proposal modes

Trained and evaluated additional heads:

```text
v33:
    feature_mode = embed
    head_hidden = 256
    max_residual = 1.0
    lr = 3e-4
    epochs = 6
    result:
        off15 step8 temp1.2 avg = 950.514809
        off20 step8 temp1.2 avg = 946.183083

v35:
    v13-style logit head, seed=3407
    max_residual = 1.0
    lr = 5e-4
    epochs = 6
    result:
        off15 step8 temp1.2 avg = 952.571570
        off20 step8 temp1.2 avg = 947.764819
```

Diagnosis:

```text
v33 has a stronger off20 branch than v31, but it mostly substitutes for v13/v31 instead of adding a new route family.
v35 does not provide useful new complementarity.
The current bottleneck is missing new proposal families, not just more capacity in the same family.
```

#### Automatic fair-K portfolio search

Added:

```text
mode_proposal_v1/search_fair_portfolios.py
mode_proposal_v1/auto_search_portfolios.py
```

These scripts search deployable candidate portfolios under the same total K=50 budget.

Best 4-mode result found:

```text
Portfolio:
    v4b off10 step10
    v4b off10 step12
    v31 off15 step8 temp1.2
    v13 off15 step8 temp1.2

K = 10 + 10 + 15 + 15 = 50

Result:
    avg = 932.567397
    delta vs base50 = -3.257595
    median delta = -0.779009
    trimmed mean delta = -2.548248
    wins/losses/ties = 527 / 463 / 10
    source counts:
        v4b off10 step10 = 226
        v4b off10 step12 = 210
        v31 off15        = 278
        v13 off15        = 286

Files:
    LOGS/Codex_Res_mode_proposal_v1/portfolio_best_v4b10s10_v4b10s12_v31off15_v13off15_k50.csv
    LOGS/Codex_Res_mode_proposal_v1/portfolio_best_v4b10s10_v4b10s12_v31off15_v13off15_k50.summary.json
```

Best base-fallback-preserving 4-mode result:

```text
Portfolio:
    base15
    v4b off10 step10
    v4b off10 step12
    v31 off15 step8 temp1.2

K = 15 + 10 + 10 + 15 = 50

Result:
    avg = 932.752368
    delta vs base50 = -3.072624
    median delta = -0.845623
    wins/losses/ties = 531 / 455 / 14

Files:
    LOGS/Codex_Res_mode_proposal_v1/portfolio_base15_v4b10s10_v4b10s12_v31off15_k50.csv
    LOGS/Codex_Res_mode_proposal_v1/portfolio_base15_v4b10s10_v4b10s12_v31off15_k50.summary.json
```

Best 5-mode result:

```text
Portfolio:
    v4b off5
    v4b off10 step10
    v8 off10 step10
    v4b off10 step12
    v31 off15 step8 temp1.2

K = 5 + 10 + 10 + 10 + 15 = 50

Result:
    avg = 932.437677
    delta vs base50 = -3.387316
    median delta = -0.873303
    trimmed mean delta = -2.669047
    wins/losses/ties = 532 / 455 / 13
    source counts:
        v4b off5         = 108
        v4b off10 step10 = 212
        v8 off10         = 198
        v4b off10 step12 = 188
        v31 off15        = 294

Files:
    LOGS/Codex_Res_mode_proposal_v1/portfolio_best5_v4b5_v4b10s10_v8off10_v4b10s12_v31off15_k50.csv
    LOGS/Codex_Res_mode_proposal_v1/portfolio_best5_v4b5_v4b10s10_v8off10_v4b10s12_v31off15_k50.summary.json
```

Tried max 6 branches under K=50; it did not improve over the 5-mode result.

#### v4b-style seed variants

Because the best portfolios rely on small-budget v4b windows, trained two v4b-style seed variants:

```text
v36:
    max_residual = 0.75
    feature_mode = logit
    filter = base_prob <= 0.15 or rank >= 10
    seed = 3407
    result:
        off10 step10 avg = 956.740753
        off10 step12 avg = 957.646120

v37:
    same as v36, seed = 2026
    result:
        off10 step10 avg = 956.265753
        off10 step12 avg = 957.833051
```

Diagnosis:

```text
The seed variants are weaker than original v4b and do not enter the top fair-K portfolios.
Original v4b already represents this small-budget proposal family well.
```

Current best deployable result:

```text
base POMO-50 avg = 935.824993
best 5-mode offline proposal portfolio avg = 932.437677
improvement = -3.387316
```

Current bottleneck:

```text
Increasing mode count and small budget diversity helps, but saturates near -3.4.
To reach -5 or better, we likely need a genuinely new proposal family:
    1. learned instance-conditional mode allocation trained on buffer-side branch wins,
    2. a stronger offline branch objective than masked BC/residual,
    3. or a proposal branch trained to optimize finite-budget novelty directly.
```

#### v4b/v8 follow-up

Tested whether the best small-budget v4b family improves with temperature:

```text
v4b off5 temp1.2:
    avg = 979.819392
v4b off10 step10 temp1.2:
    avg = 957.120818
v4b off10 step12 temp1.2:
    avg = 958.189640
v4b off10 step10 temp0.8:
    avg = 956.759616
```

Diagnosis:

```text
Temperature diversity does not help v4b.
The useful diversity comes from residual step window and proposal family, not from sampling temperature.
```

Because v8 entered the best 5-mode portfolio but had only one evaluated mode, evaluated additional v8 modes:

```text
v8 off5 step10:
    avg = 969.891198
v8 off10 step8:
    avg = 956.345167
v8 off10 step12:
    avg = 957.704588
v8 off15 step10:
    avg = 949.622775
```

Best portfolio after v8 sweep:

```text
Portfolio:
    v4b off5
    v4b off10 step10
    v8 off10 step10
    v4b off10 step12
    v13 off15 step8 temp1.2

K = 5 + 10 + 10 + 10 + 15 = 50

Result:
    avg = 932.387961
    delta vs base50 = -3.437031
    median delta = -0.873303
    trimmed mean delta = -2.584148
    wins/losses/ties = 528 / 454 / 18
    source counts:
        v4b off5         = 108
        v4b off10 step10 = 208
        v8 off10 step10  = 220
        v4b off10 step12 = 195
        v13 off15        = 269

Files:
    LOGS/Codex_Res_mode_proposal_v1/portfolio_best5_v4b5_v4b10s10_v8off10s10_v4b10s12_v13off15_k50.csv
    LOGS/Codex_Res_mode_proposal_v1/portfolio_best5_v4b5_v4b10s10_v8off10s10_v4b10s12_v13off15_k50.summary.json
```

Updated current best deployable result:

```text
base POMO-50 avg = 935.824993
best mode-conditioned offline proposal portfolio avg = 932.387961
improvement = -3.437031
```

Interpretation:

```text
The best result now uses five small/medium proposal modes.
This supports the mode-conditioned proposal decoding hypothesis:
offline data is useful when it creates complementary candidate modes,
not when it directly overwrites the base actor.

However, the gain is still below the desired -5 to -10 range.
The next version should move beyond fixed global portfolios:
    train an instance-conditional allocator on buffer-side branch wins,
    or train a new proposal family with an objective explicitly targeting missing finite-budget routes.
```
