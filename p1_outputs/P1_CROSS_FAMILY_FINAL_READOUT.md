# P1 实例族与排名稳定性审计｜最终版

## 结论先说
- **ETP：稳定。** 20 个 75% action-pool 子实例的 planner 排名中位 Kendall τ = **1.000**。
- **π-Base：不稳定。** τ = **0.346**。
- **Ring：不稳定。** τ = **0.183**。
- **Module：不稳定。** τ = **0.189**。

预锁判据是 τ < 0.4 记“排名不稳定”，τ ≥ 0.6 记“稳定”。因此三张外部图都落在不稳定区，而 ETP 明显稳定。

## ETP：P1 已完整跑通
仓库里的 compact 副本确实有 CRC 损坏，但后来按历史证据恢复了 scientific content，并且只有在 **compact hash 与 manifest hash 同时精确匹配冻结值** 后才接受：
- recovered compact SHA256：`3273bb7a5ee4a8697d4e266bd3cf666e22718b57f29e42d4d8b2c1aa7436c7a6`
- recovered manifest SHA256：`fcb56cfd5d43666de836472db06e69e7c4e265fb513f397f8c3e4dd73ddf3243`
- 1213 actions；1200 successes；13 fails
- **P1 新增 solver 调用 = 0**

ETP 20 个子实例中：
- FINAL 相对 random：平均 **-21.40 calls**，95% CI **[-22.35, -20.40]**；20/20 更好。
- EPAG：平均 **-21.40 calls**；20/20 更好。
- Graph19：平均 **-21.45 calls**；20/20 更好。
- FINAL / EPAG 的 90% calls 几乎始终维持在 7 左右，说明 ETP 的 7-vs-30 主信号不是靠某一个 action pool 巧合撑出来的。

## 三个外部域：单池冠军很不稳
最清楚的例子：
- π-Base **Small-first**：平均比 random 少 **11.09 calls**，95% CI **[-13.37, -8.83]**，20/20 更好。
- π-Base **Target-UCB**：平均反而多 **7.05 calls**，95% CI **[4.29, 9.40]**。
- Ring 原 Gold 的 **ε-Ridge** 优势在重采样中反转：平均 **+1.65 calls**，95% CI **[0.23, 3.07]**。
- Module **Target-UCB** 仍有小而稳定的优势：平均 **-0.82 calls**，95% CI **[-1.40, -0.24]**。

## 贵方法 K′=5 补充检查
每个贵方法批次都先 exact reproduce 原 Gold，再对子池测试。负数表示比 random 少调用。
- π-Base：MCTS **+3.35**；CP-SAT **+4.07**。
- Ring：MCTS **+9.56**；CP-SAT **+9.24**。
- Module：MCTS **-0.69**；CP-SAT **-0.81**。

Ring 的两种贵 planner 在 5 个子实例上都明显更差；Module 则保持小幅优势；π-Base 不稳定/偏差。这进一步支持：**复杂 planner 本身没有跨域通用优势，真正决定效果的是该域里可预测的 residual-value 信号。**

## P1 科研判决
这轮把故事收紧了：

**ETP 的强路由结果具有实例子采样稳定性；外部三域的“哪个 planner 最好”高度依赖实例族。**

所以论文里可以更强地保留 ETP 作为主证据；π-Base / Ring / Module 不再承担“证明某个算法普适领先”的任务，而承担更准确的作用：证明 hidden residual value 在不同数学图上的可预测结构不同，同时暴露单池排行榜的脆弱性。

这与当前主线“直接学习 query-conditioned residual value / outcome，再做严格 adaptive routing”一致。
