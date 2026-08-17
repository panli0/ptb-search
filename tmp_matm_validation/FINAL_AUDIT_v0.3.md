# MATM ALFWorld 自然 terminal-run 验证：最终审计 v0.3

## 最终裁定

**NATURAL_TERMINAL_RUN_PATTERN_NOT_CONFIRMED（自然 terminal-run 模式未确认）。**

这次补到了可逐步分析的真实 ALFWorld 轨迹，但仍不能把受控实验里的“连续做同一方向越久，越倾向换线”升级为自然任务中的已复现规律。

## 数据与冻结分析

- 公共数据：`toeunkim/matm-trajectories` 的 ALFWorld `population_runs.parquet`。
- 数据 SHA256：`626e2e6351d763739b0e2695a1bc442e1c851c1153c44301017739e3bd1155aa`。
- A/B 两类任务共 672 条：no-retrieval 112；5 个 retrieval 策略各 112，共 560。
- corrected adapter v0.2 修复了两个确定性标签问题：MATM 的 observation 是动作前状态，动作 i 的结果在 observation i+1；以及完成某个 object-subgoal 后必须从“原方向仍未解决”主分析中排除。
- GitHub Actions run：`32043528834`；artifact：`9292417361`。

## 冻结结果

### A：pick-two（找两个同类物体并放入目标容器）

no-retrieval：K1 switch=0/94；K6+=1/64；任务等权 K6+-K1=+1.19 pp，但 K6 只有 14 个独立 model×task 单元，原 gate 不通过。

untouched retrieval-enabled：冻结 v0.2 名义结果 K1=1/467（0.21%），K6+=7/315（2.22%），任务等权差约 +1.32 pp，原 trajectory-count gate 名义 PASS。

但是独立性审计后，K6+ 实际只有 **19 个 model×task 单元**；7 个 K6+ switch 只来自 **3 个独立 model×task 单元**。其中一个 Llama-3.3-70B × pencil 轨迹在 5 个 retrieval 深度里重复出现相同长程换线模式。按 model×task 做 paired sign-flip，单侧 p=0.125。

K 曲线也不单调：`1=0.21%, 2=0.00%, 3=1.28%, 4=0.00%, 5=4.88%, 6+=2.22%`。K5 最高，K6+ 反而下降。因此不能写“run 越长，switch 概率越高”的自然规律。

另外，这 14 个 clean switch 后 5 步的 task score progress 为 0/14；continue 为约 22.3%。这是观察性后果，不是因果比较，但它至少没有显示这些 switch 是“换线后立刻更有用”。

### B：look-at-light（找物体 + 台灯）

两个 cohort 都 **没有 K6+ clean decision**，所以完全无法检验长 terminal-run。

## 为什么 v0.1 的正信号作废

第一次 adapter 把动作和 observation 当成同一步结果，且没有正确排除已经完成的 object-subgoal；这会把失败/成功、resolved/unresolved 错位。v0.2 纠正后重新跑 frozen cohorts，因此只保留 v0.2 及本审计。

## 对课题的影响

受控 terminal-run 因果结果仍保留：在人工严格匹配状态里，K1→K6 能改变下一步选择。MATM 真实轨迹只提供一个**稀疏、方向相符但不足确认**的 A-family 信号；第二 family 没有长 run。

所以论文级表述应保持：**terminal-run 是实验室受控因果现象；自然生态有效性目前仍未确认。** 这不会推翻总账里更强的 selection / retention / long-run value 脆弱性主线。

## 停止规则

本轮自然轨迹挖掘到此停止，不再用同一 MATM bank 调标签或阈值。若以后要升级自然证据，应使用新 benchmark / 新轨迹，并从一开始记录每步 workstream、是否 unresolved、候选替代方向与未来 outcome。
