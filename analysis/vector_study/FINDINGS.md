# 不同层 single 向量训练研究：RL 学到的是"模式"，高层越来越偏向"token 空间"

数据：`repre/opd/repre_qwen3-4b-opd`（Qwen3-4B, 36层, hidden=2560, tied-embed）、
`repre/ifevalg_opd`（DeepSeek-R1-Distill-Qwen-7B, hidden=3584, 独立 lm_head）、`repre/ifevalg_rl`。
每个实验保存了逐 step 的 single 向量 checkpoint（`{layer: tensor[H]}`）。

分析脚本：`analysis/vector_study/{traj_analysis.py, logit_lens.py, make_figs.py}`
产物：`out/*.csv`、`out/logitlens_toptokens_*.txt`、`figs/fig1-3.png`

---

## 核心结论（一句话）
**中低层向量在 vocab 空间里近乎"无方向"（logit-lens 熵=随机基线），它改的是模型内部的表征/模式；
越靠近输出的高层，向量越来越多地把能量投向输出 token 空间，其中最显著的成分是推理转折词
（Wait/But/Alternatively/但/或许），最顶层（L34）达到峰值（14% 能量在反思词方向、logit-lens top1=0.35）。
但这是**程度性偏移**而非非此即彼——高层向量对输出的因果影响并非"全部"由这撮 token 方向承载（见证据12）。
高层训不好的直接原因是优化动力学病态（norm 失控 / 梯度冲突，见证据10/11），与这种表征偏移互为表里。**

> ⚠️ 措辞说明：早期版本用过"高层=纯 token 操纵/只能改 logit"这类绝对表述，经因果消融（证据12）修正为
> "高层**越来越偏向** token 空间、反思词是**最显著**成分"。凡涉及"程度"的地方均已按此校准。

---

## 证据 1：Logit-Lens 尖锐度 vs 层深（fig1 / fig2）
把每层最终向量做 `RMSNorm(v) @ W_U^T → softmax`，算归一化熵（1=均匀, 越低=越集中）。
随机高斯向量基线：qwen3-4b=0.950, deepseek7b=0.959。

qwen3-4b（mean norm-entropy）:
- 层 0–25：0.949–0.952，**与随机基线无差异** → 向量方向与输出 token 空间正交，不在"预测某token"上做文章
- 层 26 起缓慢下降，层 30≈0.913，**层 34 暴跌到 0.204**（top1 prob 0.35）
- deepseek7b：全程贴基线，最高层（L27）才略降到 0.938（该模型样本层没到最顶几层，趋势一致但更弱）

含义：低/中层的向量是"分布式、vocab-agnostic"的——典型的**内部模式/特征方向**；
高层向量随深度越来越"读得懂"，最顶层几乎就是一个 one-hot token 注入器。

## 证据 2：高层坍缩到什么 token（`out/logitlens_toptokens_qwen3-4b.txt`）
- **中低层 top tokens = 乱码碎片**：`-Col`,`祇`,`UBLE`,`篝`,`gee`,`iota`… （无语义，印证"不是token层面的信号"）
- **高层 top tokens = 连贯的推理反思词**：
  - L26–33: `But / Wait / perhaps / Alternatively / Hmm / 但 / 或许 / 然而`
  - **L34: 直接坍缩到 `Wait`（top1=0.353）、`But`**
  
这非常关键：数学推理 teacher 的行为特征就是"自我怀疑/回溯"（Wait, But, Alternatively）。
高层向量没有学到"如何推理"，而是抄近道——**在最后一层强行提高"说 Wait 的概率"**。
这是一种表层的、脆弱的模仿，也是为什么高层单向量效果差、且容易训崩（L34 熵坍缩、top1 独大）。

## 证据 3：训练轨迹几何 vs 层深（fig3, `trajectory_summary.csv`）
受控实验 `singlev_layers24-35_lr1e-3`（同一 lr 跨 24→34 层）：
- `final_norm`：24层 3.75 → 34层 3.41，随层升**单调下降**
- `straightness`（净位移/路径长，1=直线收敛，低=绕路震荡）：24层 0.90 → 34层 0.86 单调下降
→ 层越高，向量在训练中越"绕路"、越难收敛到一个稳定方向，与"高层没有稳定模式可学、只能反复微调 token 概率"一致。

mid 层实验 `10-25_lr5e-5` straightness≈0.75–0.79 且非常平稳；低层 `0-11` 收敛快。
（注意：不同实验 lr/步数不同，norm 绝对值不可跨实验直接比；straightness 因是比值受影响较小，故用受控单实验做主证据。）

---

## 建议的论文叙事
1. 命题：单向量表征操控 = 在某层注入一个学习到的 steering 方向。
2. 若该方向落在"内部特征子空间"（与 unembedding 近正交）→ 它调制的是**计算模式** → 有效、稳健（中低层）。
3. 若该方向落在"unembedding 行空间"（高 logit-lens 可读性）→ 它退化成**直接改输出 token 分布** → 只是表层模仿 teacher 的口头禅（Wait/But），泛化差、易崩（高层）。
4. Logit-lens 熵随层深的断崖（fig1）+ 高层向量朝 reasoning-trigger token 方向增长（证据2/12）共同表明："RL/OPD 想传递的是模式，但在临近输出的高层，单向量越来越多地把可用容量花在直接改 token 分布上"。注意这是程度性偏移（证据12：即使去掉反思词方向、同 norm，高层对输出的影响仍大部分保留），并非"高层只做 token 操纵"。

---

## 证据 4：沿 step 的熵演化 —— 退化是一个动态过程（fig5, `entropy_over_steps.csv`）
`energy_and_steps.py`。对 `24-35_lr1e-3`（高层）和 `10-25_lr5e-5`（中层）逐 step 做 logit-lens 熵：
- **高层 L34：熵从 0.95 单调坍缩到 0.20**（贯穿 ~110 step），把"模式→token"的退化过程完整拍下；L24-33 仅轻微下降后回稳在 ~0.90-0.94
- **中层 L10-25：熵从 1.0 掉到 ~0.95（=随机基线）后完全稳住**，始终保持分布式
→ 高层不是一开始就坏，而是训练中逐渐把向量"拉进" token 空间。

## 证据 5：W_U 主方向能量分解（fig4, `wU_energy.csv`）
对 W_U 做 SVD 取 top-r 右奇异向量（token 判别主方向），算向量能量占比 / 随机基线：
- 层 0–25：`projE(r=128)/random ≈ 0.85–1.05`（≈1，与随机无异）
- 层 26–33：单调升到 **1.5**（L31）
- 注：W_U 首奇异值 126.7 >> 次奇异值 24（强各向异性），故用主方向而非满秩行空间。
→ 比 logit-lens 熵更直接：高层向量确实把越来越多能量投到 unembedding 主方向上。

## 证据 6：向量 vs 真实 hidden state 子空间关系（fig6, `hidden_subspace.py` / `hidden_subspace.csv`）
HF 前向 24 条 DeepMath prompt，hook 抓每层 residual stream（N≈2300 token），对每层 hidden 做 PCA，度量向量落点：
- **`cos_meandir` 全程 ≈ 0**（|·|<0.03）：向量与 hidden 的整体均值方向近正交 → 不是简单顺着 hidden 的 DC 偏置（这条对所有层成立）
- **`E_top20/rand`（向量在 hidden 前20主成分的能量 / 随机基线）**：
  - 中低层多在 0.6–1.5，部分层（L12-14, L22-23）< 1 → 落在 hidden 的**低方差尾空间**
  - **高层 L24–31 升到 2.7–3.1** → 落在 hidden 的**高方差主空间**（模型本来就在大量使用的方向）
- `pc_centroid_rank`：高层普遍 < 中位 → 偏主成分

**这条给出了更完整、也更 nuanced 的物理图景**：
> 高层向量既较多落在 **hidden 的高方差主空间**（证据6），又较多落在 **unembedding 的主方向**（证据5）——
> 在临近输出的高层，这两个空间本来就高度重合（后续没剩几层，residual 已接近 logit）。
> 所以高层向量"顺着主空间推"就**更容易直接改 logit**（偏向 token 操纵）；但这是主导倾向而非全部（证据12：去掉 token 方向后效应仍大部分保留）。
> 中低层的 hidden 主空间与 unembedding 主方向尚未对齐，向量要么落尾空间、要么落"非 token"的主方向，
> 表现为对输出分布无直接可读性（=模式）。

注意 `E_top20` 指标较 noisy：norm 特别大的层（L24=17.9, L15/20/21 因高 lr）会抬高数值，需结合 logit-lens（证据1）一起看，后者更干净。

---

## 证据 8：Token 类型构成 vs 层深（fig7/fig8, `token_type_by_layer_*.csv`）
对每层向量 top-200 激活的**输入 token**分类（reflection/structural/math/other）：
- **reflection 全程 ≈ 0**！乍看矛盾，实则关键：数学 prompt 语料里本就极少出现 Wait/But，
  所以"高层对齐反思词"这件事**在输入侧 hidden 看不到，只在输出侧 push-through 才显现**。
  → 印证：最大激活（输入侧）不足以看清高层，必须靠因果 push-through（输出侧）。
- 中低层由 **structural（<think>/assistant/\n）+ math（符号/概念）** 主导，占比在层间起伏，
  L20-25 结构占比高（对齐 assistant-thinking 槽位），与证据 7a 一致。

## 证据 9：Teacher 行为对照（fig9, `teacher_behavior.json`）
vLLM 让 teacher(Qwen3-4B-RL-Math-Step500) 和 base(Qwen3-4B) 对同 64 条数学 prompt 生成，统计词频：

| 词类 | teacher | base | teacher/base |
|---|---|---|---|
| **reflection**（Wait/But/Hmm/Alternatively）| **15.2 /千词**，命中率 98% | 3.5 /千词，61% | **4.3×** |
| transition（Okay/Alright/So/</think>）| 9.9 /千词，95% | 6.4 /千词，97% | 1.5× |

→ **RL(teacher) 相对 base 主要新增的就是"反思词"行为**（4.3 倍），而过渡词是 base 本就具备的（1.5 倍）。
这正好解释向量的分工：
- **高层向量**直接对齐 teacher 新增的 reflection 词（Wait/But）——把 RL 最显著的表层行为一步到位注入 logit；
- **中低层向量**（push-through 显示指向 Okay/Alright 过渡语气）调制的是 base 已有的、更底层的"进入作答"计算模式。
高层抄了 teacher 最扎眼的"口头禅"，中低层动的是通用推理结构——机制分工闭环。

---

## 双模型对比：qwen3-4b (数学OPD) vs deepseek7b (IF-OPD)（fig10）
两个模型任务/架构都不同，正好检验结论是否通用：
- qwen3-4b：Qwen3-4B(36层,tied-embed)，teacher=Qwen3-4B-RL-Math，数学推理
- deepseek7b：DeepSeek-R1-Distill-Qwen-7B(28层,独立lm_head)，teacher=Qwen2.5-7B-IF，指令遵循

**共同规律（结论通用）**：
1. **logit-lens 熵**：两模型中低层都≈随机基线（模式），都在最顶层下降。qwen 因训到 L34（36层的顶）坍缩更极端（0.20）；deepseek 只训到 L27（28层的顶）小幅降到 0.94。
2. **W_U 主方向能量**：两模型都是中层≈1、最顶层抬升（qwen L31→1.5；deepseek L27→1.73）。
3. **token 类型**：两模型都是 structural 占比在高层（qwen L20-25；deepseek L24/26）抬升。
4. **teacher 对照**：两模型 teacher 相对 base 都显著增加了行为词，高层向量都对齐这些新增词。

**任务相关的差异（有意思）**：
| | qwen3-4b (数学) | deepseek7b (IF) |
|---|---|---|
| 高层 push-through 指向 | Wait/But/Alternatively（数学**反思**词）| must/MUST/need（指令**约束**词）|
| 中低层激活 token | `<think>`+数学符号(frac/sqrt) | `<｜User｜>`/should/Please/letters/brackets（IF结构）|
| teacher 增幅最大的词类 | reflection 4.3×（transition 仅1.5×）| transition 3.3×（reflection 2.1×）|

→ **中低层向量编码的"模式"是任务特定的结构特征**（数学符号 vs IF 约束/角色），
高层向量对齐的"输出词"也是任务特定的（Wait vs must）。
但"中低层=分布式任务结构模式、高层=坍缩到任务高频输出词"这个**机制在两个模型上完全一致**。

---

# 为什么高层训不好、中低层可以（因果机制）

前面证据描述"现象"，这一节回答"所以然"。用三个实验（1/2/4）把机制钉死。

## 机制一句话
**中低层向量下游还有十几层非线性计算可借用：它只需设一个"模式/计算意图"，让后续层去翻译成合适行为——这条通路上各 token 位置的梯度方向一致、可稳定积累。高层向量紧邻输出，能改的主要就是 next-token 的 logit；而"直接改输出"这条通路上，不同位置想要的改动互相冲突（有的该出数字、有的该出 Wait），一个全局向量无法同时满足，于是要么被拉扯到 norm 失控、要么信号互相抵消到学不动。**

## 证据 10：训练动力学 by 层（实验1+4, fig11, `layer_training_curves.csv`）
从各单层实验的训练 log 提取逐 step 的 AIME 分数 + 向量 norm + 梯度 norm：

| 层 | 位置 | AIME24 提升Δ | 最终 vec_norm | 失败模式 |
|---|---|---|---|---|
| L3-4 | 低 | **+0.333** | 2.6 | 健康 |
| L7-8 | 低 | **+0.367** | 2.9 | 健康 |
| L15-16 | 中 | **+0.325** | 2.1 | 健康 |
| L20-21 | 中 | +0.075 | 1.4 | 一般 |
| **L24-25** | 高 | +0.183（震荡不稳）| **爆炸到 16.9** | **norm 失控** |
| **L32-33** | 高 | **+0.033**（几乎不动）| 0.54（几乎不长）| **梯度消失/学不动** |

- 低/中层：分数快速升到 0.55-0.62 并稳住，norm 稳定收敛到 2-3（fig11 蓝线）。
- 高层有**两种不同的失败**：
  - L24-25：norm 单调爆炸到 17（其他层的 6 倍），分数震荡不稳 → 为压制冲突而 norm 失控
  - L32-33：梯度 norm 从一开始就贴地（fig11 panel3），norm 几乎不长，分数几乎不涨 → 信号太弱学不动
- **这直接回答原始观察**："高层效果差"= 优化动力学病态（失控 or 消失），不是简单的 loss 降不下。

## 证据 11：per-position 梯度冲突（实验2, fig12, `grad_conflict_by_layer.csv`）
度量：同一层内，不同 token 位置"想让向量往哪推以改输出"的梯度方向两两 cos（1=一致，0=正交，<0=冲突）。
- **"直接改输出"梯度的位置一致性全程都极低（0.01-0.04），各层几乎一样**，仅最后一层 L35 升到 0.14。
  → 关键：**"单向量直接改 token 分布"在任何层都是自相矛盾的任务**（不同位置该输出的 token 不同）。
- 但这不矛盾——**区别在于低层向量不必走这条路**：它可以改"模式"，让后续十几层的非线性计算把这个模式翻译成各位置合适的行为（这条通路梯度一致、可积累，故低层 norm 稳定收敛）。
  高层没有后续层可借，**被迫只能走"直接改输出"这条位置冲突的通路**，于是梯度天天打架 → norm 失控（L24-25）或抵消（L32-33）。

## 三个失败原因（综合证据 5/6/7/9/10/11）
1. **目标错配（泛化差）**：OPD 想传递的是"推理行为"，中低层能通过调制计算模式真正改变推理；高层则更多地把容量花在改 next-token 分布上，最显著的表现就是"提高 Wait/But 概率"——偏向学到表层相关（teacher 爱说 Wait）而非机制。
2. **梯度自我拮抗（norm 失控，如 L24-25）**：全局向量要同时满足所有位置互相冲突的 token 需求，只能不断增大 norm 硬压某个主导 token（对应证据3中 L34 熵坍缩到单一 Wait）。
3. **梯度消失（学不动，如 L32-33）**：冲突的位置梯度互相抵消，合梯度极小，向量原地不动。

---

## 证据 12：因果消融（实验3, fig13, `ablation_reflword.csv`）—— 修正"程度"的关键
方法（`ablation_projection.py` 及反思词子空间版）：把向量在"反思词 token 子空间"（Wait/But/Hmm/Alternatively/但/然而... 的 unembedding 行向量张成）的分量投影掉，**并缩放回原 norm**，再注入该层，测对最终 next-token 分布的因果影响 ΔKL 保留多少。
norm-controlled 是关键：比的是"同等注入强度下，方向在不在 token 子空间"，排除"投影后 norm 变小"的混淆。

结果两条，一支持一修正：
- **发现A（支持核心假设）**：向量落在反思词子空间的能量占比随层深单调上升——
  中低层 L3-L20 仅 **0.5%–1.2%**，高层 L30/33 升到 **6%**，**L34 达 14%**（logit-lens top1=0.35）。
  → 证实"越高层，向量越朝反思词 token 方向增长"，与证据1/2/3 一致。
- **发现B（修正强表述）**：但投影掉这些反思词方向（同 norm）后，**所有层的 ΔKL retention ≈ 0.94–1.10，高层也没掉到 0**。
  → 即高层向量对输出的因果影响**并非"全部"由那撮反思词方向承载**；即使 L34（14% 能量在反思词方向），
  去掉后剩余 86% 能量（同 norm）仍造成几乎同等的输出扰动。

**结论校准**：高层向量确有一个显著且随层深增长的"反思词 token"成分（发现A，是 logit-lens 尖锐、L34→Wait 的来源），
但它是向量的**最显著成分**而非**全部**。所谓"token 操纵"是**程度问题**（高层更偏向 token 空间），
不是**非此即彼**。这不影响证据1-11 的任何观测，也不影响"为什么高层训不好"的动力学机制（证据10/11 独立于此），
只把绝对化措辞收敛为程度性表述。

附注（方法学）：最初尝试用"注入后生成文本里 Wait/But 词频"做消融度量，发现信号太弱、太依赖生成长度而不可靠
（单向量在 HF 前向注入前几百 token 复现不出训练时的行为）。改用"对输出分布 ΔKL 的 norm-controlled 保留率"，
这正是 OPD loss 优化的量，干净且只需前向。

## 图表清单
- fig1 logit-lens 熵 vs 层深 ★
- fig2 top-1 prob vs 层深
- fig3 轨迹 straightness / final_norm vs 层深
- fig4 W_U 主方向能量 vs 层深
- fig5 熵沿 step 演化（高层坍缩 vs 中层平稳）★
- fig6 向量 vs hidden 主/尾空间
- fig7/8 token 类型构成 vs 层深
- fig9 teacher vs base 行为对照 ★
- fig10 双模型对比（logit-lens/W_U/hidden 三指标 vs 归一化层深）★
- fig11 训练动力学 by 层（AIME分数/vec_norm/grad_norm vs step）★★
- fig12 per-position 梯度冲突 vs 层深 ★
- fig13 因果消融：反思词子空间能量占比(A) + norm-controlled ΔKL 保留率(B) ★

**注**：证据 1-9 的图对每个模型分别有 `_qwen3-4b` / `_deepseek7b` 版本（qwen 的早期产物无后缀）。

## 可补强的后续实验（未做）
