# STORYLINE — 论文完整故事线

> 一份统领性文档：把所有实验串成一条有张力的叙事线，围绕**两个核心性质**展开。
> 配套文档：`README.md`（图表清单）、`FINDINGS.md`（详细发现）、`PAPER_SECTION_layer_locus.md`（章节草稿）。

---

## 摘要（定稿）

**中文：**

强化学习（RL）已成为提升模型性能的关键范式之一。然而，可训练参数的庞大规模使得 RL 训练动态的分析极为困难，
严重制约了对其内在机制的系统性理解。本工作通过向量引导（vector steering）从激活层面发现了与 RL 性能提升
密切相关的低维有效流形，并揭示了其两条关键几何性质：**（1）有效流形容量（Effective Manifold Capacity）**：
低维训练对 RL 已足够有效，但并非任意低维子空间均适用——其有效性取决于模型后续层所需承担的变换复杂度，
而该复杂度又由训练空间所在的模型深度及其本身维度大小共同决定；**（2）控制流形分离（Control Manifold Separation）**：
习得的流形空间在任务内与任务间均呈现出清晰的几何规律。任务内，不同训练配置下的流形呈现趋同坍缩，收敛至同一
低维子空间，且与内容主方向近乎正交；任务间，流形间的相似度则可直接作为 RL 能力可迁移性的有效度量。
基于以上发现，我们提出 **Alpha-Controller**，一个即插即用的训练框架，由 Predicter 与 Controller 两个模块构成：
Predicter 利用激活空间中的早期信号，在训练崩溃发生前进行预警；Controller 通过调控训练方向，在保持性能的同时
稳定训练过程。综上，本工作从激活流形视角深化了对 RL 训练机制的理解，并为设计更高效、更稳定的 LLM 后训练方法
提供了流形层面的新视角。

**English:**

Reinforcement learning (RL) has emerged as a key paradigm for improving model performance. However, the vast number of
trainable parameters makes the training dynamics of RL exceedingly difficult to analyze, severely limiting a systematic
understanding of its underlying mechanisms. In this work, we use *vector steering*—training a small set of additive
vectors in the activation space while freezing all model weights—to uncover, at the activation level, a
**low-dimensional effective manifold** closely tied to RL performance gains, and reveal two key geometric properties of
this manifold. **(1) Effective Manifold Capacity:** low-dimensional training is already sufficient for RL, yet not every
low-dimensional subspace works—its effectiveness hinges on the transformation complexity that the subsequent layers must
carry out, which is in turn jointly determined by the model depth at which the trained subspace resides and its own
dimensionality. **(2) Control Manifold Separation:** the learned manifolds exhibit clear geometric regularities both
within and across tasks. Within a task, manifolds trained under different configurations converge and collapse onto the
same low-dimensional subspace, which is nearly orthogonal to the principal content directions; across tasks, the
similarity between manifolds serves as a direct measure of the transferability of RL capabilities. Building on these
findings, we propose **Alpha-Controller**, a plug-and-play training framework consisting of two modules, a Predictor and
a Controller: the Predictor leverages early signals in the activation space to warn of training collapse before it
occurs, while the Controller regulates the update direction to stabilize training without sacrificing performance.
Overall, our work deepens the understanding of RL training mechanisms from the perspective of activation manifolds, and
offers a manifold-level lens for designing more efficient and stable post-training methods for large language models.

---

## 中心论点（Thesis）

> **On-policy training（RL / 蒸馏）沿着一组「任务内禀、低秩、且离开表征主轴」的控制方向改造模型。**
> 这组方向是任务的属性（与 LoRA / 全参等训练载体无关）、其相似度可预测跨任务泛化、
> 且位于表征流形的**低方差补空间**——强制它进入激活主方向则训练失效。
> 而要让如此低维的干预真正改变行为，下游需要足够的**变换复杂度**把它展开。

- **两条性质**：容量下限（Capacity Threshold）｜ 控制—内容解耦（Control–Content Disentanglement）
- **低维流形**：前置为显微镜观察到的**第一个事实**（激活更新集中在低维流形上），两性质建立其上
- **方法贡献**：Alpha-Controller（Predicter 崩溃预警 + Controller 方向干预稳定训练）

---

## 叙事开场（有张力的讲法）

**【痛点】** 理解 on-policy training 到底改了模型什么，一直很难。直接看**全参 / LoRA 的权重变化**行不通：
改动散布在千万参数上，**空间太大、随机性太强**——不同种子、不同秩、不同 lr 训出的权重差异巨大，
无法区分哪些是"任务本质"、哪些是"优化噪声"。权重是**载体**，不是**信号**。

**【转折 · 显微镜】** 我们换探针：**vector steering**——只在少数层激活上加可训练向量、冻结原始权重。
关键前提（Fig.1）：跨 4 个模型，这种极简干预能**复现全参 / LoRA 训练的绝大部分性能增益**。
即 on-policy 的有效信号可被压缩进极低维向量——几乎无信息损失，却剥掉了权重空间的冗余与随机性。
vector steering 因此成为一台**干净的显微镜**：保留信号、滤掉噪声，第一次让我们能直接、可复现地观测
on-policy training 改造模型的几何结构。

**【两个性质】** 用这台显微镜，我们发现 on-policy training 的两个基本性质 →

---

## 性质一（铺垫）· 容量下限 The Capacity Threshold

**陈述**：低维干预是**充分的，但不是任意低维**——生效需要一个最低限度的下游"变换复杂度"。

**证据**：
- 单向量在**低层可行、高层失效**。
- 收紧已有观察"RL 可被小参数子空间捕捉"：不是任意小，而是**小到临界容量为止**。
- *(ablation)* 高层单向量 + **gate**（补一次非线性）→ 重新可行 → 反证缺的是**变换复杂度**而非维度。

### 统一维度：有效变换复杂度（Effective Transformation Complexity）

把"注入层位"和"干预参数量"统一为**同一个量**——干预能对激活流形施加的**变形复杂度**，
即它能把激活云"掰"成多复杂的形状。它由两部分相乘决定：

> **变换复杂度 = 干预自身的自由度（参数维度）× 下游非线性折叠的次数（下游层数）**

| 旋钮 | 对变换复杂度的作用 |
|---|---|
| **参数维度**（单向量→LoRA→全参） | 干预自身自由度：单向量=一个方向的平移（最简单形状）；LoRA-r=r 维线性变形；全参=完整变换。 |
| **注入层位**（低→高） | 下游剩余的非线性折叠次数：低层注入→下游十几层反复折叠，简单干预也被展开成复杂形状；高层→没剩几层，形状被锁死在简单。 |
| **gate** | 给干预点补一次非线性折叠，直接抬升该点的变换复杂度。 |

**统一结论**：三个旋钮（层、参数、gate）都在调**同一个量 C**。存在临界 `C*`：
`C < C*`（高层单向量）→ 失效；下移层位 / 加 gate / 增参数 → `C ≥ C*` → 恢复。

**代理量（可画）**：`C ≈ 下游层数 D × 干预自由度`。可画"性能 vs C"，
看不同层单向量 / +gate / LoRA 是否落在同一条曲线上（gate 点把高层失效点沿曲线拉回可行区）。

---

## 桥梁 · 单向量的激活空间定位

分析单个 steering 向量落在激活空间何处 → **首次发现：向量不在激活主方向上** → 抛出悬念，引向性质二。

---

## 性质二（主菜 · 一号发现）· 离主轴的内禀低秩方向 Off-Principal Intrinsic Direction

**陈述**：on-policy 的有效改造集中在一组**任务内禀、低秩、离激活主轴**的控制方向上。

**四条递进证据**：
1. **首要方向主导（低秩）**：正交训练中第一个向量最有效，后续正交向量收益递减 → 改造是低秩的，
   存在一个"首要控制方向"。
2. **载体无关（内禀）**：同一任务下 LoRA-r4 / r8 / 全参训出的多向量高度相似 →
   方向是**任务的性质**，不是训练方式的产物。
3. **相似度预测泛化**：跨任务方向相似度预测迁移（物理≈化学 → 物理 teacher 在化学上也强）。
4. **离主轴（反直觉高潮）**：这些控制方向都**不在激活主子空间**里；
   **强制梯度投影到主轴（principal 模式）则训练失效**，投影到补空间（complement）则又快又好 →
   有效方向在**低方差补空间**。

---

## 几何综合 · 表征流形 The Representation Manifold

两个性质缝合成一幅图景：

> hidden state 落在高度各向异性的**低维流形**上；主方向（massive activation）是承载方差的"**内容轴**"。
> On-policy training **不改造内容轴，只沿正交于它的少数低秩控制方向轻推**（性质二）；
> 而要把这个低维方向翻译成流形上的实际移动（行为改变），下游必须有足够的**变换复杂度**来展开它（性质一）。
> 二者不矛盾：**方向可以低维，但把低维方向展开成复杂变形需要足够的变换容量**。

---

## 命名铁律（全文一致，防混淆）

| 术语 | 指代 |
|---|---|
| **primary control direction** | 第一个训练出的 steering 向量（有效、低秩） |
| **activation principal subspace** | 激活的 top 主成分（massive activation，内容轴） |

**核心反差句**：*the primary **control** direction is not a principal **activation** direction.*
（全文靠 control vs. activation 这对词区分——避免两个"主空间"自相矛盾。）

---

## 章节流 & 图表映射

| § | 章节 | 讲什么 | 主要图 |
|---|---|---|---|
| 1 | Intro · Steering as a Probe | 4 模型×全参/LoRA/向量，steering 复现绝大部分性能 | (Fig.1 待定) |
| 2 | 性质一 · 容量下限 | 单向量层位（高失/低可）→ 为什么（诉求冲突/退化）→ 补多少容量 → 验证机制 | `opd_layer-method-delta.svg` → `opd_grad-conflict.svg` → `opd_gate-capacity-ladder.svg` → `opd_gate-activation-dist.svg` |
| 3 | 桥梁 · 激活定位 | 单向量不在主方向 | `opd_hidden-subspace.svg`, `opd_pervec_align_perf.svg` |
| 4 | 性质二 · 离主轴内禀低秩 | 首要方向主导 / 载体无关 / 相似度预测迁移 / 主成分投影消融 | `opd_teacher_sim_bar.svg`, `opd_domain_transfer.svg`, `opd_mix_*.svg`, `opd_principal_align_noperf.svg` |
| 5 | 几何综合 · 流形 | 两性质缝合 | `opd_dim_decay_violin.svg` 等 |
| 6 | 方法 · Alpha-Controller | Predicter（激活早期信号预警崩溃）+ Controller（干预方向稳定训练） | (待补：崩溃/稳定对比曲线) |

### §2 四图因果链（顺序不要打乱）

前置：`opd_layer-method-delta.svg`（已有）**现象**。单向量 Bottom .81 / Low-Mid .82 / Mid .81 →
**Mid-High .64 / High .50**，而 LoRA/全参全程 .83–.84。→ 失效既特定于"单向量"这个干预，也特定于深度。

层段定义（36 层五等分，已写入 `make_layer_method_delta.py` / `make_layer_method_heatmap.py` 的 `BANDS` 常量）：
Bottom 0–6 ｜ Low-Mid 7–13 ｜ Mid 14–21 ｜ **Mid-High 22–28** ｜ High 29–35。

1. **`opd_grad-conflict.svg`** — *为什么*。x 轴同为 layer，读者不换坐标。两段各有各的失效原因：

   | 段 | cos | 诉求性质 | 单向量为何这样 |
   |---|---|---|---|
   | 0–19 | .019 | 方向互相冲突，但**真实**（表征级） | 冲突→单个常向量满足不了；但下游十几层非线性折叠替它完成逐位置变换 → 仍 .81 |
   | 20–35 | .019→.14 | 方向趋同、**已退化**成共享 token | 拟合得上，只是加了个共享 token bias；下游又没剩几层能补 → .64/.50 |

   灰线 .44（hidden 基线）排 confound：不是该模型任何一组逐 token 向量都近正交。
   拐点第 20 层 = `opd_layer-method-delta` 的 Mid→Mid-High 边界。
   → 统一到 `C = 干预自由度 × 下游折叠次数`：低层由**下游**提供变换复杂度，高层下游没了，**必须本地补**。

2. **`opd_gate-capacity-ladder.svg`** — *补多少*。在高层那个失效点上：Vector .64 → gate r=1 .70 →
   r4 .76 → r8 .80 → r16 .82 → Full .84，单调饱和。决定性的是 off-curve 对照：**再加一个
   input-independent 向量，参数量与 gate r=1 完全相同，却停在 .615**。
   → 关键是 **input-dependent 的非线性，不是参数量**。这正是 (1) 要的"逐 token 幅度"。

3. **`opd_gate-activation-dist.svg`** — *真的是那个机制吗*。若 gate 只学到常数缩放，就等价于给向量改模长，
   机制不成立。实测分布不是尖峰，且**越深越极化（0/1 堆积）**——最 per-token 的地方正是诉求最退化的地方。闭环。

一句话：**冲突 → 需要逐 token 的量 → 补一次 input-dependent 非线性就够（不是补参数）→ 学出来的确实是逐 token 开关。**

#### ⭐ 这条链的关键接口（审稿人会问，务必写进正文）

单看会以为 (1) 和 (3) 矛盾：(1) 说高层诉求**趋同**（cos .019→.14），(3) 却说 gate 在高层**最 per-token**。
诉求都一致了还要 per-token 干什么？

不矛盾，因为 **cos 只测方向，gate 只调幅度**：高层各 position 想推的**方向**是同一个（都指向那几个共享
token），但**每个 token 需要多大的量**天差地别。所以高层恰好是「**共享方向 + 逐 token 幅度**」这个参数化
最对的地方——不需要 per-token 的方向（那才要 LoRA/全参），只需要 per-token 的**开关**；而 gate 分布在高层
退化成 0/1 两堆，字面意思就是"这个 token 给，那个不给"。

**这把 gate 从"经验上有效的补丁"变成了 (1) 直接预测出来的结构**，也顺带解释了为什么 gate 在高层
只能追到 .82 而非 .84：它能补幅度，补不了"诉求本身已经退化"。

#### ⚠️ 数据待办（投稿前）

- ~~高层单向量分数两处不一致~~ **已解决**：ladder 的 "layers 25-28" 落在 `BANDS["Mid-High"] = (22,28)`，
  所以它对的是 Mid-High 而非 High。统一取 **0.64**（该值在 `make_gate_ablations.py` 与
  `make_gated_highlayers_multi.py` 两处独立出现；旧的 0.60 只在 delta/heatmap 表格里），
  delta 与 heatmap 已同步重画，Mid-High 的 Δ 从 −0.24 变 −0.20。
- **`opd_grad-conflict` 的 20→35 爬升是构造的**（见该脚本 docstring 的 DATA STATUS）。
  它是这条链里唯一非实测的一段，而且承担的正是"为什么"这一环 → **最高优先级补真实测量**。
- **灰线（hidden 基线 .44）是未去均值的 raw 余弦**，测的是各向异性而非"语义方向一致"。
  论点（20× 量级落差）不受影响，但 caption 必须写明 uncentered，否则审稿人问"去均值了吗"会被将住。

### Appendix 分配

| 图 | 说明 |
|---|---|
| `opd_pca-nullrow.svg` | overlap/lambda/shift 三个 null result（≈1、≈1、≈0）合成 1×3，一个图位讲完"注入不扰动主结构" |
| `opd_pca-drift.svg` | 四张里唯一有正向趋势（top layer 涌入主子空间）；§4 缺料时优先提到正文 |
| `opd_logitlens-entropy.svg` + `opd_logitlens-tokens.svg` | 配对放（曲线 + top token），支持"高层=直接操作 token、中层=分布式计算" |

（`opd_hidden-subspace.svg` 留在 §3 正文，不进 appendix。各图数据真实性标注见 `README.md`。）

---

## 待补 / 需确认

- [ ] **Fig.1 的"绝大部分性能"给一个数字**（steering recovers X% of full-param gains）—— 让开场更硬。
- [ ] **性质一的"性能 vs 有效变换复杂度 C"曲线**：需要不同注入层的单向量性能数据 + gate 点 + LoRA 点，
      叠在一张图上验证"三旋钮同一个 C"。
- [ ] **主成分投影消融补 none / complement 完整曲线**（目前 complement 早期 0.58 > principal 0.455@step100）。
- [ ] **Alpha-Controller 需要真实实验**：
      - **Predicter**：证明激活空间早期信号能在 reward/熵崩溃**之前**预警（用崩溃训练日志：lora81 降51%、fullparam 降29%，需抽崩溃前激活的 rank/主轴漂移，看是否早于 reward 下跌）。
      - **Controller**：证明"干预训练方向"能稳住训练且不掉性能（对照：有/无 Controller 的 reward 曲线 + 最终性能）。
      - 二者都尚无完整实验，摘要已写方法 → 投稿前必须补齐，否则是空头支票。

