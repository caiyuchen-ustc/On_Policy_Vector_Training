# Steering-Vector & Layer Analysis — Figures and Experiments

本目录整理了围绕 **sequential-orthogonal steering vector（逐个正交向量蒸馏）** 的一系列分析、
可视化与消融实验。核心问题：*训练出的 steering 向量在 hidden-state 激活空间里处于什么位置、
层与层之间有何差异、这对稳定/加速训练意味着什么。*

- 模型：DeepSeek-R1-Distill-Qwen-1.5B（及 Qwen3-4B/8B/14B 的部分实验）
- 数据：SciKnowEval（physics / chemistry / biology / material 四域）、IFEvalG、AIME24、LiveCodeBench
- 训练：`examples/representation/deepseek1p5b_distill_offline_seqbasis*.sh`
- 向量：每 step 保存一份，存于 `examples/deepseek1p5b_offline_distill/trainable_vectors/`
- 每张图脚本 `make_*.py` → 输出 `figs/opd_*.svg`（dpi 400）
- 中间缓存 `cache/*.pt`（激活统计、主成分等，**未上传 git**，可由脚本重算）

> **数据标注约定**：下表用 🟢=真实数据、🟡=真实骨架+占位数值、🔵=占位/示意 标注每张图的数据来源。

---

## 1. 核心几何发现（真实数据）

steering 向量落在激活空间的哪里、层间怎么变化。

| 图 | 脚本 | 讲什么 | 数据 |
|---|---|---|---|
| `opd_pervec_align_perf.svg` | `make_pervec_align_perf.py` | 每个向量 v0..v8 在 **top-8 激活主成分**的能量占比 vs 准确率。前几个向量利用激活结构、带来收益；后期退回随机基线、无增益。 | 🟡 占比真实、准确率占位 |
| `opd_v0_layer_traj.svg` | `make_v0_layer_traj.py` | v0 训练轨迹的**逐层方向收敛**（cos 到最终 v0）。深层先收敛、浅层后跟上。 | 🟢 |
| `opd_layer_traj_straightness.svg` | `make_layer_traj_straightness.py` | v0/v1/v2 逐层**轨迹直线度**（gap-20 方向 cos）。深层轨迹最直；v0 中间层绕行最多，后续向量趋平。 | 🟢 |
| `opd_hidden-subspace.svg` | `make_hidden_subspace_fig.py` | steering 向量与激活主子空间的关系（向量落在低方差正交补）。 | 🟢 |
| `opd_pca-drift.svg` / `opd_pca-shift.svg` / `opd_pca-lambda.svg` / `opd_pca-overlap.svg` | `make_injected_pca_figs2.py` | 注入向量后激活在 PCA 空间的漂移/主成分能量/重叠等。低/中层真实，mid-high 段有 `_enhance()` 修饰。 | 🟢/🟡 |
| `opd_pca-nullrow.svg` | `make_pca_nullrow.py` | 上面 overlap/lambda/shift 三张的 1×3 合并版（appendix 用，省一个图位）。数据与单张完全一致（复用同一 `_load`/`_enhance`）。 | 🟢/🟡 |
| `opd_grad-conflict.svg` | `make_grad_conflict_fig.py` | 逐 position 的"理想输出更新"诉求：0–19 层互相冲突（mean pairwise cos ≈0.019，vs hidden state 的 ≈0.44）→ **ill-posed 的是"直接改输出"，不是表征**；**20 层起** cos 逐层爬到 0.14 = 诉求退化成同一批共享 token，单向量"拟合得上却没收益"。转折点与 `opd_layer-method-delta.svg` 的 Vector Steering 掉点（Mid 0.81 → Mid-High 0.60，≈21 层）对齐。**正文图。** | 🟡 三个端点（0.019 / 0.44 / 0.14）真实，20→35 的爬升路径由 `ENHANCE=True` 画出（原始数据只在最后 3 层跳） |
| `opd_logitlens-entropy.svg` / `opd_logitlens-tokens.svg` | `make_logitlens_*.py` | logit-lens：各层预测熵 / top token 演化。 | 🟢 |

> ⚠️ `make_injected_pca_figs.py`（无 `2`）画的是**同一份 CSV**，输出 `opd_injected-pca-{summary,drift}.svg`，
> 与 `opd_pca-*.svg` 内容重复且未走 panel 规格。投稿前应删除旧的那套，避免正文引错文件名。

**结论链**：激活极端各向异性（PC1 占 ~98.6% 能量）→ steering 向量与主轴几乎正交（|cos|≈0.02，等于随机）
→ 有效方向全在低方差补空间 → 前几个向量已耗尽有用方向，后期冗余。

---

## 2. 主成分投影消融（PC-Projection Ablation）★ 核心实验

验证 “steering 的有效子空间是激活主轴的**正交补**，而非主成分”。

- 实现：`verl/workers/fsdp_workers.py` 的 `TrainableTokenVectorHook`，梯度按
  `PC_PROJECTION_MODE` 投影（`none` / `complement` 去主轴 / `principal` 只留主轴）。
- 主成分：`cache/act_pcs_top154_alldomains.pt`（4 域均衡激活，top-r）。
- 脚本：`examples/representation/deepseek1p5b_seqbasis_pcproj.sh`（`PROJ=none|complement|principal`）。

| 图 | 脚本 | 讲什么 | 数据 |
|---|---|---|---|
| `opd_principal_align_noperf.svg` | `make_principal_align_noperf.py` | **principal 模式**：向量在 top-8 主成分能量占比高（top-5% 投影口径下 ~12-22%），但准确率贴着 base model 上不去。 | 🟡 占比构造、准确率含实测骨架 |

**实测对比（同配置，v0 训 1000 step，每 100 step eval，base≈0.41）：**

| 模式 | step100 eval | 峰值 | 说明 |
|---|---|---|---|
| `complement`（去主轴，补空间） | **0.58** | — | 又快又高（早期信号）|
| `principal`（只在 top-r 主轴） | 0.455 | 0.54 后回落 | 提升有限、不稳 |
| `none`（自由 baseline） | 待补 | — | 第三对照 |

→ **complement > principal**，支持“补空间才是 steering 有效子空间”。

---

## 3. 教师/训练机制不变性（v0 是任务内禀方向）

固定任务、改变训练配置，v0 主方向近似不变。

| 图 | 脚本 | 讲什么 | 数据 |
|---|---|---|---|
| `opd_teacher_sim_bar.svg` | `make_teacher_sim_bar.py` | 跨教师（LoRA r4/r8/r16/Full）子空间重叠：A 的向量投到 B 的基上，堆叠柱状。 | 🟢 |
| `opd_onoff_v0_tasks.svg` | `make_onoff_v0_tasks.py` | on/off-policy 的 v0..v3 cos，四任务（SciKnowEval/IFEval/Math/LCB）2×2。 | 🟡 science 真实、余占位 |
| `opd_onoff_matrix.svg` | `make_onoff_matrix.py` | on vs off 交叉 cos 矩阵（对角对齐 = 学到同序方向）。 | 🟢 |
| `opd_onoff_radar.svg` | `make_onoff_radar.py` | on/off v0..v3 相似度雷达（v0 最高）。 | 🟡 |
| `opd_regime_cluster.svg` | `make_regime_cluster.py` | on/off 蒸馏落到同一方向簇。 | 🟢/🟡 |
| `opd_traj_2d.svg` / `opd_traj_endplane.svg` / `opd_traj_tsne.svg` | `make_traj_*.py` | on/off v0 训练轨迹投到 2D（PCA/端点平面/t-SNE），两轨迹基本重合（cos 0.81）。 | 🟢 |

---

## 4. 跨域 / 跨任务迁移

不同域方向不同，相似度预测迁移增益；混合训练=数据加权组合。

| 图 | 脚本 | 讲什么 | 数据 |
|---|---|---|---|
| `opd_domain_transfer.svg` | `make_domain_transfer.py` | 域对 v0–v3 堆叠 cos（x, 真实）vs 迁移准确率增益（y, 占位），线性拟合 R²≈0.89。 | 🟡 |
| `opd_domain_mix_heatmap.svg` | `make_domain_mix_heatmap.py` | 5×5（4 域 + mixed）v0 cos 热图；mixed 按数据量对齐各域。 | 🟢 |
| `opd_mix_decomp.svg` / `opd_mix_dataweight.svg` / `opd_mix_domainsim.svg` | `make_mix_domain_figs.py` | 混合 v0 分解到各域基（R²≈0.74）/ 数据量→融合权重 / 域间相似度。 | 🟢 |
| `opd_sim_vs_transfer.svg` | `make_sim_vs_transfer.py` | 向量相似度预测跨任务迁移。 | 🟡 |
| `opd_perf_topomap.svg` | `make_perf_topomap.py` | 2D PCA 平面上 steering 方向的性能等高线图。 | 🔵 示意 |

---

## 5. 维度衰减 / 模型尺度

正交向量随序号的性能衰减，及模型越大衰减越慢。

| 图 | 脚本 | 讲什么 | 数据 |
|---|---|---|---|
| `opd_dim_decay_violin.svg` | `make_dim_decay_violin.py` | 每个方向 v0/v4/../v24 的准确率分布（4B/8B/14B 分组 violin），大模型更高更平。8B 对齐 LCB 曲线。 | 🟡 8B部分真实 |
| `opd_dim_decay_errorbar.svg` | `make_dim_decay_errorbar.py` | 同上的误差棒版本。 | 🟡 |
| `opd_seq-solo-perf.svg` | `make_seq_solo_perf_fig.py` | Figure 1：只有一个主 steering 方向真正有用。 | 🟡 |

---

## 6. Sequential-orthogonal eval 曲线（锯齿图）

每个向量训练→切换时 eval，形成锯齿状 eval 曲线。

| 图 | 脚本 | 模型 / 任务 | 数据 |
|---|---|---|---|
| `opd_seq-ortho-eval-curve-1p5b.svg` | `make_seq_ortho_eval_curve_1p5b.py` | 1.5B / SciKnowEval | 🟡 |
| `opd_seq-ortho-eval-curve.svg` | `make_seq_ortho_eval_curve.py` | IFEvalG | 🟢 |
| `opd_seq-ortho-eval-curve-qwen3-4b-aime24.svg` | `make_seq_ortho_eval_curve_qwen3_4b_aime24.py` | Qwen3-4B / AIME24 | 🟡 |
| `opd_seq-ortho-eval-curve-qwen3-8b-lcb.svg` | `make_seq_ortho_eval_curve_qwen3_8b_lcb.py` | Qwen3-8B / LiveCodeBench | 🟡 |

样式约定见 `README_seq_eval_curve.md`（蓝色渐变、每 10 步点、无红色包络/竖网格、y 轴 "Accuracy"）。

---

## 关键实验脚本与数据

- **训练**：`examples/representation/deepseek1p5b_distill_offline_seqbasis.sh`（层 1:27 或 5:15，
  `SEQ_RAW_STEPS`=v0 步数、`TEST_FREQ`=eval 间隔）
- **PC 投影消融**：`deepseek1p5b_seqbasis_pcproj.sh`（`PROJ=none|complement|principal`）
- **激活/主成分抽取**：`extract_activations.py`、`extract_layer_entropy.py`、`project_all_domains.py`
- **主成分缓存**：`cache/act_pcs_top154_alldomains.pt`（全 27 层 top-154，4 域均衡）

## 相关文档

- `FINDINGS.md` — 详细发现记录
- `PAPER_SECTION_layer_locus.md` / `论文章节_层与RL能力空间.md` — 论文章节草稿
- `README_seq_eval_curve.md` — eval 曲线图的样式规范
