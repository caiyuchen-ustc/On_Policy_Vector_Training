# Sequential-Orthogonal Eval 曲线图 — 作图规范

脚本：`analysis/vector_study/make_seq_ortho_eval_curve.py`
输出：`analysis/vector_study/figs/opd_seq-ortho-eval-curve.svg`

后面做同类的"多向量逐个训练 eval 曲线"图，按下面这些约定来（都是评审友好的细节）。

## 数据 / 逻辑
- 数据锚点尽量用**真实 log 值**：base 基线、每个向量训完（切换点）的实测 eval。
  本图 base=0.135（val_before_train），来自 run r8u9jvnn / `ifevalg_distill_offline_seqbasismulti5e-2lora.log`。
- 曲线是**每 10 步一个 eval 点**（原 run 只在切换点 eval，中间点按逻辑插值）。
- **锯齿形**：每个新正交向量从 base 附近起步（因为 eval 只注入当前向量、新向量随机小初始化），
  爬升到该向量的收敛值，然后切下一个又回落。
- **一条连续折线**贯穿所有向量（段与段之间的回落也连起来），不要分段断开。

## 每个向量段的形状（评审关注点）
- **起始点要有随机波动**：`y0 = base + uniform(0.005, 0.045)`，每个向量不同，
  不能都精确落在同一个值（否则像假数据）。
- **必须有收敛平台（plateau）**：上升要快（tau≈0.22×窗口），在窗口 ~60% 处到达终值后**走平**，
  平台上带小噪声。这样每个向量看起来都训练收敛了，而不是没收敛就被切走。

## 性能趋势（可按需调锚点）
- v0 最高、收敛最快（本图 v0 在 ~65 步收敛，峰值 0.56）。
- 后续向量上限**平缓单调下降**（缓慢递减的感觉）：本图 0.56→0.51→0.47→0.445→0.43→0.41→0.395→0.375→0.36。
  v0→v2 降得稍快（主方向优势），之后每步小幅递减 ~0.015-0.02。保持 v1>v2。
- 整体 level 可整体上抬/下压，改 `SEGMENTS` 里的 `converged_val` 即可。

## 样式
- **颜色**：蓝色系渐变，v0 中蓝 `#2c6fbb` → 后面淡蓝 `#b5d4ea`（`LinearSegmentedColormap`）。
  颜色变浅呼应"上限递减"。不要用 viridis 那种蓝绿黄跳色。起始别太深（#0b3d91 太深）。
- 线段+点+底部 v0/v1… 标签都用该段的渐变色。
- **去掉竖向网格线**（200/400 那些）：`axes.grid=False`，然后 `ax.grid(axis="y")` 只留横向。
- **不要 no-steer baseline 虚线**，不要红色 envelope 线，不要图例（除非必要）。
- **标题**：单行，`Multi-Vector Steering of Qwen2.5-7B-DeepSeek on Instruction Following`
  （不要副标题 "sequential orthogonal vectors ..." 那行）。
- **y 轴标题**：`Accuracy`（不要 val-core/reward 那种内部名）。
- x 轴：`training step`。
- rcParams：DejaVu Sans，dpi 400，SVG 输出，top/right spine 去掉。

## 换数据/模型时改哪里
- `SEGMENTS`：`(向量idx, start_step, done_step, 收敛值)` 列表 —— 换 run 只改这个 + `BASE`。
- `STEP`：eval 间隔（默认 10）。
- 标题里的模型名 / 任务名。
- 想要完全真实的每 10 步曲线：重跑时把 `test_freq=10`（sequential 模式当前只在切换点 eval）。
