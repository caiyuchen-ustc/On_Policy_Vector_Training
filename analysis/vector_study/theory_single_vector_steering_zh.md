# RLVR 低维向量转向的理论

*附录 —— ICLR 风格。本附录给出正文经验发现背后的完整推导：（i）单个输入无关的转向向量即可恢复
RLVR 蒸馏收益的绝大部分；（ii）顺序学习的正交向量对同一算子执行收缩（deflation），表现为一个主导模态
加迅速衰减的边际收益；（iii）有效方向位于激活协方差的低方差正交补空间中，而非其主子空间。所有结论均在
明确编号的假设下陈述，证明完整。*

*核心结果。单向量的恢复率有两种等价读法——一种是信噪比（定理 A.5），一种是三个可解释常数的显式下界
（定理 A.6）：*
$$
\rho=\frac{\mathrm{SNR}}{1+\mathrm{SNR}}
\qquad\text{以及}\qquad
\rho\ \ge\ \frac{\kappa}{(1+\eta)(1+\varepsilon)},
$$
*其中 $\varepsilon$ 衡量主子空间侵入度（离主成分程度），$\eta$ 衡量补空间中的残差复杂度，$\kappa$ 衡量
样本间更新的一致性。每个常数各自对应一个实验（§A.9）。*

---

## A.1 记号与设定

考虑一个冻结的预训练语言模型，其解码层为 $\{f_\ell\}_{\ell=1}^{L}$，隐藏维度为 $d$，词表大小为 $V$，
（绑定或非绑定的）解嵌入矩阵为 $U\in\mathbb R^{V\times d}$。一个*上下文* $x\sim\mathcal D$ 表示一个 token
位置连同其前缀，采样自蒸馏数据分布 $\mathcal D$。对于基座模型：

$$
\mathbf h_\ell(x)\in\mathbb R^{d}\ \ (\text{第 }\ell\text{ 层隐藏状态}),\qquad
\mathbf z_0(x)=U\,\mathbf h_L(x)\in\mathbb R^{V}\ \ (\text{logits}),\qquad
p_0(\cdot\mid x)=\mathrm{softmax}(\mathbf z_0(x)).
$$

教师模型（经 RLVR 训练）产生 $\mathbf z_T(x)$ 与 $p_T(\cdot\mid x)$。**向量转向**在受控层集合
$\mathcal S\subseteq\{1,\dots,L\}$ 每层输出加一个可训练、输入无关的向量 $\boldsymbol\delta_\ell$，即
$\mathbf h_\ell\mapsto\mathbf h_\ell+\boldsymbol\delta_\ell$，骨干权重全部冻结。为清晰起见先推导第 $\ell$
层单点情形，一般情形见注 A.2。记 $\|v\|_A^2:=v^\top A v$，$\lambda_i,\sigma_i$ 为第 $i$ 大特征/奇异值。

蒸馏目标为对教师的期望前向 KL：

$$
\mathcal L(\boldsymbol\delta)\;=\;\mathbb E_{x\sim\mathcal D}\,
\mathrm{KL}\!\big(p_T(\cdot\mid x)\,\big\|\,p_{\boldsymbol\delta}(\cdot\mid x)\big).
\tag{A.1}
$$

### 假设

此处集中列出五条假设，各自在首次使用处给出动机。

> **假设 A.1（局部 / 信赖域区域）。** 向量转向工作在基座模型邻域内，logits 的一阶展开与 KL 的二阶
> （Fisher）展开精确：记 $J_x=\partial\mathbf z(x)/\partial\mathbf h_\ell(x)$，有
> $\mathbf z_{\boldsymbol\delta}(x)=\mathbf z_0(x)+J_x\boldsymbol\delta+o(\|\boldsymbol\delta\|)$，且 KL
> 三阶余项为 $o(\|J_x\boldsymbol\delta-\mathbf t(x)\|^2)$。这正是 RLVR 的 KL 惩罚所诱导的小步长区域
> （见 §A.6，\citet{PNT} 的门 I）。

> **假设 A.2（低熵可验证奖励）。** 教师 logit 偏移
> $\mathbf t(x,\cdot):=\mathbf z_T(x,\cdot)-\mathbf z_0(x,\cdot)$ 分解为共享秩一分量加小残差：
> $\mathbf t(x,\cdot)=a(x)\,\mathbf s+\mathbf r_\perp(x,\cdot)$，满足
> $\mathbb E_x\|a(x)\mathbf s\|_{F_x}^2\gg\mathbb E_x\|\mathbf r_\perp(x,\cdot)\|_{F_x}^2$。

> **假设 A.3（度量齐性）。** 表示度量 $G_x$（定义 A.1）近似输入无关，$G_x\approx\bar G=:G$，且 $G$ 在
> $\operatorname{span}\{\boldsymbol\Delta(x):x\in\mathcal D\}$ 上正定。

> **假设 A.4（离主成分定位）。** 令 $P$ 投影到激活协方差 $C$（定义 A.3）的 top-$m$ 主子空间，$Q=\mathbf I-P$。
> 主/补分解在 $G$ 度量下近似正交，
> $\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2\approx\mathbb E_x\|P\boldsymbol\Delta(x)\|_G^2+\mathbb E_x\|Q\boldsymbol\Delta(x)\|_G^2$，
> 且有效偏移集中于补空间：
> $\mathbb E_x\|P\boldsymbol\Delta(x)\|_G^2\le\varepsilon\,\mathbb E_x\|Q\boldsymbol\Delta(x)\|_G^2$，
> $0\le\varepsilon\ll1$。

> **假设 A.5（补空间中的共享方向）。** 存在单位向量 $\mathbf w\in\operatorname{Im}(Q)$，$\|\mathbf w\|_G=1$，
> 使 $Q\boldsymbol\Delta(x)=a(x)\,\mathbf w+\boldsymbol\xi(x)$，其中 $a(x)\ge0$，$\mathbb E[\boldsymbol\xi(x)]=0$，
> $\mathbb E\langle\boldsymbol\xi(x),\mathbf w\rangle_G=0$。定义
> $$
> \kappa:=\frac{\mathbb E[a(x)]^2}{\mathbb E[a(x)^2]}\in(0,1],
> \qquad
> \eta:=\frac{\mathbb E\|\boldsymbol\xi(x)\|_G^2}{\mathbb E[a(x)^2]}\ge0 .
> $$

假设 A.1 是标准 KL 信赖域假设；A.2 是唯一实质性建模假设，专属可验证奖励（§A.3 导出）。A.4–A.5 是离主成分
几何的激活空间形式，把 A.2 精化为三个可分别测量的常数（$\varepsilon,\eta,\kappa$），仅用于定理 A.6 的
显式下界。

---

## A.2 蒸馏归约为加权最小二乘

分类分布族是指数族，对数配分 $A(\mathbf z)=\log\sum_v e^{z_v}$，$\nabla A=p$，Fisher-Hessian：

$$
F(\mathbf z)=\nabla^2 A(\mathbf z)=\operatorname{diag}(p)-pp^\top\ \succeq\ 0 .
\tag{A.2}
$$

**引理 A.1（KL 是 Bregman 散度；局部二次型）。**
*对任意 logits，$\mathrm{KL}(\mathrm{softmax}(\mathbf z')\|\mathrm{softmax}(\mathbf z))=D_A(\mathbf z\|\mathbf z')$，
且在假设 A.1 下，*
$$
\mathrm{KL}\!\big(p_T(\cdot\mid x)\,\|\,p_{\boldsymbol\delta}(\cdot\mid x)\big)
=\tfrac12\big(J_x\boldsymbol\delta-\mathbf t(x)\big)^\top F_x\big(J_x\boldsymbol\delta-\mathbf t(x)\big)
+o\big(\|J_x\boldsymbol\delta-\mathbf t(x)\|^2\big),
\quad F_x:=F(\mathbf z_T(x)),\ \mathbf t(x)=\mathbf z_T(x)-\mathbf z_0(x).
\tag{A.3}
$$

*证明。* 指数族两成员间 KL 等于对数配分在对应自然参数处的 Bregman 散度；代入 $A,\nabla A=p$ 得恒等式。
将 $D_A$ 关于第一变元在 $\mathbf z_T(x)$ 处 Taylor 展开、用 $\nabla^2 A=F$ 及 A.1 的一阶 logit 展开得 (A.3)。$\square$

**定义 A.1（表示度量与目标偏移）。**
*定义* **拉回 Fisher 度量** *与* **最优表示偏移**：
$$
G_x:=J_x^\top F_x\,J_x,
\qquad
\boldsymbol\Delta(x):=\arg\min_{\mathbf u}\ \|J_x\mathbf u-\mathbf t(x)\|_{F_x}^2
=G_x^{-1}J_x^\top F_x\,\mathbf t(x)\ (\text{当 }G_x\text{ 可逆}).
$$

**命题 A.2（蒸馏 = 加权最小二乘）。**
*在假设 A.1 下，至多相差常数与 $o(1)$，*
$$
\mathcal L(\boldsymbol\delta)
=\tfrac12\,\mathbb E_x\big(\boldsymbol\delta-\boldsymbol\Delta(x)\big)^\top
G_x\big(\boldsymbol\delta-\boldsymbol\Delta(x)\big)+\text{const}.
\tag{A.4}
$$

*证明。* 由 (A.3)，$\mathrm{KL}=\tfrac12\|J_x\boldsymbol\delta-\mathbf t(x)\|_{F_x}^2$。将
$\mathbf t(x)=J_x\boldsymbol\Delta(x)+\mathbf r_x$，$\mathbf r_x\perp_{F_x}\operatorname{col}(J_x)$
（定义 A.1），交叉项消失，故
$\|J_x\boldsymbol\delta-\mathbf t(x)\|_{F_x}^2=\|J_x(\boldsymbol\delta-\boldsymbol\Delta(x))\|_{F_x}^2+\mathbf r_x^\top F_x\mathbf r_x$。
第一项 $=\|\boldsymbol\delta-\boldsymbol\Delta(x)\|_{G_x}^2$；第二项与 $\boldsymbol\delta$ 无关。取 $\mathbb E_x$。$\square$

**注 A.2（一般注入位置）。** 多点 $\mathcal S$ 时堆叠 $\boldsymbol\delta$，$J_x$ 取从所有位置到 logits 的
雅可比，(A.4) 逐字成立，$G_x=J_x^\top F_x J_x$；单点最后一层即 $J_x=U$。

命题 A.2 把单向量转向重述为：*在输出敏感度度量 $G$ 下，用一个常向量拟合以输入为指标的教师偏移云
$\{\boldsymbol\Delta(x)\}$*。

---

## A.3 RLVR 教师偏移是低熵奖励倾斜

**引理 A.3（KL 正则 RL 最优解是指数倾斜）。**
*目标 $\max_{\pi}\ \mathbb E_{y\sim\pi}[r(x,y)]-\beta\,\mathrm{KL}(\pi\|p_0)$ 严格凹，唯一最优在*
$$
\pi^\star(y\mid x)=\frac{p_0(y\mid x)\,e^{r(x,y)/\beta}}{Z_\beta(x)},
\qquad
\log\frac{\pi^\star(y\mid x)}{p_0(y\mid x)}=\tfrac1\beta\,r(x,y)-\log Z_\beta(x).
\tag{A.5}
$$

*证明。* 严格凹（KL 凸、奖励项仿射）；对 $\sum_y\pi=1$ 引入乘子 $\lambda$，稳定性
$r-\beta(\log\pi-\log p_0)-\beta-\lambda=0$ 给出 $\pi\propto p_0 e^{r/\beta}$；归一并取对数。$\square$

**机制性推论（假设 A.2）。** 对可验证 $r\in\{0,1\}$，所有高 reward completion 沿*同一*方向整体抬升：
教师偏移同号、共享、低熵，而非任意高秩重写。将其建模为秩一加残差
$\mathbf t(x,\cdot)=a(x)\mathbf s+\mathbf r_\perp(x,\cdot)$ 即假设 A.2。它*非*普适：非 RL 教师给出高秩
$\mathbf t$，违背 A.2——与正文"只有 RL 教师可被单向量捕获"一致。拉回穿过 $J_x$ 得表示空间细针
$\boldsymbol\Delta(x)=a(x)\mathbf w+\tilde{\boldsymbol\Delta}(x)$，横向散布很小。

---

## A.4 单向量恢复：两种等价形式

**命题 A.4（最优转向向量）。**
*(A.4) 的最小值点为 $\boldsymbol\delta^\star=\bar G^{-1}\mathbb E_x[G_x\boldsymbol\Delta(x)]$；在假设 A.3 下
$\boldsymbol\delta^\star=\boldsymbol\mu:=\mathbb E_x\boldsymbol\Delta(x)$。*

*证明。* 凸二次型，令梯度 $\mathbb E_x[G_x(\boldsymbol\delta-\boldsymbol\Delta)]=0$；A.3 下 $G$ 提出。$\square$

**定义 A.2（恢复比）。**
$\rho:=\dfrac{\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2-\mathbb E_x\|\boldsymbol\Delta(x)-\boldsymbol\delta^\star\|_G^2}{\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2}\in[0,1].$

### A.4.1 信噪比形式

**定理 A.5（恢复 = 信噪比）。**
*在假设 A.1、A.3 下，记 $\tilde{\boldsymbol\Delta}(x)=\boldsymbol\Delta(x)-\boldsymbol\mu$，*
$$
\boxed{\;\rho=\frac{\|\boldsymbol\mu\|_G^2}{\|\boldsymbol\mu\|_G^2+\mathbb E_x\|\tilde{\boldsymbol\Delta}(x)\|_G^2}
=\frac{\mathrm{SNR}}{1+\mathrm{SNR}},\qquad
\mathrm{SNR}:=\frac{\|\boldsymbol\mu\|_G^2}{\mathbb E_x\|\tilde{\boldsymbol\Delta}(x)\|_G^2}.\;}
\tag{A.6}
$$
*特别地 $\rho\ge0.85\iff\mathrm{SNR}\ge5.\overline{6}$。*

*证明。* $\boldsymbol\delta^\star=\boldsymbol\mu$ 时 $\mathcal L(\boldsymbol\mu)=\tfrac12\mathbb E_x\|\tilde{\boldsymbol\Delta}\|_G^2$
（交叉项零），平行轴恒等式给
$\mathbb E_x\|\boldsymbol\Delta\|_G^2=\|\boldsymbol\mu\|_G^2+\mathbb E_x\|\tilde{\boldsymbol\Delta}\|_G^2$。代入定义 A.2。$\square$

**推论 A.6b（谱形式）。** *令 $\mathbf u(x)=G^{1/2}\boldsymbol\Delta(x)$，$\Sigma_M=\mathbb E_x[\mathbf u\mathbf u^\top]$
为白化偏移算子 $M$ 的二阶矩。则 $\rho=\|\boldsymbol\mu\|_G^2/\operatorname{tr}\Sigma_M\le\sigma_1^2(M)/\sum_i\sigma_i^2(M)$，
当 $\boldsymbol\mu$ 与 $\Sigma_M$ 首特征向量对齐（RLVR 细针情形）时取等。*

*证明。* $G^{1/2}\boldsymbol\mu\boldsymbol\mu^\top G^{1/2}\preceq\Sigma_M$ 得 $\|\boldsymbol\mu\|_G^2\le\lambda_1(\Sigma_M)$；
$\operatorname{tr}\Sigma_M=\sum_i\sigma_i^2(M)$。$\square$

### A.4.2 显式三常数下界

定理 A.5 把一切压进一个 SNR。假设 A.4–A.5 把它拆成三个可分别测量的常数——主成分侵入度 $\varepsilon$、
补空间残差 $\eta$、样本间一致性 $\kappa$——每个对应一个实验（§A.9）。

**定理 A.6（离主成分单方向下界）。**
*在假设 A.1–A.5 下，*
$$
\boxed{\;\rho\ \ge\ \frac{\kappa}{(1+\eta)(1+\varepsilon)}.\;}
\tag{A.7}
$$

*证明。* 由定理 A.5，$\rho=\|\boldsymbol\mu\|_G^2/\mathbb E_x\|\boldsymbol\Delta(x)\|_G^2$。分别对分子下界、分母上界。

*分子。* 因 $\boldsymbol\mu=\mathbb E[\boldsymbol\Delta(x)]$ 且 $\mathbb E[\boldsymbol\xi(x)]=0$，用 $Q$ 投影（A.5）得
$Q\boldsymbol\mu=\mathbb E[a(x)]\,\mathbf w$。由 $\|\mathbf w\|_G=1$ 及 $\|\boldsymbol\mu\|_G^2\ge\|Q\boldsymbol\mu\|_G^2$
（A.4 的 $G$-正交分解），
$$
\|\boldsymbol\mu\|_G^2\ \ge\ \mathbb E[a(x)]^2\ =\ \kappa\,\mathbb E[a(x)^2].
$$

*分母。* 由 A.4 的 $G$-正交分解，$\mathbb E_x\|\boldsymbol\Delta\|_G^2\le(1+\varepsilon)\,\mathbb E_x\|Q\boldsymbol\Delta\|_G^2$。
由 A.5 及 $\mathbb E\langle\boldsymbol\xi,\mathbf w\rangle_G=0$，
$$
\mathbb E_x\|Q\boldsymbol\Delta\|_G^2=\mathbb E\|a(x)\mathbf w+\boldsymbol\xi(x)\|_G^2
=\mathbb E[a(x)^2]+\mathbb E\|\boldsymbol\xi(x)\|_G^2=(1+\eta)\,\mathbb E[a(x)^2].
$$
故 $\mathbb E_x\|\boldsymbol\Delta\|_G^2\le(1+\varepsilon)(1+\eta)\mathbb E[a(x)^2]$。相除，$\mathbb E[a(x)^2]$ 约去，得 (A.7)。$\square$

**推论 A.7（变异系数形式）。**
*因 $\kappa=1/(1+\mathrm{CV}(a)^2)$，$\mathrm{CV}(a)^2=\operatorname{Var}(a)/\mathbb E[a]^2$，*
$$
\rho\ \ge\ \frac{1}{(1+\mathrm{CV}(a)^2)(1+\eta)(1+\varepsilon)} .
$$

**解读。** 单向量恢复绝大部分收益，需*三个*条件同时成立：更新**离主成分**（$\varepsilon$ 小——即监测为
PSI 的量，§A.7）；补空间残差**低维**（$\eta$ 小——高层增大的量，§A.8）；逐样本修正**强同向**
（$\mathrm{CV}(a)^2$ 小，即 $\kappa\to1$——由低熵可验证奖励提供，§A.3）。定理 A.5 与 A.6 一致：都归结为
"$G$ 度量下一个主导模态"，但 A.6 把 SNR 归因到三个物理来源。

---

## A.5 顺序正交多向量转向即为收缩

正文按顺序、相互正交约束学习每层有序集合 $\{\mathbf v^{(0)},\dots,\mathbf v^{(K)}\}$：自由范数训 $\mathbf v^{(0)}$
至收敛并冻结；对每个 $k$ 在每步前把梯度投影到 $\{\mathbf v^{(j)}\}_{j<k}^\perp$（正文式 3）。

**定理 A.8（多向量转向 = 截断 SVD）。**
*在假设 A.1、A.3 及损失 (A.4) 下，该过程依次返回 $\Sigma_M$ 的前几个特征向量（等价 $M$ 的前几个左奇异
向量），且*
$$
\rho_K=\frac{\sum_{i\le K}\sigma_i^2(M)}{\sum_i\sigma_i^2(M)},
\qquad \rho_K-\rho_{K-1}=\frac{\sigma_K^2(M)}{\sum_i\sigma_i^2(M)} .
\tag{A.8}
$$

*证明。* 在自由幅度、受限子空间 $\mathcal V$ 上最小化 (A.4)，选出 $\Sigma_M$ 在 $G^{1/2}\mathcal V$ 上的首
特征向量（Rayleigh 商）。正交约束限制到前 $k$ 个的正交补；由 Courant–Fischer 即整体第 $(k{+}1)$ 个特征
向量——带收缩的正交迭代一步。捕获能量叠缩为特征值部分和。$\square$

**特征对应。** $\mathbf v^{(0)}$ 主导（$\rho_0>0.85$ = 一个大奇异值 = 定理 A.5/A.6）；边际递减
（$\sigma_K^2/\sum_i\sigma_i^2$）；大模型 ⇒ 谱更平 ⇒ 衰减更慢；低维但非一维（$\varepsilon$-秩 $>1$：噪声
地板之上若干共享子技能模态）。常向量无法拟合逐 token 残差 $\tilde{\boldsymbol\Delta}(x)$；定理 A.8 刻画对
共享、输入无关可实现部分的恢复（§A.8 精确给出边界）。

---

## A.6 有效方向回避主子空间

正文：学习向量在 top-8 激活 PC 中仅携带约 $0.52\%$（随机基线 $8/d$）能量，尽管这些 PC 持 $>99\%$ 方差
（top-1 PC $=98.8\%$）。

**定义 A.3（方差 vs. 敏感度度量）。**
$C=\mathbb E_x[\mathbf h\mathbf h^\top]$（激活协方差；首特征向量即*主子空间*，投影 $P$）与 $G=U^\top F U$
（输出敏感度，定义 A.1）。$C$ 沿表示*变化*最大处大；$G$ 沿*输出*最敏感处大。

**命题 A.9（度量失配 ⇒ 补空间转向）。**
*最优方向是 $G$ 的（$\boldsymbol\mu$ 加权）首特征向量（定理 A.5–A.6）。若 top-$C$ 方向落在 $G$ 的小特征值
子空间中，则 $\boldsymbol\delta^\star$ 与激活主子空间近乎正交，即 A.4 中 $\varepsilon$ 很小。*

*证明。* 满足 $\|v\|_G\approx0$ 的方向无法最大化 Rayleigh 商，被排除于 $\operatorname{span}(\boldsymbol\delta^\star)$；
前提断言 top-$C$ 方向满足 $\|v\|_G\approx0$。$\square$

*前提为何成立。* 主导激活 PC 是高范数共模/偏置方向（参与比 $\approx1$；top-1 PC $=98.8\%$）。softmax 对常
logit 偏移不变，对 $Uv$ 接近全一 logit 方向的隐藏方向近乎不变；此类 $v$ 满足 $FUv\approx0$，故 $\|v\|_G\approx0$：
承载几乎全部 $C$ 却几乎不承载 $G$。强制在此更新无收益反注入粗粒度扰动（正文主子空间消融）。这是定理 A.6
中小 $\varepsilon$ 的几何来源。

### 一个独立的权重空间佐证

命题 A.9 是由度量失配得到的*激活空间*陈述。\citet{PNT} 通过不同机制，在*权重空间*得到相同的离主成分结论。

> **归属说明。** ⟨待核⟩ 陈述转述 \citet{PNT}，引用前应对照原文。引理 A.10 自足。

**三门机制 ⟨待核⟩。** \citet{PNT} 论证 RLVR 权重更新集中于权重矩阵顶奇异方向（权重*主成分*）之外，几乎
保持权重谱，机制为：**(I) KL 锚点**——KL 惩罚限制更新范数（支撑我们的 A.1）；**(II) 几何**——曲率集中于
主成分，故范数受限更新流入低曲率补空间；**(III) 精度**——bf16 抹去亚精度的主成分更新。

**引理 A.10（权重离主成分 ⇒ 激活离主成分）。**
*对一层 $\mathbf h=\phi(W\mathbf a)$，$D=\operatorname{diag}(\phi')$，SVD $W=\sum_i s_i\mathbf p_i\mathbf q_i^\top$，
投影 $P_W^{\mathrm{prin}}$（顶权重主成分）与 $P_C^{\le k}$（top-$k$ 激活 PC），一阶诱导偏移
$\boldsymbol\Delta_{\mathbf h}(x)=D\,\Delta W\,\mathbf a(x)$ 满足*
$$
\|P_C^{\le k}\boldsymbol\Delta_{\mathbf h}\|
\le\|P_C^{\le k}DP_W^{\mathrm{prin}}\|\,\|\Delta W\mathbf a\|
+\|P_C^{\le k}D(\mathbf I-P_W^{\mathrm{prin}})\Delta W\mathbf a\| .
$$
*若 (i) $\|P_W^{\mathrm{prin}}\Delta W\|$ 小（门 I–III），(ii) top-$k$ 激活 PC 即高增益方向
$\operatorname{col-span}\{\mathbf p_i\}_{i\le k}$（故 $P_C^{\le k}D(\mathbf I-P_W^{\mathrm{prin}})\approx0$），
则 $\|P_C^{\le k}\boldsymbol\Delta_{\mathbf h}\|$ 小：激活偏移离激活主成分（$\varepsilon$ 小）。*

*证明。* 按 $P_W^{\mathrm{prin}}$ 分解 $\Delta W$，左乘 $P_C^{\le k}D$，三角不等式。第一项由 (i) 小，第二项
由 (ii) 小。$\square$

条件 (ii) 即命题 A.9 所用的实测 $98.8\%$ 对齐。故 \citet{PNT} 的门（提供 (i)）连同该对齐（提供 (ii)），
从独立前提复现命题 A.9——从而定理 A.6 中的小 $\varepsilon$：
$$
\underbrace{\text{KL + 曲率 + 精度}}_{\text{\citet{PNT}，权重}}
\Rightarrow
\underbrace{\Delta W\perp\text{权重主成分}}_{\text{门 II}}
\overset{\text{引理 A.10}}{\Rightarrow}
\underbrace{\boldsymbol\Delta_{\mathbf h}\perp\text{激活主成分}}_{\text{命题 A.9, }\varepsilon\downarrow}.
$$

---

## A.7 预测性监测推论（Alpha-Stabler）

用 top-$k$ 激活 PC $\mathcal U_\ell$ 固定基座主子空间，$P_\ell=\mathcal U_\ell\mathcal U_\ell^\top$。对批均值
$\bar{\mathbf h}_\ell^{(t)}$ 与偏移 $\boldsymbol\Delta_\ell^{(t)}=\bar{\mathbf h}_\ell^{(t)}-\bar{\mathbf h}_\ell^{\mathrm{base}}$，
定义**主子空间侵入度**：

$$
\mathrm{PSI}_\ell^{(t)}=\frac{\|P_\ell\boldsymbol\Delta_\ell^{(t)}\|^2}{\|\boldsymbol\Delta_\ell^{(t)}\|^2}\in[0,1].
\tag{A.9}
$$

**推论 A.11（PSI 即经验 $\varepsilon$）。** *至多相差 $G$-vs-欧氏加权，$\mathrm{PSI}_\ell^{(t)}$ 估计假设 A.4 的
侵入比 $\varepsilon/(1+\varepsilon)$。由命题 A.9，健康更新 $\mathrm{PSI}\approx0$；持续升高标志 $\varepsilon$
增大——更新能量迁入高方差、低效应主子空间——由定理 A.6 直接劣化恢复下界 $\rho$。*

*证明。* $\mathrm{PSI}$ 是 A.4 分解下 $\mathbb E\|P\boldsymbol\Delta\|^2/\mathbb E\|\boldsymbol\Delta\|^2=\varepsilon/(1+\varepsilon)$
的欧氏对应；命题 A.9 给 $\|P_\ell\boldsymbol\delta^\star\|_G\approx0$ 故健康时 $\mathrm{PSI}\approx0$；升高增大
(A.7) 中 $(1+\varepsilon)$ 因子。$\square$

这使 **Predicter** 成为 $\varepsilon$（并经引理 A.10 成为权重主成分漂移）的直接读数。**Controller** 裁剪主
成分分量 $\tilde{\mathbf h}_\ell=\mathbf h_\ell-\alpha P_\ell\boldsymbol\Delta_\ell^{(t)}$，
$\alpha=\max\{0,1-\sqrt{\tau/\mathrm{PSI}_\ell^{(t)}}\}$，在保留有效补空间分量 $(\mathbf I-P_\ell)\boldsymbol\Delta_\ell$
的同时恢复小 $\varepsilon$——与 §A.5 同一正交补原理，动态施加。

---

## A.8 有效性区域

定理 A.6 在 $\varepsilon,\eta,\mathrm{CV}(a)^2$ 任一增大时劣化。正文各有体现：

- **高层 ⇒ $\eta$ 增大。** 靠近输出剩余非线性不足以把常偏移展开成随上下文的修正，所需偏移变输入相关，补
  空间残差 $\boldsymbol\xi(x)$（从而 $\eta$）增大，$\rho$ 下界变弱——即高层失效。近乎二值门控（正文图 4b）
  是 $a(x)\to$ 常数，A.5 的退化情形。秩-$r$ 门控修正正是靠引入最小输入依赖来减小 $\eta$。
- **远距 / 非 RL 教师 ⇒ $\mathrm{CV}(a)^2$ 增大且 A.2 失效。** 修正变多方向、不再同号，$\kappa$ 降低；非 RL
  教师高秩倾斜直接违背 A.2。
- **强制更新入主子空间 ⇒ $\varepsilon$ 最大。** 丢弃 $Q\boldsymbol\Delta$、仅保留 $\varepsilon$ 控制的部分，
  $\rho$ 塌缩（主子空间消融）。

三常数因而统一解释成功与失败两个区域。

---

## A.9 结果汇总与经验对照

| 预测 | 结果 | 常数 | 状态 |
|---|---|---|---|
| 蒸馏 $\equiv$ 度量 $G$ 下加权最小二乘 | 命题 A.2 | — | 在 A.1 下精确 |
| RL logit 偏移 $=$ 奖励倾斜 $r/\beta$ | 引理 A.3 | — | 精确 |
| $\rho=\mathrm{SNR}/(1{+}\mathrm{SNR})$；$\rho>0.85\Rightarrow\mathrm{SNR}\gtrsim5.7$ | 定理 A.5 | — | ✔ $>85\%$ 恢复 |
| 恢复 $=$ 首模态能量比 | 推论 A.6b | — | ✔ |
| $\rho\ge\kappa/((1{+}\eta)(1{+}\varepsilon))$ | 定理 A.6 | $\varepsilon,\eta,\kappa$ | ✔ 三因子 |
| 多向量收益 $=$ 奇异值部分和（收缩） | 定理 A.8 | — | ✔ $\mathbf v^{(0)}$ 主导、衰减 |
| 大模型 ⇒ 谱更平 ⇒ 衰减更慢 | 定理 A.8 | — | ✔ 跨规模 |
| 有效向量 ⟂ top 激活 PC | 命题 A.9 | $\varepsilon\downarrow$ | ✔ 能量 $0.52\%$ |
| 强制更新到 top PC ⇒ 无收益 | 命题 A.9 | $\varepsilon\uparrow$ | ✔ 消融 |
| 权重离主成分 ⇒ 激活离主成分 | 引理 A.10 | $\varepsilon\downarrow$ | ✔ \citet{PNT} ⟨待核⟩ |
| PSI $\approx\varepsilon/(1{+}\varepsilon)$；预测崩溃 | 推论 A.11 | $\varepsilon$ | ✔ PSI 崩溃曲线 |
| 高层需输入相关性 | §A.8 | $\eta\uparrow$ | ✔ 层与门控 |
| 非 RL / 远距教师 ⇒ 失效 | §A.8 | $\kappa\downarrow$ | ✔ 教师类型 |

**总结。** KL 正则的 RLVR 沿低熵、可验证奖励方向倾斜基座策略（引理 A.3）；拉回表示空间使教师$\to$基座偏移
成为细针状云（假设 A.2）。最优输入无关向量即该云的度量均值——等价于白化偏移算子的首奇异方向——恢复比例
$\rho=\mathrm{SNR}/(1{+}\mathrm{SNR})$（定理 A.5），并有显式下界 $\kappa/((1{+}\eta)(1{+}\varepsilon))$
（定理 A.6）。三常数隔离了机制：RLVR 更新**离主成分**（$\varepsilon\!\downarrow$，命题 A.9 / 引理 A.10，
即 PSI 信号）、补空间**近一维**（$\eta\!\downarrow$）、修正**同向**（$\kappa\!\uparrow$）。顺序正交向量对
同一算子收缩（定理 A.8）。一句话：*RLVR 的有效更新先是离主成分的，再在离主成分补空间中近似秩一。*

---

### 恒等式一览

$$
\rho
=\underbrace{\frac{\|\boldsymbol\mu\|_G^2}{\|\boldsymbol\mu\|_G^2+\mathbb E_x\|\tilde{\boldsymbol\Delta}\|_G^2}}_{\text{偏差-方差}}
=\underbrace{\frac{\mathrm{SNR}}{1+\mathrm{SNR}}}_{\text{信号比}}
=\underbrace{\frac{\sigma_1^2(M)}{\sum_i\sigma_i^2(M)}}_{\text{首模态能量（对齐时）}}
\ \ \ge\ \ \underbrace{\frac{\kappa}{(1+\eta)(1+\varepsilon)}}_{\text{三常数下界}},
\qquad
\rho_K=\frac{\sum_{i\le K}\sigma_i^2(M)}{\sum_i\sigma_i^2(M)} .
$$
