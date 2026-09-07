# 为什么单个 Steering Vector 能恢复 RLVR 的绝大部分增益

本节给出一个可以直接写入论文正文或附录的证明框架，用以解释为什么在 RL with Verifiable Rewards (RLVR) 中，单个输入无关的 steering vector 往往足以恢复绝大部分性能增益。整套论证分为三步。首先，我们把蒸馏目标在激活空间中化为一个局部二次型拟合问题。其次，我们证明 KL 正则 RL 的 teacher 不是任意目标分布，而是 base policy 关于 reward 的指数倾斜。最后，受《The Path Not Taken: RLVR Provably Learns Off the Principals》所强调的 off-principal 现象启发，我们引入两个几何假设：其一，RLVR 的有效更新主要位于 activation principal subspace 的正交补；其二，在该补空间中，teacher shift 由一个共享主方向主导。基于这两点，可以严格推出最优单向量 steering 恰为平均 hidden-state shift，并且其可恢复比例具有显式下界。

## 1. 设定与局部蒸馏几何

固定一个冻结的 base model，并选定插入层 $\ell$。对任意输入上下文 $x$，记 $h_\ell(x)\in\mathbb R^d$ 为第 $\ell$ 层 hidden state，记 $z_0(x)\in\mathbb R^V$ 为 base logits，记 $p_0(\cdot\mid x)=\mathrm{softmax}(z_0(x))$ 为 base next-token distribution。进一步记 $z_T(x)$ 与 $p_T(\cdot\mid x)$ 分别为 teacher logits 与 teacher distribution。

单向量 steering 在层 $\ell$ 注入一个常数向量 $\delta\in\mathbb R^d$：

$$
h_\ell'(x)=h_\ell(x)+\delta.
$$

对应的蒸馏目标写为

$$
\mathcal L(\delta)
=
\mathbb E_{x\sim\mathcal D}
\big[
\mathrm{KL}(p_T(\cdot\mid x)\Vert p_\delta(\cdot\mid x))
\big],
$$

其中 $p_\delta(\cdot\mid x)$ 是注入 steering vector 后得到的模型分布。

### 假设 1（局部二次型区域）

在所选插入层附近，steered logits 关于 hidden perturbation 满足一阶展开

$$
z_\delta(x)=z_0(x)+J_x\delta+o(\|\delta\|),
$$

其中 $J_x=\partial z(x)/\partial h_\ell(x)$ 是从插入层到 logits 的 Jacobian。并且，在 teacher logits 附近，KL divergence 可以由其 Fisher 二阶展开近似：

$$
\mathrm{KL}(p_T\Vert p_\delta)
=
\frac{1}{2}
\big(J_x\delta-t(x)\big)^\top
F_x
\big(J_x\delta-t(x)\big)
+o\big(\|J_x\delta-t(x)\|^2\big),
$$

其中

$$
t(x)=z_T(x)-z_0(x),
\qquad
F_x=\mathrm{diag}(p_T(\cdot\mid x))-p_T(\cdot\mid x)p_T(\cdot\mid x)^\top.
$$

假设 1 描述的正是 KL 正则所诱导的小步长 trust-region 区域，也是后续局部理论成立的基础。

### 定义 1（Teacher Shift 的 Pullback 表示）

定义 pullback metric

$$
G_x=J_x^\top F_x J_x,
$$

并将 teacher 的输出空间目标位移拉回到 hidden space，定义为

$$
\Delta(x)
=
\arg\min_{u\in\mathbb R^d}
\big(t(x)-J_xu\big)^\top F_x\big(t(x)-J_xu\big).
$$

当 $G_x$ 可逆时，有显式表达式

$$
\Delta(x)=G_x^{-1}J_x^\top F_x t(x).
$$

向量 $\Delta(x)$ 表示：在局部二次几何下，最能复现 teacher logit shift 的 hidden-state perturbation。

### 命题 1（蒸馏目标化为激活空间二次拟合）

在假设 1 下，蒸馏目标满足

$$
\mathcal L(\delta)
=
\frac{1}{2}
\mathbb E_x
\big[
(\delta-\Delta(x))^\top G_x(\delta-\Delta(x))
\big]
+\mathrm{const}+o(1).
$$

#### 证明

由假设 1，

$$
\mathrm{KL}(p_T\Vert p_\delta)
=
\frac{1}{2}
\big(J_x\delta-t(x)\big)^\top F_x\big(J_x\delta-t(x)\big)
+o\big(\|J_x\delta-t(x)\|^2\big).
$$

由定义 1，$\Delta(x)$ 是 hidden space 中该二次型的最优解，因此存在残差 $r_x$，使得

$$
t(x)=J_x\Delta(x)+r_x,
$$

并且 $r_x$ 与 $J_x$ 的列空间在 $F_x$ 度量下正交。于是

$$
\big(J_x\delta-t(x)\big)^\top F_x\big(J_x\delta-t(x)\big)
=
\big(J_x(\delta-\Delta(x))\big)^\top F_x\big(J_x(\delta-\Delta(x))\big)
+r_x^\top F_x r_x,
$$

因为交叉项在 $F_x$-正交性下为零。第一项进一步化为

$$
(\delta-\Delta(x))^\top J_x^\top F_x J_x(\delta-\Delta(x))
=(\delta-\Delta(x))^\top G_x(\delta-\Delta(x)).
$$

对 $x$ 取期望即得结论。$\square$

命题 1 说明，单向量 steering 的本质不是在参数空间里做神秘压缩，而是在 pullback metric $G_x$ 下，用一个常数向量去拟合一簇输入相关的 hidden-state teacher shift $\{\Delta(x)\}$。

## 2. RLVR Teacher 不是任意 Teacher，而是 Reward Tilt

命题 1 对任意 teacher 都成立，因此真正决定“单向量为什么够用”的，是 RLVR teacher 本身的结构，而不是 surrogate 形式本身。

### 引理 1（KL 正则 RL 的最优解是指数倾斜）

对每个上下文 $x$，考虑 KL 正则 RL 目标

$$
\max_{\pi(\cdot\mid x)}
\Big[
\mathbb E_{y\sim\pi(\cdot\mid x)} r(x,y)
-\beta\,\mathrm{KL}(\pi(\cdot\mid x)\Vert p_0(\cdot\mid x))
\Big].
$$

其唯一最优解为

$$
\pi^*(y\mid x)
=
\frac{1}{Z_\beta(x)}
p_0(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big),
$$

其中

$$
Z_\beta(x)=\sum_y p_0(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big).
$$

#### 证明

对约束 $\sum_y\pi(y\mid x)=1$ 引入拉格朗日乘子 $\lambda$，定义

$$
\mathcal J(\pi,\lambda)
=
\sum_y \pi(y\mid x)r(x,y)
-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{p_0(y\mid x)}
+\lambda\Big(\sum_y\pi(y\mid x)-1\Big).
$$

对 $\pi(y\mid x)$ 求偏导并令其为零，得到

$$
r(x,y)-\beta\Big(\log\pi(y\mid x)-\log p_0(y\mid x)+1\Big)+\lambda=0.
$$

故有

$$
\log\pi(y\mid x)
=
\log p_0(y\mid x)+\frac{r(x,y)}{\beta}+c(x),
$$

其中 $c(x)$ 吸收 $\lambda$ 与常数项。指数化并由归一化约束确定 $c(x)$，即得结论。$\square$

引理 1 立即推出

$$
\log\frac{\pi^*(y\mid x)}{p_0(y\mid x)}
=
\frac{1}{\beta}r(x,y)-\log Z_\beta(x).
$$

因此，RLVR teacher 并不是任意目标模型，而是 base model 关于 reward 的指数倾斜。特别地，当 $r(x,y)\in\{0,1\}$ 时，所有高 reward completion 都会被沿同一方向整体抬升，这正是低维性的第一重来源：teacher 相对 base 的变化天然具有同号、低熵、共享的结构。

## 3. Off-Principal 定位与补空间单主方向结构

接下来需要把“reward tilt 的低熵性”转化为“单向量即可恢复大部分 hidden-space shift”的定理。为此，我们引入受 off-principal 观点启发的几何假设。

记插入层 hidden state 的 activation covariance 为

$$
C=\mathbb E_x\big[(h_\ell(x)-\bar h)(h_\ell(x)-\bar h)^\top\big].
$$

令 $P$ 为 $C$ 的 top-$m$ principal subspace 的投影，记 $Q=I-P$ 为其补空间投影。

### 假设 2（Metric Homogeneity）

pullback metric 在样本间变化较慢，即存在固定半正定矩阵 $G$，使得 $G_x\approx G$。并进一步假设 $G$ 在 $\{\Delta(x):x\in\mathcal D\}$ 的线性张成空间上正定。于是局部 surrogate 可以写成

$$
\mathcal L(\delta)
\approx
\frac{1}{2}\mathbb E_x\|\delta-\Delta(x)\|_G^2+\mathrm{const},
\qquad
\|u\|_G^2:=u^\top Gu.
$$

### 假设 3（Off-Principal Localization）

在 $G$ 度量下，principal/complement 分解满足近似正交，即

$$
\mathbb E_x\|\Delta(x)\|_G^2
\approx
\mathbb E_x\|P\Delta(x)\|_G^2+\mathbb E_x\|Q\Delta(x)\|_G^2.
$$

并且有效 teacher shift 主要集中在 principal complement 中：

$$
\mathbb E_x\|P\Delta(x)\|_G^2
\le
\varepsilon\,\mathbb E_x\|Q\Delta(x)\|_G^2,
\qquad
0\le \varepsilon \ll 1.
$$

这正是 off-principal 观点在 activation space 中的抽象表达，也与你的实验现象一致：有效 steering direction 在 top activation PCs 上的能量接近随机基线，而一旦强制投到 principal subspace 中，性能几乎无法提升。

### 假设 4（补空间中的共享主方向）

存在单位向量 $w\in\mathrm{Im}(Q)$，满足 $\|w\|_G=1$，使得

$$
Q\Delta(x)=a(x)w+\xi(x),
$$

其中标量系数 $a(x)$ 非负，残差满足 $\mathbb E[\xi(x)]=0$，且

$$
\mathbb E\langle \xi(x),w\rangle_G=0.
$$

定义

$$
\kappa:=\frac{\mathbb E[a(x)]^2}{\mathbb E[a(x)^2]},
\qquad
\eta:=\frac{\mathbb E\|\xi(x)\|_G^2}{\mathbb E[a(x)^2]}.
$$

这里，$\kappa\in(0,1]$ 衡量样本间更新是否同号且彼此一致；$\eta\ge 0$ 衡量在去除补空间主方向后，仍然残留多少多方向复杂度。

## 4. 主定理：单向量恢复率的显式下界

现在可以给出主结论。

### 定理 1（Off-Principal 单主方向条件下的单向量可恢复性）

在假设 1 与假设 2 下，局部 surrogate 的最优常数 steering vector 为平均 teacher shift

$$
\delta^*=\mu:=\mathbb E_x[\Delta(x)].
$$

定义局部 teacher-shift energy 的恢复比例

$$
\rho
:=
\frac{\mathbb E_x\|\Delta(x)\|_G^2-\mathbb E_x\|\Delta(x)-\delta^*\|_G^2}
{\mathbb E_x\|\Delta(x)\|_G^2}.
$$

则有

$$
\rho
=
\frac{\|\mu\|_G^2}{\mathbb E_x\|\Delta(x)\|_G^2}
=
\frac{\mathrm{SNR}}{1+\mathrm{SNR}},
\qquad
\mathrm{SNR}:=\frac{\|\mu\|_G^2}{\mathbb E_x\|\Delta(x)-\mu\|_G^2}.
$$

若进一步满足假设 3 与假设 4，则

$$
\rho
\ge
\frac{\kappa}{(1+\eta)(1+\varepsilon)}.
$$

#### 证明

由假设 2 与命题 1，局部 surrogate 可以写成

$$
\mathcal L(\delta)
\approx
\frac{1}{2}\mathbb E_x\|\delta-\Delta(x)\|_G^2+\mathrm{const}.
$$

对 $\delta$ 求梯度，得到

$$
\nabla_\delta \mathcal L(\delta)
=
G\Big(\delta-\mathbb E_x[\Delta(x)]\Big).
$$

由于 $G$ 在相关子空间上正定，故最优解唯一，且

$$
\delta^*=\mu=\mathbb E_x[\Delta(x)].
$$

下面计算恢复比例。由 centered second moment 分解，

$$
\mathbb E_x\|\Delta(x)\|_G^2
=
\|\mu\|_G^2+\mathbb E_x\|\Delta(x)-\mu\|_G^2.
$$

因此

$$
\rho
=
\frac{\|\mu\|_G^2}{\|\mu\|_G^2+\mathbb E_x\|\Delta(x)-\mu\|_G^2}
=
\frac{\mathrm{SNR}}{1+\mathrm{SNR}}.
$$

接下来证明显式下界。由 $\mu=\mathbb E[\Delta(x)]$ 与 $\mathbb E[\xi(x)]=0$，有

$$
Q\mu
=
\mathbb E[Q\Delta(x)]
=
\mathbb E[a(x)]w.
$$

因此

$$
\|\mu\|_G^2
\ge
\|Q\mu\|_G^2
=
\mathbb E[a(x)]^2
=
\kappa\,\mathbb E[a(x)^2].
$$

另一方面，由假设 3，

$$
\mathbb E_x\|\Delta(x)\|_G^2
\le
(1+\varepsilon)\mathbb E_x\|Q\Delta(x)\|_G^2.
$$

再由假设 4 及 $\mathbb E\langle \xi(x),w\rangle_G=0$，有

$$
\mathbb E_x\|Q\Delta(x)\|_G^2
=
\mathbb E_x\|a(x)w+\xi(x)\|_G^2
=
\mathbb E[a(x)^2]+\mathbb E\|\xi(x)\|_G^2
=
(1+\eta)\mathbb E[a(x)^2].
$$

将以上三式合并，得到

$$
\rho
=
\frac{\|\mu\|_G^2}{\mathbb E_x\|\Delta(x)\|_G^2}
\ge
\frac{\kappa\,\mathbb E[a(x)^2]}{(1+\varepsilon)(1+\eta)\mathbb E[a(x)^2]}
=
\frac{\kappa}{(1+\eta)(1+\varepsilon)}.
$$

定理得证。$\square$

### 推论 1（大恢复率的显式条件）

在定理 1 的条件下，进一步有

$$
\rho
\ge
\frac{1}{(1+\mathrm{CV}(a)^2)(1+\eta)(1+\varepsilon)},
$$

其中

$$
\mathrm{CV}(a)^2:=\frac{\mathrm{Var}(a)}{\mathbb E[a]^2}.
$$

#### 证明

由

$$
\kappa=\frac{\mathbb E[a]^2}{\mathbb E[a^2]}=\frac{1}{1+\mathrm{CV}(a)^2},
$$

再结合定理 1 即得结论。$\square$

推论 1 把结论解释得非常直接：单个 steering vector 之所以能恢复大部分 RLVR 增益，本质上依赖于三个条件同时成立：更新是 off-principal 的（$\varepsilon$ 小），补空间中的残差复杂度低（$\eta$ 小），以及不同样本上的有效修正具有强同向性（$\mathrm{CV}(a)^2$ 小）。

## 5. 为什么 RLVR 比一般 Teacher 更满足这些条件

定理 1 本身并不依赖 teacher 的来源，但引理 1 揭示了为什么 RLVR 是特别有利的情形。

第一，binary verifiable reward 产生的是低熵的 exponential tilt，而不是任意高维重写，因此高 reward 行为被统一上调，这直接支持较大的 $\kappa$，即不同样本诱导的 hidden-state correction 更容易同号且互不抵消。

第二，off-principal 相关分析以及你的 principal-subspace projection 实验共同表明：KL 正则 RL 倾向于远离主方向学习，这正对应较小的 $\varepsilon$。

第三，一旦把 principal content directions 去掉，剩余 complement 更像是一个 control subspace 而不是 content subspace。在这个子空间中，teacher shift 更容易集中于一个共享主方向附近，并只保留较小残差，因此 $\eta$ 更小。

这也解释了为什么单向量 steering 对 RLVR 有效，却通常难以压缩一般 non-RL teacher：后者未必满足同向性、off-principal 定位，或者补空间中的单主方向结构。

## 6. 与实验现象的直接对应

上述定理可以直接解释你的主要实验结果。

第一，principal projection 为什么失败。因为强制投到 principal subspace 中，等价于丢弃了真正承载 RLVR 能力改造的 $Q\Delta(x)$ 分量，只保留由 $\varepsilon$ 控制的微弱部分，因此恢复率自然塌缩。

第二，为什么 teacher 离 student 越远，单向量越不够。因为此时所需 correction 更加输入相关、更加多方向，体现为补空间残差 $\xi(x)$ 增大，即 $\eta$ 变大，恢复率下界随之变弱。

第三，为什么高层单向量 steering 更容易失败。因为离输出太近时，下游非线性不足以把一个输入无关的常数 shift 展开成所需的输入相关修正，于是残差复杂度上升，同样表现为 $\eta$ 变大。你的 gated/rank 实验之所以能救回来，本质上就是通过引入最小程度的输入依赖来减小这个残差项。

## 7. 证明逻辑总图

如果把整套证明压缩成一句话，其核心逻辑是：

> RLVR 先把学习限制在一个小的 KL trust region 内，再把 base policy 沿 reward 方向做指数倾斜；而有效更新主要位于 activation principal complement 中，并且在该 complement 内近似由一个共享主方向主导，因此最优单向量 steering 就是这簇 teacher shift 的均值方向。

更具体地说，逻辑顺序是以下四步：

1. `蒸馏问题 -> 激活空间拟合问题`。命题 1 说明，单向量训练等价于在 $G$ 度量下用常数向量拟合 $\Delta(x)$ 的云团。
2. `RLVR teacher -> 低熵 reward tilt`。引理 1 说明 teacher 相对 base 的变化天然是同号、共享、低熵的，而不是任意高维目标。
3. `off-principal -> 真正有效的自由度只在补空间里`。假设 3 表明主方向并不承载 RLVR 的核心更新，因此问题被降到 complement 中。
4. `补空间近一维 -> 单向量足够`。假设 4 说明在 complement 中，teacher shift 近似为 $a(x)w+\xi(x)$。于是均值方向就是 $w$，而恢复率由三个因素控制：off-principal 程度 $\varepsilon$、补空间残差复杂度 $\eta$、样本间同向性 $\kappa$。

因此，最终结论不是“RL 在全空间里神奇地 rank-1”，而是更精确的：

> **RLVR 的有效更新先是 off-principal 的，再在 off-principal complement 中近似 rank-1。**

这正是单个 steering vector 能恢复绝大部分 RLVR 增益的数学原因。

## 8. 可直接写进论文的小结段落

单个 steering vector 之所以能够恢复 RLVR 的绝大部分增益，并不是因为 RL update 在整个 activation space 中全局 rank-one，而是因为 KL 正则 RL 先将学习限制在 base model 附近的局部 trust region 内，随后 RLVR teacher 作为 base policy 关于 reward 的指数倾斜，诱导出一个低熵、强同向的 hidden-state target shift。进一步地，受 off-principal 几何结构约束，该有效 shift 主要集中于 activation principal subspace 的正交补，并在该补空间中由一个共享主方向主导。于是，最优单向量 steering 恰为平均 hidden-state shift，其局部可恢复比例满足

$$
\rho\ge\frac{\kappa}{(1+\eta)(1+\varepsilon)}.
$$

其中，$\varepsilon$ 衡量主子空间侵入程度，$\eta$ 衡量补空间中的残差复杂度，$\kappa$ 衡量样本间更新的一致性。因此，当 RLVR update 满足 off-principal、补空间低残差以及样本间强同向这三项条件时，单个 steering vector 自然可以恢复绝大部分 RL 增益。
