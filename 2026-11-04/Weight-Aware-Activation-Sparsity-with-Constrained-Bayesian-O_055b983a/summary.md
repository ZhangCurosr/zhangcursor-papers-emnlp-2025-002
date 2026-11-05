---
title: "Weight-Aware-Activation-Sparsity-with-Constrained-Bayesian-O"
source: https://aclanthology.org/2025.emnlp-main.57.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:10:30"
field: "大模型推理加速与压缩"
keywords: ["activation sparsity", "model compression", "large language models", "training-free", "Bayesian optimization", "inference acceleration", "weight-aware pruning"]
innovations: ["将权重L1范数联合激活幅度用于稀疏化阈值决策，推导更紧的稀疏化误差上界", "约束TPE贝叶斯优化实现块间单调递增稀疏率分配，兼顾性能与搜索效率", "定制Triton GPU内核支持非均匀块级稀疏与结构感知阈值，实现端到端加速"]
benchmarks: ["WikiText2", "C4", "ARC-Easy", "ARC-Challenge", "HellaSwag", "PIQA", "Winogrande", "MMLU", "GSM8K"]
---

# 论文速读：Weight-Aware Activation Sparsity with Constrained Bayesian Optimization Scheduling for Large Language Models

## 一句话总结
提出 **WAS（Weight-Aware Activation Sparsity）**，一种训练无关的激活稀疏框架，通过将权重L1范数纳入稀疏化阈值决策，并结合约束贝叶斯优化为不同Transformer块分配差异化稀疏率，在LLaMA/Mistral系列模型上实现了最高1.68×的端到端推理加速，且在75%极端稀疏率下大幅领先SOTA基线。

## 研究问题与动机
- **现代LLM丧失自然稀疏性**：传统ReLU模型激活天然稀疏，但现代LLM普遍采用GLU类激活（如SwiGLU），不再具备此特性，需人工引入稀疏化机制。
- **现有方法忽略权重-激活耦合**：CATS、TEAL等仅依据激活值幅度设定阈值，未考虑被零化激活值所对应权重的重要性，导致稀疏化误差估计偏差。
- **统一稀疏率忽视块级敏感性差异**：已有方法对所有Transformer块施加相同稀疏率，但实证表明浅层块对稀疏更敏感，深层块容忍度更高，统一策略非最优。
- **高稀疏率下性能崩溃**：在75%稀疏率等极端场景下，现有训练无关方法（如WANDA、TEAL）perplexity急剧恶化，缺乏有效的自适应分配机制。

## 核心贡献（创新点）
- **Weight-aware阈值决策**：首次将权重列L1范数与激活幅度联合纳入稀疏化mask判断，从稀疏化误差的上界推导得出两者同等重要，与TEAL等纯幅度方法的本质区别在于误差界更紧。
- **约束贝叶斯优化实现块间稀疏分配**：基于"浅层敏感→深层容忍"的单调敏感性趋势，设计约束TPE搜索策略（后续块稀疏率不低于前一块），相比无约束优化更高效且避免高维空间的局部最优。
- **自定义GPU Triton内核支持端到端加速**：将权重感知mask生成、非均匀块级稀疏率、子结构（Q/K/V）差异化阈值直接嵌入计算流，在A800上实现最高1.68×wall-clock加速，且兼容通用Transformer架构。

## 方法详解
- **稀疏化误差分析**：从$L = \|\mathbf{xW}^T - \mathbf{x}'\mathbf{W}^T\|_1$出发，利用三角不等式推导上界$\sum_i |x_i - x'_i| \times \|\mathbf{W}^T_{i,:}\|_1$，证明稀疏化代价同时取决于激活值大小与对应权重范数。
- **Weight-aware mask**：$\text{mask}(x_i, \mathbf{W}) = \mathbb{I}(|x_i| \times \|\mathbf{W}^T_{i,:}\|_1 \geq \tau)$，列向L1范数可预计算，几乎零额外开销。
- **阈值确定**：利用LLM激活值（经权重缩放后）仍保持零均值对称单峰分布的特性，通过校准集10条2048-token序列估计CDF，由目标稀疏率$p$反推$\tau_p$。
- **块间分配（Inter-Block）**：以WikiText2 perplexity为目标，使用TPE采样器执行50次迭代；约束条件为$s_i \geq s_{i-1}$（单调递增）且全局平均接近目标$r$，有效缩减搜索空间。
- **块内分配（Intra-Block）**：因阈值操作不可微，STE梯度回传失败，改用贪心搜索（Algorithm 2）：按组件参数量比例分配稀疏增量，每次选择MSE下降最小的组件增加稀疏率，直至达到块级目标。
- **GPU内核设计**：在DejaVu基础上扩展，支持FP16 SplitK累加、缓存感知调度（优先复用激活、降低块专属权重优先级）、非均匀稀疏及子结构差异化阈值。

## 实验与结果
- **评测模型**：LLaMA-2（7B/13B/70B）、LLaMA-3（8B/70B）、LLaMA-3.1（8B/70B）、Mistral-7B。
- **评测数据集**：WikiText2、C4（perplexity）；ARC-E/Challenge、HellaSwag、PIQA、Winogrande、MMLU（5-shot）、GSM8K（推理/知识）。
- **基线方法**：CATS（仅40%有效）、WANDA（静态权重剪枝）、TEAL（当前SOTA训练无关激活稀疏）。
- **Perplexity结果**：
  - **40%/60%**：WAS与TEAL基本持平或略优；LLaMA-2-7B @60% WikiText2：WAS=6.56 vs TEAL=6.80。
  - **75%极端稀疏**：WAS显著领先，LLaMA-2-7B WikiText2 perplexity **12.76 vs TEAL 42.15**（降低29.39），C4 **15.56 vs 51.34**（降低35.78）。
- **推理任务结果**：75%稀疏率下WAS在LLaMA-3-8B等模型上较TEAL提升**1.50%~5.70%**绝对增益（AVG指标）。
- **硬件加速**：A800 GPU单batch decoding，60%稀疏率最高**1.52×**加速，75%稀疏率最高**1.68×**加速。
- **消融**：Table 3表明weight-aware（↓22.94）和TPE块间分配（↓21.02）各自贡献显著，组合效果最优；Table 4验证约束TPE优于无约束版本且收敛更快（50次 vs >100次）。

## 相关工作脉络
- **TEAL（Liu et al., 2024）**：当前训练无关激活稀疏SOTA，仅用激活幅度阈值，本文在其基础上引入权重感知与块级差异化分配，核心差异在于稀疏化误差模型的完整性。
- **WANDA（Sun et al., 2023）**：静态权重剪枝方法，利用权重L1范数评估重要性；本文借鉴其范数思想但应用于激活稀疏而非权重剪枝，适用场景完全不同。
- **CATS（Lee et al., 2024）**：仅对MLP层gate输出做相对幅度稀疏化，影响范围局限于W_up/W_down子集，本文对全模型所有组件进行全面稀疏化。
- **DejaVu（Liu et al., 2023）**：训练型方法，用轻量MLP预测低激活神经元并剪枝；本文完全无需训练，部署成本更低。
- **ReLUfication / ProSparse**：通过替换激活函数+继续预训练恢复稀疏性，计算代价高昂；本文训练无关，直接适用于已部署模型。
- **OWL（Yin et al., 2023）**：基于特征异常值的权重剪枝逐层调整稀疏率；本文面向激活稀疏且使用贝叶斯优化而非启发式规则。

## 局限性与未来方向
- 当前方法假设激活分布在零附近对称单峰，对非对称/多峰激活分布的模型泛化性未知。
- 仅探索训练无关场景；论文自述未来可将WAS框架与训练结合，有望进一步压缩性能损失。
- 未验证在MoE架构、Vision-Language模型等非标准Transformer结构上的适用性。
- 校准集仅用10条序列，在分布偏移场景下的阈值稳定性有待检验。

## 研究启发与可借鉴点
- **误差上界推导指导稀疏化设计**：从理论角度证明权重-激活耦合的重要性，比纯经验启发更具说服力；此分析范式可迁移至其他压缩方法（如低秩近似、量化）。
- **约束优化降维策略**：将单调先验知识嵌入贝叶斯优化约束，既提升搜索效率又改善收敛稳定性，对高维超参调优有普遍参考价值。
- **块级敏感性分析与分层分配**：通过系统实验揭示 Transformer 块深度的敏感性梯度，启发我们可在其他结构化模型（如Diffusion、MoE）中做类似的位置敏感性剖析。
- **训练无关 + 定制内核的工程闭环**：从理论方法到Triton kernel落地的完整链路值得借鉴，尤其适合本团队关注部署效率的研究方向。
- **权重范数预计算的零开销设计**：列向L1范数只需一次性预计算，推理期无额外延迟，此技巧可复用于其他需要动态阈值的方法。

## 关键术语表
- **Activation Sparsity**：在推理时根据输入动态将不重要激活值置零，从而跳过对应权重计算以加速推理。
- **Weight-Aware Sparsification**：将权重重要性（列L1范数）与激活幅度联合评估，以更准确地估计稀疏化误差。
- **Constrained Bayesian Optimization**：在搜索空间中引入先验约束（如单调性）的贝叶斯优化，用于高效分配块间稀疏率。
- **TPE（Tree-structured Parzen Estimator）**：基于核密度估计的贝叶斯优化变体，擅长高维超参搜索，本文选用它替代Gaussian Process。
- **Inter-Block / Intra-Block Sparsity**：分别指不同Transformer块之间的稀疏率分配，与单个块内部各子组件（Q/K/V/MLP等）的稀疏率分配。
- **Greedy Search for Intra-Block Allocation**：通过逐步增加稀疏率并选择MSE退化最小的组件迭代分配块内稀疏度。
- **Triton Kernel**：基于Triton编译器的自定义GPU计算内核，本文用于高效实现权重感知的动态稀疏计算。
- **Training-free Sparsification**：无需重新训练或微调，直接对预训练模型施加稀疏化策略的方法。

## 可复现要素
- **代码开源**：https://github.com/HITSZ-Miao-Group/WAS
- **模型权重**：LLaMA-2、LLaMA-3、LLaMA-3.1、Mistral（官方开源，需申请访问）
- **数据集**：WikiText2、C4（公开）；ARC、HellaSwag、PIQA、Winogrande、MMLU、GSM8K（公开benchmark）
- **关键超参**：TPE迭代次数 $N_{\text{trials}}=50$；约束范围 $[r-0.05, r+0.05]$；校准集10条2048-token序列；贪心搜索步长 $\alpha$（论文附录D未明确数值）；GPU：A800，TP2（70B模型）
