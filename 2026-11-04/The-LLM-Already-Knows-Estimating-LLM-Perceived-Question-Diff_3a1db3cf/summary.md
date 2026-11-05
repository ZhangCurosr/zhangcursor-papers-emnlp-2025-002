---
title: "The-LLM-Already-Knows-Estimating-LLM-Perceived-Question-Diff"
source: https://aclanthology.org/2025.emnlp-main.61.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:07:21"
field: "大语言模型推理效率优化"
keywords: ["LLM difficulty estimation", "hidden representations", "Markov chain", "value function", "adaptive inference", "Self-Consistency"]
innovations: ["仅利用目标LLM初始隐藏状态估计难度，测试时零生成开销", "将自回归生成过程建模为Markov链并用TD学习训练价值函数", "基于难度估计动态分配推理预算的自适应采样策略"]
benchmarks: ["MMBench", "ScienceQA", "MathVista", "StrategyQA", "gsm8k", "commonsenseQA"]
---

# 论文速读：The-LLM-Already-Knows-Estimating-LLM-Perceived-Question-Diff

## 一句话总结
本文提出了一种仅利用目标LLM自身隐藏表示（无需生成任何输出token）来估计LLM感知题目难度的轻量级方法，通过将token级生成过程建模为马尔可夫链并定义价值函数实现高效准确的难度预测，同时验证了其在自适应推理策略中的效率提升效果。

## 研究问题与动机
- **核心问题**：如何准确估计目标LLM对输入问题的感知难度，以支持难度感知的性能评估、训练和推理策略自适应。
- **现有方法不足**：
  - 基于重复采样的方法（如AG、Self-Consistency）需要多次生成输出，计算开销大
  - 使用辅助LLM评估（如LLMs-Ranking）无法准确捕捉目标模型自身对难度的感知
  - 微调目标模型的方法（如HaluSearch）可能损害模型的通用能力与鲁棒性
- **动机依据**：隐藏表示能提供更细粒度的模型预测逻辑信息；图1中t-SNE可视化显示easy/hard问题的最后一层隐藏表示在特征空间中呈现明显分离趋势。

## 核心贡献（创新点）
- **提出基于隐藏表示的零生成难度估计方法**：与现有依赖输出一致性或辅助模型的方法本质不同，仅利用目标LLM对输入的初始隐藏状态即可预测难度，测试时零生成开销。
- **将自回归生成过程建模为马尔可夫链并定义价值函数**：首次将RL中的Bellman方程引入LLM内部状态空间，通过TD误差训练轻量级价值网络逼近期望输出质量。
- **设计三种难度感知的重复采样自适应策略**：将估计的难度阈值用于动态调整Self-Consistency、Best-of-N、Self-Refine的推理预算，在保持/提升准确率的同时显著减少token消耗。
- **验证跨模态、跨模型规模、跨域的泛化能力**：在6个文本/多模态数据集上验证有效性，证明方法对不同模型大小（3B-72B）稳定适用。

## 方法详解
- **马尔可夫链建模**：将LLM的自回归生成分为状态序列 $\{s_t\}$，其中 $s_t = \{H_t, y_t\}$，$H_t$ 为包含到时刻t所有token隐藏表示的上下文嵌入序列，$y_t$ 为生成token；转移函数由目标模型 $f_\theta$ 定义。
- **奖励函数定义**：仅在生成EOS时返回基于输出序列 $\mathbf{y}$ 的ORM奖励 $R(s_t) = \text{Reward}(\mathbf{y})$，中间状态奖励为0。
- **价值函数与Bellman方程**：$V(s_t) = \mathbb{E}[R(s_t) + \gamma V(s_t + 1)]$，初始状态价值 $V(s_0)$ 直接反映输入问题被模型感知的难度（$V(s_0) \leq \tau$ 判定为困难）。
- **训练目标**：采用TD学习最小化平方TD误差 $\mathcal{L}_{\text{TD}} = \mathbb{E}[\sum_t \delta_t^2]$，其中 $\delta_t$ 根据是否到达EOS分别使用bootstrapped目标或直接奖励。
- **网络架构**：使用两层全连接神经网络 $\hat{F}_\phi$ 逼近价值函数，输入为最后一层隐藏状态 $h_t$。

## 实验与结果
- **数据集**：MMBench、ScienceQA、MathVista（多模态）；StrategyQA、gsm8k、commonsenseQA（纯文本）；RLHF-V、VLFeedback（开放-ended）。
- **模型**：Qwen2.5-VL-7B-Instruct、InternVL3-8B，以及不同规模Qwen2.5-VL（3B-72B）。
- **主要结果**：
  - MMBench（Qwen2.5-VL-7B）：ROC-AUC **94.15%**，Macro-F1 **80.68%**，显著优于AG（81.04%/72.17%）和LLMs-Ranking（44.28%/51.43%）
  - ScienceQA（Qwen2.5-VL-7B）：ROC-AUC **93.09%**，Macro-F1 **79.48%**
  - 测试时间成本：Figure 2显示本方法耗时最低（约1.5-1.8分钟/400样本），HaluSearch-Gen需5+分钟
  - 自适应推理：在ScienceQA的SC策略下，在更少token消耗下达到最高准确率；MathVista的BoN/SR策略下准确率提升显著
  - 跨模型规模：Table 4显示3B到72B规模上Macro-F1稳定在66-80区间
  - 开放-ended任务：RLHF-V上ROC-AUC达82.84%（Qwen2.5-VL-7B），VLFeedback上达92.16%

## 相关工作脉络
- **AG (Lee et al., 2025)**：基于目标模型输出一致性的难度估计，需多次采样生成；本文仅用输入隐藏状态，零生成开销
- **LLMs-Ranking (Wang et al., 2024b)**：使用辅助LLM直接评估难度；本文利用目标模型自身内部表示，更准确反映目标模型的感知
- **HaluSearch-Gen/Critic (Cheng et al., 2025)**：微调目标模型使其具备难度评估能力；本文避免微调，不损害通用能力
- **Self-Consistency / Best-of-N / Self-Refine**：传统重复采样策略固定预算；本文引入难度自适应动态分配推理预算
- **LLM表示研究 (Kong et al., 2024; Zhang et al., 2024b)**：关注对齐、安全性、可解释性；本文首次将隐藏表示系统用于难度估计的RL价值函数建模

## 局限性与未来方向
- **访问限制**：需要获取目标LLM的token级隐藏表示，对闭源系统不可用
- **单轮对话局限**：当前方法仅针对单轮输入，未扩展至多轮对话场景
- **跨模型泛化有限**：Table 5显示直接在Qwen和LLaMA之间迁移价值函数性能下降明显（ROC-AUC从~80降至~47）
- **未见领域泛化**：论文指出跨域泛化是未来方向，目前仅在相同架构不同规模上验证

## 研究启发与可借鉴点
- **内部表示作为难度信号**：隐藏状态已包含模型对问题不确定性的隐式编码，无需输出即可预测，为"无生成推理"场景提供新思路
- **MDP形式化LLM生成过程**：将自回归生成视为Markov决策过程，用价值函数建模"从当前内部状态出发期望获得的质量"，可迁移至其他生成质量预测任务
- **轻量级价值网络训练范式**：用TD误差而非端对端梯度训练，仅需提取最后一层隐藏状态，训练成本极低（Table 6：1.56分钟 vs HaluSearch-Gen的5.32分钟）
- **自适应推理预算分配框架**：难度估计→阈值判定→动态选择单采样/多采样策略的流程，可直接复用于其他采样增强方法
- **跨模型训练分离策略**：为每个目标模型单独训练轻量价值函数，避免直接迁移的性能损失，同时保持训练效率

## 关键术语表
- **Hidden Representations**：LLM在processing输入时各层神经元激活的高维向量，包含模型对输入的语义理解与推理状态
- **Markov Chain（马尔可夫链）**：描述状态转移概率的随机过程，此处用于形式化LLM逐token生成的内部状态演化
- **Value Function（价值函数）**：在RL中估计从某状态出发期望累积奖励的函数，本文用于预测从给定隐藏状态出发模型产生正确输出的期望质量
- **TD Error（时序差分误差）**：强化学习中预测值与bootstrap目标之间的差异，本文用于训练价值网络的监督信号
- **Outcome Reward Model (ORM)**：仅对最终输出序列打分而无需中间过程奖励的评估模型，本文用于生成训练标签
- **Self-Consistency**：通过多次采样CoT推理路径并投票选择最常见答案的推理增强方法
- **Best-of-N**：生成N个候选答案并选择评分最高的一个的推理策略
- **Difficulty-Aware**：根据输入问题预估难度动态调整计算资源分配的策略设计原则

## 可复现要素
- **数据集**：MMBench、ScienceQA、MathVista、StrategyQA、gsm8k、commonsenseQA、RLHF-V、VLFeedback（公开数据集）
- **代码/权重**：论文未提及代码开源声明
- **关键超参**：
  - 训练集划分：MMBench/MathVista为50%训练/45%测试/5%验证；其他数据集95%训练/5%验证
  - 阈值τ：在验证集上确定
  - 网络架构：两层全连接神经网络
  - 学习率：$1 \times 10^{-4}$
  - 采样温度T：0.5
  - 独立推理尝试次数：3次（足够，见Table 7）
  - 实现环境：Python 3.9.21, PyTorch 2.5.1, NVIDIA A100 GPU
