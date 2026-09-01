---
title: "Parallel-Continuous-Chain-of-Thought-with-Jacobi-Iteration"
source: https://aclanthology.org/2025.emnlp-main.47.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 17:00:57"
field: "高效大语言模型推理"
keywords: ["Chain-of-Thought", "Continuous Reasoning", "Parallel Inference", "Jacobi Iteration", "Efficient LLM", "Implicit CoT", "Self-Distillation"]
innovations: ["将Jacobi迭代引入连续CoT，实现隐式思维token的并行更新而非序列解码，打破自回归依赖", "提出统一框架PCCoT，通过单一超参T等价覆盖Pause Tokens、连续CoT、iCoT等方法", "揭示迭代次数非单调效应及隐式token间奇偶交错相似性模式，发现训练后token不收敛现象"]
benchmarks: ["GSM8K-Aug", "GSM8K-Aug-NL"]
---

# 论文速读：Parallel-Continuous-Chain-of-Thought-with-Jacobi-Iteration

## 一句话总结
本文提出 Parallel Continuous Chain-of-Thought (PCCoT)，通过在隐式思维 token（latent thought tokens）上执行 Jacobi 迭代，将原本序列依赖的连续链式思考过程转为并行更新，在数学推理任务上实现可比或更优性能的同时，节省约 50% 的训练与推理时间，并表现出更好的训练稳定性。

## 研究问题与动机
1. **连续 CoT 的效率瓶颈**：连续链式思考通过连续隐式向量（latent thought tokens）进行隐式推理，能有效节省推理 token 数；但隐式 token 间存在序列依赖，无法并行训练，导致训练时间长、推理速度慢。
2. **已有方法无法打破序列依赖**：现有连续 CoT 方法均采用序列解码方式逐步生成每个隐式 token，从根本上受限于前向传递的串行性。
3. **隐式 token 数量难以扩展**：使用更多隐式 token 意味着需要更多前向传递步数，导致训练不稳定且性能下降（作者实验中发现），限制了连续 CoT 的扩展能力。
4. **标准 CoT 存在训练-推理间隙**：标准 CoT 训练中基于金标中间步骤预测下一个 token，推理时却自回归生成，错误会级联传播；连续表示天然规避了这一 gap。

## 核心贡献（创新点）
1. **提出 PCCoT 框架**：将非线性 Jacobi 迭代引入连续 CoT，所有隐式思维 token 通过并行迭代同步更新，替代传统的序列解码。与已有工作的本质区别在于打破了隐式 token 间的序列化依赖，实现训练/推理的双重加速。
2. **建立统一理论联系**：证明 PCCoT 统一了多个已有方法——当 $T=0$ 时等价于 Pause Tokens，当 $T=c$ 时等价于连续 CoT，给出完整的计算图等价性数学归纳法证明（Appendix A）。
3. **揭示迭代次数的非单调效应**：实验发现增加迭代次数 $T$ 并非单调提升性能（在 $T=3$ 附近达到最佳，之后反而下降），并解释这与训练不稳定性相关，为后续工作提供了重要的调参经验。
4. **发现隐式 token 的非收敛特性**：训练后的隐式 token 在多轮 Jacobi 迭代中不再收敛（与随机初始化模型的行为相反），表明 PCCoT 的成功并非源于"快速收敛到不动点"，颠覆了直觉假设。
5. **揭示隐式 token 间的结构化相似模式**：发现隐式 token 间存在奇偶交错（interleaved）的相似性模式，提示 token 间存在隐藏的结构化依赖关系，为理解连续推理机制提供了新线索。

## 方法详解
### 核心算法设计
1. **初始化阶段**：给定查询序列 $x = (x_1, x_2, \ldots, x_n)$，在其后附加一个可学习的 `<bot>` token 和 $c$ 个占位 `<latent>` token，构成完整输入序列送入 Transformer 模型 $f$：
   $$[h_{n+1}^{(1)}; \ldots; h_{n+c+1}^{(1)}] = f([E_{x_1}; \ldots; E_{x_{n+c+1}}])$$

2. **Jacobi 迭代更新**：对接下来的 $T$ 轮额外迭代，每一轮将上一轮最后 $c$ 个隐式 token 的隐藏状态作为本轮的输入嵌入（替代原始 `<latent>` embedding），并行更新所有隐式 token：
   $$[h_{n+1}^{(t+1)}; \ldots; h_{n+c+1}^{(t+1)}] = f([E_{x_1}; \ldots; E_{x_{n+1}}; h_{n+1}^{(t)}; \ldots; h_{n+c}^{(t)}])$$
   其中 $t = 1, 2, \ldots, T$。关键：每轮迭代所有 $c$ 个隐式 token **并行**计算，而非逐个生成。

3. **答案生成**：经过 $T$ 轮迭代后，附加 `<eot>` token，以最后一个迭代周期的隐式 token 隐藏状态为上下文，用标准 Transformer 自回归方式生成答案 token。

4. **与连续 CoT 的等价性（Appendix A 证明）**：通过数学归纳法证明，当 $T \geq c$ 时，PCCoT 的计算图与使用 $c$ 个隐式 token 的连续 CoT 完全等价——第 $i$ 个隐式 token 在第 $i$ 轮迭代后达到固定点。

5. **训练方法（CODI 框架）**：采用 CODI（Shen et al., 2025）自蒸馏框架：教师任务学习标准 CoT 并施加交叉熵损失，学生任务学习连续 CoT 仅对答案 token 施加交叉熵损失，并通过 L1 损失蒸馏教师预测分布；实验中不使用 CODI 中的 MLP 增强层以保证公平对比。

6. **关键超参**：GPT-2 Small 上 $c=24$（隐式 token 数）、$T=3$（额外迭代数）；Llama3.2-1B-Instruct 上同样设置；LoRA rank $r=128$，alpha=32。

## 实验与结果
### 数据集
- **GSM8K-Aug**：基于 GSM8K 扩展，385k 训练集，推理步骤使用纯数学方程
- **GSM8K-Aug-NL**：同上，推理步骤使用自然语言

### 评估基线
Implicit CoT、Pause Tokens、Continuous CoT（同框架训练）、CODI、Coconut、iCoT（文献引用值）

### 主要结果（GPT-2 Small）
| 方法 | GSM8K-Aug | GSM8K-Aug-NL |
|------|-----------|---------------|
| CoT | 44.1 | 34.8 |
| Continuous CoT | $48.24 \pm 1.61$ | $45.06 \pm 2.58$ |
| **PCCoT（Ours）** | **$\mathbf{49.48 \pm 0.31}$** | **$\mathbf{49.23 \pm 0.80}$** |

- PCCoT 在两个数据集上均取得最高准确率，且标准差显著更小（训练更稳定）
- 训练时间从 24.91h 降至 13.72h（节省 ~45%）；推理时间从 0.443s 降至 0.199s（节省 ~55%）

### 主要结果（Llama3.2-1B-Instruct）
| 方法 | GSM8K | GSM8K-NL |
|------|-------|----------|
| Continuous CoT | $50.47 \pm 0.68$ | $48.47 \pm 1.40$ |
| **PCCoT（Ours）** | **$53.35 \pm 0.18$** | **$\mathbf{50.72 \pm 1.39}$** |

### 消融与发现
- 迭代次数 $T$ 非单调：$T=3$ 附近最优，$T>6$ 时性能波动增大且训练不稳定
- 固定隐式 token 初始化（不可训练）仍能得到合理性能（47.31% vs 49.48%），表明模型对初始化具有鲁棒性
- PCCoT 优于"用 CoT 解码方式的 PCCoT"，验证了连续推理避免训练-推理间隙的假设

## 相关工作脉络
1. **Continuous CoT（Coconut, Hao et al. 2024）**：首次提出用连续隐式向量替代离散推理 token；PCCoT 在其基础上引入 Jacobi 迭代实现并行化，$T=c$ 时与其等价。
2. **Pause Tokens（Goyal et al. 2024 ICLR）**：在输入序列中插入可训练离散 pause token 以扩展计算深度；PCCoT 用连续隐式 token + 并行迭代替代，$T=0$ 时与其等价。
3. **Implicit CoT / iCoT（Deng et al. 2023, 2024）**：前者用知识蒸馏让模型学习隐式推理，后者进一步去除所有推理 token；PCCoT 保留了隐式推理空间但大幅提升效率，$c=0$ 时接近 iCoT。
4. **CODI（Shen et al. 2025）**：单模型自蒸馏框架，同时学习显式 CoT 和连续 CoT；本文将其作为训练方法，而非方法本身。
5. **Jacobi Decoding（Santilli et al. 2023）**：将 Jacobi 迭代用于自回归语言模型的并行解码；与 PCCoT 的关键区别在于该方法涉及离散 token 解码，而 PCCoT 全程无离散解码，因此可统一用于训练和推理。
6. **Looped Transformers / Reasoning with Latent Thoughts（Saunshi et al. 2025）**：从理论角度分析 looped transformer 自然诱导隐式推理；PCCoT 提供了实用的训练框架，将其思想落地为可操作的并行方案。

## 局限性与未来方向
1. **训练仍慢于标准 CoT**：当前基于 CODI 的蒸馏训练策略导致 PCCoT 训练时间仍远超标准 CoT（GPT-2 Small 上 13.72h vs 4.39h），需探索更高效的训练范式。
2. **隐式 token 行为机制尚不清晰**：训练后隐式 token 在多轮迭代中不收敛、奇偶交错相似模式等现象缺乏理论解释，阻碍了对方法的深入理解和扩展。
3. **当前仅在数学推理数据集验证**：CLUTRR、CommonsenseQA、StrategyQA 等数据集因样本量不足无法区分方法性能，需在更大规模、更多样化任务上验证泛化能力。
4. **扩展到大模型的可行性待验证**：论文仅在 GPT-2 Small 和 Llama3.2-1B-Instruct 上验证，未尝试更大模型规模，Scaling law 行为未知。

## 研究启发与可借鉴点
1. **Jacobi 迭代替代序列依赖的计算范式**：将 Jacobi 迭代用于打破自回归序列依赖的思路可迁移到其他需要隐式中间表示的任务（如隐式规划、状态估计），为并行化序列生成提供新思路。
2. **隐式表示避免训练-推理间隙的洞察**：PCCoT 优于标准 CoT 解码版本的现象，证实了连续表示在规避自回归错误传播方面的优势，这一思路可用于改进其他 CoT 变体的训练策略。
3. **迭代次数非单调性的经验法则**：$T$ 不宜过大（超过 $c/4$ 左右性能开始下降），这一发现提示设计并行推理方法时需权衡并行度与迭代深度，避免"过度迭代导致训练不稳定"。
4. **统一框架的设计哲学**：PCCoT 通过单一超参 $T$ 统一了 Pause Tokens、连续 CoT、iCoT 等多个方法，这种"超参连续化"的建模思路可为消融现有方法家族提供通用范式。
5. **隐式 token 初始化鲁棒性分析**：固定初始 embedding 后模型仍保持较高性能（仅下降 2.17%）的结果，为后续研究提供了安全初始化的参考基线。

## 关键术语表
**Latent Thought Token（隐式思维 token）**：连续 CoT 中表示中间推理步骤的连续向量，替代标准 CoT 中的离散文本 token，支持紧凑且高效的隐式推理。
**Jacobi Iteration（雅可比迭代）**：一种数值分析方法，在多变量方程组中利用上一轮的各分量值并行更新所有分量；本文将其应用于隐式 token 的并行更新。
**CODI（Compressing Chain-of-Thought into Discrete-continuous space via self-distillation）**：一种自蒸馏训练框架，单模型同时担任教师（标准 CoT）和学生（连续 CoT）角色，通过 L1 损失蒸馏知识。
**Pause Token（暂停 token）**：在输入序列中插入的可训练离散 token，允许模型在特定位置进行额外计算，是 PCCoT 在 $T=0$ 时的特例。
**Implicit CoT（iCoT）**：通过知识蒸馏让模型学习隐式推理并直接生成答案，无需任何显式或隐式推理 token。
**Training-Inference Gap（训练-推理间隙）**：标准自回归模型在训练中基于金标前驱 token 预测、推理时基于自身生成的前驱 token 预测，导致分布不一致的现象。

## 可复现要素
- **数据集**：GSM8K-Aug 和 GSM8K-Aug-NL（MIT 许可，385,620 训练样本），论文声明可用
- **代码**：已开源，GitHub 地址 https://github.com/whyNLP/PCCoT
- **预训练权重**：论文未提及独立开源权重，需自行从 GPT-2 Small 和 Llama3.2-1B-Instruct 微调
- **关键超参**：
  - LoRA rank $r=128$，alpha=32，dropout=0.1，target modules=[attn, ffn]
  - 优化器：AdamW，GPT-2 学习率 $3\times10^{-3}$，Llama 学习率 $8\times10^{-4}$
  - batch size=128，GPT-2 训练 40 epochs，Llama 训练 10 epochs
  - PCCoT: $c=24$ 隐式 token，$T=3$ 额外迭代
  - 连续 CoT 基线：12 个隐式 token（论文发现比 CODI 原 paper 的 6 个效果更好）
  - 推理策略：greedy decoding
- **实现基础**：HuggingFace Transformers + FlashAttention-2
