---
title: "Thinking-Out-Loud-Do-Reasoning-Models-Know-When-They-re-Righ"
source: https://aclanthology.org/2025.emnlp-main.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:07:40"
field: "大语言模型可靠性评估"
keywords: ["calibration", "reasoning models", "uncertainty quantification", "reinforcement learning", "large language models"]
innovations: ["首次系统比较指令微调/SFT推理/RL推理三类模型的口语化校准性能", "揭示推理训练导致的'U型'校准轨迹与知识边界感知下降", "分析推理链长度与置信度校准的非线性关系"]
benchmarks: ["AIME 2024/2025", "SimpleQA", "FreshQA", "GPQA-Diamond", "SuperGPQA", "LiveBench-Reasoning"]
---

# 论文速读：Thinking-Out-Loud-Do-Reasoning-Models-Know-When-They-re-Righ

## 一句话总结
本文系统研究了大型推理模型（LRMs）的**口语化校准能力**（即模型表达自身确定性的可靠性），发现基于推理轨迹的SFT与强化学习可阶梯式提升复杂推理任务上的校准质量，但可能导致模型对事实性知识的边界感知下降（"推理税"）。

## 研究问题与动机
1. **核心问题**：具备自我反思行为的推理模型（如DeepSeek-R1、o1等）是否真正更了解自身的知识边界？其口语化置信度表达是否与真实正确率更好对齐？
2. **现有方法不足**：先前研究表明多数指令微调LLM在各类任务上系统性高估自身确定性（Xiong et al., 2024），且校准性能对提示设计高度敏感（Tian et al., 2023; Yang et al., 2024），缺乏稳定可靠的内部不确定性度量。
3. **动机缺口**：尽管LRMs已展现出类人自我反思行为（如"Wait, but..."），但这类行为如何影响其校准特性尚属空白，亟需在统一框架下比较不同训练范式的校准表现。

## 核心贡献（创新点）
1. **系统性校准基准**：首次在统一实验设置下比较指令微调模型、SFT推理模型、RL推理模型三类架构在数学、事实性、科学推理、一般推理四大领域的校准与失败预测能力，填补了跨训练范式校准对比的空白。
2. **阶梯式改进发现**：揭示SFT于推理轨迹可显著提升推理密集型任务的准确率与校准质量，而推理RL能在跨域（如coding/math训练对science评测）持续改善校准，体现RL的泛化优势。
3. **"推理税"量化**：发现小型SFT推理模型在事实性基准上校准恶化且显著减少"I don't know"响应，表明过度优化推理能力可能损害模型对自身知识边界的意识。
4. **推理链长度分析**：提出Thoughtology视角，实证分析推理链长度与准确率、置信度、校准误差（ECE）的关系，发现极长推理链（>10k tokens）在小型模型上引入额外不确定性。

## 方法详解
- **模型分类**：划分为三类——Instruct模型（Qwen2.5-14B/32B, DeepSeek-V3）、SFT推理模型（R1-Distill-Qwen-14B/32B，基于800k蒸馏轨迹）、RL推理模型（DeepCoder-14B, Skywork-OR1-32B, DeepSeek-R1）。
- **评估基准**：数学（AIME 2024/2025）、事实性（SimpleQA, FreshQA）、科学推理（GPQA-Diamond, SuperGPQA）、一般推理（LiveBench-Reasoning）。
- **提示策略**：
  1. Vanilla CoT（基础思维链）
  2. Vanilla CoT + probability mass（要求输出0.0–1.0概率分数）
  3. Self-reflection（两轮对话：首轮作答，二轮评估自身置信度）
- **核心指标**：
  - **校准误差**：Expected Calibration Error (ECE) = $\sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{avgConf}(B_m)|$，及自适应版本(ACE)
  - **失败预测**：AUROC、AUPRC-P（正例）、AUPRC-N（负例）
  - **知识边界感知**："not attempted"响应比例

## 实验与结果
- **数学推理**：R1-Distill-Qwen-32B在AIME上准确率从9.67%跃升至65.7%，ECE从0.752降至0.240；DeepSeek-R1达到68.0%准确率与0.136 ECE（最强）。
- **科学推理**：Skywork-32B在SuperGPQA上达50.4%准确率，ECE仅0.298；DeepSeek-R1在GPQA-Diamond上68.7%准确率，ECE仅0.082。
- **事实性基准**：小型SFT模型（如R1-Distill-Qwen-14B）在SimpleQA上准确率5.69% vs 指令模型6.04%，ECE却从0.625恶化至0.719；但DeepSeek-R1（大模型）在SimpleQA上29.7%准确率与0.324 ECE显著优于DeepSeek-V3的21.4%与0.515。
- **关键结论**："U型"校准轨迹——指令微调→SFT推理（校准下降）→RL推理（校准恢复）；模型规模是关键调节变量，大模型更能从推理训练获益。

## 相关工作脉络
1. **Wei et al. (2024) SimpleQA**：评估o1在事实性任务上的校准，发现o1校准更好且"I don't know"比例更高；本文延伸对比了更多开源LRM，揭示训练阶段的细微差异。
2. **Xiong et al. (2024) 广泛校准研究**：指出多数指令微调LLM系统性高估确定性；本文进一步区分了不同后训练范式（SFT vs RL）对校准的非对称影响。
3. **Tian et al. (2023) 提示设计敏感性**：证明校准对提示高度敏感；本文系统比较三种提示策略，发现RL模型校准更稳健。
4. **Chu et al. (2025) SFT记忆vs RL泛化**：提出SFT"记忆"而RL"泛化"的理论框架；本文实证校准维度支持这一观点，并细化到不同任务领域。
5. **Song et al. (2025) Hallucination Tax**：发现RL后模型更频繁尝试回答不可解题；本文独立量化了该现象在开源模型中的规模与后果。
6. **Marjanovic et al. (2025) Thoughtology**：初步探索推理链长度与模型行为的关系；本文扩展了该分析，引入校准误差作为第三个维度。

## 局限性与未来方向
- **局限**：
  1. 未评估代码推理任务（多数模型无法同时输出代码片段与置信度）
  2. 事实性基准评估未包含人工验证，依赖自动判定
  3. 模型规模有限（最大70B），难以断言超大模型行为
- **未来方向**：
  1. 设计能鼓励适当拒答（abstention）的奖励函数，缓解"幻觉税"
  2. 探索混合训练策略（如事实性SFT+推理RL）以平衡能力与校准
  3. 研究推理链长度与置信度校准的非线性关系，优化生成长度控制

## 研究启发与可借鉴点
1. **阶梯式训练分析框架**：将后训练分解为"Instruct→SFT→RL"三阶段并系统评估校准，为理解训练过程对模型可靠性的影响提供了清晰的分析范式。
2. **跨域泛化检验**：在coding/math训练后评估science/factuality校准，揭示了RL带来的校准泛化能力，可推广至其他领域的鲁棒性检验。
3. **失败预测分解**：同时报告AUROC与AUPRC-P/N，区分模型区分正确/错误预测的能力，为可靠性评估提供更细粒度的诊断。
4. **知识边界量化**：将"I don't know"响应率作为校准的子维度，为评估模型自我意识提供了可操作的指标。
5. **提示策略对比实验**：系统比较Vanilla CoT、概率分数输出、自我反思三种策略，发现SR在事实性任务上效果因数据集而异，提示设计需因任务定制。

## 关键术语表
- **Verbalized Confidence/Calibration**：通过语言直接输出的置信度估计及其与真实正确率的一致性程度
- **ECE (Expected Calibration Error)**：按置信度分箱后，各箱内平均置信度与实际准确率的加权平均绝对偏差
- **ACE (Adaptive Calibration Error)**：基于数据分布动态调整分箱边界的校准误差，确保每箱样本数相等
- **AUROC/AUPRC**：评估模型利用置信度区分正确/错误预测的二分类性能指标，前者关注整体排序，后者关注正例召回
- **Reasoning Tax/幻觉税**：过度优化推理能力导致的校准退化与知识边界意识丧失现象
- **Self-Reflection Prompting**：两轮对话提示策略，第二轮要求模型对自身答案进行置信度评估
- **SFT Distillation**：使用强模型生成的长推理轨迹对弱模型进行监督微调的过程
- **Thoughtology**：研究推理链长度与模型行为、性能、置信度之间关系的分析视角

## 可复现要素
- **数据集**：所有评估基准均公开（AIME, SimpleQA, FreshQA, GPQA-Diamond, SuperGPQA, LiveBench-Reasoning）
- **代码/权重**：模型权重全部开源（Qwen2.5系列、DeepSeek-R1/Distill系列、DeepCoder、Skywork-OR1）；推理代码使用Huggingface Transformers与API
- **关键超参**：temperature=0.6/0.7（见表6），top_p=0.8–1.0，序列长度限制3K–16K；所有模型最多生成32,000新token用于推理链
- **提示模板**：完整模板见附录A，可直接复现三种提示策略
