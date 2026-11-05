---
title: "User-Feedback-in-Human-LLM-Dialogues-A-Lens-to-Understand-Us"
source: https://aclanthology.org/2025.emnlp-main.133.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:08:58"
field: "大语言模型对齐与用户交互"
keywords: ["implicit feedback", "user-LLM interaction", "model alignment", "SFT", "feedback semantics", "WildBench"]
innovations: ["构建首个密集标注的真实对话隐式反馈数据集并进行系统分析", "提出利用反馈语义内容进行响应重生成而非仅依赖极性的训练方法", "揭示反馈信号在简单与复杂任务上的相反效果，为 noisy real-world data 利用提供警示"]
benchmarks: ["MT-Bench", "WildBench", "LMSYS-chat-1M", "WildChat"]
---

# 论文速读：User-Feedback-in-Human-LLM-Dialogues-A-Lens-to-Understand-Us

## 一句话总结
本文系统研究了从真实用户-LLM交互日志中隐式获取的用户反馈，分析了其发生模式与成因，并探索利用反馈语义内容（而非仅极性）进行模型蒸馏改进的效果；实验表明该信号对简单基准MT-Bench有效，但在复杂基准WildBench上效果有限甚至有害。

## 研究问题与动机
- 实际部署的LLM会长期与用户交互，但显式请求反馈具有侵入性，如何有效利用隐式反馈信号成为关键问题。
- 已有工作（如Don-Yehiya et al., 2024）仅使用反馈极性（正/负）训练模型，但本文发现单纯"促进正向、抑制负向"可能导致模型退化。
- 现有标注数据稀疏且不完整（仅标注部分turn），缺乏对完整对话中反馈动态的系统分析。
- 用户反馈的内容语义是否能为模型提供更精准的改进信号，值得深入研究。

## 核心贡献（创新点）
- 构建了首个密集标注的隐式反馈数据集：在109条真实对话（LMSYS 75条 + WildChat 34条）中对每个用户turn进行完整反馈标注，提供了比先前工作更全面的分布统计。
- 提出基于in-context示例的LLM反馈检测prompt模板：相比Prior工作，稀疏设置下二分类准确率从41.4%提升至81.1%，密集设置下细粒度分类准确率翻倍以上。
- 揭示了反馈发生规律与负面成因：反馈在长对话后段高频出现（占后期turn一半以上）；引发正向反馈的用户prompt反而质量更低且更toxic（涉及jailbreak），拒绝行为仅占<3%，不是负反馈主因。
- 提出了利用反馈语义进行响应重生成（Regeneration w/ Semantics）的方法：对收到负反馈的实例，用强LLM结合反馈内容重新生成回答用于SFT训练，而非仅依赖初始用户输入。
- 发现反馈信号的复杂性与噪声问题：在简单任务（MT-Bench）上引入反馈语义有显著提升（Vicuna-7b从6.37升至6.86），但在复杂任务（WildBench）上反而下降（从28.90降至23.38）。

## 方法详解
**反馈分类体系**：遵循Don-Yehiya et al.的五类ontology——正向反馈、Rephrasing（重述）、Make Aware without Correction（指出错误）、Make Aware with Correction（指出错误+修正）、Ask for Clarification（请求澄清）、无反馈。

**密集标注流程**：作者阅读Prior工作的标注指南后，对75条LMSYS和34条WildChat对话进行完整标注（共433个用户turn），多标签情况按优先级规则取单标签（Make Aware with Correction > Make Aware without Correction > Ask for Clarification > Rephrasing）。

**自动检测方法**：使用GPT-4o-mini配合带in-context示例的新prompt模板，对完整对话中每个用户turn输出n-1维的反馈标签向量。

**响应重生成方法**：
- Regeneration w/ Semantics：$\mathbf{m}_i^{sem} = \phi(\mathbf{u}_i, \mathbf{m}_i, \mathbf{u}_{i+1})$，即用强LLM结合初始请求、初始回复和负反馈重新生成
- Baseline Regeneration from Scratch：$\mathbf{m}_i^{scra} = \phi(\mathbf{u}_i)$，仅用强LLM从原始用户输入重新生成
- 使用Reward Model进行成对评估，分别在有/无反馈上下文条件下打分

**训练流程**：收集D_neg（含负反馈子对话）和D_rand（随机采样），各1K条；用GPT-4o-mini重生成20K条对话用于SFT训练，学习率5e-6，A100 GPU约2小时。

## 实验与结果
**数据集**：LMSYS-chat-1M（评测导向，较短，含边界案例）和WildChat（真实需求导向，较长，任务复杂）。

**自动检测性能**（Table 2）：
- Sparse设置：二分类准确率81.1% vs Prior的41.4%，F1提升显著
- Dense设置：细粒度分类49.0% vs Prior的22.3%，准确率翻倍

**成对评估结果**（Table 3，LMSYS）：
- Better模型生成的m^scr对Weak模型初始回复win rate达88%（无反馈评估）
- m^sem在Eval w/ fb下89% win rate，但在Eval w/o fb下仅61%
- Weak模型自我改进（m_{i+1} vs m_i）win rate仅58%（WildChat为74%）

**SFT训练结果**（Table 4，主要数据）：
- **MT-Bench**（简单任务）：SFT on m^sem with D_neg效果最好，Vicuna-7b达6.68±0.03（vs base 6.09），Mistral-7B达6.32±0.03（vs base 3.09）
- **WildBench**（复杂任务）：SFT on m^scr with D_rand效果最好，Vicuna-7b达28.90±3.38，而SFT on m^sem仅24.47（LMSYS）和23.38（WildChat），明显低于baseline

**KTO基线对比**：纯极性监督在MT-Bench上未见提升（Vicuna-7b保持6.09），WildBench上反而下降至21.33（vs base 26.0）。

**核心结论**：反馈语义有助于简单任务，但在复杂任务上可能是噪声；直接蒸馏强模型输出比利用反馈语义更有效。

## 相关工作脉络
- **Don-Yehiya et al. (2024)**：首次系统研究隐式反馈的分类与利用，但仅使用极性信号进行KTO训练；本文扩展至语义利用和密集标注分析。
- **Shaikh et al. (2025)**：从grounding角度定义七类用户响应ontology，与本文有重叠但不完全兼容（"Next Turns"和"Follow-ups"不被视为反馈）。
- **Ethayarajh et al. (2024, KTO)**：基于非配对偏好数据的对齐方法；本文将其作为基线对比。
- **Madaan et al. (2024, Self-Refine)**：通过自我反馈迭代优化模型生成；本文不同在于利用真实用户反馈而非模型自我反思。
- **Chang et al. (2025)**：指出多轮交互中LLM表现低于单轮，且受限于人机双方能力上限。
- **Gao et al. (2024)**：从用户编辑行为推断隐式偏好；本文聚焦自然对话中的语言反馈。

## 局限性与未来方向
- 假设所有对话位置的反馈同等重要，但实际不同阶段反馈作用不同（如修正答案 vs 确认目标达成）。
- 仅将反馈视为对最近模型回复的评价，忽略了用户可能希望修改更早回复的场景。
- 未解决"谁的观点应该被对齐"的问题——不同用户可能有不同偏好（如详细解释vs简洁回答）。
- 训练数据存在隐私泄露和偏见放大的风险，仅建议用于研究目的。
- 反馈检测在密集设置下细粒度准确率仍有限（~49%），存在噪声。

## 研究启发与可借鉴点
- **数据质量优先于信号复杂性**：利用反馈语义的提升有限，直接蒸馏强模型（m^scr from D_rand）在复杂任务上表现更稳定，提示 noisy real-world data 需谨慎处理。
- **区分任务复杂度设计评估**：简单基准（MT-Bench）与复杂基准（WildBench）呈现相反趋势，提醒我们在设计反馈学习框架时需考虑任务难度分层。
- **in-context learning提升反馈检测**：通过few-shot示例可将自动检测准确率大幅提升，为大规模隐式反馈挖掘提供了实用方案。
- **positive feedback可能隐藏风险**：正向反馈与用户prompt toxic性和jailbreak意图正相关，警示单纯追求positive feedback可能放大有害行为。
- **成对评估的双重条件设计**：Eval w/ fb和Eval w/o fb的对比有效分离了"反馈语义帮助"与"模型能力提升"的贡献，值得借鉴。

## 关键术语表
- **Implicit User Feedback**：用户在多轮对话中未明确表达满意度，但通过后续输入间接反映对前一轮模型回复不满的信号。
- **Regeneration w/ Semantics (m^sem)**：利用强LLM结合初始请求、初始回复和用户负反馈内容重新生成改进回答的方法。
- **Regeneration from Scratch (m^scr)**：仅基于初始用户请求、忽略历史回复和反馈，用强LLM重新生成回答的基线方法。
- **D_neg / D_rand**：分别指包含负反馈的子对话数据集和随机采样的子对话数据集，用于SFT训练对比。
- **KTO (Kahneman-Tversky Optimization)**：基于前景理论的模型对齐方法，直接利用非配对偏好数据进行训练。
- **WildBench**：包含1024道来自WildChat的真实复杂用户问题的评测基准，平均499 tokens，复杂度评分4.31。
- **MT-Bench**：80道人工构建的双轮问题基准，平均91 tokens，复杂度3.85，侧重常见任务类型。
- **Make Aware with Correction**：用户不仅指出模型错误，还提供修正指令的反馈类型，语义信息最丰富。

## 可复现要素
- **数据集**：LMSYS-chat-1M和WildChat均为公开数据集；本文新增的109条密集标注数据已公开（论文声明" Our dataset and code is shared publicly"）。
- **代码**：论文声明开源，具体仓库链接见脚注3。
- **关键超参**：SFT学习率5e-6，每轮约2小时（单卡A100），训练数据量20K条对话。
- **模型配置**：Base模型为Vicuna-7b和Mistral-7b；重生成模型为GPT-4o-mini。
- **评估设置**：使用GPT-4作为LLM-Judge，报告5次随机初始化的均值和方差；WildBench随机采样500题。
