---
title: "Seeing-is-Believing-but-How-Much-A-Comprehensive-Analysis-of"
source: https://aclanthology.org/2025.emnlp-main.74.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:06:40"
field: "多模态大模型可靠性与校准"
keywords: ["视觉语言模型", "不确定性量化", "校准", "口语化置信度", "多模态对齐", "提示工程"]
innovations: ["首次系统评估VLM在多模态场景下的口语化校准行为，揭示视觉推理模型的校准优势与跨模态校准鸿沟", "提出VCAP两阶段提示策略，显式分离视觉感知与任务推理以改善多模态置信度校准"]
benchmarks: ["MMMU-Pro", "VideoMMMU", "Visual SimpleQA", "MathVista", "MathVision", "IsoBench"]
---

# 论文速读：Seeing-is-Believing-but-How-Much-A-Comprehensive-Analysis-of-Verbalized-Calibration-in-Vision-Language-Models

## 一句话总结
本文对当前主流视觉-语言模型（VLMs）的**口语化校准（verbalized calibration）**进行了系统性综合评估，发现大多数模型存在严重的置信度校准偏差；同时提出**VCAP（Visual Confidence-Aware Prompting）**，一种通过两阶段提示显式分离视觉理解与任务执行的策略，在多个开源模型上显著改善了校准质量。

## 研究问题与动机
- **核心问题：** 现有 LLM 口语化不确定性的校准研究已较充分，但 VLMs 在多模态环境下表达置信度的校准行为缺乏系统评估。
- **动机一（模型差异）：** 指令微调模型、文本中心推理模型、视觉中心推理模型三者间的校准表现差异尚不清楚。
- **动机二（模态敏感性）：** 将指令嵌入图像（vision-only setting）是否会导致模型置信度估计进一步恶化，尚待验证。
- **动机三（跨模态一致性）：** 语义等价但呈现模态不同的输入（如公式 vs 图像图），模型给出的置信度是否保持一致，现有研究未见系统回答。

## 核心贡献（创新点）
1. **首个全面的 VLM 口语化校准评估：** 覆盖 3 类模型、4 个任务域、3 种评估配置（通用/嵌入式指令/语义对齐），填补了 VLM 多模态校准系统性评测的空白。
2. **发现视觉推理模型的显著优势：** 证明视觉中心推理模型（如 o3/o4-mini）在所有基准上均表现出更优校准，与仅文本推理或指令微调模型形成鲜明对比，为"以视觉为中心的推理有助于不确定性感知"提供了实证依据。
3. **揭示跨模态校准鸿沟：** 发现即使内容一致，VLM 在视觉输入下的校准显著差于文本输入，且在嵌入式指令场景下准确度与校准度同步下降，量化了模态不对齐的影响。
4. **提出 VCAP 两阶段提示策略：** 通过"先描述视觉输入并给出置信度，再基于此完成最终任务"的分步机制，在多个开源模型（Qwen2.5-VL 7B/72B、InternVL3 等）上实现优于 Top-K 和 Self-Reflection 的校准增益。

## 方法详解
- **评估指标：Expected Calibration Error（ECE）**
  公式：$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |\operatorname{acc}(B_m) - \operatorname{avg}\mathbf{Conf}(B_m)|$，共 $M=10$ 个置信度分箱。
- **三种评估配置：**
  - **General Evaluation：** 文本指令 + 图像/视频输入，涵盖 MMMU-Pro、VideoMMMU、Visual SimpleQA、MathVista/MathVision。
  - **Embedded Instruction Evaluation：** 问题完整嵌入图像（vision-only），使用 MMMU-Pro 的 vision split，测试纯视觉指令下的校准变化。
  - **Semantically Aligned Modalities Evaluation：** 使用 IsoBench 数据集，同一语义内容分别以视觉（图像图表、棋局）和文本（LaTeX、PGN）呈现，隔离模态效应。
- **VCAP（Visual Confidence-Aware Prompting）：** 两阶段多轮提示策略。
  - 第一阶段：要求模型详细描述视觉输入并给出独立置信度评分（0-100%），聚焦纯感知阶段，降低认知负载。
  - 第二阶段：要求模型结合第一阶段的视觉描述及对应置信度完成最终任务并输出最终答案和总置信度，实现感知-推理解耦。
- **基线对比方法：**
  - **CoT（vanilla）：** 单轮思维链提示。
  - **Top-K（K=3）：** 生成多个候选答案并各自评分，选最高置信度输出。
  - **Self-Reflection：** 先作答，再单独一轮反思并给出置信度。

## 实验与结果
- **数据集：** MMMU-Pro（多学科图像理解）、VideoMMMU（视频理解）、Visual SimpleQA（事实性）、MathVista/MathVision（数学视觉推理）、IsoBench（模态对齐测试）。
- **评估模型（14 个）：** GPT-4.1、GPT-4o、GPT-5(medium)、o1、o3、o4-mini；Qwen2-VL/2.5-VL（7B/72B）、InternVL3 78B、Kimi-VL-A3B Instruct/Thinking、Skywork-R1V/R1V2 38B。
- **主要结果：**
  - **通用评估（Table 1）：** 大多数指令模型和文本推理模型 ECE > 0.25，呈系统性**过度自信**。o3 在绝大多数任务中 ECE 最低（如 MMMU-Pro: 0.047，Visual SimpleQA: 0.085）。
  - **嵌入式指令评估（Figure 3 / Table 9 Appendix）：** 几乎所有模型的准确率和校准度均出现下降；o3/o4-mini 虽有退化但仍显著优于其他模型。
  - **语义对齐评估（Table 2 / Table 12）：** 在 IsoBench 中，视觉输入普遍比文本输入校准更差（数学领域尤为明显），仅 o3/o4-mini 保持较小差距。
  - **VCAP 效果（Table 3）：** 在 IsoBench 上，VCAP 相比 vanilla CoT、Top-K、Self-Reflection 均带来**校准提升**；例如 Qwen2.5-VL 72B All 上 ECE 从 0.431 降至 0.365，ACC 从 53.7% 升至 57.0%；InternVL3 78B 上 ECE 从 0.412 降至 0.331。
- **最强模型：** OpenAI o3 在多数基准上 ECE 最低（MMMU-Pro: 0.047，Visual SimpleQA: 0.085，IsoBench Mathematics-text: 0.075），展现出最优校准能力。

## 相关工作脉络
1. **Xiong et al. (2024) / Tian et al. (2023)：** 在 LLM 上研究口语化不确定性和 Top-K 提示法，本文将其扩展到多模态 VLM 场景并验证有效性。
2. **Zeng et al. (2025)：** 系统评估推理模型（reasoning models）的校准表现，本文在 VLM 层面补充了"视觉推理模型同样优于文本推理模型"的跨模态证据。
3. **Groot & Valdenegro Toro (2024)：** 小规模日语 VLM 数据上的校准研究；本文扩展到大尺度、多任务、多模态的通用评测框架。
4. **Fu et al. (2024) — IsoBench：** 提出模态对齐基准，本文将其用于系统分析"语义等价但模态不同"时的校准差异，深化了对模态鸿沟的理解。
5. **Fadeeva et al. (2023/2024) / Vashurin et al. (2024)：** 基于采样的不确定性量化方法；本文聚焦无需额外计算开销的"反射型"口语化方法，适合闭源 VLM 部署场景。
6. **Zhao et al. (2025)：** 针对目标级别的 VLM 校准微调；本文从全局视角揭示"视觉推理训练"本身对校准的积极作用。

## 局限性与未来方向
- **局限性一：** o3/o4-mini 等表现优异的视觉推理模型为闭源，其内部训练细节未知，难以深入分析其优势来源。
- **局限性二：** 目前"视觉推理"类别仅有这两个 OpenAI 模型达到生产级性能，无法在同一类别中比较不同模型家族的表现差异。
- **局限性三：** 未对下游应用中常见的高难度场景（如视频理解中的时序视觉定位）进行专门实验。
- **未来方向：** 探索视频理解的时序视觉定位校准；设计针对 VLM 的校准感知微调方法；进一步研究语言式不确定性表达（LVU）与数值式（NVU）在多模态中的表现差异。

## 研究启发与可借鉴点
1. **三阶段评估设计思路可迁移：** 通用指令/嵌入式指令/语义对齐的配置框架，可用于评测其他 VLM 特性（如幻觉率、鲁棒性）的跨模态一致性。
2. **VCAP 的两阶段解耦范式可复用：** 在任意需要置信度输出的多模态任务中，可将"感知-推理"分步策略作为即插即用的校准提升模块，无需重新训练。
3. **视觉推理模型的校准优势为模型选择提供参考：** 若任务涉及高不确定性感知需求（如医疗影像、自动驾驶），应优先考虑经过视觉推理训练的模型架构。
4. **LVU vs NVU 的对比实验设计（Appendix F）：** 可借鉴于评估文本类 LLM 中不同不确定性感官表达方式的效果差异。

## 关键术语表
- **Verbalized Calibration（口语化校准）：** 模型通过自然语言显式表达其对答案置信度的估计，并与实际准确率对齐的程度。
- **ECE（Expected Calibration Error，期望校准误差）：** 衡量模型预测置信度与实际准确率之间差异的指标，值越低表示校准越好。
- **Visual Reasoning Models（视觉推理模型）：** 将视觉输入直接融入多步推理过程的模型（如 OpenAI o3/o4-mini），而非仅将图像作为辅助信息。
- **VCAP（Visual Confidence-Aware Prompting）：** 本文提出的两阶段提示策略，先让模型描述视觉内容并给出感知置信度，再据此完成最终任务推理。
- **Modality Misalignment（模态不对齐）：** 同一语义内容以不同模态（文本/图像）呈现时，模型输出表现不一致的现象。
- **Embedded Instruction（嵌入式指令）：** 将问题指令直接嵌入图像中（而非通过文本输入），测试模型纯视觉理解能力。
- **IsoBench：** 专为测试多模态模型在模态转换（图文同构）下性能对齐而设计的基准，涵盖数学、游戏、科学、算法四个领域。
- **Top-K Prompting（Top-K 提示）：** 让模型生成 K 个候选答案并各自评分，取置信度最高的作为最终输出的提示策略。

## 可复现要素
- **数据集：** MMMU-Pro（公开）、VideoMMMU（公开）、Visual SimpleQA（公开）、MathVista（公开）、MathVision（公开）、IsoBench（公开）。
- **代码/权重：** 模型权重方面，Qwen2.5-VL、InternVL3、Kimi-VL、Skywork-R1V 系列权重公开；o3/o4-mini 为闭源 API。论文未提供专用代码仓库链接。
- **关键超参：** vLLM 推理引擎，采用默认 SamplingParams；ECE 分箱数 M=10；Top-K 中 K=3；视频统一采样 32 帧；LLM 输出格式通过 XML 标签 `<confidence>` 提取。
- **置信度提取：** 少量偏离格式的响应使用 GPT-4.1 标准化；缺失置信度的样本（<2%）被剔除。
