---
title: "ViClaim-A-Multilingual-Multilabel-Dataset-for-Automatic-Clai"
source: https://aclanthology.org/2025.emnlp-main.21.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:09:57"
---

# 论文速读：ViClaim-A-Multilingual-Multilabel-Dataset-for-Automatic-Clai

## 一句话总结
本文发布ViClaim，首个面向短视频转录文本的多语言、多标签主张检测数据集（覆盖英/德/西语3种语言、6个主题、1,798个视频、17,116句标注）。通过定制标注平台与MACE软标签机制缓解主观歧义，并在多语言预训练模型上建立基线，验证了模型在域内可实现高宏F1，但跨主题迁移时性能显著下降。

## 研究问题与动机
- 现有虚假信息检测高度集中于书面文本（如Twitter帖子）或视觉篡改分析，短视频口语转录的细粒度语义分析几乎空白。
- 口语转录在句法结构、语用特征与上下文依赖上与书面语差异显著，直接复用书面语Claim检测模型性能骤降（附录F显示F1从0.693跌至0.32）。
- 现有视频虚假信息研究多为全视频二分类任务（判断视频是否造假），未将“主张识别（Claim Detection）”作为独立前置子任务建模。
- 缺乏支持跨语言、跨主题迁移评估的资源，难以系统度量视频主张检测的领域泛化瓶颈。

## 核心贡献（创新点）
- 提出ViClaim数据集：首个覆盖多语言、多主题、句子级多标签的视频转录主张标注库，填补短视频口语语义分析的基准空白。
- 设计配套标注工具与软标签范式：开发可视化标注平台，引入MACE贝叶斯信任模型生成soft-label，将标注者分歧转化为概率化监督信号，适配主观边界模糊任务。
- 建立多语言基线与双协议评测体系：微调XLM-RoBERTa-Large及三款开源LLM，同步提供Cross-Valid（测性能上限）与Leave-Topic-Out（测域迁移）两套评估协议。
- 量化口语-书面语模态鸿沟：通过冻结的CheckThat!书面语模型在ViClaim上的大幅衰退，实证验证现有文本Claim资源不可直接迁移至视频转录场景。

## 方法详解
- **数据构建管线**：人工筛选≤90秒YouTube Shorts，使用AssemblyAI生成转录、SpaCy切句；12名多语种标注员独立完成，每人分配600个视频，支付€500/人。
- **多标签体系**：每句可分配FCW（可核查事实）、FNC（不可核查/无公共兴趣事实）、OPN（观点/推测/情绪）及None，允许多标签共存。
- **软标签生成**：采用MACE计算每位标注者的信任权重$w_i$，软标签定义为$y_{\mathrm{soft}} = \sum_i w_i \cdot a_i$（$a_i \in \{0,1\}^3$为硬标注向量），保留类别共存的不确定性。
- **模型输入与训练**：输入拼接形式为`[完整转录文本; 目标句]`，为分类器提供上下文；XLM-RoBERTa-Large采用标准微调，Falcon-7B/Mistral-7B/Llama3.2-3B采用QLoRA；损失函数为输出与软标签的交叉熵。
- **评估协议**：(1) Cross-Valid：按语言与主题分层的5折交叉验证；(2) Leave Topic Out：轮留出单一主题作测试集，训练集仅含其余5主题；以各标签AUC+Youden’s J统计量寻优阈值，macro F1为早停指标。

## 实验与结果
- **数据集规模**：1,798视频 / 17,116句，EN/DE/ES三语各约600视频，6主题（US Elections 2024、War in Ukraine、Migration、European Union、General US Society、League of Legends）。
- **Cross-Valid结果**：XLM-RoBERTa-Large表现最佳，FCW F1=0.899、FNC F1=0.776、OPN F1=0.836，宏观F1达0.896；FCW因数据充裕且易识别得分最高，FNC因歧义性略低。
- **Leave Topic Out结果**：与训练分布相近的US Society表现最佳（F1≈0.798）；远离政治语境的League of Legends下降最明显（F1≈0.690~0.721），跨主题落差约0.1~0.2。
- **最强结果与提升幅度**：XLM在Cross-Valid下取得最优宏F1 0.896；相较直接迁移的CheckThat!书面语基线（F1 0.32）提升超过170%，但跨主题测试时仍出现约10%~20%的性能衰减，凸显领域适配必要性。
- **闭源模型对比**：Grok2/o4-mini/o3-mini经prompt测试F1均低于0.78，显著弱于
