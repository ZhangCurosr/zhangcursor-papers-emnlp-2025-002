---
title: "SEA-Supervised-Embedding-Alignment-for-Token-Level-Visual-Te"
source: https://aclanthology.org/2025.emnlp-main.55.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:05:24"
field: "多模态表示学习"
keywords: ["多模态大语言模型", "视觉-语言对齐", "token级对齐", "对比学习", "adapter优化", "小模型"]
innovations: ["提出token级监督对齐方法，利用预训练VLM语义标签引导adapter将视觉token映射到LLM嵌入空间", "设计相似度加权采样与局部窗口采样策略，平衡语义连续性与对比学习有效性", "无需额外训练数据或推理开销，显著提升小模型跨模态理解能力"]
benchmarks: ["VQAv2", "TextVQA", "GQA", "ScienceQA-IMG", "MMBench", "POPE", "VizWiz", "MM-Vet", "CapsBench", "OCRBench", "MMMU"]
---

# 论文速读：SEA-Supervised-Embedding-Alignment-for-Token-Level-Visual-Te

## 一句话总结
SEA提出了一种token级别的监督对齐方法（Supervised Embedding Alignment），通过在预训练阶段利用CLIP等预训练视觉-语言模型的语义标签，通过对比学习将视觉token精确映射到LLM嵌入空间，有效缓解了现有MLLM中图像级对齐不足导致的语义失真和模态表示差距问题，且无额外训练数据或推理开销。

## 研究问题与动机
1. **语义失真问题**：传统adapter将视觉patch映射到LLM嵌入空间后，视觉token的语义信息严重失真（如"蓝莓"被错误映射为"bluejeans"），迫使语言模型消耗额外容量补偿表征偏差。
2. **模态表示差距**：adapter处理后的视觉token与LLM原生输入嵌入空间存在显著距离（t-SNE可视化可见明显分离），模型难以有效利用视觉输入进行推理。
3. **小模型问题更严重**：较小语言模型（如Gemma-2B）容量有限，视觉感知与语言性能之间的权衡更为严峻，传统对齐方法对其提升有限。
4. **现有方法粒度不足**：当前对齐多依赖图像级或区域级监督（如optimal transport、回归技术），无法捕捉fine-grained语义信息以实现最优视觉-语言整合。

## 核心贡献（创新点）
1. **首次提出token级别预训练对齐范式**：通过在预训练阶段为每个视觉patch生成连续语义标签并进行对比学习，精确引导adapter将视觉token映射到LLM嵌入空间；与现有方法本质区别在于从粗粒度图像级监督转向细粒度token级监督。
2. **相似度加权采样+局部采样策略**：设计基于相似度的多标签加权采样保持视觉token的连续语义表示，并采用k×k局部窗口采样增强对比学习有效性；与已有工作本质区别在于解决了视觉token语义多样性与对比学习样本多样性之间的平衡问题。
3. **显著改善小模型性能且无额外开销**：SEA在不增加训练数据和推理成本的前提下，使Gemma-2B平均性能提升7.61%，同时保持语言能力不下降；与已有方法本质区别在于仅修改adapter训练目标而无需改变模型架构。
4. **揭示adapter对齐质量与语言能力的关联**：系统分析表明改善对齐可同时提升视觉感知和保留语言能力（缓解instruction-tuning期间的语言退化）；这是首次将adapter对齐质量问题与语言性能退化建立量化关联。

## 方法详解
**整体框架**：SEA在预训练阶段对adapter引入token级对比学习目标，与自回归语言建模损失联合优化。

**语义标签提取**：
- 使用预训练视觉-语言模型（如CLIP）的vision encoder f和text encoder h
- 构建约400万词的词汇表W，提取每个视觉patch v_i与词汇表中所有词的余弦相似度
- 选取相似度最高且大于0的top-n个词作为该patch的语义标签（论文中n=10）

**Token-Level对齐损失**：
- **相似度加权采样**：将相似度分数归一化为采样概率 $S_{norm}^i = S_i / sum(S_i)$，按概率采样标签
- **局部采样策略**：在k×k窗口内采样，确保同一batch内共享标签的patch只保留一个
- 对于多词标签 $w_i$，计算其在LLM嵌入空间的特征：$t_i = \frac{1}{M}\sum_{k=1}^{M}\Psi(w_i^k)$
- 对比学习损失：$\mathcal{L}_a = -\frac{1}{2N}\sum_{i=1}^{N}(\log\frac{\exp(\phi(x_{vi}, t_i)/\tau)}{\sum_{j=1}^{N}\exp(\phi(x_{vi}, t_j)/\tau)} + \log\frac{\exp(\phi(t_i, x_{vi})/\tau)}{\sum_{j=1}^{N}\exp(\phi(t_i, x_{vj})/\tau)})$，其中$\phi$为归一化余弦相似度，$\tau$为可学习温度参数

**总损失函数**：$\mathcal{L} = \mathcal{L}_g + \lambda\mathcal{L}_a$，其中$\mathcal{L}_g$为标准自回归语言建模损失，$\lambda$为权重因子平衡两者

## 实验与结果
**数据集与设置**：
- 预训练数据：Cambrian-1的2.5M caption对；消融实验使用LLaVA-Pretrain的CC-595K
- 指令微调：Cambrian-7M或LLaVA-Instruct混合数据集（656K）
- 视觉编码器：CLIP-ViT-L@336px、SigLIP-ViT-SO@384px、DINOv2-L@224px
- 语言模型：Gemma-2B、Phi-3-mini-4k、Llama3-8B、Vicuna-7B/13B

**主要结果**（SEA-PRIME，Table 1）：
- **Gemma-2B**：VQAv2=81.0，VQAT=60.7，MMB=68.8，POPE=87.8，VizWiz=61.9，MM-Vet=38.0
- **Phi3-3.8B**：VQAv2=80.7，MMB=72.6，VizWiz=61.9，MM-Vet=46.8
- **Llama3-8B**：VQAv2=83.1（最高），SQA-I=79.0，MMB=76.0，VizWiz=64.7，MM-Vet=46.0
- **Vicuna-13B**：SQA-I=80.9（最高），MM-Vet=48.8

**消融结果**（Table 2，SEA-LlaVA vs LlaVA）：
- LlaVA→SEA-LlaVA：VQAv2从78.8→79.1，VQAT从58.3→58.9，SQA-I从67.9→69.4，VizWiz从45.7→48.8
- Gemma-2B+SEA：VQAv2从72.5→76.6（+4.1），MMB从54.0→59.5（+5.5）
- **小模型平均提升7.61%**

**细粒度感知任务**（Table 3）：SEA在CapsBench（90.4 vs 88.0）、COCO Captions CIDEr（88.7 vs 84.8）、OCRBench（336 vs 319）上均有提升

**跨编码器迁移**（Table 5）：将CLIP提取的语义标签用于DINOv2训练，VQAv2从71.4提升至74.0

## 相关工作脉络
1. **LLaVA（Liu et al., 2023）**：SEA的直接基线，采用图像级自回归损失训练adapter；SEA在其基础上引入token级对比学习，显著提升对齐质量。
2. **AlignGPT（Zhao et al., 2024）**：通过相似性token分配改善跨模态对齐；SEA与之本质区别在于直接从语义标签出发而非通过分配预测。
3. **Rethinking MLLMs（Lin et al., 2024）**：引入segmentation/OCR增强视觉token；SEA聚焦于adapter本身的语义对齐而非token构造方式。
4. **Mini-Gemini（Li et al., 2024）**：面向小模型的视觉语言模型；SEA与其正交，SEA专注于对齐方法改进，可结合Mini-Gemini的分辨率策略。
5. **CLIP/SigLIP视觉-语言预训练**：SEA利用这些模型提取语义标签，但不重复其图像-文本级对比学习，而是将标签迁移到LLM嵌入空间进行token级对齐。
6. **Cambrian-1（Tong et al., 2024）**：提供大规模训练数据；SEA在其数据基础上改进adapter训练目标。

## 局限性与未来方向
1. **静态标签选择**：当前使用固定数量的top-n标签，未来可探索根据视觉内容复杂度动态调整标签数量的策略。
2. **安全性对齐未探索**：语义对齐与安全对齐之间的关系有待研究，可能影响实际应用。
3. **仅限图像模态**：目前方法仅适用于静态图像，扩展到视频、音频等其他模态是重要方向。
4. **词汇表构建依赖**：约400万词词汇表通过语料库分析提取，可能遗漏某些领域特定词汇。

## 研究启发与可借鉴点
1. **语义标签提取策略可复用**：利用预训练VLM（如CLIP）提取token级语义标签的思路，可迁移到其他需要细粒度对齐的多模态任务（如文档理解、医学影像分析）。
2. **局部采样策略适用于大规模对比学习**：k×k窗口采样+重复标签去重的设计，解决了对比学习中样本冗余问题，可推广至其他视觉-语言对齐任务。
3. **对齐质量与语言能力的关联分析值得借鉴**：论文引入TACS指标量化对齐质量，并与POPE等下游性能建立关联，为后续研究提供了可复用的评估框架。
4. **小模型对齐优化的启示**：SEA对小模型提升显著（7.61%）的发现，提示后续工作应重点关注轻量级MLLM的对齐策略设计，这对边缘部署场景具有重要价值。
5. **跨编码器标签迁移策略**：将CLIP语义标签直接用于DINOv2训练并获得性能提升，表明语义监督与特定编码器解耦，这一思路可扩展到其他领域自适应场景。

## 关键术语表
**SEA（Supervised Embedding Alignment）**：本文提出的token级监督对齐方法，通过对比学习引导视觉token精确映射到LLM嵌入空间。
**TACS（Token Alignment Consistency Score）**：本文提出的对齐质量评估指标，衡量视觉token与文本token的余弦相似度。
**Adapter**：连接视觉编码器和语言模型的模块化组件，负责将视觉特征转换到LLM嵌入空间。
**Semantic Label**：通过CLIP等预训练VLM为每个视觉patch提取的语义标签，用作token级对齐的监督信号。
**Similarity-Weighted Sampling**：基于视觉patch与语义标签相似度概率分布的采样策略，保持语义连续性。
**Localized Sampling**：在k×k窗口内采样的策略，确保对比学习样本的空间多样性。
**MLLM（Multimodal Large Language Model）**：融合视觉和语言的大规模多模态模型，如LLaVA、Qwen-VL等。
**Modality Representation Gap**：视觉token嵌入空间与LLM原生文本嵌入空间之间的表征差距。

## 可复现要素
- **数据集**：Cambrian-1（2.5M caption对）、Cambrian-7M（指令微调）；消融实验使用LLaVA-Pretrain（CC-595K）和LLaVA-Instruct混合数据集（656K）；论文未明确说明是否公开，但Cambrian数据集为开源
- **代码**：已开源，地址为https://github.com/YuanyangYin/SEA
- **权重**：论文未明确说明权重是否公开，提到SEA-PRIME系列模型
- **关键超参**：top-n=10（语义标签数量），τ=0（温度参数），k=2（局部采样窗口2×2），λ（对齐损失权重，论文未明确给出具体值）
- **训练配置**：AdamW优化器，cosine学习率调度，1 epoch，8×H800 GPU；Stage 1仅优化adapter，Stage 2优化LLM和adapter（SEA-PRIME额外fine-tune vision encoder，lr=2e-6）
