---
title: "Scaling-Rich-Style-Prompted-Text-to-Speech-Datasets"
source: https://aclanthology.org/2025.emnlp-main.180.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:13:23"
field: "风格可控语音合成"
keywords: ["text-to-speech", "style-prompted TTS", "speech dataset", "style annotation", "large-scale speech", "classifier-free guidance"]
innovations: ["基于感知说话人相似度（VoxSim）的 intrinsic rich tag 自动传播方法", "三级联（表达性过滤→语义匹配→音频LLM声学匹配）的 situational rich tag 自动标注 pipeline"]
benchmarks: ["Consistency MOS", "Naturalness MOS", "Tag Recall", "Intelligibility MOS", "WER"]
---

# 论文速读：Scaling-Rich-Style-Prompted-Text-to-Speech-Datasets

## 一句话总结
本文提出了 **ParaSpeechCaps**，首个大规模支持丰富风格标签（59个标签，涵盖说话人级 intrinsic 和话语级 situational）的语音数据集，通过两种创新性的自动标注方法将数据规模扩展至 2427 小时；基于该数据集微调 Parler-TTS 后，在风格一致性和语音自然度上分别取得 +7.9% 和 +15.5% 的提升。

## 研究问题与动机
1. **现有风格标注数据集规模不足**：已有 rich style tag 数据集（如 Expresso、EARS）规模仅 47–60 小时，无法满足 TTS 模型的大规模训练需求。
2. **现有大规模数据集仅覆盖基础标签**：如 Parler-TTS、PromptSpeech 等虽使用万小时级数据，但仅支持 gender、pitch、speed 等可通过信号处理工具直接提取的 basic tags，无法控制 guttural、nasal、pained 等抽象 rich tags。
3. **Intrinsic（说话人级）标签难以自动扩展**：intrinsic tags 需逐说话人标注，且存在主观性，已有工作缺乏将少数人工标注自动传播至大规模无标注数据的可靠方案。
4. **Situational（话语级）标签自动标注困难**：一方面互联网规模语音数据中表达性speech占比较低；另一方面，现有自动分类器（如 emotion2vec）仅覆盖 8 种情绪，无法支持本文所需的 26 种丰富 situational tags。

## 核心贡献（创新点）
1. **提出 ParaSpeechCaps 数据集**：首个同时覆盖 rich intrinsic（33 tag）和 rich situational（26 tag）的大规模开源 TTS 风格标注数据集，总规模 2.9k 小时、45k 说话人，填补了大型多风格数据集的空白。
2. **Intrinsic 标签的感知说话人相似度传播方法**：利用 VoxSim 感知说话人嵌入模型，将已标注 VoxCeleb 说话人的 intrinsic tags 传播至 Emilia 语料中感知相似的说话人（阈值 cosine ≥ 0.8），实现 9× 至 2427 小时的数据扩展。
3. **Situational 标签的三段式级联自动标注方法**：结合 DVA 分类器的 Expressivity Filtering、sentence embedding 的 Semantic Matching、以及 Gemini 1.5 Flash 的 Acoustic Matching，解决"表达性样本稀缺"与"现有分类器标签覆盖有限"两大难题。
4. **验证自动标注数据质量与 Parler-TTS 微调效果**：人工评测显示自动标注数据的质量与人工标注数据相当；基于 PSC-Scaled 微调的 Parler-TTS 在 Consistency MOS 和 Naturalness MOS 上较最佳 baseline 分别提升 7.9% 和 15.5%。
5. **详细的消融实验与设计选择分析**：对 intrinsic 传播使用的嵌入模型（VoxSim vs. WavLM Large）及 situational 三级联步骤逐一消融，证明各设计决策的必要性。

## 方法详解

### 数据集构建流程
整体流程分为两步：(1) 构建小规模人工标注基集 PSC-Base（282 小时）；(2) 通过自动标注方法将其扩展为 PSC-Scaled（2427 小时）。所有 style tags 经 Mistral-7B-Instruct-v0.2 转换为自然语言 style prompt。

#### 3.1 PSC-Base（人工标注部分）
- **数据来源**：Expresso（47h）、EARS（60h）及 VoxCeleb 子集（594 位名人）。
- **Intrinsic 标注**：通过 Amazon Mechanical Turk 众包，为每位说话人采集 5 份标注，保留至少 2/3 位 annotator 一致认可的 tags（dev/train vs. holdout 标准不同）。通过 GPT-4 补充 VoxCeleb 名人的 accent 信息。
- **Situational 标注**：复用 Expresso 和 EARS 的已有标注，并 remap 至本文 tag 词汇表。
- **Holdout 集设计**：从 VoxCeleb 中选取 64 位说话人确保每个 rich intrinsic tag 均有 2 男 2 女，用于后续模型评估。

#### 3.2 PSC-Scaled（自动标注部分）
**Intrinsic 标签扩展（基于 VoxSim 传播）：**
- 对每位 VoxCeleb 已标注说话人及每位 Emilia 说话人，抽取 10 个随机 utterance 计算 median perceptual embedding（使用 VoxSim）。
- 设定 cosine similarity 阈值 0.8（对应 VoxSim 5/6 分），将 VoxCeleb 说话人的 intrinsic tags 传播至匹配的 Emilia 说话人，排除 clarity 相关 tags（因 VoxSim 对该类 tag 区分度低）。

**Situational 标签扩展（三级联流程）：**
1. **Expressivity Filtering**：使用 Wagner et al. (2023) 的 DVA（Dominance-Valence-Arousal）分类器，保留任一维度值 < 0.35 或 > 0.75 的 utterance；进一步按 emotion-specific A/D/V 方向阈值进行二次过滤（如 angry 需 high dominance/arousal、low valence）。
2. **Semantic Matching**：使用 SFR-Embedding-Mistral 对 transcript 和查询语句进行 embedding，按 cosine similarity 排序；过滤掉含 emotion-specific keyword（如 transcript 含 "angry" 但语义不匹配）的样本。
3. **Acoustic Matching**：对 top 100k 候选样本，使用 Gemini 1.5 Flash 进行音频 LLM 评分，要求仅关注 tone 而非 content，保留 Likert 5 分样本。

#### 3.3 Basic Tags 自动提取
- Gender：使用数据集元数据 + GPT-4 + 分类器获取。
- Pitch/Speed/Noise：分别使用 PENN、g2p（phonemes/second）、Brouhaha（SNR）计算后按阈值分桶，使用 Dataspeech 库实现。

## 实验与结果

### 数据集统计
- **PSC-Base**：282 小时，覆盖 28 rich intrinsic + 23 rich situational + 8 basic tags。
- **PSC-Scaled**：2427 小时（intrinsic 2518h + situational 298h），总计 2.9k 小时、45k 说话人。
- **标签分布**：intrinsic tags 覆盖 28 个 rich 标签（pitch、texture、clarity、volume、rhythm、accent 六大类）；situational tags 覆盖 26 个 rich 标签（18 种 emotion + 8 种 expressiveness）。

### 自动标注质量验证（人工评测）
| 数据集 | Intrinsic Tag Recall ↑ | Situational Tag Recall ↑ |
|---|---|---|
| PSC-Base | 48.7% | 68.1% |
| PSC-Scaled | 50.3% | 71.3% |
| Std. Embedder（ablation） | 45.3% | — |
| w/o Expressivity（ablation） | — | 61.0% |
| w/o Semantic（ablation） | — | 66.1% |
| w/o Acoustic（ablation） | — | 63.3% |

**结论**：PSC-Scaled 在两类标签上的 recall 均与 PSC-Base 相当；三种 ablation 均导致性能下降，证明三级联步骤缺一不可；使用标准 WavLM Large 嵌入（Std. Embedder）效果劣于 VoxSim，印证感知说话人相似度模型的重要性。

### TTS 模型实验（Parler-TTS 微调）
主要评测集：246 个 tag-balanced 样本（holdout split）+ 240 个组合 prompt 样本。

| 模型 | CMOS ↑ | Intr TR ↑ | Sit TR ↑ | NMOS ↑ | IMOS ↑ | WER ↓ |
|---|---|---|---|---|---|---|
| Ground Truth | 4.42 | 88.7% | 88.6% | 4.36 | 4.28 | 7.93 |
| Parler-TTS（base） | 3.05 | 33.0% | 21.2% | 2.85 | 4.31 | 4.62 |
| +LTTSR（仅basic tags） | 3.07 | 33.7% | 22.4% | 2.95 | 4.44 | 4.47 |
| +LTTSP,Exp,EARS | 3.55 | 40.7% | 69.7% | 3.10 | 4.19 | 7.14 |
| **Base（Ours）** | 3.75 | 63.6% | 68.1% | 3.27 | 4.05 | 9.14 |
| **Scaled（Ours）** | **3.83** | **69.5%** | **75.4%** | **3.58** | 4.07 | 8.63 |

**核心结论**：
- 相对于最强 baseline（+LTTSP,Exp,EARS），Base 模型 Consistency MOS 提升 **+5.6%**、Naturalness MOS 提升 **+5.5%**；Scaled 模型进一步提升至 **+7.9%**（CMOS）和 **+15.5%**（NMOS）。
- Intrinsic Tag Recall 从 40.7% 大幅提升至 69.5%，证明本文人工标注的 intrinsic 数据质量优于 LibriTTS-P。
- 组合风格 prompt 评估中，Scaled 模型同时生成两类标签的比例最高，且在部分成功时更倾向于保留 intrinsic tag。
- 推理阶段引入 classifier-free guidance（CFG）可进一步提升所有模型的 Consistency MOS。
- Intelligibility MOS 下降的原因：模型更忠实于 slurred、stammering、非美式口音等天然难懂的风格标签，属于"忠实风格呈现"而非质量退化。

## 相关工作脉络
1. **Parler-TTS (Lacombe et al., 2024)**：开源的 style-prompted TTS 基线，使用 45k 小时 Librispeech + basic tags（gender/pitch/speed/noise）。本文在其基础上引入 rich tags 并大幅扩展数据规模，定位差异在于从 basic 标签升级为 rich tags。
2. **LibriTTS-P (Kawamura et al., 2024)**：对 LibriTTS-R 添加 rich intrinsic tags，但规模仅 0.6k 小时且无 situational tags。本文使用其作为对比 baseline（+LTTSP,Exp,EARS）。
3. **Expresso (Nguyen et al., 2023) / EARS (Richter et al., 2024)**：小规模的 rich situational 标注数据集（47h / 60h），本文直接复用其标注并进行 remap，同时贡献了新的 VoxCeleb intrinsic 标注。
4. **SpeechCraft (Jin et al., 2024)**：使用 emotion2vec 等分类器自动标注 8 种情绪，扩展至 2.4k 小时，但仅覆盖 basic 情绪标签。本文在三段式 pipeline 中融合语义+声学信号，覆盖 26 种丰富 tags。
5. **PromptTTS / PromptTTS2 (Guo et al., 2022; Leng et al., 2023)**：使用合成情感数据或改进架构，依赖商业 TTS 系统生成数据，受限于合成语音的自然度和说话人覆盖。本文全部基于真实语音数据进行标注和训练。
6. **AudioBox (Vyas et al., 2023)**：结合自动扩展 basic tags 与人工标注 rich tags 数据集，但未系统性解决 rich intrinsic tags 的自动传播问题，也未达到本文的数据规模。

## 局限性与未来方向
1. **语言覆盖限制**：当前实验仅限英语，未来可扩展至多语言场景，包括语音语言本身和风格 prompt 语言两个维度。
2. **数据集偏差**：rich tags 之间存在隐含相关性（如 shrill 与 female 正相关、guttural 与 male 正相关），可能固化人口统计学偏见，限制风格与身份的多样化组合；缓解偏见是重要的未来方向。
3. **缺乏自动评测指标**：风格一致性和语音质量评估仍依赖昂贵的主观 MOS 评测，阻碍快速实验迭代和大规模评估，未来需探索自动风格评测指标。

## 研究启发与可借鉴点
1. **感知说话人相似度用于标签传播**：VoxSim 嵌入相比标准说话人验证嵌入（WavLM Large）更适用于 rich intrinsic tag 的跨说话人传播，说明"主观感知相似性"与"声学特征相似性"的区别在风格标注任务中具有关键价值，可迁移至其他语音属性迁移场景。
2. **三级联（Expressivity → Semantic → Acoustic）处理丰富标签的策略**：先通过 DVA 筛选高表达性样本解决"数据不足"问题，再用文本语义匹配扩大覆盖范围，最后用音频 LLM 做精确过滤解决"分类器标签覆盖不足"问题——这一通用 pipeline 设计可迁移至其他需要丰富标注的大规模语音数据集构建任务。
3. **Inference-time Classifier-Free Guidance 无需训练即可提升风格一致性**：论文发现即使训练时未使用 dropout-based CFG，推理时加入 CFG 仍能稳定提升风格一致性，提示未来 work 可进一步探索不依赖额外训练开销的风格控制增强方法。
4. **数据集设计的 tag-balanced holdout 评估策略**：通过确保每个 rich tag 都有均衡的男女说话人代表，构建 tag-balanced 评估集，避免了数据集分布偏差对评测结果的影响，值得在风格可控 TTS 研究中借鉴。

## 关键术语表
- **ParaSpeechCaps**：本文提出的大规模语音风格标注数据集，包含 59 个 rich style tags，总计 2.9k 小时（282h 人工 + 2427h 自动）。
- **Intrinsic tags**：与说话人身份绑定的说话风格标签（如 shrill、guttural、British accent），跨话语持久存在，需逐说话人标注。
- **Situational tags**：话语级风格标签（如 angry、whispered、enthusiastic），针对单条 utterance 标注，反映说话情境。
- **Rich tags vs. Basic tags**：Rich tags 是主观抽象描述（需人工标注），Basic tags 是可通过信号处理直接提取的客观标签（如 gender、pitch level、speed level）。
- **VoxSim**：Perceptual speaker similarity 模型，衡量人类感知层面的说话人相似性，本文用于 intrinsic tag 传播。
- **DVA（Dominance-Valence-Arousal）**：情绪的三维表征空间，本文用 DVA 分类器筛选高表达性语音片段。
- **SFR-Embedding-Mistral**：基于 Mistral 的 sentence embedding 模型，用于 transcript 与 target tag 的语义匹配检索。
- **Consistency MOS / Naturalness MOS**：人工评测风格一致性与语音自然度的 Mean Opinion Score（5分制 Likert scale）。

## 可复现要素
- **数据集**：ParaSpeechCaps 已开源（https://github.com/ajd12342/paraspeechcaps）。
- **代码/权重**：代码和模型权重已开源发布。
- **模型架构**：基于 Parler-TTS（880M 参数，Flan-T5-Large 文本编码器 + DAC decoder LM）。
- **关键超参**：Base 模型 4× NVIDIA A40，140k 步，peak LR 8×10⁻⁵；Scaled 模型 4× NVIDIA H100，2×420k 步，LR 8×10⁻⁵ → 4×10⁻⁵；batch size=32，weight decay=0.01；推理使用 temperature=1.0，repetition penalty=1.0，CFG scale=1.5，最大 2580 tokens。
- **基础语料**：Emilia（45k hr English）、VoxCeleb1+2、Expresso、EARS、LibriTTS-R/P。
- **辅助模型**：VoxSim、SFR-Embedding-Mistral、Gemini 1.5 Flash、Mistral-7B-Instruct-v0.2、PENN、g2p、Brouhaha、Dataspeech。
