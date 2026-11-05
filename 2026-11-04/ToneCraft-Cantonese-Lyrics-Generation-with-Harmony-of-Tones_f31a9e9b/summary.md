---
title: "ToneCraft-Cantonese-Lyrics-Generation-with-Harmony-of-Tones"
source: https://aclanthology.org/2025.emnlp-main.18.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:07:49"
field: "音乐信息检索与歌词生成"
keywords: ["Cantonese lyrics generation", "tone-melody harmony", "constrained text generation", "large language model", "Harmony-Aware Loss", "character-level control"]
innovations: ["CTP-Mapping: 将0243音乐理论转化为字符-声调-音高映射表，消除对齐数据依赖", "Harmony-Aware Loss: 融合交叉熵与和声正则项的复合损失，显式优化声调-旋律对齐", "Polish Process: 局部不和谐检测与masked prediction重排序的后处理策略"]
benchmarks: ["Harmony", "Consistency", "Harmonicity", "Listenability", "Ambiguity"]
---

# 论文速读：ToneCraft-Cantonese-Lyrics-Generation-with-Harmony-of-Tones

## 一句话总结
本文提出 **ToneCraft** 框架，利用大语言模型生成符合粤语声调与旋律和谐的歌词，通过字符级控制、符号化旋律编码与自研的 Harmony-Aware Loss（HAL），在无需对齐数据的情况下实现了歌词声调与旋律轮廓的对齐，客观与主观评估均显著优于现有基线。

## 研究问题与动机
1. **粤语歌词生成中的"和声"（Harmony）难题**：粤语有九种音调轮廓与六个声调，歌词必须与旋律的音高变化相匹配，否则会导致听感歧义（如"花鞋"误听为贬义词）。现有工作主要关注音节节奏对齐，忽视了"音高-声调"对齐这一粤语特有的核心约束。
2. **候选空间受压缩小多样性**：和声规则大幅减少每个位置的可选字符，如何在强约束下保持歌词多样性与语义质量是核心挑战。
3. **对齐数据稀缺**：粤语歌曲缺少大量旋律-歌词对齐标注数据，传统 Seq2Seq 方法难以直接应用，需探索无对齐数据的解决方案。

## 核心贡献（创新点）
1. **ToneCraft 框架**：提出首个基于 LLM 的粤语和声歌词生成系统，支持字符级 melody-aligned 生成，可推广至其他声调语言。
2. **Harmony-Aware Control（HAC）算法**：整合字符级控制、符号化旋律引导与任务特定损失（HAL），在保持生成灵活性的同时实现声调-旋律对齐；首次将音乐理论中的 0243 方法转化为可学习的映射表（CTP-Mapping），消除对齐数据依赖。
3. **抛光策略（Polish Process）**：设计局部不和谐检测与重排序机制，通过 masked prediction + 联合评分（和声一致性 + 上下文连贯性）提升输出质量，兼顾歌唱流畅度与语义清晰度。

## 方法详解
### 1. 音高映射（Pitch Mapping）
- **A2R-Mapping**：基于四分位数将绝对 MIDI 音高分段为 4 个相对音级（$P_1$ 低 → $P_4$ 高），按句级别计算。
- **CTP-Mapping**：逆向工程 0243 方法（"do-re-mi-fa"对应 T4→0, T6→2, T3/T5→4, T1/T2→3），构建"字符-声调-相对音高"映射表，从纯歌词语料中自动提取音高信息，无需对齐标注。

### 2. 字符级控制（Character-level Control）
- **Tokenizer 修改**：跳过中文字符合并，确保每个 token 仅对应单个汉字。
- **Language Model Head**：对禁用 token（多字符 token）的 logits 置为 $-\infty$，强制单字生成。
- **专用音高 token**：引入 `<p0>`~`<p4>` 分别表示标点与 4 个相对音级。

### 3. Harmony 度量与 Harmony-Aware Loss（HAL）
- **Harmony 公式**：
  $$
  Harmony = \frac{1}{n} \sum_{i=1}^{n} e^{-|\hat{p}_i - p_i|}
  $$
  其中 $\hat{p}_i$ 为预测音高，$p_i$ 为标签音高，值越大表示对齐越好。
- **HAL 设计**：
  $$
  \mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i) + \ln\left(1 - \frac{1}{n}\sum_{i=1}^{n} e^{-|\hat{p}_i - p_i|}\right)
  $$
  第一项为标准交叉熵保持语言质量，第二项为和声正则项，推动预测音高向标签音高收敛。梯度分析证明该项能产生指向正确音高的校正力。

### 4. 混合训练（Mixed Training）
- 在 BELLE 基础指令数据上微调以恢复因字符级控制导致的中文理解能力下降。
- 引入 `fill_num` 变量区分任务类型：仅当输入包含 pitch tokens 时启用 HAL，否则使用普通交叉熵，避免训练不稳定。

### 5. 抛光策略（Polish Process）
- **不和谐检测**：预测音高与实际音高偏差 ≥2 级或轮廓冲突的位置。
- **Masked Prediction**：掩码整词，仅用上下文预测候选替换。
- **重排序**：
  $$
  S(c) = \alpha \cdot S_{harm}(c) + (1-\alpha) \cdot S_{cont}(c)
  $$
  其中 $S_{harm}$ 评估局部旋律趋势一致性，$S_{cont}$ 为 Wenzhong-GPT2-110M 的 log-likelihood 衡量上下文连贯性。

## 实验与结果
### 数据集
- **粤语歌词**：飞水粤语歌词网站（6000+ 首歌曲）
- **普通话歌词**：中文流行歌曲（扩展多样性）
- **BELLE**：基础指令微调数据
- **粤语字典**：Cantonese Standard Pronunciation Lexicon（获取准确声调标注）

### 评估指标
- **客观**：Harmony、Consistency（Spearman 秩相关）、Avg Sim、Min Sim、MaD1/2、MiD1/2
- **主观**（10 名粤语母语者，1-5 分）：Harmonicity、Listenability、Ambiguity、Diversity、Fluency、Coherence、Poeticness

### 主要结果
| 模型 | Harmony↑ | Consistency↑ | Harmonicity↑ | Listenability↑ | Ambiguity↓ |
|------|----------|--------------|--------------|----------------|------------|
| Llama3 (Finetuned) | 0.479 | 0.061 | 2.45 | 2.41 | 2.66 |
| SmBART-3 | 0.862 | 0.059 | 3.36 | 2.90 | 2.08 |
| **ToneCraft(Llama3)** | **0.973** | **0.953** | — | — | — |
| **ToneCraft(Qwen2, Polished)** | **0.929** | **0.858** | **4.11** | **3.86** | **1.26** |
| Human | 0.671 | 0.791 | — | — | — |

- **ToneCraft** 在 Harmony 与 Consistency 上远超所有基线，主观评估中 Harmonicity（4.11）、Listenability（3.86）、Ambiguity（1.26，越低越好）均为最优。
- 多样性和文本质量（Fluency、Coherence）也达到最高水平，抛光后进一步提升可听性与降低歧义。
- 消融实验表明：字符级控制 + HAL + 专用音高 token 三者协同效果最佳，其中 Tokenizer/LM Head 修改对 Qwen2 提升最显著。

## 相关工作脉络
1. **SongNet / Charpoet / ChipSong**：传统约束歌词生成方法，主要关注格式、音节数与词级长度控制，未涉及音高-声调对齐，不适用于粤语和声需求。
2. **DeepRapper / SongRewriter**：推理阶段调整词概率提升多样性，但约束复杂度远低于本文的和声规则。
3. **SmBART-3 / Chen & Teufel (2024)**：基于多语言的韵律敏感歌词生成，但采用序列到序列架构，依赖对齐数据，且未考虑粤语六声系统。
4. **Unilg / Songmass**：使用模板或预训练语言模型结合韵律约束，但仍以音节级对齐为主，缺少字符级和声建模。
5. **IComposer (Lee et al., 2019)**：中文歌曲创作系统，提供真人对比基准，但侧重 Mandarin 而非粤语，且使用对齐数据训练。
6. **TTS 领域（FastSpeech/VITS）**：本文讨论 melody-lyric alignment 与 TTS phoneme-to-spectrogram alignment 的结构相似性，提出可借鉴 duration predictor 进行一对一建模。

## 局限性与未来方向
1. **简化映射假设**：当前采用音符与汉字的一一对应，实际歌曲中一个音节可能跨越多个音符（多对一）或一个音符覆盖多个音节（一对多）。
2. **忽略时值（Duration）信息**：未考虑音符时长对歌词生成的影响，无法建模节奏层面的对齐。
3. **未处理入声字（checked tone）**：粤语九声中的入声（-p, -t, -k 韵尾）具有短促特征，当前模型仅使用六声抽象，可能影响细节质量。
4. **未来方向**：引入音符时值信息、结合音素特征（如入声标记）进行节奏感知生成、扩展到 Mandarin 及其他声调语言（如闽南语、吴语）。

## 研究启发与可借鉴点
1. **领域知识驱动的数据增强**：CTP-Mapping 将音乐理论（0243 方法）转化为可学习的映射表，展示了如何将专家经验嵌入深度学习 pipeline 以克服数据稀缺问题，该方法论可迁移至其他需要领域先验的任务。
2. **自定义 Loss 与 LLM 微调的结合**：HAL 的设计巧妙地在标准交叉熵之外叠加了任务特定的正则项，且通过数学分析证明其梯度方向正确性，为"神经符号融合"提供了可复用的范式。
3. **字符级控制的实现技巧**：通过修改 tokenizer merges + LM head logits masking 实现强制单字生成，而非完全 redesign tokenizer，是一种轻量且有效的策略。
4. **双阶段生成（生成 + 抛光）**：Polish 流程中的 masked prediction + 重排序策略可在不重新训练模型的前提下提升局部质量，适用于对局部一致性要求高的文本生成任务。
5. **混合训练稳定性技巧**：使用 `fill_num` 变量动态区分任务类型并选择性应用不同 loss，解决了多任务训练中 loss 冲突导致 NaN 的问题，具有工程参考价值。

## 关键术语表
- **和声（Harmony）**：粤语歌词声调轮廓与旋律音高轮廓之间的对齐程度，决定歌曲的可唱性与语义清晰度。
- **0243 方法**：粤语歌词创作的经典音高映射规则，将六个声调映射到四个相对音高（do/re/mi/fa），确保声调与旋律匹配。
- **CTP-Mapping（Char-Tone-Pitch Mapping）**：本文提出的字符-声调-相对音高映射表，通过逆向工程 0243 方法从纯歌词数据中提取音高信息。
- **A2R-Mapping（Absolute-to-Relative Mapping）**：基于四分位数的音高分段方法，将绝对 MIDI 值映射为 4 个相对音级。
- **Harmony-Aware Loss（HAL）**：本文提出的复合损失函数，融合交叉熵与和声正则项，显式优化歌词-旋律对齐。
- **Polish Process**：局部不和谐检测与重排序后处理流程，通过 masked prediction + 联合评分提升歌词质量和可唱性。
- **Five-Level Mark**：赵元任提出的五度标记法，用 [1-5] 数值描述声调轮廓，用于量化声调与旋律的对齐程度。
- **入声（Checked Tone）**：粤语九声中由 -p/-t/-k 韵尾触发的短促变体（第 7/8/9 声），与第 1/3/6 声共享音高轮廓但发音更短促。

## 可复现要素
- **数据集**：粤语歌词（飞水歌词网站）、普通话歌词、BELLE 指令数据集、粤语字典；代码与部分数据已开源（https://github.com/purepasserby/ToneCraft）
- **代码**：已开源
- **权重**：论文未明确提及开源权重，需查看 GitHub 仓库
- **关键超参**：LoRA rank=128，学习率 1e-4，batch size=1（梯度累积 8 步），训练 3 epochs，bfloat16 混合精度，单卡 RTX 3090（24GB）
