---
title: "Towards-Controllable-Speech-Synthesis-in-the-Era-of-Large-La"
source: https://aclanthology.org/2025.emnlp-main.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:08:09"
field: "可控语音合成与大模型"
keywords: ["可控语音合成", "Text-to-Speech", "大语言模型", "零样本TTS", "指令驱动生成", "语音tokenization"]
innovations: ["首篇可控TTS系统综述，三轴分类架构/策略/特征", "提出Gemini驱动的instruction-following/naturalness/expressiveness三维自动化评测", "揭示从标签到自由指令的四阶段演进并验证统一指令接口优势"]
benchmarks: ["Gemini-based controllability eval (Table 6)", "Zero-shot vs Instruction-based对比", "NISQA/UTMOS vs MLLM评估Pearson对比 (Table 7)"]
---

# 论文速读：Towards Controllable Speech Synthesis in the Era of Large Language Models: A Systematic Survey

## 一句话总结
本文是**首篇系统综述可控TTS**的论文，从模型架构、控制策略与特征表示三个维度梳理了从传统标签控制到LLM指令引导的演进脉络，并在Zero-shot与Instruction-based任务上评测了多个开源/商业系统，验证了Gemini驱动的自动化评估框架的有效性。

## 研究问题与动机
1. **可控性成为工业刚需**：影视配音、游戏、机器人、虚拟助手等场景要求对音色、情绪、语速、风格等细粒度属性进行精准控制，但此前缺乏系统性综述。
2. **既有综述存在视角盲区**：早期综述（Zen et al., 2009; Triantafyllopoulos et al., 2023）聚焦音质与声学建模，忽略了"可控性"这一核心维度，也未覆盖自然语言提示（PromptTTS、CosyVoice）等新兴范式。
3. **LLM时代的方法论突变**：VALL-E 系列、CosyVoice、VoxInstruct 等将TTS重构为条件语言建模任务，离散语音token + LLM解码器成为新主流，亟需统一分类框架。
4. **评估体系滞后于方法发展**：传统客观指标（MCD、WER、PESQ）难以捕捉"指令遵循度"与"表达力"，需要面向可控性的新评估范式。

## 核心贡献（创新点）
1. **首个可控TTS全景分类体系**：从模型架构（NAR/AR/LLM）、控制策略（Style Tagging / Reference Prompt / Natural Language Description / Instruction-Guided）、特征表示（Continuous vs Discrete）三轴组织近100篇工作，填补综述空白。
2. **揭示"从标签到自由指令"的演进规律**：梳理出控制范式从离散标签→语音参考→文本描述→自然语言指令的四阶段跃迁，并论证LLM集成是实现直觉化控制的必然路径。
3. **提出面向可控性的Gemini自动化评测管道**：以instruction following / naturalness / expressiveness三维对10个系统（8开源+2商用）进行评测，证明MLLM评估比NISQA/UTMOS更贴近人类偏好（Pearson 0.12–0.17 vs -0.10–-0.17）。
4. **系统性对比连续特征与离散token的Trade-off**：指出连续表示（MelS/LinS）保真度高但计算昂贵；离散token（EnCodec/SoundStream）兼容LLM训练且泛化强，但存在信息损失。
5. **公开维护动态论文库与更新站**：在https://github.com/imxtx/awesome-controllabe-speech-synthesis持续收录，为社区提供可追踪的资源入口。

## 方法详解
**1. 模型架构两分法**
- **非自回归（NAR）**：Transformer（FastSpeech/FastPitch）、VAE（CLONE/GenerSpeech）、Diffusion（NaturalSpeech 2/3、DEX-TTS）、Flow（Audiobox/P-Flow/E2 TTS）。核心公式：$\arg\max_\theta P(\mathbf{y}|\mathbf{x};\theta)$，并行生成，推理快。
- **自回归（AR）/LLM-based**：Decoder-only Transformer + 语音tokenizer（EnCodec/SoundStream/HuBERT）。VALL-E 开创两阶段管线（粗粒度token → 细粒度refinement），后续工作（Spark-TTS、CosyVoice、F5-TTS）在tokenizer与对齐上迭代。核心公式：$\arg\max_\theta \prod_{t=1}^T P(y_t|y_{<t},\mathbf{x};\theta)$。

**2. 控制策略四象限**
- **Style Tagging**：离散标签（angry/happy）或连续参数（guiding scale factor）直接调制latent/prosody。局限：表达多样性受限，只能覆盖预定义属性子集。
- **Reference Speech Prompt**：输入几秒参考语音，通过adaptive normalization / 多层style adapter 实现zero-shot voice cloning与风格迁移（MetaStyleSpeech、StyleTTS-ZS、MegaTTS 2）。
- **Natural Language Description**：用文本描述音色/情绪/环境（PromptTTS、InstructTTS、VoiceLDM）。引入variation network 建模prompt残差。
- **Instruction-Guided Control**：将TTS重构为统一instruction-to-speech任务，单条自然语言同时承载内容与风格（VoxInstruct、CosyVoice）。支持插入/删除/替换等free-form editing（VoiceCraft、InstructSpeech）。

**3. 特征表示与解耦**
- **连续表示**：MelS/LinS/Waveform，保留细粒度声学细节，适合GAN/VAE/Flow/Diffusion。
- **离散token**：经RVQ-GAN（EnCodec/SoundStream/HiFi-Codec）或SSCP（HuBERT/Wav2Vec 2.0）量化，便于LLM建模与跨模态对齐，但牺牲部分高保真度。
- **属性解耦**：对抗训练（辅助分类器）与信息瓶颈（低容量编码器分支）结合KL正则/量化，分离speaker/emotion/prosody/content。

**4. 评估设计**
- 客观：MCD（<4 优，>6 失真）、FDSD、WER、Cosine（ Speaker sim.）、PESQ。
- 主观：MOS/CMOS/AB-Test。
- 新型MLLM评测：Gemini按3维1-5分打分，Pearson相关在instruction following(0.12)、naturalness(0.17)、expressiveness(0.14)上均优于NISQA/UTMOS。

## 实验与结果
**数据集**：涵盖Tag-based（IEMOCAP/RECOLA/RAVDESS/CMU-MOSEI）、Description-based（PromptSpeech/Parler-TTS/SpeechCraft）、Dialogue（Taskmaster-1/MagicData-RAMC/DailyTalk）三类共16个公开资源（见Table 3）。

**主要评测结果（Table 6，Gemini 2.5 Flash 1-5分）**
- **Zero-shot基线**：Vevo自然度最高（4.43±0.55）、表达力最强（4.32±0.75）；Spark-TTS最弱（3.68/3.83）。
- **Instruction-based提升**：所有模型全维上涨；CosyVoice以4.81/4.92/4.78领跑三项；MiniMax TTS与EmoVoice均超4.6。
- **关键结论**：即使最弱的Instruction方法（VoxInstruct 4.45）仍优于最强Zero-shot基线，印证"统一指令接口"范式优势。
- **自动化评测可靠性**：Table 7显示Gemini管道与人类偏好的Pearson（0.12–0.17）显著高于NISQA/UTMOS（-0.10–0.01），说明任务定制评估更重要。

## 相关工作脉络
1. **VALL-E (Wang et al., 2023a)**：开创neural codec language model用于zero-shot TTS，本文将其定位为LLM-based TTS奠基作，后续VALL-E X/R/2、Spark-TTS、CosyVoice等均在tokenizer与对齐上迭代。
2. **PromptTTS (Guo et al., 2023)**：首次用文本prompt描述5类属性，开创description-based控制；本文将其归入"自然语言描述"象限，并与InstructTTS/VoxInstruct形成渐进关系。
3. **StyleTTS 2 (Li et al., 2023)**：利用大SLM蒸馏time-varying style diffusion，实现human-level零样本TTS；本文突出其在"风格解耦+连续特征"路线的代表性。
4. **NaturalSpeech 2/3 (Shen et al., 2024; Ju et al., 2024)**：latent diffusion + factorized codec，实现无需参考的零样本合成；本文将其置于Diffusion+NAR路线，并与VoiceLDM等环境控制工作对比。
5. **F5-TTS (Chen et al., 2024d)**：基于flow-matching的NAR TTS，直接用ConvNeXt v2增强text-speech对齐；本文归类为"Flow-based 零样本"新范式。
6. **CosyVoice (Du et al., 2024)**：监督语义token + flow-matching + LLM指令控制，本文实验证实其在三项可控性指标上领先，定位为本综述"当前SOTA代表"。

## 局限性与未来方向
**论文自述局限**
1. **可控属性交互建模不足**：现有工作多将emotion/speaker/prosody视为独立变量，未系统刻画其耦合关系。
2. **效率维度缺失**：instruction/description驱动方法依赖大型LLM codec与跨模态架构，计算开销高昂，本文未深入讨论推理优化。
3. **社会影响未覆盖**：deepfake生成、对抗攻击等伦理风险仅简要提及，未展开。
4. **边界研究未纳入**：speech enhancement/separation/pretraining/speech-to-speech translation等互补方向未讨论。

**未来方向（论文提出）**
1. **细粒度指令驱动合成与编辑**：开发支持多尺度精细控制的解耦表示，减少多次试错。
2. **多模态表达性合成**：整合图像/视频输入生成叙事性语音，应用于影视游戏。
3. **零样本长语音与对话合成**：维持情绪一致性，突破上下文感知与情感连贯瓶颈。
4. **大规模数据集生成**：利用预训练分析模型+ChatGPT自动生成多场景描述与编辑指令。

## 研究启发与可借鉴点
1. **可控TTS的"四象限分类法"可迁移**：架构×控制策略×特征表示的三维taxonomy为其他生成任务（视频/音乐）的特征解耦综述提供方法论模板。
2. **MLLM作为自动化评测代理的实验设计**：用Gemini按任务定制rubric打分，优于通用指标（NISQA/UTMOS）的思路可直接复用于其他可控生成基准构建。
3. **连续特征与离散token的trade-off权衡框架**：论文对MelS/LinS vs EnCodec/SoundStream的利弊剖析，可为团队在"保真度-效率-可控性"三者间的架构选型提供决策依据。
4. **instruction-to-speech的统一范式**：将内容与风格融合进单条自然语言prompt，简化工程接口，可借鉴到语音代理、多模态交互系统中。
5. **动态论文库维护机制**：开源维护可追踪的文献库（github + 定期更新），适合科研团队作为领域情报追踪的标准实践。

## 关键术语表
**Controllable TTS**：能对音色、情绪、语速、风格等属性进行细粒度干预的文本到语音合成技术。
**Zero-shot TTS**：无需目标说话人微调，仅凭几秒参考语音即可克隆音色并合成任意文本。
**Neural Codec Language Model**：将波形经RVQ-GAN量化为离散token，再由LLM进行条件语言建模的TTS范式（VALL-E系列）。
**Flow Matching**：流匹配生成技术，通过学习从噪声到数据的概率流实现一步/少步高效合成（F5-TTS/E2 TTS使用）。
**Style Tagging**：用离散标签（如angry/happy）或连续因子直接调制语音属性的早期控制方法。
**Instruction-Guided Synthesis**：将TTS重构为"自然语言指令→语音"的统一生成任务，内容+风格由单prompt指定（VoxInstruct/CosyVoice）。
**Attribute Disentanglement**：通过对抗训练/信息瓶颈将speaker/emotion/prosody/content等属性解耦到独立潜变量。
**FDSD**：Fréchet DeepSpeech Distance，基于预训练ASR模型特征空间的分布距离，评估合成语音感知质量。

## 可复现要素
- **数据集**：多数公开可获取（IEMOCAP/RAVDESS/Common Voice/AISHELL-3/GigaSpeech/SpeechCraft/Parler-TTS等，见Table 3），部分如PromptSpeech数据集规模未公开。
- **代码/权重**：论文维护了开源方法汇总（Appendix Table 5/Appendix A.3），主流tokenizer（EnCodec/SoundStream/HiFi-Codec）均有开源仓库；具体模型代码分散在各论文实现中。
- **关键超参**：论文未集中列出统一超参表，各方法超参差异较大（如Diffusion步数、Flow匹配学习率、RVQ层数等），需参看原论文。
- **评测环境**：Gemini评测使用Gemini 2.5 Flash，10个模型各合成20条（10英+10中），prompt如Appendix A.5给出。
