---
title: "Seeing-More-Saying-More-Lightweight-Language-Experts-are-Dyn"
source: https://aclanthology.org/2025.emnlp-main.28.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:06:34"
field: "多模态大模型效率优化"
keywords: ["video-language model", "token compression", "dynamic compression", "multimodal large language model", "video understanding", "language-aware compressor"]
innovations: ["利用轻量语言专家生成软caption token，通过描述长度自适应控制视觉token压缩比", "提出语义密度感知监督，以强LVLM精炼caption引导CapPruner学习内容自适应压缩", "三阶段渐进训练结合中间层hidden state，实现49% FLOPs降低的同时保持竞争力性能"]
benchmarks: ["MVBench", "Video-MME", "MSVD-QA", "MSRVTT-QA", "TGIF-QA", "ActivityNet-QA", "VSI-Bench", "LongVideoBench"]
---

# 论文速读：Seeing-More-Saying-More-Lightweight-Language-Experts-are-Dynamic-Video-Token-Compressors

## 一句话总结
论文提出 LangDC（Language-aware Dynamic Token Compressor），一种基于轻量级语言专家 CapPruner 的动态视频 token 压缩器，通过语言描述的长度自适应地调节视觉 token 压缩比，在将 FLOPs 降低 49% 的同时保持与强基线 VideoGPT+ 竞争力的视频理解性能。

## 研究问题与动机
1. **LVLM 计算瓶颈**：大视频语言模型通过双二次复杂度处理海量视觉 token，严重阻碍实际部署，亟需高效的 token 压缩策略。
2. **固定压缩比的缺陷**：现有方法（Q-Former、Resampler、C-Abstractor、LDP 等）采用固定压缩比，忽略视频片段间语义密度的显著差异，导致信息丰富片段表征不足、静态片段计算浪费。
3. **Oracle 分析的启示**：Table 1 中 MVBench 的 Oracle 实验显示，不同视频/任务的最优 token 数差异极大（如 Moving Count 任务需 757.8 个 token，而 State Changes 仅需 156.5 个），证明静态压缩策略无法适配多样化内容。
4. **人脑类比**：人类描述视觉场景时遵循"看见更多 → 说得更多"的模式，复杂场景用更多词汇表达细节，简单场景则简略描述，语言长度与语义密度存在天然对应关系。

## 核心贡献（创新点）
1. **提出 LangDC 语言感知动态 token 压缩框架**：用轻量级语言模型生成软 caption token 作为视觉表示，通过描述长度自适应控制压缩比，区别于 Q-Former/Resampler 等固定步长下采样方法。
2. **提出语义密度感知监督（Semantic Density-Aware Supervision）**：利用强 LVLM（LLaVA-OneVision）提取各片段关键视觉线索，经 LLM 精简后作为监督信号，使 CapPruner 学习"内容越丰富 → 描述越长 → 分配 token 越多"的动态规则。
3. **三阶段渐进式训练策略**：Cross-modal Pretraining → CapPruner Pretraining → SFT，其中 CapPruner 利用中间层 hidden state（第 15 层）而非最后一层，避免过度抽象导致的表征退化。
4. **高效且性能不劣**：相较 VideoGPT+ 降低 49% FLOPs（49.85T → 25.15T），MVBench 仅下降 1.6%，Video-MME w/ subs 提升 1.4%，且平均 token 数从 3328 降至 1068。

## 方法详解
**整体架构（Figure 2）**：基于 VideoGPT+ 构建，包含双视觉编码器（CLIP-ViT-L/14-336 + InternVideo2-stage-2-1B）、投影层、基础裁剪器（AvgPooling 4×4）、语言专家 CapPruner、后投影层和 LLM（Qwen-2.5-3B）。输入视频被划分为 4 个片段，每片段采样 16 帧，经编码器后送入 CapPruner 动态压缩。

**CapPruner 设计**：
- 由轻量语言模型（初始化自 Qwen-2.5-0.5B）和两个投影层构成。
- **语言建模头（Language Modeling Head）**：将 hidden state 映射到词表，生成 soft caption tokens，控制 token 数量；"padding" token 表示该片段已充分压缩。
- **后投影层（Post Projector）**：将 hidden state 维度对齐至 LLM embedding，支持端到端微调与推理。
- 实验发现中间层（第 15 层）hidden state 最优，浅层语义不足、深层过拟合任务特定抽象。

**语义密度感知监督**：
- 教师模型 LLaVA-OneVision 对视频片段生成详细 caption。
- 利用 Qwen2.5-7B 去除冗余词汇、连接词和推测性内容，得到精炼的 supervisory signal。
- 约束 teacher description 仅来自指令微调数据集片段，保证与 VideoGPT+ 公平对比。

**三阶段训练**：
1. **Cross-modal Pretraining**：仅训练视觉编码器到 CapPruner 和 LLM 的投影层，冻结其余参数。
2. **CapPruner Pretraining**：用基础 caption 数据集（CC-595K）预训练 CapPruner，再用语义密度感知 caption 微调，同步更新投影层。
3. **Supervised Finetuning**：LoRA (rank=128) 应用于 LLM，投影层全参数训练，其余冻结；采用 teacher forcing 提升效率。

## 实验与结果
**数据集与基线**：
- 选择题基准：MVBench（14 个子任务）、Video-MME（w/o subs / w/ subs）。
- 开放问答基准：MSVD-QA、MSRVTT-QA、ActivityNet-QA、TGIF-QA，由 GPT-3.5-Turbo-0613 辅助评估。
- 核心对比基线：VideoGPT+、Video-LLaVA、VideoChat2、Chat-UniVi-V1.5、LDPv2、Resampler、C-Abstractor 等。

**主要结果**：
- **Video-MME**：LangDC（3B 参数）44.3（无字幕）/ 51.3（有字幕），超越 VideoGPT+（44.5 / 49.9），字幕场景 +1.4%。
- **MVBench**：LangDC 57.1 vs VideoGPT+ 58.7，↓1.6%，FLOPs 49.85T → 25.15T（↓49%）。
- **开放 VideoQA**：MSVD-QA 74.0%（↑1.6%）、TGIF-QA 76.8%（↑2.2%），MSRVTT-QA 59.9%（持平）。
- **效率优势**：平均 token 数 1068（动态），仅为原始 3328 的 32%；GPU 显存和延迟均低于池化基线（Figure 4）。
- **动态压缩验证**：Table 5 显示不同子任务分配的 token 数差异显著（Action Antonym: 143.2, Object Existence: 184.7, Episodic Reasoning: 249.1, State Change: 257.2）。
- **泛化能力**：Table A3 在 VSI-Bench（室内新场景）和 LongVideoBench（长视频）上匹配或超越 VideoGPT+。

**消融实验**：
- CapPruner 预训练至关重要：移除后准确率从 54.52% 降至 45.40%（Table 7）。
- 语义监督信号带来稳定增益：Table 8 显示 w/ caption 在 pooling 和 LangDC 上均有提升。
- 组合策略：LangDC + LDPv2 以 748 token 实现 54.13% 平均准确率，优于 Resampler（832 token, 49.0%）。

## 相关工作脉络
1. **Q-Former / Blip-2（Li et al., 2023a）**：通过 cross-attention 将可学习 token 与视觉特征交互实现固定压缩；LangDC 的核心区别在于用语言长度替代固定步长，实现内容自适应压缩。
2. **Resampler（Alayrac et al., 2024; Li et al., 2024e）**：引入固定数量 trainable tokens；LangDC 的 token 数随片段语义密度动态变化，避免信息贫乏片段的冗余或丰富片段的欠表征。
3. **LDP / LDPv2（Chu et al., 2023, 2024）**：卷积下采样保留空间结构，但压缩比固定；LangDC 可兼容 LDPv2 作为 base pruner，通过 CapPruner 进一步动态优化。
4. **C-Abstractor（Cha et al., 2024）**：局部感知投影器，静态降维；LangDC 引入语言专家的"见多言多"先验，与任务无关地自适应调整。
5. **PixelML / Token Merging（Ren et al., 2023; Bolya et al., 2022）**：通道合并或 token 合并的简单策略；LangDC 通过软 caption 保留更丰富的语义信息而非裸特征池化。
6. **长视频理解工作（LongVLM、MovieChat、Video-XL）**：侧重扩展上下文窗口；LangDC 从压缩侧降低输入负荷，二者正交可互补。

## 局限性与未来方向
1. **规模扩展未验证**：实验仅覆盖 1.5B/3B LLM，CapPruner 在更大参数规模（如 7B+）下的架构缩放效应尚未探索。
2. **单压缩比限制**：当前每个片段仅生成一套软 caption tokens，缺乏多粒度/多比率表征，可能对细粒度视频 QA 任务适应性不足。
3. **教师模型依赖**：语义密度感知监督依赖 LLaVA-OneVision + Qwen2.5-7B 管线，推理时虽无需教师，但训练成本较高；自蒸馏或弱监督改进是潜在方向。
4. **仅验证视频理解**：未在图像理解或跨模态生成任务上测试 LangDC 的通用性。

## 研究启发与可借鉴点
1. **语言长度作为压缩比的代理变量**：将 token 数量与描述长度绑定是一种优雅且可微的自适应机制，可迁移至图像、点云等多模态压缩场景。
2. **软 caption token 表示**：用语言模型 hidden state 替代传统的池化/采样特征，保留了更密集的语义信息，为 multimodal connector 设计提供新思路。
3. **语义密度感知监督范式**：用强教师模型生成结构化监督信号并经由 LLM 精炼，可用于训练任何需要"内容自适应"的视觉编码器或压缩模块。
4. **中间层 hidden state 的选择**：实验表明最深层并非最优，中间层在语义丰度与任务脱钩之间取得平衡，这一观察对设计轻量语言专家具有普遍参考价值。
5. **与 LDP/Resampler 等正交模块组合**：LangDC 可作为 plug-and-play 模块叠加到现有压缩器上，为后续工作快速构建基线提供模块化思路。

## 关键术语表
**LangDC**：Language-aware Dynamic Token Compressor 的缩写，本文提出的语言感知动态视频 token 压缩器框架。
**CapPruner**：Caption Pruner 的缩写，LangDC 核心组件，一个基于轻量语言模型的动态 token 裁剪器，通过生成软 caption tokens 实现自适应压缩。
**Soft Caption Tokens**：由语言模型 hidden state 投影得到的连续视觉表征，取代传统离散 token，携带语义密度信息。
**Semantic Density-Aware Supervision**：语义密度感知监督，利用强 LVLM 提取并精炼 video segment 的 caption 作为 CapPruner 训练目标，引导其学习"内容越丰富 → 分配 token 越多"的规则。
**Oracle Token Count**：在 MVBench 每个测试样本上选择能获得正确响应的最高压缩比（最少 token 数），用于揭示不同视频/任务对 token 数量的差异化需求。
**Base Pruner**：基础裁剪器（本文采用 AvgPooling 4×4），与 CapPruner 并行工作，两者输出在时间维度聚合后输入 LLM。
**Cross-modal Pretraining**：LangDC 三阶段训练的第一步，仅训练视觉编码器到 CapPruner 和 LLM 的投影层，建立跨模态对齐。

## 可复现要素
- **数据集**：MVBench、Video-MME、MSVD-QA、MSRVTT-QA、ActivityNet-QA、TGIF-QA、VSI-Bench、LongVideoBench 均为公开基准。
- **代码**：已开源，见 https://github.com/NIneeeeeem/LangDC。
- **权重**：模型权重论文未明确声明开源状态，代码仓库应包含训练脚本与预训练权重获取方式（需访问仓库确认）。
- **关键超参**：
  - 帧数：16 frames
  - 片段数：4 segments
  - CapPruner 最大压缩 token 数：128 tokens/segment
  - CapPruner hidden state 层数：第 15 层
  - LLM：Qwen-2.5-3B；CapPruner 初始化：Qwen-2.5-0.5B
  - SFT LoRA rank：128
  - Base Pruner：AvgPooling 4×4
  - 教师模型：LLaVA-OneVision + Qwen2.5-7B（精炼 caption）
