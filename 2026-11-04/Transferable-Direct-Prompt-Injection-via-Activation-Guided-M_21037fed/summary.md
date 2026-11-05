---
title: "Transferable-Direct-Prompt-Injection-via-Activation-Guided-M"
source: https://aclanthology.org/2025.emnlp-main.102.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:08:19"
field: "大语言模型安全与对抗攻击"
keywords: ["Direct Prompt Injection", "Black-box Attack", "Activation-guided", "Energy-based Model", "MCMC Sampling", "Transferable Vulnerability"]
innovations: ["基于代理模型激活值训练EBM评估对抗提示质量", "Token-level MCMC采样实现无梯度黑盒攻击", "跨模型与跨任务高迁移性直接提示注入"]
benchmarks: ["CYBERSECEVAL3", "Tensor Trust Dataset", "五款主流LLM（Qwen2.5/Qwen2/Llama-3.1/Llama-3/GPT-4o-mini）"]
---

# 论文速读：Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling

## 一句话总结
本文提出一种基于激活值（activations）引导的**无梯度黑盒直接提示注入（DPI）攻击框架**，通过代理模型内部激活值训练能量基模型（EBM）来评估攻击提示质量，再结合token-level MCMC采样自适应优化提示。该方法无需查询目标模型，在五款主流LLM上实现了49.6%的攻击成功率（ASR），跨模型与跨任务均展现出优异的迁移性。

## 研究问题与动机
- **白盒/灰盒方法的实战局限**：现有DPI攻击多依赖梯度信息（如GCG、AutoDAN），但在黑盒场景（如云服务）中无法获取模型参数或响应logits，实用性差。
- **现有黑盒方法的缺陷**：人工 crafted 提示（如StruQ）随机性强、稳定性差；基于查询的fuzzing方法（如PromptFuzz）需大量调用目标模型，迁移性不足。
- **缺乏可解释性与安全评估手段**：当前攻击生成过程“黑盒化”，难以揭示LLM内在的安全漏洞模式，不利于系统性防御验证。
- **自然性与攻击强度的平衡**：现有方法生成的对抗提示往往人类可读性差（高困惑度），易被过滤机制检测。

## 核心贡献（创新点）
- **首创激活值引导的EBM评估框架**：利用代理模型（Qwen2.5/Llama-3.1）的内部激活值训练二分类EBM，以能景能量隐式建模成功攻击提示的分布。区别于仅依赖输出概率的方法，本文方法从语义表征层面指导生成。
- **Token-level MCMC采样策略**：采用BERT作为提议模型，在保持提示自然性的同时自适应搜索高质量攻击变体。区别于GCG的贪心后缀优化，本方法支持整句结构的探索与接受-拒绝迭代。
- **免查询的黑盒攻击范式**：完全无需访问目标模型，仅通过离线生成的攻击提示即可触发跨模型攻击，解决了云服务场景下频繁查询违规的问题。
- **系统性评估与可解释性分析**：在7个任务、5款LLM（含闭源GPT-4o-mini）上验证了模型迁移性与任务迁移性，并通过PCA可视化揭示了激活空间与攻击成功率的强相关性（Pearson r=-0.979）。

## 方法详解
方法分为三阶段：
1. **模板数据集构建**
   - 从Tensor Trust竞争数据集中解耦攻击提示为前缀（Prefix）、中缀（Infix）、后缀（Suffix）三部分。
   - 使用Qwen2.5-7B-Instruct收集20,000条`(激活值x, 攻击标签y)`样本，经成功率筛选得到85×35×85的最终模板集。
2. **能量基模型训练**
   - 以两层MLP（1024→256）为分类器，最大化交叉熵损失训练。
   - 依据Grathwohl et al. (2020)理论，分类器logits隐式定义能量函数：$E_\theta(x) = -\log[\exp(f_\theta(x)[0]) + \exp(f_\theta(x)[1])]$。
   - 选取第25层激活层训练的EBM作为最优评估器。
3. **MCMC采样优化**
   - 以种子提示为初始候选，每轮随机选取一个token位置，用BERT MLM重采样生成新候选。
   - 计算新旧候选的能景得分及接受概率：$p(X'|X) = \min\left(\frac{e^{-E(X')} p_{MLM}(X_i|X_{/i})}{e^{-E(X)} p_{MLM}(X'_i|X_{/i})}, 1\right)$。
   - 迭代$T$步（$T$等于提示总token数），保留历史最低能量样本作为最终攻击提示。

## 实验与结果
- **数据集**：Tensor Trust攻击数据集（训练/测试任务分离）+ CYBERSECEVAL3的7个标准DPI任务。
- **基线对比**：Human Experts、Initial Prompts、GCG-Inject（白盒）、AutoDAN-GA-Inject（灰盒）、PromptFuzz（黑盒）。
- **模型迁移结果**（Table 1）：
  - Ours(Qwen2.5)平均ASR达**49.6%**，较人工提示提升**34.6%**；对Llama3.1维持44.4% ASR，而GCG-Inject骤降至5.55%。
  - 唯一成功案例突破GPT-4o-mini防护（人工提示ASR=0%）。
- **任务迁移结果**（Table 2）：
  - 在未见过任务上保持**36.6%** ASR，显著高于AutoDAN的20.6%与人工的32.53%。
- **自然性**：生成提示PPL=127.68，与人工（98.90）和AutoDAN（130.78）同量级，远低于GCG（17464.28），具备绕过基于困惑度过滤的能力。

## 相关工作脉络
- **GCG-Inject / Neural Exec**：白盒梯度攻击，依赖完整模型访问，本文方法在黑盒设定下性能超越并避免过度拟合特定架构。
- **AutoDAN-GA-Inject**：灰盒遗传算法优化，需观察目标响应分布；本文仅用代理模型激活值，无需交互。
- **PromptFuzz / GPT-Fuzzer**：基于蒙特卡洛树搜索的黑盒fuzzing，需大量查询；本文离线采样后直接投放。
- **COLD-Attack**：logit空间Langevin动力学，需白盒访问；本文转向激活空间实现无梯度控制。
- **Representation Engineering (Zou et al., 2023a)**：揭示激活编码安全概念；本文将其应用于攻击分布学习而非单概念操纵。
- **Mix & Match (Mireshghallah et al., 2022)**：无参数MCMC文本生成框架；本文创新性地将其与EBM结合，用于黑盒攻击提示优化。

## 局限性与未来方向
- **自然性与攻击强度的权衡**：当前PPL水平虽通过人工可接受阈值，但仍有提升空间，未来可微调提议模型或引入更强约束。
- **未应对基于文本分类器的防御**：如InjecGuard等guardrail模型，未来需研究绕过检测机制的策略。
- **代理模型选择依赖**：EBM性能受限于代理模型（Qwen2.5/Llama-3.1）的激活语义丰富度，跨架构泛化能力待验证。
- **单轮优化深度有限**：MCMC迭代步数固定为token总数，复杂任务可能需要更深入的搜索策略。

## 研究启发与可借鉴点
- **激活值作为代理信号的可迁移性**：证明中间层激活能捕获跨模型共享的脆弱性模式，可用于其他黑盒攻击（如后门注入、越狱）的设计。
- **EBM+MCMC的采样范式**：将分类器隐式能量模型与无梯度MCMC结合，为任何需要“质量评估-自适应生成”的任务提供通用框架。
- **组件解耦与重组策略**：Prefix/Infix/Suffix分离后随机组合，大幅提升攻击提示多样性，可推广至其他对抗样本构造场景。
- **安全评估基准价值**：CYBERSECEVAL3的任务设计与LLM-based/keyword混合评估协议可直接复用，为后续红队测试提供标准化流程。
- **可解释性驱动的攻击设计**：激活空间可视化揭示的“成功攻击方向”，启发防御方针对性地修改模型表征以阻断攻击路径。

## 关键术语表
- **Direct Prompt Injection (DPI)**：通过直接输入渠道（如用户提示）注入恶意指令，覆盖系统预设指令的攻击方式。
- **Energy-based Model (EBM)**：通过能量函数隐式定义概率分布的模型，本文用作评估对抗提示质量的评分器。
- **Markov Chain Monte Carlo (MCMC) Sampling**：基于接受-拒绝准则的随机采样方法，本文用于在提示空间高效搜索高质量攻击变体。
- **Activation**：Transformer块间的隐藏状态，本文利用其丰富语义指导攻击生成。
- **Cross-model Transferability**：攻击提示对未见模型保持高效性的能力，本文核心评价指标。
- **Cross-task Transferability**：攻击提示对未参与训练的任务保持有效性的能力。
- **Perplexity (PPL)**：衡量文本自然度的指标，越低表示越接近人类语言分布。
- **Surrogate Model**：用于提取激活值的替代模型，本文指Qwen2.5-7B-Instruct和Llama-3.1-8B-Instruct。

## 可复现要素
- **数据集**：Tensor Trust攻击数据集（公开竞赛数据）、CYBERSECEVAL3任务（论文附录提供7个任务说明）。
- **代码/权重**：论文未声明开源仓库或代码链接。
- **关键超参**：MCMC迭代步数=提示token总数；batch size=20；EBM训练：学习率0.0003、AdamW优化、100 epoch、batch size=256；类权重按样本标签调整。
