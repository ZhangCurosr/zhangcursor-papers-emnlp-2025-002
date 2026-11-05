---
title: "SwarmAgentic-Towards-Fully-Automated-Agentic-System-Generati"
source: https://aclanthology.org/2025.emnlp-main.93.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:15:56"
field: "多智能体系统自动化生成"
keywords: ["Multi-Agent System", "Agent Generation", "Particle Swarm Optimization", "LLM-based Agent", "Automated Workflow Design", "Open-ended Planning"]
innovations: ["首次实现零种子/零模板的完全自动化代理系统生成", "将PSO重新形式化为语言驱动的符号优化过程", "提出融合失败记忆、个人最优与全局最优的三源速度更新机制"]
benchmarks: ["TravelPlanner", "Natural Plan", "Creative Writing", "MGSM"]
---

# 论文速读：SwarmAgentic-Towards-Fully-Automated-Agentic-System-Generati

## 一句话总结
SwarmAgentic 是首个完全自动化的代理系统生成框架，通过语言驱动的粒子群优化（PSO）从零开始构建代理并联合优化其功能与协作策略，无需人工干预；在六个真实开放任务上取得 SOTA，其中 TravelPlanner 相对 ADAS 提升 +261.8%。

## 研究问题与动机
1. **现有框架缺乏完全自主性**：现有 agentic system 生成框架要么依赖预定义模板/种子 agent，要么无法从零生成 agent，限制了适应性与可扩展性。
2. **结构化约束阻碍开放任务**：Trip Planning、Travel Planner 等开放型任务需要高层规划与系统级协调，人工设计代理结构"既繁重又不可扩展"。
3. **已有方法各自只能覆盖部分自治维度**：SPP 缺乏自生成与协作重构；EvoAgent / AgentSquare 支持功能优化但协作结构固定；ADAS / AutoAgents 等均需种子 agent 或模板，无法 from-scratch 生成。
4. **缺少语言空间的群体优化机制**：传统 PSO 优化连续向量，难以直接迁移到非微分、组合式的结构化系统配置搜索空间。

## 核心贡献（创新点）
1. **提出 SwarmAgentic 框架，实现无人工干预的从 scratch 到 fully optimized 的完整闭环**；与 EvoAgent/ADAS 等仅做局部突变或基于种子的搜索本质不同，SwarmAgentic 零种子、零模板，全程由 LLM 驱动语言空间搜索。
2. **将 PSO 重新形式化为符号化语言驱动过程（language-driven PSO）**；传统 PSO 用数值加减更新速度/位置，本文用 LLM 将速度和位置更新翻译为结构化的文本变换，使搜索空间成为离散的结构化表示空间。
3. **引入 Failure-Aware Velocity Update 三源合一的更新机制**（失败驱动 + 个人最优 + 全局最优）；与单层反馈（如 ADAS 的 meta-agent 改写）不同，本文同时记忆历史失败、对比个人历史上最佳、对齐群体历史上最佳，三者经 LLM 综合后产生更新方向。
4. **在六类真实开放任务上验证全自动化有效性，且开销低于 ADAS**；TravelPlanner 相对 ADAS 提升 +261.8%，整体训练成本节省 ~21.3%，验证成本节省 ~48.9%。

## 方法详解
- **系统表示**：每个粒子 $i$ 在时刻 $t$ 表示一个候选系统 $S_i^{(t)}=(\mathcal{A}_i^{(t)}, \mathcal{W}_i^{(t)})$，其中 $\mathcal{A}$ 为 agent 集合，每个 agent 三元组 $(I,R,P)$ 对应角色标识、职责、执行策略；$\mathcal{W}$ 为协作结构（有向任务流水线）。
- **粒子初始化**：使用 LLM 按不同温度（低/中/高）生成多样化初始系统，保障 exploration 与 exploitation 的平衡。
- **Flaw Identification（缺陷识别）**：LLM 根据执行错误集 $\mathcal{E}_i^{(t)}$ 诊断两类缺陷——agent 缺陷（缺失/冗余/策略模糊）和协作结构缺陷（缺失/冗余步骤、上下文不完整、输出错配）。
- **Failure-Aware Velocity Update**（式 3–6）：
  - $c_f r_f F(v_i^{(t)}) = \text{LLM}_{\text{fail}}(v_i^{(t)}, f_i^{(t)}, f_i^{(t+1)})$：失败驱动项，识别上一轮无效修改，避免重复错误。
  - $c_p r_p (p_i^* - x_i^{(t)}) = \text{LLM}_{\text{pers}}(x_i^{(t)}, p_i^*, f_i^{(t+1)})$：个人最优项，向自身历史上最佳进化。
  - $c_g r_g (g - x_i^{(t)}) = \text{LLM}_{\text{glob}}(x_i^{(t)}, g, f_i^{(t+1)})$：全局最优项，学习群体历史最佳配置的可迁移改进。
- **Position Update**（式 7）：$x_i^{(t+1)} = \text{LLM}_{\text{pos}}(x_i^{(t)}, v_i^{(t+1)})$，执行两类适配：① agent 层面（增删改角色/职责/策略）；② 协作结构层面（重排/增删/修改进程步骤与输入输出依赖）。
- **评估**：每轮迭代后按任务 fitness function $J(S)$ 更新个人/全局最优，直至达到最大迭代数或满足停止条件。

## 实验与结果
- **任务**：TravelPlanner（长程规划）、Natural Plan（Trip/Meeting/Calendar Scheduling）、Creative Writing、MGSM（数学推理）共 6 个。
- **基线**：Direct、CoT、Self-Refine、SPP、EvoAgent、ADAS。
- **模型配置**：Optimizer = GPT-4o-mini-0718；Executors = GPT-3.5-turbo、GPT-4o、Claude-3.5-sonnet、DeepSeek-V3、Gemini-1.5-Pro；5 粒子，10 迭代；ADAS 30 迭代。
- **关键数字**：
  - TravelPlanner：SwarmAgentic vs ADAS（GPT-4o）Final = 32.2 vs 8.9，相对提升 **+261.8%**；Hard Constraint Macro 66.7 vs 27.8。
  - Creative Writing：8.2–8.5 vs 最优基线 7.3（ADAS）。
  - MGSM（GPT-4o）：88.4 vs ADAS 87.0，仍为最高。
  - Trip Planning（GPT-4o）：13.1 vs ADAS 3.1；Meeting Planning：56.0 vs 43.0；Calendar Scheduling：82.0 vs 66.0。
- **消融**：移除 Failure-Driven Adjustments 下降 +35.5%，移除 Agent-Level Adaptation 下降 +17.7%，移除 Collaborative Structure Reconfiguration 下降 +8.1%；粒子和迭代数均呈单调改善。
- **跨模型迁移**：GPT-4o-mini 优化的系统直接迁移至 Gemini-1.5-Pro、Claude-3.5-sonnet、DeepSeek-V3 仍全面超基线。
- **成本**：Training cost 比 ADAS 低 ~21.3%，Validation cost 低 ~48.9%。

## 相关工作脉络
1. **SPP（Wang et al., 2023）**：多 persona 协作，但依赖固定脚本模板，无法 from-scratch 生成；SwarmAgentic 在三个自治维度均打勾。
2. **EvoAgent（Yuan et al., 2024）**：基于进化算法突变 agent 行为，但协作拓扑固定；SwarmAgentic 可同时重构角色与流程。
3. **ADAS（Hu et al., 2024）**：Meta-agent 在代码空间搜索，但搜索始于 7 个手工种子 agent；SwarmAgentic 零种子启动。
4. **Agent Square（Shang et al., 2024）**：在 16 个预定义模块类型中重组；SwarmAgentic 不依赖任何预定义模块库。
5. **MODEL SWARMS（Feng et al., 2024）**：同样使用 PSO，但在模型权重/概率空间做 expert interpolation；SwarmAgentic 搜索的是可执行的 multi-agent 系统结构与协议。
6. **Agent Symbolic Learning / GödelAgent**：文本梯度做单 agent 递归自改进；SwarmAgentic 扩展到 multi-agent 系统的联合优化。

## 局限性与未来方向
1. **缺少归纳偏置（inductive priors）**：纯语言驱动初始化在结构化环境中可能收敛慢；未来可结合约束引导搜索。
2. **LLM 幻觉风险**：系统性错误可能在优化循环中累积传播；需要外部知识 grounding 或选择性 human-in-the-loop 验证。
3. **纯文本环境限制**：不支持多模态感知与物理交互，难以直接用于 embodied agent 场景。

## 研究启发与可借鉴点
1. **PSO → 语言空间的映射思路**：将连续空间的速度/位置更新翻译为"语义操作序列"，为其他组合优化问题（如 workflow 合成、prompt engineering）提供通用范式。
2. **三源合一的 velocity update（失败 + 个人最佳 + 全局最佳）**：比单一反馈机制更具鲁棒性；可迁移到基于 LLM 的算法合成、代码生成等迭代优化任务。
3. **温度分层初始化 + 并行粒子**：兼顾多样性与收敛效率；在资源受限场景下可通过调节粒子数和迭代数灵活权衡。
4. **跨模型可迁移性**：在廉价模型上优化系统结构后迁移到强模型验证，可显著降低成本；适用于"优化-部署"分离的工业流程。
5. **LLM 作为"结构化 diff"生成器**：`LLM_fail / LLM_pers / LLM_glob / LLM_pos` 等模块化 prompt 设计可直接复用为其他 agent 自动化框架的组件。

## 关键术语表
**SwarmAgentic**：基于群体智能的完全自动化多代理系统生成框架，零人工干预。
**Failure-Aware Velocity Update**：融合历史失败经验、个人最优、全局最优三项的 LLM 驱动更新机制。
**Particle（粒子）**：编码为一个候选代理系统的结构化语言表示（角色 + 协作流程）。
**Flaw Identification**：LLM 根据执行错误诊断 agent 缺陷与协作结构缺陷的过程。
**Personal Best / Global Best**：粒子自身历史最优系统 / 群体历史最优系统，用于引导搜索收敛。
**Agent-Level Adaptation**：在粒子更新时对单个 agent 的角色、职责、执行策略进行增删改。
**Collaborative Structures Reconfiguration**：在粒子更新时重排/增减/修改工作流步骤及其输入输出依赖。
**Temperature-controlled Initialization**：通过低/中/高三种采样温度生成多样化初始粒子，平衡探索与利用。

## 可复现要素
- **数据集**：TravelPlanner、Natural Plan、Creative Writing、MGSM——均使用公开数据，未见额外私有数据集声明。
- **代码**：已开源，见 github.com/SwarmAgentic。
- **模型**：Optimizer = GPT-4o-mini-0718；Executors 含 GPT-3.5-turbo-0125、GPT-4o-0806、Claude-3.5-sonnet-0620、DeepSeek-V3、Gemini-1.5-Pro。
- **关键超参**：粒子数 = 5；最大迭代 = 10；ADAS 对比使用 30 迭代；训练集：TP 9 条、NP 各 1 条难度示例、CW 5 条用于训练。
- **Prompt**：附录 D.3 公开全部 prompt 模板（可完全复现）。
