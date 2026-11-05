---
title: "SafeScientist-Enhancing-AI-Scientist-Safety-for-Risk-Aware-S"
source: https://aclanthology.org/2025.emnlp-main.116.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:13:41"
---

# 论文速读：SafeScientist: Enhancing AI Scientist Safety for Risk-Aware Scientific Discovery

## 一句话总结
本文提出 SafeScientist 框架与 SciSafetyBench 基准，首次针对多智能体科研代理的全流程（输入提示、分组讨论、工具调用、论文撰写）集成多层安全监控机制，使 AI 科学家在面临高风险或恶意提示时能主动拒答并守住伦理底线，安全性较传统科研代理框架提升约 34.69% 且不损害产出质量。

## 研究问题与动机
- **核心问题**：LLM 驱动的自动化科学发现代理在加速研究的同时，面临恶意利用、有害偏见放大及危险知识传播等严重伦理与安全威胁。
- **现有研究不足 1**：现有 LLM 安全工作多聚焦单模型对抗攻击、预训练数据偏见或单一防御手段（如安全微调、运行时检测），缺乏对多智能体科研环境中涌现性风险的系统性建模与防护。
- **现有研究不足 2**：当前主流 AI Scientist 框架（如 AI Scientist、Agent Laboratory 等）几乎仅关注研究能力生成，未内置覆盖输入拦截、协作监督、工具合规与学术伦理的端到端安全防线。
- **现实需求**：亟需面向科学 AI 场景的专用安全评估基准与鲁棒防御框架，以支撑负责任、可信赖的自主科研系统落地。

## 核心贡献（创新点）
- **提出 SafeScientist 全链路安全科研框架**：首次将 Prompt Monitor、Agent Collaboration Monitor、Tool-Use Monitor 与 Paper Ethic Reviewer 串联至科研代理的完整工作流，实现从源头到输出的闭环管控；与已有工作本质区别在于将安全机制深度嵌入多阶段科研流水线，而非仅依赖外部事后过滤或单一模型微调。
- **构建 SciSafetyBench 多维度科学安全基准**：涵盖 6 个学科领域的 240 项高风险发现任务及 30 种科学工具的 120 个工具级风险场景，并按直接恶意、间接恶意、无意后果、任务固有四类风险源系统化标注；区别于通用 LLM 安全基准，本基准紧贴真实科研语境与实验操作规范。
- **设计系统化对抗攻击体系并验证防御鲁棒性**：引入 7 种查询注入攻击（含低资源翻译、Base64、载荷拆分、DAN、DeepInception 及组合）与两类恶意代理攻击，证明语义+结构双轨融合策略可显著提升对高级逃逸手法的拦截率；与仅依赖单一检测模型的工作相比，本文采用多视角分级拦截（pass/warning/reject）机制。

## 方法详解
- **Pipeline 架构**：用户指令经领域与任务类型解析后，动态激活专家智能体集合（领域研究员、综述撰写者、实验规划者）进行最多 3 轮分组讨论，随后调用检索模块与 30 种科学仿真工具完成信息收集与实验模拟，最终由写作模块生成结构化论文草稿。
- **Prompt Monitor（输入层）**：双阶段融合检测。LLaMA-Guard-3-8B 评估提示词语义意图并输出安全风险标签与解释；SafeChecker 扫描已知攻击模式（越狱、角色扮演等），覆盖 17 类风险并输出 pass/warning/reject 三级判定；任一模块标记拒绝即拦截，warning 允许探索但需标注风险。
- **Agent Collaboration Monitor（协作层）**：在智能体讨论环节部署专注伦理与安全的监控智能体，实时审视交互内容，对潜在恶意引导或有害共识提供纠正性伦理干预。
- **Tool-Use Monitor（工具层）**：部署
