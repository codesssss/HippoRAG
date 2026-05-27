# Fact-Witness 这个概念在 §4 method 里到底是什么

> 给导师看的简介。目标是让导师能判断:这个概念是否该保留 / 改名 / 怎么改。
> 写作时间:2026-05-15

---

## 一、它在方法里**承担的角色**(一句话)

**Fact-witness = OpenIE 抽取的 (h, r, t) 三元组,在 method 里不作为推理单元,而作为"两篇文档之间的 transition 是否合法"的可审计依据。**

换句话说:

- 它**不是**新的事实抽取(我们直接用 HippoRAG-style OpenIE,不动抽取器)
- 它**不是**reasoning unit(reader 仍然消费自然语言段落,不消费三元组)
- 它**是**一个 admission rule(准入规则):一条 doc-to-doc 边能不能进入检索图,看 OpenIE substrate 里是否有 fact 能见证这次跳转

---

## 二、它跟周围概念的关系

```
┌─────────────────────────────────────────────────────────────┐
│  Offline indexing 阶段(HippoRAG-style)                      │
│                                                              │
│   document corpus D                                          │
│      ↓ (OpenIE)                                              │
│   triples F = { (h, r, t) }  ← 这就是 fact-witness 集合      │
│      + provenance index: src(f) → which document(s)         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Query-time retrieval                                        │
│                                                              │
│   构建 query-local graph G_q = (V_q, A_q)                    │
│                                                              │
│   关键准入规则:                                              │
│     一条 arc (d_i, d_j) ∈ A_q 当且仅当                       │
│     存在 fact f ∈ F 能"见证"这条 doc-to-doc transition       │
│                                                              │
│   "见证" 意味着:                                              │
│     - f 来自 d_i 或 d_j(provenance 锚定)                     │
│     - f 的某个 argument 跟 query 里的实体相关                │
│     - f 的另一个 argument 出现在另一篇 document 里           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  最终输出:ranked evidence list R_q(在 G_q 上跑 local-PPR)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、为什么要单独造这个概念(它解决什么问题)

### 问题:传统 GraphRAG 的两条路线都不够

| 路线 | 怎么做 | 问题 |
|---|---|---|
| **静态 KG 推理** (ToG, Relink) | 把 OpenIE 三元组当作 KG,在 KG 上做推理(traversal / repair) | reader 不读三元组,读自然语言 doc;KG 推理跟 doc 检索之间有 gap |
| **Query-time chain mining** (NeocorRAG) | 检索完 doc 后,在 retrieved text 上**重新调 LLM 抽 chain** | query-time LLM 调用昂贵且不稳定;chain 是事后拼出来的,不是检索时就保证的 |

**两条路线的共同特征**:把 fact / chain **当作推理对象本身**。

### 我们的处理:把 fact 降级为"准入证据"

- 我们**仍然让 reader 读 document**(不偏离主流 RAG)
- 我们**不在 query time 重抽 chain**(不偏离 retrieval-time 效率)
- 但我们用**预抽好的 OpenIE fact** 来**约束哪些 doc-to-doc 跳转能进入检索图**

这样:
- ✅ Chain 在**检索图内部就成形**,不是事后从文本拼
- ✅ 每条 doc-to-doc 边都**可追溯到一个具体 fact 和它的 source sentence**(可审计)
- ✅ Query-time **不需要 LLM 重抽**(只 LLM 抽 question 一次,不抽 corpus)

**"witness" 这个词的来源**:
- 法律/逻辑上,"witness" 指 **能证明某事成立的可追溯依据**
- 我们用它来强调:**fact 不是 reasoning unit,是 transition 的依据**

---

## 四、它在代码里的真实形态(不夸大)

| 论文里写 | 代码里对应 |
|---|---|
| `fact-witness substrate F` | OpenIE triples + provenance index(直接复用 HippoRAG indexing) |
| `witness set W_q(d_i, d_j)` | `local_graph.py` 里 admission 检查时枚举的 fact 集合 |
| Sentence-grounded witness | `SENTENCE_GROUNDED_TRANSITION` edge type |
| Argument-grounded witness | `SAME_SUBJECT / SAME_OBJECT / ROLE_BRIDGE / TITLE_ROLE_GROUNDING` edge types(共 4 种,论文抽象成 1 种 mode) |

**论文写的 5 种 edge → 抽成 2 种 mode**,这是 abstraction,不是编造。

---

## 五、术语 candidates(请导师定夺)

我不坚持任何一个术语,但请导师从以下选项里挑一个,或给一个新的:

| Option | 候选词 | 论证 |
|---|---|---|
| A | **保留 "fact-witness substrate"** + 加一句定义 | 突出 "fact 作为 transition 见证" 这个角色,这是我们的贡献定位 |
| B | **OpenIE-grounded transition layer** | 强调 OpenIE 来源 + transition 用途,但失去 "witness" 的可审计感 |
| C | **Document-grounding fact layer** | 强调 fact 把 doc 锚回 source 的功能,但失去 transition 的方向性 |
| D | **Relational evidence layer** | 跟 NeocorRAG 词汇撞,可能被认为是 NeocorRAG 同类工作 |
| E | **导师推荐的术语** | (留空) |

**我目前的立场**:倾向 A(保留 + 加定义),理由是 "witness 这个词本身有学术地位"(逻辑/法律领域常用),且它精确描述了我们的角色——fact 作为 doc-to-doc 跳转的可追溯依据,跟 ToG/Relink 的 "reasoning substrate" 形成精确对照。

但如果导师觉得 "fact-witness" 听起来像我们造的术语(因为它确实是组合词),换成 B 或导师指定的术语都可以。

---

## 六、真正想让导师判断的两个问题

**问题 1:这个概念在论文 §4.1 里值不值得单独成节并起一个名字?**

- 如果**值得**,我们就保留它(无论叫什么词),§4.1 主要讲 "OpenIE substrate 的角色重定位"
- 如果**不值得**,我们就把 §4.1 改成纯 HippoRAG-style offline indexing 描述,不提 witness 概念,§4.2 admission rule 直接说 "an arc is admitted when supported by an OpenIE fact"

**问题 2:这个概念是否应该出现在 method 主线里?**

- **方案 A**:它是核心机制,§4.1 单独讲,§4.2 admission rule 围绕它
- **方案 B**:它是辅助概念,只在 §4.2 一句话提一下,不单独成节
- **方案 C**:完全不提,只用 "OpenIE-extracted facts" 这种通用表述

我的判断:方案 A 是当前论文主张,方案 B 是退一步妥协,方案 C 是完全去掉这个概念,跟 HippoRAG 的差别只剩 "在 doc graph 上跑 local-PPR"——这样跟 HippoRAG 2 的差别会非常小。

---

## 七、一句话总结

> **Fact-witness substrate 是一个 framing 决定**:把 OpenIE fact 从 "推理对象" 降级为 "doc-to-doc transition 的可追溯依据"。这个 framing 让 EvidenceFlow 能跟 ToG/Relink/NeocorRAG 三条线**精确切边**:静态 KG 推理把 fact 当推理对象,query-time chain mining 让 LLM 重抽,我们让 fact 在 indexing 时就预抽好,在 retrieval 时只起准入作用。
>
> 这个 framing 本身可以保留也可以放弃。如果保留,它是论文的 §4.1 核心概念,需要一个稳定的名字(目前叫 "fact-witness substrate")。如果放弃,§4.1 就退回普通 OpenIE indexing 描述,论文的 graph-based novelty 会大幅减弱。

请导师定夺。
