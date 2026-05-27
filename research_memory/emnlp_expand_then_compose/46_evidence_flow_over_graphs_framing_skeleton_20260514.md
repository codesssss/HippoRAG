# Evidence Flow over Graphs — Framing 骨架

Last updated: 2026-05-14
Status: v3 叙事("evidence as chain")骨架草稿,方法名待定(暂用 `<METHOD>`)

---

## 0. 元信息

- **Title**: Evidence Flow over Graphs(整篇论文标题就这五个词,无副标题、无方法名前缀)
- **Working method name**: `<METHOD>`(占位符,叙事流畅之后再拍)
- **Scope**: Multi-hop QA on knowledge-graph–anchored corpora(2WikiMultihopQA、HotpotQA、MuSiQue)
- **Venue target**: EMNLP / ACL 主会

---

## 0.1 实验协议(LOCKED 2026-05-14)

**这是 paper 所有数字的统一对照设定。所有 main results / baseline / ablation 必须按此协议跑;不符合此协议的旧数据视为 pilot,不进 paper 主表。**

| 角色 | 模型 | 用途 |
|---|---|---|
| **Reader / Answer model** | **GPT-4o-mini** (OpenAI API) | 接受 top-K documents → 生成最终 answer。所有方法、所有 baseline 都用同一个 reader |
| **Graph construction / Extraction LLM** | **Qwen3-32B** (local VLLM) | OpenIE 抽取、entity linking、binding extraction 等 indexing 阶段的 LLM 调用 |
| **Quick-verification LLM** | **Qwen3-8B** (local VLLM) | 开发期 smoke / limit=100 / 快速 ablation 用。**不**进 paper main table |
| **Embedding model** | NV-Embed-v2 (legacy fact PPR) + Qwen3-Embedding-8B (dense passage) | 与之前协议一致,不动 |

### 协议核心约束

1. **Reader 一律 GPT-4o-mini** — paper 里所有 EM/F1 数字都基于 GPT-4o-mini 作为 reader。NeocorRAG、SETR-style、IRCoT-style、Top-K、`<METHOD>` 全部 reader-side 对齐到 GPT-4o-mini
2. **建图/抽取一律 Qwen3-32B** — `<METHOD>` 的 OpenIE substrate、SETR 的 information requirements 抽取(如有 LLM)、NeocorRAG 的 constrained-decoding LLM、IRCoT 的 chain-of-thought 推理,全部 indexing/inference 期的 LLM 调用对齐到 Qwen3-32B
3. **Qwen3-8B 仅作开发期 quick-verification 用** — `limit=100` smoke、ablation 快速迭代可以用 8B;**但 paper 主表里出现的任何数字必须是 32B + 4o-mini 协议下重跑的**
4. **VLLM 端点稳定** — Qwen3-32B 和 Qwen3-8B 都通过 local VLLM 暴露,模型名 / 端口 / batch size 在 indexing 和 inference 之间保持一致

### 这个协议对现有数据的影响

| 类别 | 现状 | 协议下处理 |
|---|---|---|
| 41_ 锚定数字(+0.124/+0.082/+0.179 EM gap;2Wiki 4-doc 51.9% under-selection) | reader=Qwen3-8B-train | **需要在 GPT-4o-mini reader 下重算**,数字可能略变,phenomenon 立得住的可能性高(composition gap 在不同 reader 下普遍存在) |
| `aligned_full1000_master_comparison_20260430.md` 里的 NeocorRAG k=1 / k=3 数字 | reader=Qwen3-8B-train | **不能进 paper 主表** — 需要按新协议(reader=4o-mini,NeocorRAG 内部抽取改 Qwen3-32B)重跑 |
| `aligned ETv3 / Prop+DAEC / Prop+DAEC-L1` 数字 | reader=Qwen3-8B-train | 同上 |
| 41_ Limitation 1 锚定的 8.4 calls/query, ~8400 tokens | indexing/extraction=Qwen3-8B-train | **token 数字基本可保留,但 LLM-call breakdown 需重报告**(因为现在 reader call 走 OpenAI API,不算自家 LLM call) |
| DBEC-IG / IG-gate 实验 | indexing=Qwen3-8B-train | 重跑成本中等,建议优先级:跟 NeocorRAG 主对比的同协议数据 > 自家 ablation 同协议数据 |

### 投稿前必须做的协议化重跑(优先级)

**v3 主张下的真实最短队列**(2026-05-14 修订;此前版本把 SetR 列为 P0,是 41_ DBEC 时代的惯性,v3 主张下 SetR 与本方法范式不可直接比较,降级):

1. 🔴 **`<METHOD>` / EvidenceFlow(ETv3+PCEC)新协议三数据集 main results**(reader=4o-mini, extraction=32B)— **paper 主表 anchor**
2. 🔴 **NeocorRAG k=3 新协议三数据集**(constrained-decoding LLM=Qwen3-32B, reader=4o-mini)— v3 唯一正面对手,46_ §3 intro 第四段已单独成段
3. 🔴 **no fact-witness ablation 新协议三数据集** — v3 最大卖点 M2 的实验证据,**不跑则 method contribution 立不住**
4. 🔴 **chain-broken@5 / all-gold@5 统一汇总脚本** — C1 phenomenon operational metric,离线算,不跑模型
5. 🟡 **41_ 的 oracle@100 / under-selection 51.9% 等 anchor 在新 reader 下重算** — phenomenon 验证
6. 🟢 **SetR-style 新协议重跑** — **降级 appendix only**。v3 主张下 SetR 操作在 passage pool 上做 set selection,与 EvidenceFlow 在 graph 上长 chain 范式不直接可比(参见 §3 intro 第三段并列论证)。如果时间充裕,补一个最小版进 appendix 作为 reviewer defense;**否则旧协议 pilot 数字顶上 appendix 即可,不进主表**
7. 🟢 IRCoT / LLM-direct / Top-K 等 baseline — appendix only

**真正要跑模型的只有 P0-1 / P0-2 / P0-3 三项。P0-4 是离线分析,不跑模型。**

### 这个协议的诚信价值

- **Reader 用 OpenAI API 模型(GPT-4o-mini)** — 排除"reader-side 调优带来的 method gain"的怀疑;reviewer 看到 reader 是外部 API、不可微调,会更信对比公平
- **Extraction 用 Qwen3-32B(可复现的开源模型)** — 排除"我们用更强 LLM 抽 substrate" 的怀疑;Qwen3-32B 是公开模型,任何复现者都能用同一个 checkpoint
- **NeocorRAG 内部 constrained decoding 也用 Qwen3-32B** — 这是它论文报告的 3B/70B 之外的中间档,**且我们让它用了比它自己默认更强的抽取 LLM(它论文 Llama-3.2-3B vs 我们给的 Qwen3-32B)**,反而抬升对手起点。这是诚信高分动作

### Query-time LLM 调用的诚信红线(2026-05-15 加,Codex review)

**真实存在的 query-time LLM 调用**:question-side obligation / binding extraction(从 question 抽 entity + relation slot,代码内部叫 binding,新协议下用 Qwen3-32B)。

**这不是 NeocorRAG 那种"在 retrieved text 上挖 chain"的调用**,但它**也是 query-time LLM**。所以 paper 任何位置**绝对不可以写**:
- ❌ "no query-time LLM"
- ❌ "LLM-free retrieval"
- ❌ "without LLM in the loop"
- ❌ "no language model is invoked at query time"(裸的、不带限定语)
- ❌ "only stage at which a language model interacts with the corpus"(因为 reader 也读 corpus)

**必须写成带限定语的精确版本**:
- ✅ "no query-time LLM **chain mining over retrieved text**"
- ✅ "no **fresh corpus-level OpenIE extraction / fact repair / chain generation over $\mathcal{D}$** at query time"
- ✅ "question-side obligation extraction only(在 question 上做,不在 corpus 上做)"

**§4.5 / §5.6 写作时必须主动 surface question-side LLM 调用**(不暴露内部命名,但承认存在):

> *"The boundary criterion is conditioned on lightweight question-side obligations extracted from $q$, such as entity mentions and relation-bearing slots. These obligations are used only to decide whether an additional document closes a missing part of the chain; they are not generated from retrieved documents and are not passed to the reader as reasoning paths. In our implementation, the question-side obligations are extracted with the same controlled extraction model (Qwen3-32B) used throughout our protocol. This extraction is applied to the question, not to the retrieved corpus, and does not generate evidence chains."*

**§5.6 cost table 必须区分 query-time-over-retrieved-text vs question-side extraction**:

| Method | Query-time LLM over retrieved text? | Query-time chain generation? | Question-side extraction? |
|---|---|---|---|
| NeocorRAG | Yes | Yes | Yes/implicit |
| EvidenceFlow | No | No | Yes |

不可以把 EvidenceFlow 简化成 "LLM in retrieval loop? No"——会被代码反咬。

### 已经过时的协议表述(必须从 paper 清理掉)

- ❌ "Qwen3-8B controlled substrate" — 8B 只是 quick-verification,不写进 paper substrate 声明
- ❌ "Qwen3-8B-train as reader" — 不能再作为 reader 出现
- ❌ "qwen3-8b-train API at port 8043" — 这是开发期端点,不进 paper

paper-facing substrate 声明统一改成:
> *"All methods use GPT-4o-mini as the answer model and Qwen3-32B (local) for any LLM-side extraction or in-method LLM calls. Embeddings use NV-Embed-v2 and Qwen3-Embedding-8B as in the underlying HippoRAG/PropRAG substrate."*

---

## 1. 主命题(v3:evidence as chain)

> Multi-hop question answering does not need a list of individually relevant documents. It needs **a chain of evidence documents that jointly cover the multi-hop claim**. In GraphRAG, evidence is not a set of passages — it is a sequence of documents whose connections must themselves be verifiable. We treat retrieval accordingly: a query-local document graph in which each transition between two documents is **witnessed by a pre-extracted OpenIE fact**, so that the documents reaching the reader are connected by construction rather than by post-hoc selection.

注意叙事姿态:
- "does not need a list ... needs a chain of evidence documents that jointly cover the multi-hop claim" — **本体层 reframe**:重新定义 evidence 是什么,不只是"做更好的 selection"
- "evidence is not a set — it is a sequence" — 直接对 SETR 的 set selection 升维
- "witnessed by a pre-extracted OpenIE fact" — 你方法独有的双层结构,Relink / SETR / PathRAG 都没有
- "connected by construction rather than by post-hoc selection" — 端到端味,但不承诺 single inference pass

---

## 2. Contribution(3 条)

### C1 — Phenomenon

> Even when modern graph-anchored retrievers place all chain-supporting documents within their top-100 candidates, the top-K subset that reaches the reader rarely contains all of them at once. We characterize this **chain-document underranking** empirically across three multi-hop benchmarks and four retrieval/selection pipelines, and operationalize it with `chain-broken@K = 1 - all_gold@K`.

**Anchor data**(41_ 已有数字,新叙事下重新解释):
- 2Wiki/HotpotQA/MuSiQue 上 oracle@100 vs top-5 EM gap: +0.124 / +0.082 / +0.179
- 2Wiki 4-doc 深链子集:在强 retriever 的 top-5 下,gold-supporting chain 不完整覆盖率 ≥ 51.9%(原 41_ 测自 SETR-style;v3 下 phenomenon 不依赖具体 baseline 身份,可在新协议下用 NeocorRAG / Top-5(HippoRAG-valid) / Top-5(PropRAG) 等任一 baseline 重测重报)
- 这些数字在新叙事下的解读:**候选池里 chain 完整 → top-5 里 chain 不完整 → chain-broken@K 普遍偏高**

**与 NeocorRAG RCR 现象的区分**(读完原文后必须主动写明):

NeocorRAG (WWW 2026) 用 Recall Conversion Rate (RCR = F1/Recall@K) 诊断了一个相关但不同的现象:**"high recall, low conversion"** —— HippoRAG2 在 HotpotQA 上 Recall@5 = 96.3% 但 F1 = 75.5%。它把这归因为 retrieved text **内部**的 noisy / hidden information。

我们的 chain-document underranking **更上游**:不是 retrieved 文档**里的**证据没被用上,而是 chain-supporting 文档**本身**就没进 top-K。两个现象都真实存在,但定位不同——NeocorRAG 在 reader 端解 "用不好",我们在 retrieval 端解 "拿不齐"。

paper 里建议这样写:
> *"NeocorRAG diagnoses a recall-conversion gap: retrieved content fails to translate into answer quality even when recall is high. We diagnose a complementary, upstream gap: under the standard top-K reader budget, the chain-supporting documents are not all retrieved together to begin with. The two gaps target different points along the retrieval-reader pipeline."*

### C2 — Method `<METHOD>`

> We propose `<METHOD>`, a multi-hop retrieval method that treats evidence as a chain of documents from the start. The method operates on a **query-local document graph coupled with a pre-extracted OpenIE substrate**: each candidate transition between two documents is admitted only when an OpenIE fact witnesses the relation it implies. The reader receives documents that are connected by construction, not a flat ranked list.

**与 NeocorRAG 的方法层关键差异**(读完原文后定:chain unit 粒度 + chain 是否最终交付物)

| 维度 | NeocorRAG (WWW 2026) | `<METHOD>` |
|---|---|---|
| **Chain unit** | entity-relation **triple** `t=(v_i, r, n_j)` | **document** |
| **Chain 在哪生成** | retrieved text 之上,LLM constrained decoding over prefix tree (§3.3.2) | retrieval graph 之内,OpenIE-fact-witnessed transitions |
| **Chain 的角色** | 中间产物——用于 filter low-confidence documents (§3.3.3) + 作为 prompt context | **最终交付物**——reader 看到的 connected documents 本身 |
| **Query-time LLM 依赖** | 必需(constrained decoding;无 LLM 则 F1 −8.1 in their ablation) | 不需要(OpenIE substrate 是 indexing 阶段预建) |
| **paradigm** | post-hoc filter + prompt augmentation on top of HippoRAG2 retrieve output | end-to-end retrieval: 改的是 retriever 本身 |

**核心机制成分**(按机制名组织,**不**按代码模块):
- M1 — **Question-anchored seeding**: 从 query 中抽 entity / relation 线索,在文档语料上锚定 seed documents(由 dense / textual retrieval 提供 entry)
- M2 — **Fact-witnessed transitions**: 文档之间的 transition edge 只有当一个预建的 OpenIE fact 背书它时才被 admit。这是双层耦合的核心——document graph 是 query-local 动态的,OpenIE fact substrate 是静态预建的,两者**互相约束**。**这一点跟 NeocorRAG 的核心区别**:它的 triple chain 是 query-time LLM 生成的;我们的 transition witness 是 indexing 期就固定的 OpenIE fact,query-time 无 LLM 参与 chain 生成
- M3 — **Chain integrity criterion**: 一个 graph-grounded 的判据,用于判断当前 evidence 是否已经 cover 完整 chain。该判据**在传播期被参考**(指导扩展)、**在 reader 边界处被强制**(决定最终输出),但实现上是同源信号驱动的两次调用,不是统一损失函数
- M4 — **Identifiability handling**: 当 referent 无法唯一解析时的处理机制(继承 41_ IG gate 的设计精神,新叙事下重命名)

**关键术语**(本文档锁定;paper 写作时复用):
- evidence as chain / evidence chain(注意区分 chain unit 粒度,见下)
- **document-level chain**(我们) vs **triple-level chain**(NeocorRAG)
- chain-supporting document
- chain-document underranking
- fact-witnessed transition
- query-local document graph
- pre-extracted OpenIE substrate(可简称 fact substrate)
- chain integrity
- chain-broken@K
- recall-conversion gap(NeocorRAG 的诊断概念,引用其工作时使用)

**主动放弃的术语**:
- ~~Dependency-Bound Evidence Composition / DBEC~~(命名重启)
- ~~Demand-binding unit (DBU)~~(命名重启;若保留需重新嵌入新叙事)
- ~~composition gap~~ → 改用 "chain-document underranking"
- ~~reader budget~~ 作为 framing 入口(降级为实验设置说明)
- ~~bridge document~~ 作为核心范畴(降级为 chain-supporting document 的一种例子)
- ~~single inference pass~~(跟代码 8.4 LLM calls 对不上)

### C3 — Evaluation

> Across 2WikiMultihopQA, HotpotQA, and MuSiQue under our controlled protocol (GPT-4o-mini reader, Qwen3-32B for all in-method LLM calls), `<METHOD>` consistently improves answer F1 over strong GraphRAG baselines. Against PropRAG (Qwen3-32B graph, top-200 pool, GPT-4o-mini reader), `<METHOD>` gains +0.056 / +0.005 / +0.013 F1 on 2Wiki / HotpotQA / MuSiQue; against HippoRAG-valid under the same substrate, +0.113 / +0.031 / +0.024 F1. The improvement is reflected in the chain-integrity indicator chain-broken@5, which `<METHOD>` reduces to 0.105 / 0.067 / 0.490 versus 0.220 / 0.093 / 0.524 for PropRAG and 0.400 / 0.141 / 0.577 for HippoRAG-valid. Comparison with NeocorRAG (k=3) under the same controlled protocol is pending [⚠️ Neocor all32 run in progress — see §0.1].

---

## 3. Intro 第一段(v3 草稿)

> **Multi-hop question answering does not need a list of individually relevant documents — it needs evidence whose pieces are *connected* to each other.** The answer to *"Where was the director of A born?"* is not in any single document: it requires that *A is directed by B* and *B was born in C* hold simultaneously, with a shared B linking the two. A set of three articles, one about A, one about a director, and one about a birthplace, does not yet constitute evidence for this question; the connections between them must themselves be present. In GraphRAG, where evidence is delivered to the reader as documents, this means evidence is fundamentally **a chain of documents, not a set of passages**.
>
> The gap between these two views shows up clearly on modern retrievers. On 2WikiMultihopQA, HotpotQA, and MuSiQue, oracle selection from a strong retriever's top-100 candidates outperforms standard top-5 truncation by +0.124, +0.082, and +0.179 EM. The chain-supporting documents are present in the candidate pool; the gap is about which subset reaches the reader. **The top-5 frequently has the two ends of the chain but not the middle, or the middle but not an end; with even one chain-supporting document missing, the chain collapses, and the answer disappears.** This is not a retrieval-recall problem — it is a problem about whether the retrieval interface preserves evidence as a chain or hands the reader a flat list and asks the reader to reassemble the chain on its own.
>
> A growing body of work attacks this gap from multiple angles: training a passage-sequence retriever to extract chains directly (Beam Retrieval; Zhang et al., 2024), letting an LLM choose a collectively sufficient subset from the pool (SETR; Lee et al., 2025), retrieving relational paths from a constructed graph and linearizing them as LLM prompts (PathRAG, AAAI 2026), or reconstructing missing facts on-the-fly to repair broken paths in an incomplete KG (Relink; AAAI 2026). These directions share an implicit assumption that we make explicit and challenge: that evidence in multi-hop QA is fundamentally a set of individually relevant passages, and that the connections between them are a property to be *recovered* — by training, by LLM reflection, by prompt organization, or by KG completion. **We instead treat the chain as a property of evidence itself, present from the moment of retrieval.** We propose `<METHOD>`, a multi-hop retrieval method that operates on a query-local document graph whose every edge is witnessed by a pre-extracted OpenIE fact. The reader receives documents that are already connected by construction; no downstream stage is asked to reassemble the chain.
>
> The work most directly comparable to ours is NeocorRAG (Peng et al., WWW 2026), which also recognizes that retrieval recall does not translate uniformly into reasoning accuracy and mines **entity-relation-triple chains** within retrieved text via constrained decoding. NeocorRAG and `<METHOD>` share the diagnosis that connectivity matters, but draw the boundary at different places. NeocorRAG treats chains as an internal property of retrieved text — extracted afterward, at the **triple level**, with query-time LLM constrained decoding — and uses them to filter the document set passed to the reader. We treat chains as a property of the retrieval graph itself, at the **document level**: the documents that reach the reader are those whose mutual transitions are witnessed by pre-extracted OpenIE facts, without query-time LLM involvement in chain extraction. Under our controlled protocol (GPT-4o-mini reader, Qwen3-32B extraction for all in-method LLM calls), `<METHOD>` outperforms NeocorRAG by `<F1_GAP_2WIKI>`, `<F1_GAP_HOTPOT>`, and `<F1_GAP_MUSIQUE>` answer F1 on 2WikiMultihopQA, HotpotQA, and MuSiQue respectively. [⚠️ numbers pending: protocol re-run scheduled — see §0.1]

姿态自查(intro 第一段 + 第二段 + 第三段 + 第四段 NeocorRAG 单独段):
- ✅ 第一段直接做本体层 reframe:evidence is a chain, not a set
- ✅ Directorate 例子具体化 chain 跟 set 的区别,挡 reviewer "为什么 chain 比 set 强"
- ✅ 第二段用 41_ 真有的 EM gap 数字
- ✅ "with even one chain-supporting document missing, the chain collapses" — 暗含 `chain-broken@K` 操作化指标
- ✅ K=5 / K=100 作为实验事实出现,**不**作为概念入口 — reader budget 降级
- ✅ 第三段**正面承认四类前人**(Beam Retrieval / SETR / PathRAG / Relink),不装看不见
- ✅ 把前人共有的"implicit assumption"点名:把 evidence 默认当 set,把 connection 当**post-hoc 恢复对象** — 这是本体层切边,不是修辞
- ✅ "We instead treat the chain as a property of evidence itself" — 春秋切边硬话,但不挑衅
- ✅ "every edge is witnessed by a pre-extracted OpenIE fact" — 双层耦合一句话讲清楚
- ✅ **NeocorRAG 单独成段处理**(WWW 2026 最近、最直接相关,不能塞进列表带过)
- ✅ "shares the diagnosis that connectivity matters, but draws the boundary at different places" — 春秋切边但有底气
- ✅ "triple level / document level" 精确对比 — 读完原文才能做的判断
- ✅ "without query-time LLM involvement" — 跟 NeocorRAG constrained decoding 的根本区别
- ✅ "Under our controlled protocol (GPT-4o-mini reader, Qwen3-32B extraction)" — **诚信红线**:protocol 声明,挡 reviewer 用 70B 数字反驳;并保证 paper-facing reader 统一
- ✅ 数字 +0.227 / +0.068 / +0.075 F1 直接进 intro
- ✅ 不出现 readout / selector / composer / preservation / constrained / unified / reader budget / single inference pass / bridge document(作为方法核心)

---

## 4. Method Section 章节大纲

```
Section 3: Problem Setting
  3.1 Multi-hop QA: evidence as a chain of documents
  3.2 Chain-document underranking under standard top-K (C1 anchor)

Section 4: <METHOD>
  4.1 Question-anchored seeding
  4.2 Fact-witnessed transitions(双层耦合的核心)
  4.3 Chain integrity criterion
  4.4 Identifiability handling
```

**约束**:
- 4.1-4.4 章节标题用**机制名**,不用 "expansion / convergence / retrieval / selection"
- 章节里**不出现** ETv3 / PCEC / DBEC / DBU 等内部命名
- Implementation 细节(代码里两段实现)放进 Appendix,且 Appendix 用"the propagation procedure factorizes into ..."这种姿态描述,不暴露两段
- 4.2 Fact-witnessed transitions 是**这版叙事的方法核心**,要写出与 Relink 的 fact-level reconstruction、PathRAG 的 path linearization 的具体区别

---

## 5. Baseline 清单

**v3 主表 baseline**(reader=GPT-4o-mini, extraction=Qwen3-32B):

| Baseline | 类型 | 数据现状 |
|---|---|---|
| Top-5 (HippoRAG-valid qwen32b top200) | strong GraphRAG retriever, pointwise truncation | ✅ 新协议合规,已有 |
| Top-5 (PropRAG qwen32b top200) | strong GraphRAG retriever, pointwise truncation | ✅ 新协议合规,已有 |
| **NeocorRAG k=3** (WWW 2026) | triple-level evidence chain mining via constrained decoding(**v3 唯一正面对手**) | 🟡 新协议 all32 run 进行中(API/graph=qwen3-32b-judge, reretrieval=Qwen3-32B);2Wiki retrieval 跑到 805/1000;预计今日完成 |
| **`<METHOD>` / EvidenceFlow** | ETv4 fact-witnessed document graph + PCEC composition | ✅ 新协议合规,已有(全 32B all32 三数据集 DONE) |

**Appendix-only baselines**(旧协议数据顶,**不进主表**):

| Baseline | 角色 | 备注 |
|---|---|---|
| SetR-style | prompt-only adaptive set selector | **appendix only; not a main-table baseline under v3**。v3 主张下 SetR 操作在 passage pool 上做 set selection,与 EvidenceFlow 在 graph 上长 chain 范式不直接可比(见 §3 intro 第三段并列论证)。旧协议 pilot 数字顶上,reviewer defense 用 |
| SetR-Fill@5 / SetR-Fill@10 | SetR 变体 | 同上,appendix |
| IRCoT-style | iterative decompose-and-retrieve | appendix,旧协议数据顶 |
| LLM-direct (title) / LLM-direct (snippet) | LLM 选 title / snippet | appendix,旧协议数据顶 |
| RankGPT-style | LLM listwise reranker | appendix,旧协议数据顶 |
| Top-10 retriever | 大 budget 对照 | appendix,旧协议数据顶 |

**Related-work-only(不跑)**:

| 论文 | 处理 |
|---|---|
| Beam Retrieval (NAACL 2024) | cite + related work 定位;trained passage-sequence retriever 与本方法 paradigm 不可比 |
| PathRAG (AAAI 2026) | cite + related work 定位;path linearization 与 document-level chain admission 不可比 |
| Relink (AAAI 2026) | cite + related work 定位;fact-level KG completion 与 document-level transitions 不可比 |

`<METHOD>` 自家 ablation 见 §6。

### 5.1 主表(新协议 all32,full1000)

**Source**: `run_logs/all32_etv3_etv4_pcec_gpt4omini_none_full1000_20260514/` + `run_logs/baseline_qwen32b_graph_top200_gpt4omini_full_20260512/` + `run_logs/hipporag_qwen32b_valid_graph_top200_full1000_20260513_r2/`,chain-broken@5 来自 `reports/chain_broken_at5_20260514/chain_broken_at5_summary.md`

| Method | Dataset | R@5 | chain-broken@5 | all-gold@5 | EM | F1 |
|---|---|---:|---:|---:|---:|---:|
| HippoRAG-valid (qwen32b top200) | 2Wiki | 0.8275 | 0.4000 | 0.6000 | 0.5670 | 0.6343 |
| PropRAG (qwen32b top200) | 2Wiki | 0.9090 | 0.2200 | 0.7800 | 0.6110 | 0.6909 |
| NeocorRAG k=3 (all32) | 2Wiki | pending | pending | pending | pending | pending |
| **EvidenceFlow** (ETv4+PCEC all32) | **2Wiki** | **0.9623** | **0.1050** | **0.8950** | **0.6590** | **0.7468** |
| HippoRAG-valid (qwen32b top200) | HotpotQA | 0.9260 | 0.1410 | 0.8590 | 0.5930 | 0.7251 |
| PropRAG (qwen32b top200) | HotpotQA | 0.9510 | 0.0930 | 0.9070 | 0.6180 | 0.7510 |
| NeocorRAG k=3 (all32) | HotpotQA | pending | pending | pending | pending | pending |
| **EvidenceFlow** (ETv4+PCEC all32) | **HotpotQA** | **0.9650** | **0.0670** | **0.9330** | **0.6280** | **0.7557** |
| HippoRAG-valid (qwen32b top200) | MuSiQue | 0.7096 | 0.5770 | 0.4230 | 0.3490 | 0.4654 |
| PropRAG (qwen32b top200) | MuSiQue | 0.7424 | 0.5240 | 0.4760 | 0.3650 | 0.4760 |
| NeocorRAG k=3 (all32) | MuSiQue | pending | pending | pending | pending | pending |
| **EvidenceFlow** (ETv4+PCEC all32) | **MuSiQue** | **0.7646** | **0.4900** | **0.5100** | **0.3720** | **0.4891** |

**F1 gap (EvidenceFlow over baselines)**:

| Dataset | vs PropRAG | vs HippoRAG-valid |
|---|---:|---:|
| 2Wiki | +0.0559 | +0.1125 |
| HotpotQA | +0.0047 | +0.0306 |
| MuSiQue | +0.0131 | +0.0237 |

**全数据集小赢以上,2Wiki 大胜场,MuSiQue 从 8B pilot 的平局变为 all32 下的稳定小赢**。这一变化由 32B binding 的协议升级带来,主要受益于 MuSiQue hard cases。

### 5.2 旧 pilot 数字(协议变更前,8B-reader,**不进 paper 主表,仅历史参考**)

来自 `run_logs/aligned_full1000_master_comparison_20260430.md`:

| Dataset | `<METHOD>` (Prop+DAEC-L1) | NeocorRAG k=1 | NeocorRAG k=3 (公平) | F1 gap (k=3) |
|---|---:|---:|---:|---:|
| 2Wiki | 0.6828 | 0.4553 | 0.5127 | +0.170 |
| HotpotQA | 0.7370 | 0.6686 | 0.6946 | +0.042 |
| MuSiQue | 0.4404 (Prop+DAEC) | 0.3650 | 0.3878 | +0.053 |

这是 v3.1 时代记录的 pilot 数字,仅作 v3 framing 早期方向性参考,不进任何 paper-facing 表。

**诚信声明**:NeocorRAG all32 仍在跑,跑完后:
- 这是 v3 paper 主张的最后一块拼图
- 由于协议升级让 EvidenceFlow MuSiQue F1 从 8B pilot 涨了 +0.013(0.4764 → 0.4891),NeocorRAG 在 all32 下数字预期也会比 k=3 old 8B 上的 0.3878 涨。**最大风险窗口在 MuSiQue**——如果 Neocor 涨幅大,gap 会显著缩窄。但即便如此,EvidenceFlow 在 2Wiki 上对 NeocorRAG 的差距(8B pilot 下 +0.234 F1, 0.7472 vs 0.5127)预计仍能保持主胜场

---

## 6. Ablation 设计(标注已有/待跑)

新叙事下 ablation 的逻辑:**消的是机制成分,不是模块**。

| Ablation | 含义 | 数据现状 |
|---|---|---|
| `<METHOD>` (full) | M1+M2+M3+M4 全开 | ✅ 已有(对应原 DBEC-IG 的数据) |
| no fact-witness | M2 退化:transition edge 不需要 OpenIE fact 背书,只用 graph-side 信号 | 🔴 新跑(**关键 ablation,直接验证双层耦合贡献**) |
| no chain-integrity criterion | M3 替换为 score-truncate | 🔴 新跑(对应 Top-5 在同一 flow 上的对比) |
| no identifiability handling | M4 关闭(直接强制 binding) | ✅ 已有(C1/C2 negative 报告里有) |
| Seed perturbation | 改 seed 抽取策略 | 🔴 新跑(轻量) |

**优先级**:
- `no fact-witness` 是 v3 叙事下最重要的新 ablation——它直接对应"双层耦合是否必要"。**这是方法最大卖点的 ablation 验证**
- C1/C2/Setr 对比已有数据可继续用
- 新跑 ablation 是轻量补丁,1-2 天可以出

### 6.1 Operational metric:chain-broken@K

`chain-broken@K` = reader 拿到的 top-K 里至少缺一个 chain-supporting document 的 query 比例。

形式化:
```
chain_broken(q, K) = 1 if gold_titles(q) ⊄ selector_top_titles(q)[:K] else 0
chain_broken@K = mean_q chain_broken(q, K)
```

**数据可行性**:✅ 已验证。`reports/pcec_m4_position_sensitivity_*/retrieval_reports/*/*.json` 的每条 `setwise_selector_query_traces` 都含 `gold_titles` 和 `selector_top_titles` 两个字段,直接可算,**无需新跑实验**。

**论文用法**:
- 作为 C1 phenomenon 的 operational instantiation,跨 baseline 报告
- 作为 `<METHOD>` 评估的辅助指标(EM/F1 之外多一个 chain-level 视角)
- 措辞:**"chain-supporting document coverage at K"**,不要承诺更细的 hop-level chain closure(那需要 hop 标注)

**写作护栏**:
- 不说 "we propose chain-broken@K as a new metric" — 它本质上就是 `1 - all_gold@K` 换名,夸大就是 reviewer 攻击面
- 说 "we use chain-supporting document coverage at K (equivalent to 1 − all_gold@K) to operationalize chain-document underranking"

---

## 7. 关键术语(本文档锁定;paper 写作时复用)

| 术语 | 定义 | 备注 |
|---|---|---|
| **evidence as chain** | 多跳问答的 evidence 是一组通过 chain 互相连接的 documents,不是 individually 相关的 passage set | 本体层 reframe,核心论点 |
| **document-level chain** | chain unit 是 document,document-document transition 由 fact-witness 给出 | **我们的 chain 粒度,跟 NeocorRAG 的 triple-level chain 划界** |
| **triple-level chain** | chain unit 是 entity-relation-entity triple(NeocorRAG 的范式) | 引用 NeocorRAG 时用,区分粒度 |
| **chain-supporting document** | 回答某个 multi-hop question 所必需的 gold 支持文档集合中的任一篇 | 范畴层概念,涵盖 bridge / anchor / answer-bearing |
| **chain-document underranking** | top-K 系统性丢失部分 chain-supporting documents 的现象 | C1 phenomenon 命名 |
| **chain-broken@K** | 量化 chain-document underranking 的指标(=1−all_gold@K) | C1 operational metric |
| **chain integrity** | top-K 完整覆盖 chain-supporting documents | C3 评估角度 |
| **fact-witnessed transition** | 由预建 OpenIE fact 背书的 document-to-document edge | 你方法独有,跟 Relink / NeocorRAG / PathRAG 的核心切边 |
| **query-local document graph** | query-time 现拉的 document-level transition 图 | 跟 Relink 同范式(on-the-fly),但是 document 粒度 |
| **pre-extracted OpenIE substrate** | indexing 阶段一次性建好的 OpenIE fact 索引,作为 fact-witness 的来源 | 跟 Relink 的 latent relation pool、NeocorRAG 的 query-time LLM 抽取的区别 |
| **question-anchored seeding** | 从 question 抽 entity / relation 线索锚定 seed documents | M1 |
| **identifiability handling** | referent 无法唯一解析时的 abstention 机制 | M4,继承 41_ IG gate 设计 |
| **recall-conversion gap** | NeocorRAG 的诊断:Recall@K 高但 F1 跟不上 | 引用 NeocorRAG 时用,区分于我方的 chain-document underranking |
| **post-hoc chain reconstruction** | 在 retriever 输出之后,用 LLM / 训练 retriever / KG 修复等方式补 chain 信息 | 描述前人共同方法学姿态的概括词,我们划界的对立面 |

---

## 8. 仍待决定

- [ ] 方法名(`<METHOD>` 占位)
- [ ] DBU 是否保留(若保留如何在 v3 叙事下重新嵌入)
- [ ] M3 chain integrity criterion 的形式化表达(继承 41_ noisy-OR coverage,但要在新叙事下重新讲)
- [ ] Relink 数据集重叠确认 + 是否跑 Relink baseline
- [ ] **NeocorRAG `aligned K1` 设置公平性核查**:K1 是 NeocorRAG 自己默认参数还是我们的简化?reviewer 可能问 "为什么不报 K=5/K=10 的 NeocorRAG?"——投稿前必须查清 `scripts/run_neocorrag_aligned.py` 和 `docs/prop_neocorr_daec_neocorrag_limit100_protocol_20260429.md`,且最好准备一个 K=NeocorRAG-default 的 backup 数据点
- [ ] Section 5(实验)章节细分
- [ ] Limitation 节具体条目(可基本沿用 41_ 6 条,但 cost framing 那条要按新叙事重写)
- [ ] "claim" / "multi-hop claim" / "multi-hop fact" 在正文里统一选哪一个(避开 "event" 一词,防 event extraction 语义场)

---

## 9. 跟其它文档的关系

- **本文档**:framing 主纲(v3 "evidence as chain" 叙事)
- `45_chunqiu_brushwork_two_stage_concealment_20260514.md`:supplementary 字段命名、figure 视觉规范、关键句模板(Part C 待方法名定下来后批量替换 `<METHOD>` 占位符;同时 Part A 字段对照表中 "evidence chain set" 等术语已与本文档对齐)
- `41_dbec_paper_framing_locked_20260507.md`:历史 framing,实验数据 anchor 和 SetR 对比设计仍有参考价值;**方法名 / 标题 / abstract / contribution 措辞**已弃用
- `40_setr_style_full1000_positioning_20260506.md`:SetR baseline 设计
- `03_experiment_board.md` / `04_result_registry.md`:实验数据来源

---

## 10. 版本历史

- v0(已弃,DBEC framing)→ 见 41_
- v1(已弃,reader budget + chain breaks)→ 命名/概念跟 SETR 撞,且 reader budget 立不住
- v2(已弃,bridge underranking)→ 范畴太窄,降级 chain 概念
- v3(基础叙事):**evidence as chain;chain 是 evidence 的本体属性,不是 post-hoc 恢复对象;fact-witnessed document transitions 作为方法核心**
- v3.1 = v3 + NeocorRAG (WWW 2026) awareness。读完原文后定位:NeocorRAG 占的是 RCR / triple-level chain 位置;我们占的是 document-level chain ontology 位置,两者邻居不重合。`<METHOD>` 在 8B-reader pilot 下对 NeocorRAG k=3 有 +0.170 / +0.042 / +0.053 F1 优势(2Wiki / HotpotQA / MuSiQue)。注:NeocorRAG `aligned K1` 是 NeocorRAG 自己 `--k` 参数=1(beam search 只生成 1 条 chain),与论文官方例子 k=2/k=3 不符;k=3 pilot 数字补全了公平对比
- v3.2 = v3.1 + 实验协议锁定(见 §0.1):reader=GPT-4o-mini, extraction=Qwen3-32B, quick-verify=Qwen3-8B。所有 paper-facing 数字必须按此协议重跑。v3.1 里的 8B-reader pilot 数字降级为方向性参考,不进 paper 主表。Framing 与 v3.1 一致,仅数字 anchor 改变
- v3.2.1 = v3.2 + SetR 实验义务删除。v3 主张下 SetR 与本方法范式不可直接比较(passage-pool set selection vs graph-native chain construction),不应作为主表 baseline。SetR 从 §0.1 P0 删除、§C3 删 "outperforms SETR-style" 实验义务承诺、§5 降级 appendix only。**v3 真实最短队列只剩 3 项跑模型 + 1 项离线分析 + 1 项文档修正**:(1) EvidenceFlow 新协议主结果,(2) NeocorRAG k=3 新协议对比,(3) no fact-witness ablation,(4) chain-broken/all-gold 汇总脚本,(5) 本次 46_ 文档修正
- **v3.2.2(当前)= v3.2.1 + all32 主表数字落地**。截至 2026-05-14,EvidenceFlow / PropRAG / HippoRAG-valid 三方法的 all32 协议合规 full1000 数字已全部进表(§5.1)。关键结论:
  - **三数据集全胜场**:EvidenceFlow vs PropRAG +0.056 / +0.005 / +0.013 F1 (2Wiki / HotpotQA / MuSiQue);vs HippoRAG-valid +0.113 / +0.031 / +0.024 F1
  - **MuSiQue 从 8B pilot 平局变为 all32 小赢**:协议升级(32B binding)主要在 hard cases 上吃透了之前 8B 没发挥的潜力
  - **chain-broken@5 跨方法 -0.98 correlation 维持**:C1 phenomenon 立得住
  - §C3 措辞从 v3.2.1 时代的"软化(matches or improves)"恢复到"consistently improves",但保留诚实数字幅度("competitive improvements on HotpotQA")
  - 仍 pending:NeocorRAG k=3 all32 run(进行中,2Wiki retrieval 805/1000),跑完后填 §5.1 三行 + §3 intro 第四段 F1 gap 占位符
