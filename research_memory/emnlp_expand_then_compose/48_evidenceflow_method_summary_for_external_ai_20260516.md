# EvidenceFlow 方法总结(用于让外部 AI 帮写 §2 Related Work)

> 写作时间:2026-05-16
> 用途:把这份文档复制给其他 AI,让它根据这份方法描述帮你写 §2 Related Work

---

## 1. 任务 setting

我们写一篇 **多跳问答 (multi-hop question answering)** 的论文,投 ACL/EMNLP main paper。

任务输入:
- 一个问题 $q$(通常需要跨多篇文档推理才能回答)
- 一个语料库 $\mathcal{D}$(几万到几十万篇文档)

任务输出:
- 一个 ranked evidence list $R_q$(对 $\mathcal{D}$ 中段落的排序)
- 下游 reader(LLM)消费 $R_q$ 的 top-K 段落,生成最终答案

评估数据集:**HotpotQA, 2WikiMultihopQA, MuSiQue**(标准多跳 QA benchmark)

---

## 2. 我们的方法:EvidenceFlow

### 2.1 核心思想(一句话)

> 把多跳检索建模成"在 query 诱导的子图上传播证据流"——离线建多层 OpenIE 图,query 时跑 local-PPR 给段落打分,再用关系扩展和 noisy-OR coverage 做检索增强。

### 2.2 三段式结构

#### §4.1 Offline Indexing: Document-Grounded Multi-Layer OpenIE Graph

**离线建图**:
- 用语言模型对每篇文档抽 OpenIE facts $f = (h, r, t)$
- 保留每个 fact 的 provenance $\mathrm{src}(f) \in \mathcal{D}$(来自哪篇 doc)
- 整个 fact 集合记为 $\mathcal{F}$

**多层图结构** $\mathcal{G}_{\mathcal{D}}$:
- **2 种节点类型**:
  - passage nodes(每篇 doc 一个)
  - phrase nodes(每个不同 entity surface form 一个)
- **3 种边类型**:
  - passage-phrase 边(doc 提到 entity)
  - relational 边(由 fact 诱导,带 relation label $r$ 和 provenance)
  - synonymy 边(dense 嵌入相似的 entity 别名)

**特点**:
- 图离线建一次,跨所有 query 复用
- query time 不再做 OpenIE 抽取或 fact 生成
- 沿用 HippoRAG-style indexing,不在 OpenIE extractor 本身做创新

#### §4.2 Evidence-Driven Local Graph Retrieval

**关键科学对象**:
- **Evidence flow** = $\pi_q$,一个 query-conditioned mass distribution
- 从 question anchor 出发,沿 provenance-grounded edges 传播
- 在 query-local 子图 $G_q$ 上(不是全图)计算

**Query-local 子图构造**:
- 用受控 LLM 从 question 抽 entity anchors $A(q)$(LLM 只作用 question,不作用 retrieved doc)
- 用 dense retriever 给 entry signal $\mathrm{Entry}(q)$
- 子图初始化:anchor 对应的 phrase nodes ∪ entry passages ∪ provenance 含 anchor 的 passages
- 沿 $\mathcal{G}_{\mathcal{D}}$ 的边做有界跳数扩展

**Local Personalized PageRank**(关键技术细节):
- **PPR 跑在 fact units 上**(不是 phrase/passage 上)
- 两个 fact $f_i, f_j$ 共享 activated entity 时相连
- fact $f$ 跟 $\mathrm{src}(f)$ 相连
- Seed 分布 $s_q$ 放在 argument 包含 anchor 的 fact 上

**核心公式 (Eq. 1)**:
$$\pi_q = \alpha\, s_q + (1-\alpha)\, P_q^{\!\top}\, \pi_q$$

**Passage 分数读出 (Eq. 2)**:
$$s_{\mathrm{ppr}}(d \mid q) = \frac{1}{\sqrt{|\mathcal{F}_d|}} \sum_{f \in \mathcal{F}_d} \pi_q(f), \quad \mathcal{F}_d = \{f \in \mathcal{F}: \mathrm{src}(f) = d\}$$

(passage 分数 = 来自该 doc 所有 fact 的 PPR mass 之和,除以 sqrt(fact 总数) 归一)

#### §4.3 Retrieval Enhancement with Relational Evidence

**针对 local-PPR 的两个失败模式**:
1. 处于多跳链上但远离 anchor 的桥文档,得分被压低
2. top 排名被单一实体邻域聚集,其他 query 所需 entity/relation 覆盖不足

**Enhancement 1: Relation-grounded expansion**
- 对每个高分节点,沿关系边扩展拉桥文档
- 候选分数 = $\pi_q(u)$ + 关系短语 $r$ 跟 question 的 dense 相似度

**Enhancement 2: Coverage-aware reranking** (用 noisy-OR coverage)
- 用受控 LLM 从 question 抽 question-side requirements $B(q) = \{b_1, b_2, ...\}$
- 每个 requirement 是 entity 或 relation slot
- 每个 passage $d$ 对每个 requirement $b$ 关联贡献 $\phi_{d,b} \in [0, 1]$(dense 相似度)
- **Noisy-OR coverage 公式 (Eq. 3)**:
  $$\mathrm{cov}_q(R \mid b) = 1 - \prod_{d \in R}(1 - \phi_{d,b})$$
- 贪心选择最大化跨所有 requirement 的 coverage 边际增益
- 输出:final ranked evidence list $R_q$,下游 reader 可消费任意 prefix

---

## 3. 关键设计选择(写 related work 时要强调的差异)

| 设计 | 我们怎么做 | 跟前人不同 |
|---|---|
| **OpenIE 三元组的角色** | 保留完整 (h, r, t) + provenance,作 indexing 阶段图的关系结构 | HippoRAG 解构 (h,r,t) 丢失 r,只用 entity 共现;ToG/Relink 把 (h,r,t) 当 KG reasoning substrate |
| **PPR 跑在哪** | query-local 子图上(只含相关邻域) | HippoRAG 跑在全 corpus 图上;PropRAG 不用 PR |
| **Chain 怎么形成** | 在 indexing 时已经准备好(关系边带 r 和 provenance),retrieval 时通过 PPR 形成 | NeocorRAG query time 用 LLM 在 retrieved text 上生成 chain |
| **Reranking** | 用 noisy-OR coverage 在 question-side requirements 上做 | MR 用 token diversity;submodular IR 用 set function;一般 reranker 用 cross-encoder 重新打分 |
| **Reader budget K** | 不是 method 目标,只是输出截断设置 | 一些 selection-based 方法把 K 当输入 |

---

## 4. 我们方法对应的相关工作方向(给写 §2 用)

### 主线 1:Graph-based Retrieval for Multi-hop QA(最直接对手)

- **HippoRAG (Gutiérrez et al., NeurIPS 2024)**:passage + phrase 节点 + 全图 PR,我们直接对照
- **HippoRAG 2 (Gutiérrez et al., 2025)**:HippoRAG 升级版,我们沿用其 OpenIE indexing
- **PropRAG (You et al., 2024)**:用 proposition 替代 triple,beam search 做 path discovery
- **GraphRAG (Edge et al., Microsoft Research, 2024)**:用 community summary
- **HGRAG / HyperGraphRAG**:entity hypergraph + cross-granularity diffusion

### 主线 2:KG Reasoning for QA

- **ToG / Think-on-Graph (Sun et al., ICLR 2024)**:LLM-guided KG traversal
- **Relink (Huang et al., AAAI 2026)**:Reason-and-construct,query-time fact 重建
- **KG-RAG / KAPING / GoR**:KG-augmented retrieval 一族

### 主线 3:Query-time Evidence Chain Generation

- **NeocorRAG (Peng et al., WWW 2026)**:query-time constrained decoding 抽 entity-relation chain 作 prompt 增强
- **Self-Ask / IRCoT (Trivedi et al., ACL 2023)**:interleaved reasoning + retrieval

### 主线 4:Personalized PageRank for IR

- **Topic-Sensitive PageRank (Haveliwala, WWW 2002)**:经典 PR
- **PathRetrieve / GoR**:在图上做 PR-like 排序

### 主线 5:Coverage-aware / Diversity-aware Reranking

- **MMR / Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR 1998)**:经典 diversity reranking
- **Submodular IR (Lin & Bilmes, NAACL 2011)**:用 submodular function 做 coverage
- **Noisy-OR Aggregation in IR / VQA**:noisy-OR 在 IR 早有应用

---

## 5. 我们方法的差异定位句(给写 §2 用)

> **vs HippoRAG**:Shared OpenIE indexing, but EvidenceFlow runs PPR on a query-local subgraph (not the full corpus graph) and adds a retrieval-enhancement step that addresses two failure modes of vanilla local PPR — bridge passages with low propagated mass, and top rankings dominated by single-entity neighborhoods.

> **vs PropRAG**:Both use OpenIE-derived structure, but PropRAG performs beam search over proposition paths; EvidenceFlow propagates PR mass over fact units and uses noisy-OR coverage for reranking.

> **vs ToG/Relink (KG reasoning)**:These approaches treat the extracted relations as the reasoning substrate itself, with the LLM traversing or repairing the KG at query time. EvidenceFlow keeps OpenIE on the indexing side; the reader still consumes natural-language passages, and no query-time KG traversal or fact repair is performed.

> **vs NeocorRAG**:NeocorRAG invokes a language model on retrieved text to mine entity-relation chains as prompt augmentation. EvidenceFlow avoids this query-time mining; the relational structure is built once at indexing time, and chain-aware retrieval is computed by graph propagation rather than by chain generation.

> **vs MMR / Submodular IR**:Shared coverage / saturation principle, but our coverage units are question-side entity-relation requirements (extracted from $q$ via a controlled LLM), and saturation is computed via noisy-OR over dense-similarity contributions $\phi_{d,b}$ rather than over query tokens.

---

## 6. 我们方法的核心 contribution 句(intro 风格,可以借鉴)

> We propose **EvidenceFlow**, a graph-based retrieval framework for multi-hop question answering. EvidenceFlow has three components: (1) an offline document-grounded multi-layer OpenIE graph that supplies entity anchors and relational structure; (2) an evidence-driven local graph retrieval procedure based on personalized PageRank over a query-induced subgraph; and (3) a retrieval-enhancement step that performs relation-grounded expansion and noisy-OR coverage-aware reranking on top of the local ranking. The central object is a query-conditioned mass distribution $\pi_q$ — the *evidence flow* — propagated from question anchors through provenance-grounded edges. Across HotpotQA, 2WikiMultihopQA, and MuSiQue, EvidenceFlow improves over strong graph-based baselines by addressing two failure modes of vanilla local PPR: chain-broken bridge passages and top-ranking entity-neighborhood collapse.

---

## 7. 写作纪律(请帮写 §2 时务必遵守)

1. **不夸大别家工作的 weakness** — 写 "differs from X in that ..." 不写 "X has the limitation that ..."
2. **每段引出我们的差异定位** — 段尾用一句 "In contrast, EvidenceFlow ..." 收
3. **术语跟方法对齐** — 用 "OpenIE facts"(不是 "triples"), "fact units"(不是 "fact nodes"), "evidence flow"(不是 "evidence chain")
4. **诚信措辞** — 不写 "LLM-free retrieval" 这种绝对话(我们 query time 也用 LLM 做 question-side anchor / requirement 抽取);可以写 "without query-time LLM extraction over retrieved documents"
5. **篇幅 0.8-1.2 页双栏**(750-900 词);三段式最稳,每段 250-300 词

---

## 8. 推荐的 §2 三段结构

```
§2.1 Graph-based Retrieval for Multi-hop QA  (~300 词)
  - HippoRAG / HippoRAG 2(共享 OpenIE indexing,差异:query-local + enhancement)
  - PropRAG(proposition-based,差异:不做 beam search,用 PR + coverage)
  - GraphRAG / HGRAG(community/hypergraph,差异:更轻量)
  - 段尾切边:"In contrast, EvidenceFlow ..."

§2.2 KG Reasoning and Query-time Chain Mining  (~250 词)
  - ToG / Relink(KG reasoning substrate,差异:OpenIE 留 indexing 侧)
  - NeocorRAG(query-time chain generation,差异:chain 来自 indexing 时已建好的图)
  - 段尾切边:"EvidenceFlow does not perform query-time fact generation or chain mining over retrieved text..."

§2.3 Personalized PageRank and Coverage-aware Reranking  (~200 词)
  - Topic-Sensitive PR(我们 local-PPR 的 IR lineage)
  - MMR / submodular IR(我们 noisy-OR coverage 的 reranking lineage)
  - 段尾点出我们的具体设计:"local subgraph + fact-unit propagation" / "noisy-OR over question-side requirements"
```

---

## 9. 必须 cite 的文献(BibTeX 条目)

```bibtex
@inproceedings{angeli2015leveraging,
  author = {Angeli, Gabor and Premkumar, Melvin Johnson and Manning, Christopher D.},
  title = {Leveraging Linguistic Structure for Open Domain Information Extraction},
  booktitle = {ACL},
  year = {2015}
}

@inproceedings{gutierrez2024hipporag,
  author = {Gutiérrez, Bernal Jiménez and others},
  title = {HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models},
  booktitle = {NeurIPS},
  year = {2024}
}

@article{gutierrez2025hipporag,
  author = {Gutiérrez, Bernal Jiménez and others},
  title = {From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)},
  journal = {arXiv:2502.14802},
  year = {2025}
}

@inproceedings{sun2024tog,
  author = {Sun, Jiashuo and others},
  title = {Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph},
  booktitle = {ICLR},
  year = {2024}
}

@inproceedings{huang2026relink,
  author = {Huang, ... and others},
  title = {Relink: Reason-and-Construct for Multi-hop Question Answering},
  booktitle = {AAAI},
  year = {2026}
}

@inproceedings{peng2026neocorrag,
  author = {Peng, ... and others},
  title = {NeocorRAG: ... Evidence Chain Mining for ...},
  booktitle = {WWW},
  year = {2026}
}

@inproceedings{haveliwala2002topic,
  author = {Haveliwala, Taher H.},
  title = {Topic-Sensitive PageRank},
  booktitle = {WWW},
  year = {2002}
}

@inproceedings{carbonell1998mmr,
  author = {Carbonell, Jaime and Goldstein, Jade},
  title = {The Use of MMR, Diversity-based Reranking for Reordering Documents and Producing Summaries},
  booktitle = {SIGIR},
  year = {1998}
}
```

(其它 cite 由写作 AI 根据需要补充)

---

## 10. 给外部 AI 的指令(直接 prompt)

> 根据上面 §1-§9 的方法描述,请写一份 ACL/EMNLP main paper 风格的 §2 Related Work,要求:
> 1. 三段式(Graph-based Retrieval / KG Reasoning + Chain Mining / PPR + Coverage Reranking)
> 2. 篇幅约 800-900 词(双栏约 1 页)
> 3. 每段段尾用 "In contrast, EvidenceFlow ..." 引出差异
> 4. 引用上面给出的 8 个核心文献,可以再补充你认为必要的相关工作
> 5. 严格遵守 §7 写作纪律(不夸大别家、术语对齐、诚信措辞)
> 6. 用 LaTeX 格式输出(`\section{Related Work}`, `\subsection`, `\paragraph`, `\citep{}` 等)
