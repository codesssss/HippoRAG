# 用一个具体例子讲清楚:HippoRAG 做了什么,EvidenceFlow 多做了什么

> 配合 47_ 的 Fact-Witness 简介一起看
> 写作时间:2026-05-15

---

## 设定一个真实风格的 query

**Query**:*"In which country is the capital of the country where the director of Saving Private Ryan was born located?"*
(直白点:救援大兵的导演出生在哪个国家,那个国家的首都在哪个国家?)

**正确答案**:USA(Spielberg → Cincinnati → USA)

**正确证据链(2 hop)**:

```
Doc A: "Steven Spielberg was born in Cincinnati, Ohio."
   → entity: Cincinnati
Doc B: "Cincinnati is a major city in the United States of America."
   → entity: United States of America
```

reader 必须**同时拿到 Doc A 和 Doc B**,才能拼出答案。

---

## 接下来对比两种系统在这个 query 上**实际做的事**

假设 corpus 里有这些相关文档:

```
Doc A: "Steven Spielberg was born in Cincinnati, Ohio."
Doc B: "Cincinnati is a major city in the United States of America."
Doc C: "Steven Spielberg directed Saving Private Ryan in 1998."  ← 跟 query 表面最相关
Doc D: "Tom Hanks starred in Saving Private Ryan."               ← distractor (跟 director 无关)
Doc E: "Cincinnati Reds are a baseball team."                    ← distractor (跟 query 无关)
```

---

## 一、HippoRAG 2 在这个 query 上做什么

### Offline indexing(已经做完)

HippoRAG 把 corpus 抽 OpenIE,建一张**全局静态图**:

```
phrase_node 集合:
  Spielberg, Cincinnati, Ohio, USA, Saving Private Ryan, Tom Hanks, Cincinnati Reds, ...

passage_node 集合:
  Doc A, Doc B, Doc C, Doc D, Doc E, ...

edges:
  - passage→phrase:Doc A 提到 Spielberg, Cincinnati, Ohio
                   Doc B 提到 Cincinnati, USA
                   Doc C 提到 Spielberg, Saving Private Ryan
                   ...
  - phrase→phrase synonymy:e.g., USA ↔ United States of America
```

注意:**HippoRAG 不存 (h, r, t) 三元组本身,只把它们解构成 phrase node + passage→phrase 边**。Relation 信息丢了。

### Query time

**Step 1**:从 query 抽 entities:
```
Spielberg, Saving Private Ryan
```

**Step 2**:在**整个 corpus-level 图**上跑 PPR,seed 是 query entities。

PPR 会:
- 给 query entity(Spielberg, Saving Private Ryan)及其邻居打分
- 通过 phrase→phrase synonymy 扩散
- 通过 passage→phrase 边把分数传给 passage

**Step 3**:passage 按 PPR 分数排序,取 top-K。

### 排序结果(预期)

```
1. Doc C (高分:跟 Spielberg 和 Saving Private Ryan 都连)
2. Doc A (高分:跟 Spielberg 连)
3. Doc D (中分:跟 Saving Private Ryan 连,通过 Tom Hanks 间接)
4. Doc B (低-中分:跟 Cincinnati 连,但 Cincinnati 不是 query entity!)
5. Doc E (低分:跟 Cincinnati 连)
```

### HippoRAG 的真实问题

**Doc B 排名危险**——它是答案必需的桥文档,但 query 里**没有 Cincinnati 这个 entity**!Cincinnati 是中间桥实体,query 不知道它。

PPR 能不能把分数传到 Doc B,**完全靠 phrase→phrase synonymy 扩散步数**:
- Spielberg(seed) → Cincinnati(in Doc A 提到) → Doc B
- 如果 PPR alpha 设置不当 / 步数不够 / synonymy 边不密,**Doc B 可能掉到 top-5 之外**

这就是 HippoRAG 经典的 **"chain-broken" 问题**:中间桥文档因为不是 query entity 的直接邻居,可能丢失。

---

## 二、EvidenceFlow 在同一个 query 上多做了什么

### Offline indexing

跟 HippoRAG **完全一样**,但**多保留一件事**:**保留 OpenIE 三元组本身,带 provenance**。

```
fact-witness substrate F = {
  (Spielberg, born_in, Cincinnati)         provenance: Doc A
  (Cincinnati, located_in, USA)            provenance: Doc B
  (Spielberg, directed, Saving Private Ryan) provenance: Doc C
  (Tom Hanks, starred_in, Saving Private Ryan) provenance: Doc D
  (Cincinnati Reds, is_a, baseball team)   provenance: Doc E
  ...
}
```

**HippoRAG 把 (h, r, t) 解构成图边后丢掉 r**;**EvidenceFlow 把整组 (h, r, t) 留下来**,作为 doc-to-doc transition 的"见证"用。

### Query time

**Step 1**:跟 HippoRAG 一样,从 query 抽 entities:
```
A(q) = { Spielberg, Saving Private Ryan }
```

**Step 2**:**这里开始不一样**——构造 **query-local graph G_q**(不是在全局图上跑 PPR)。

构造 G_q 时,每条 doc-to-doc 边都要**找 fact-witness**:

```
候选边:Doc A → Doc B 能不能进 G_q?

Look up F:
  (Spielberg, born_in, Cincinnati)   from Doc A   ← 包含 Spielberg(query entity)
  (Cincinnati, located_in, USA)      from Doc B
  
共享 entity:Cincinnati
Cincinnati 是 Spielberg(query entity)的邻居 → 跟 query 相关
  
✅ 准入:Doc A → Doc B 这条边有 fact-witness
```

```
候选边:Doc C → Doc D 能不能进 G_q?

Look up F:
  (Spielberg, directed, Saving Private Ryan)  from Doc C
  (Tom Hanks, starred_in, Saving Private Ryan) from Doc D
  
共享 entity:Saving Private Ryan
但这条 chain 跟 query 问的"导演出生地"无关
  
🟡 准入但低权重(通过 fact-witness,但 chain 方向不对)
```

```
候选边:Doc B → Doc E 能不能进 G_q?

Look up F:
  Doc B 的 fact:(Cincinnati, located_in, USA)
  Doc E 的 fact:(Cincinnati Reds, is_a, baseball team)
  
共享 entity:Cincinnati(浅层共指)
但 Cincinnati Reds ≠ Cincinnati (city)
  
🔴 不准入(实体共指弱,无 query-relevant chain)
```

**Step 3**:在 G_q 上跑 **local-PPR**(不是全图 PPR),seed 是 query entities + entry docs。

**Step 4**:passage 按 local-PPR 分数排序。

### 排序结果(预期)

```
1. Doc A (高分:fact-witnessed 连接到 G_q 中心)
2. Doc B (高分! ← 关键差异)
   理由:Doc A → Doc B 这条边在 G_q 里被 fact-witness 准入
        Cincinnati 这个 query 不知道的桥实体被 fact 显式连起来
        local-PPR 在 G_q 上跑时,Doc B 通过 Doc A 拿到高分
3. Doc C (高分:跟 Spielberg 直接连)
4. Doc D (低分:fact 存在但 chain 方向跟 query 无关)
5. Doc E (极低分:fact-witness 准入失败)
```

### EvidenceFlow 多做的事(具体到这个例子)

| 动作 | HippoRAG | EvidenceFlow |
|---|---|---|
| **保留 OpenIE 三元组本身** | ❌ 解构成图边丢失 r | ✅ 保留完整 (h, r, t) + provenance |
| **PPR 跑在哪** | corpus-level 全局图 | **query-local 子图 G_q**(只含 fact-witnessed 边) |
| **doc-to-doc 边的来源** | passage→phrase→passage 间接二跳 | **doc-to-doc 直接边,由 fact-witness 准入** |
| **Cincinnati 桥实体处理** | 靠 PPR 扩散步数,可能丢 | **Doc A→Doc B 直接边,Cincinnati 显式作为 fact argument** |
| **Doc D / Doc E 这种 distractor** | 也通过 phrase→passage 拿到分数 | **fact-witness 准入失败或低权重,不进 G_q 或低分** |

---

## 三、把这个例子升级成"在 HippoRAG 上多做了什么"的精确陈述

### EvidenceFlow vs HippoRAG 2 的真实差别(只有这两条)

**差别 1:OpenIE 的角色**

- HippoRAG 2:OpenIE 三元组**解构**成 `(h)-passage_link-(t)` 二部图,relation `r` **丢失**,`(h, r, t)` 整体不保留
- EvidenceFlow:OpenIE 三元组**保留完整 (h, r, t)**,作为 **doc-to-doc transition 的 fact-witness**

**差别 2:PPR 跑在哪**

- HippoRAG 2:在 **corpus-level 全局图**上跑 PPR,seed = query entities
- EvidenceFlow:**先用 fact-witness 构造 query-local 子图 G_q**,在 G_q 上跑 local-PPR

### 这两条差别带来的具体好处

| HippoRAG 2 的脆弱点 | EvidenceFlow 的应对 |
|---|---|
| 桥实体不在 query 里 → PPR 扩散可能不到桥文档 | fact-witness 把 doc-to-doc 直接边显式建立,桥实体作为 fact argument 被显式追踪 |
| 全图 PPR 受 distractor 干扰 | local-PPR 只在 fact-witness 准入的子图上跑,distractor 边被 admission 规则过滤 |
| relation `r` 丢失 → 无法判断 chain 方向 | fact 完整保留 → admission 时可以用 r 判断是否 query-relevant |

---

## 四、所以 fact-witness 这个概念到底"多做了什么"

**一句话**:

> HippoRAG 把 OpenIE 当**建图原料**(用完丢);EvidenceFlow 把 OpenIE 当**doc-to-doc 跳转的可追溯依据**(全程保留)。

**这个差别让 EvidenceFlow 在 chain-broken case 上有 baseline 不可达的提升**——这是 retrieval 层面的真实改进,不是后处理或 reranking。

**如果砍掉 fact-witness 概念**:
- §4.1 退回讲 "我们用 HippoRAG-style OpenIE indexing"(没 novelty)
- §4.2 admission rule 失去依据(只能说 "我们建一个子图",但建图准则就是 fact-witness 这个概念,砍掉它就只剩 entity 共现这种弱准则)
- 论文跟 HippoRAG 2 的差别**从两条降到一条**(只剩 local-PPR)

**如果保留 fact-witness 概念**:
- §4.1 是真实贡献:不是新算法,而是把 OpenIE 角色重定位
- §4.2 admission rule 有论文级依据:每条边都有可审计的 fact 见证
- 论文的真实增量(vs HippoRAG)是清楚的:**保留三元组完整结构 + query-local 子图 PPR**

---

## 五、给导师看的最终建议

如果导师觉得 "fact-witness" 词不学术,**不影响整个机制**——可以叫:
- "OpenIE-grounded transition" (强调来源 + 用途)
- "preserved relational layer" (强调保留完整结构)
- "transition-grounding fact set" (强调作 transition 依据)
- 任何导师推荐的术语

**但请保留这个概念在 §4.1 的位置**——它是 EvidenceFlow vs HippoRAG 的真实增量之一。砍掉这个概念,论文的 graph-based novelty 会从两条降到一条,撑不住 main paper。
