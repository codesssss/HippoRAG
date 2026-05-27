# 春秋笔法:两阶段实现的论文叙事规范

Last updated: 2026-05-14

## 用途

本文档**只补 41_ 没覆盖的三件事**:

1. **代码字段 ↔ 论文叙事**的命名映射(防 supplementary / report 暴露内部命名)
2. **Method figure** 的视觉规范(防作图时退化为两 box 串联)
3. **关键句模板**(41_ 给的是 ❌/✅ 替换表,本文档给可直接复用的完整段落)

适用范围:
- Paper draft 写作(intro / method / ablation / related work)
- Supplementary materials、report 表头、JSON dump、figure caption
- 任何对外可见的 paper-facing artifact

**不适用于**:
- 代码注释、commit message、内部 changelog——内部继续用 ETv3 / PCEC
- 团队内部讨论
- 任何 framing / 方法名 / claim / 章节结构决策——这些**全部以 41_ 为准**

## 与其它文档的关系

| 文档 | 管什么 |
|---|---|
| `41_dbec_paper_framing_locked_20260507.md` | **主纲**:方法名、标题、4 个 contribution、章节结构、claim 边界、locked-out language、6 条 limitation、abstract |
| `45_`(本文档) | **附则**:字段命名、figure 视觉规范、关键句模板 |
| `40_setr_style_full1000_positioning_20260506.md` | SetR baseline 设计与定位 |
| `00_north_star.md` | 入口指针 |

**冲突仲裁**:本文档任何内容若与 41_ 冲突,**以 41_ 为准**,本文档应同步修订。

## 41_ 已锁定的命名(本文档不重复决策,仅引用)

| 论文用语 | 含义 |
|---|---|
| **DBEC** | 方法名(Dependency-Bound Evidence Composition) |
| **DBEC-IG** | Identifiability-Gated 变体 |
| **DBEC-nobinding** | 去掉 binding 机制的 ablation |
| **DBU**(demand-binding unit) | 核心新概念:一个 (demand, entity binding) 对 |
| **Composition gap** | oracle@100 vs top-5 的 F1/EM 差距 |
| **Under-selection failure mode** | prompt-only 选择器选少于 gold 支持数的失败模式 |
| **Identifiability gate** | 当 referent 无法唯一解析时的 abstention 机制 |
| **SetR-style / SetR-Fill@5 / SetR-Fill@10** | baseline 命名 |
| **Controlled substrate** | 同 LLM、同 embedding、同 pool 的对照设置 |

**所有 paper-facing 写作都用以上术语**。本文档下面所有句式模板都基于这套术语。

---

# Part A:代码字段 → 论文叙事映射

## 一、核心原则

DBEC 在代码层面客观上是两段实现(ETv3 expander + PCEC composer)。
41_ 的 Section 4 章节结构按**机制成分**组织(4.1 demand decomposition / 4.2 binding /
4.3 noisy-OR coverage / 4.4 IG gate),已经把这两段实现彻底打散到机制视角。

**但**:supplementary、reports/ 下的 JSON、scripts/ 输出的文件名仍然带着 `etv3` / `pcec` /
`dbec` 等内部命名。如果 reviewer 在 supplementary 里看到 `pcec_admission_trace.json` 或
`etv3_native_full1000_audit.md`,前面所有 paper 叙事努力归零。

**本节规则**:**所有送审 supplementary 一律走 rename layer,使用 41_ 锁定术语;
代码内部、commit、本地 reports 不动。**

## 二、字段命名对照表

### B.1 核心数据对象

| 代码内部(保留) | 送审 supplementary 用语 | 论文正文用语 |
|---|---|---|
| `etv3_pool` / `etv3_candidates` | `candidate_pool` | retrieval pool / candidate pool |
| `etv3_output` | `pool_after_expansion`(若必须暴露) | (paper 不直接出现) |
| `pcec_selected` / `pcec_topk` | `selected_evidence_set` | the selected evidence set / final K=5 evidence |
| `pcec_admission_trace` | `selection_trace` | (paper 不直接出现) |
| `pcec_binding_cache` | `binding_cache` | binding assignments |
| `dbec_requirements` | `query_demands` | atomic demands |
| `dbec_demand_graph` | `demand_graph` | demand graph G_q |
| `pcec_coverage_score` | `coverage_score` | noisy-OR coverage |

### B.2 方法 variant 命名

| 代码内部(保留) | paper-facing(已在 41_ 锁定) |
|---|---|
| `dbec_full` / `daec_full` | **DBEC** |
| `dbec_selective` / `dbec_titleuniq` | **DBEC-IG** |
| `dbec_nobinding` / `daec_nobinding` | **DBEC-nobinding** |
| `setr_faithful` / `setr_local` | **SetR-style** |
| `setr_fill5` | **SetR-Fill@5** |
| `setr_fill10` | **SetR-Fill@10** |
| `top5_baseline` / `top10_baseline` | **Top-5** / **Top-10 retriever** |
| `ircot_local` | **IRCoT-style** |
| `llm_direct_title` / `llm_direct_snippet` | **LLM-direct (title)** / **LLM-direct (snippet)** |

### B.3 报告与日志命名

| 代码内部(保留) | 若需提交为 supplementary 时的呈现 |
|---|---|
| `run_logs/daec_llm_wiki_title_proprag_full1000_*/` | `runs/dbec/<dataset>_<pool>_<date>/`(rename layer 重打包) |
| `run_logs/daec_selective_titleuniq_*/` | `runs/dbec_ig/<dataset>_<pool>_<date>/` |
| `reports/etv3_native_full1000_audit_*/` | `reports/pool_audit_<date>/` |
| `reports/pcec_m4_position_sensitivity_*/` | `reports/budget_sensitivity_<date>/` |
| `reports/setr_faithful_proprag_full1000_*/` | `reports/setr_style_<pool>_<date>/` |

### B.4 字段命名雷区(supplementary 必须重命名)

以下字段名一旦出现在 supplementary JSON / Excel / md report 里,**直接暴露两阶段**:

| 雷区字段 | 必须改成 |
|---|---|
| `expander_output` / `expand_stage_*` | `pool_after_step_1` 或直接合并进上游 |
| `composer_output` / `compose_stage_*` | `selected_evidence_set` |
| `stage1_*` / `stage2_*` | 按 41_ Section 4 机制名重命名(demand / binding / coverage / gate) |
| `pre_compose_*` / `post_compose_*` | `before_selection` / `after_selection`(且仅用于 baseline 对比) |
| `etv3_*` / `pcec_*` 任何前缀 | 按 B.1/B.2 表替换 |

### B.5 Rename layer 实现位置(建议)

不在每个 export script 里手改,在 export 出口集中加一层:

- `scripts/export_for_supplementary.py`(待新建):读取 reports/ 下原始文件,按 B.1-B.4
  对照表重写 key 和文件名,输出到 `supplementary_packaged/`
- 投稿前**唯一**走这条 pipeline 的产物才进 supplementary

代码、commit、本地实验产物**不动**。

---

# Part B:Method Figure 视觉规范

## 一、核心约束

41_ Section 4 把方法按机制成分分(demand decomposition / binding / coverage / IG gate),
**figure 也按机制成分画,不按 ETv3/PCEC 两段画**。

## 二、Figure 1(主方法图)规范

### ❌ 自爆画法

```
[Query] → [Retriever (HippoRAG)] → [pool] → [ETv3 expand] → [PCEC compose] → [reader]
```

或者:

```
[Query] → [Stage 1: Expansion] → [Stage 2: Composition] → [Reader]
```

两种画法都把 DBEC 直接定性为两阶段方法。

### ✅ 正确画法:按机制成分组织,而不是按代码段组织

布局示意(交付 illustrator / TikZ 时按此结构):

```
                        Query q
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │  DBEC                                             │
   │                                                   │
   │   ① Demand decomposition                          │
   │      r₁ → r₂ → r₃   (demand graph G_q)           │
   │                                                   │
   │   ② Entity-anchored binding                       │
   │      (r₁, e_a) (r₂, e_b) ...  ← DBU instances    │
   │                                                   │
   │   ③ Noisy-OR coverage over DBUs                   │
   │      arg max  P(coverage | E ⊆ pool, |E| = K)    │
   │                                                   │
   │   ④ Identifiability gate (DBEC-IG)                │
   │      abstain when binding non-unique              │
   │                                                   │
   │   Retrieval pool P (fixed, provided upstream)     │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
                  Selected evidence E
                          │
                          ▼
                        Reader
```

### 关键规则

1. **DBEC 是一个大框**,内部按 41_ Section 4 的四个机制成分(①demand ②binding
   ③coverage ④gate)分列,不画 "ETv3 box" 和 "PCEC box" 两个独立方框
2. **Retrieval pool P 画在框内作为输入条件**,标注 "fixed, provided upstream"——明确
   说明 pool 不是 DBEC 自己造的(这是诚信红线:DBEC 是 fixed-pool composer,不是
   retriever)
3. **箭头从①一路向下贯穿至④,不是并行**——表达"四个机制成分按依赖关系串联",但
   都在同一个 DBEC 大框内
4. **不在任何位置写 ETv3 / PCEC**(包括 caption、legend、坐标轴标签)

### Figure 1 Caption 模板

> **Figure 1.** DBEC for multi-hop RAG composition. Given a query $q$ and a fixed
> retrieval pool $P$, DBEC (a) decomposes the query into atomic demands with
> dependency edges (§4.1), (b) grounds dependencies through entity-anchored
> bindings producing demand-binding units (§4.2), (c) optimizes a noisy-OR
> coverage objective over DBUs to select the top-$K$ evidence set (§4.3), and
> (d) gates the binding mechanism when extracted referents cannot be uniquely
> resolved (§4.4). The selected evidence set is passed to the reader for
> answer generation.

注意 caption 的几件事:
- 四个机制成分编号 (a)-(d) 对应 §4.1-§4.4——caption 跟章节直接挂钩,reviewer 一眼能映射
- 明确说 "given a query $q$ and a fixed retrieval pool $P$"——pool 是外部输入,DBEC 是
  composer 不是 retriever
- caption 不出现 "stage" / "phase" / "step" / "ETv3" / "PCEC"

## 三、Figure 2(机制分析图)规范

41_ Section 5.3 是机制分析章节,会用到 figure(under-selection 分布、conditional
slicing 等)。

### 命名规则

X 轴 / Y 轴 / legend 命名按 41_ 锁定术语:
- ❌ "PCEC selection size" → ✅ "Selected set size $|E|$"
- ❌ "ETv3 pool depth" → ✅ "Pool depth $|P|$"
- ❌ "Required for full chain" → ✅ "Required supports $n_{\text{gold}}$"

### Under-selection 图(必出)

41_ 5.3.2 锚定的数字(2Wiki 4-doc under-selection 51.9% 等)对应一张分布图。命名:

- 标题:**Under-selection rate by query depth**
- X 轴:**Required supports** $n_{\text{gold}}$(2 / 3 / 4)
- Y 轴:**Under-selection rate** $P(n_{\text{selected}} < n_{\text{gold}})$
- legend:**SetR-style**,以及 DBEC 作为参考线

## 四、Table 规范

### Main results table(对应 41_ Section 5.2)

行顺序按 41_ 锁定(已设定):

```
Top-5
IRCoT-style
LLM-direct (title)
LLM-direct (snippet)
SetR-style
SetR-Fill@5
SetR-Fill@10
─────────────
DBEC
DBEC-IG
```

**Top-5 / SetR 系列在上,DBEC 系列在下并用横线隔开**——这是 multi-hop RAG 论文的
惯例,不要打乱。

### Ablation table(对应 41_ Section 5.7)

41_ 已锁的 ablation 包含 DBEC-nobinding 和 IG gate。不要新增"Score-Truncated /
Decoupled Convergence / Pool-Initialized"这种虚构 variant——它们**在代码里不存在**,
也**不在 41_ 锁定 anchor 数据里**。

如果实验里需要新增 ablation,**先回 03_experiment_board.md 立项跑数据,然后回 41_
更新 contribution claim**,再来 45_ 加命名规范。

---

# Part C:关键句模板

41_ 给的是 ❌/✅ 替换表(12 条 locked-out language)。本节补**可直接复用的完整段落**,
用于不同位置。所有句子都基于 41_ 锁定术语(DBEC / DBU / composition gap / under-selection
等),不引入新概念。

## C.1 Abstract 引导句(若需要在 41_ abstract 外另写一版,如 200 字版本)

> *Multi-hop retrieval-augmented generation requires composing a compact evidence
> set from a fixed retrieval pool. We show that even high-recall pools leave a
> substantial composition gap under small reader budgets. We introduce DBEC, a
> train-free evidence composer that decomposes a query into demand-binding units
> and optimizes noisy-OR coverage over them. Across three multi-hop benchmarks,
> DBEC outperforms prompt-only selectors and is competitive with SetR-style
> adaptive selection overall, significantly improving deep-dependency queries
> where prompt-only selectors systematically under-select.*

**挡住的攻击**:"这是 retriever + selector 两阶段方法"——通过强调 "fixed retrieval pool"
+ "composer"(而非"selector after retrieval")明确 DBEC 是 composer 不是 retriever。

## C.2 Intro 第三段(对应 41_ Section 1 第 3 段:Existing approaches fall short)

> *Existing approaches to this composition gap fall into three families. Rerankers
> (e.g., cross-encoders) reorder a fixed top-$K$ list pointwise; they do not change
> which $K$ passages are selected once the pool is given. Iterative methods (e.g.,
> IRCoT-style decompose-and-retrieve) interleave retrieval and reasoning but
> remain pointwise within each iteration. Prompt-only adaptive selectors (e.g.,
> SetR) ask an LLM to choose a passage subset from the pool, but the selection
> signal is conditioned only on the query and passage texts; there is no explicit
> mechanism to enforce that selected passages jointly cover mutually dependent
> demands. DBEC takes a different angle: it makes the dependency structure
> explicit in the selection objective itself.*

**挡住的攻击**:"DBEC 跟 X 方法有什么本质区别"——一段话覆盖三类对手,且把区别定为
"explicit dependency structure",直接呼应方法名 Dependency-Bound。

## C.3 Section 4 开头(方法章节起点)

> *DBEC takes as input a query $q$ and a fixed retrieval pool $P$ produced by an
> upstream retriever, and outputs an evidence set $E \subseteq P$ with $|E| \le K$
> for the reader. The composition proceeds through four mechanism components:
> demand decomposition (§4.1) extracts atomic demands and their dependency edges;
> entity-anchored binding (§4.2) grounds dependency variables into demand-binding
> units; the noisy-OR coverage objective (§4.3) ranks evidence by joint demand
> coverage; the identifiability gate (§4.4) abstains from binding when the
> grounding cannot be uniquely resolved. We describe each in turn.*

**挡住的攻击**:"两阶段方法应该叫 retrieval 和 selection 两节"——本段已经把方法按
**机制成分**分四节而不是按 retrieval/selection 分两节,reviewer 看到这一段就放弃这条
攻击线。

**关键词**:"mechanism components"——这是 41_ 章节结构的正面命名,本节开宗明义。

## C.4 Section 4.2 末尾衔接(从 binding 过渡到 coverage)

> *Once demand-binding units are constructed, the selection problem reduces to
> choosing a small evidence subset that maximizes coverage over the DBU set. We
> formalize this objective next.*

**挡住的攻击**:"4.1-4.2 是 retrieval,4.3-4.4 是 selection,这就是两阶段"——本段用
"the selection problem **reduces to**" 把整段叙事框定为**一个**优化问题,中间没有
模块切换。

## C.5 Related Work(SetR 划界,对应 41_ 2.2)

> *SetR (Lee et al., 2025) similarly recognizes that multi-hop QA requires
> jointly sufficient evidence rather than individually relevant passages, and
> proposes a prompt-only adaptive set selector that asks an LLM to identify
> the information requirements of a query. DBEC differs in three respects.
> First, DBEC's atomic selection units are demand-binding units — (demand,
> entity binding) pairs — that explicitly ground dependency variables to
> referents, whereas SetR's information requirements are entity-agnostic
> textual specifications. Second, DBEC's coverage objective is computed by
> a transparent noisy-OR over DBUs, whereas SetR's selection is computed by
> a black-box LLM conditioned only on query and passage texts. Third, in
> our controlled substrate, the prompt-only selection in SetR-style exhibits
> a systematic under-selection failure on deep-dependency queries, which
> DBEC's explicit binding addresses.*

**挡住的攻击**:"DBEC 是 SetR 的 graph 变体"——一段话精确给出三处差异,且每处都
锚定到 41_ 已锁的数字证据。

## C.6 Section 5.3.3 开头(nobinding ablation 的衔接,对应 41_ Strengthening 2)

> *To verify that binding is the active mechanism, not a decorative addition,
> we evaluate DBEC-nobinding: it preserves the demand decomposition and the
> noisy-OR coverage objective, but does not condition coverage on dependency
> bindings. If binding were not load-bearing, DBEC-nobinding should match
> DBEC; if binding were universally helpful, DBEC-nobinding should
> underperform DBEC across all conditions.*

**挡住的攻击**:"为什么不消融每个模块"——本段直接告诉 reviewer "我们消融了 binding,
且消融设计本身可证伪",reviewer 没有追加质疑空间。

## C.7 Limitation 节首句(对应 41_ Section 6)

> *DBEC trades selector-side inference cost for reader-context structure: it
> uses approximately 8.4 LLM calls per query, versus 1 call for prompt-only
> selection. This is a deliberate trade — DBEC optimizes the reader-context
> budget under a fixed $K$, not selector-side efficiency.*

**挡住的攻击**:"DBEC 比 SetR 贵 8 倍,凭什么"——本段把 cost 从"方法弱点"框定为
"deliberate trade on a different resource axis",且这个 framing 已经在 41_
Strengthening 3 锁定。

---

# Part D:Supplementary 提交前自查清单

按顺序勾完才能投。**这一清单是 supplementary / paper-facing artifact 专用**,跟 41_ 的
locked-out language 自查表不重复(那个查论文正文,本表查 supplementary)。

## D.1 Paper 主文检查(跟 41_ 互补)

- [ ] 全文 grep `ETv3` / `PCEC` / `etv3` / `pcec` — 应为零
- [ ] 全文 grep `stage 1` / `stage 2` / `first stage` / `second stage` / `two-stage` —
  应为零(除非描述 baseline pipeline)
- [ ] 全文 grep `expander` / `composer`(作为名词指 DBEC 内部) — 应为零
- [ ] 全文 grep `selector` — 只在描述 SetR / 外部 baseline 时出现
- [ ] 全文 grep `readout` — 应为零
- [ ] Section 4 章节标题与 41_ 一致(4.1 demand decomposition / 4.2 binding /
  4.3 coverage / 4.4 IG gate),不出现 expansion/convergence 等替代分法
- [ ] Figure 1 是一个大框,框内四个机制成分,**不是**两 box 串联
- [ ] Figure 1 caption 不出现 ETv3 / PCEC / stage / phase

## D.2 Supplementary 内容检查

- [ ] supplementary 内所有 JSON key、md 表头按 Part A 字段对照表重命名
- [ ] supplementary 内所有文件名按 B.3 表重打包(不直接复制 `run_logs/daec_*/`
  或 `reports/etv3_*/` 原目录)
- [ ] supplementary README 不引用 internal commit / branch name(`feature/pcrs-rag-v1`
  这种)
- [ ] Code 提交版本(如有)经过 sanitize:模块 import、类名等保留(不影响功能),但
  user-facing config keys / log messages 按对照表替换

## D.3 Figure / Table 检查

- [ ] 所有 figure 的 X/Y 轴 label、legend、caption 不出现内部命名
- [ ] 所有 table 的 column header、row header 按 41_ 锁定 variant 命名
- [ ] Main results table 的 baseline 顺序与 41_ Section 5.2 锁定一致
- [ ] Ablation table 只包含 41_ 锁定的 ablation(DBEC-nobinding、IG gate、SetR-Fill@5/10、
  Top-10 retriever),不出现虚构 variant

## D.4 Rebuttal 准备(投稿时不需要,review 期触发)

41_ 的 12 条 locked-out language 已经覆盖大部分 reviewer 攻击。本节只列 41_ 未覆盖、
且本文档(45_)各 Part 已经预案过的攻击:

| Reviewer 攻击 | 回答模板位置 |
|---|---|
| "DBEC 是不是 retriever + selector 两阶段" | C.1 abstract + C.3 method 开头 |
| "DBEC 跟 SetR 本质区别是什么" | C.5 related work |
| "为什么不每个模块都消融" | C.6 nobinding ablation |
| "DBEC 比 SetR 贵 8 倍,工程上不实用" | C.7 limitation 首句 + 41_ Strengthening 3 |
| "supplementary 里出现 etv3/pcec,跟正文不一致" | **Part A 字段对照表已预防**——若仍发生,
  说明 rename layer 漏掉了,补 D.2 检查 |

---

# Part E:本文档的边界

**本文档不做**:
- 重命名 DBEC、改标题、改 contribution、改章节结构(全部以 41_ 为准)
- 引入新概念(chain sufficiency / expansion regime / convergence regime 等都不属于
  本文档管辖,且与 41_ 锁定术语冲突)
- 设计 ablation variant(以 41_ 锁定为准,新 variant 走 03_experiment_board.md)
- 决定 baseline 是否包含某篇论文(以 40_setr_style_full1000_positioning_20260506.md
  为准)

**本文档管辖**:
- 代码 ↔ 论文叙事的字段命名映射(Part A)
- Method figure 的视觉规范(Part B)
- 41_ 替换表之外的、可复用的完整段落模板(Part C)
- supplementary 提交前的字段命名自查清单(Part D)

如有冲突,**以 41_ 为准**,本文档同步修订。

---

## Status

第一版:2026-05-14。基于与 41_ 对齐原则重写。

**先决条件**:
- Part A 字段对照表需要在投稿前**配套实现** `scripts/export_for_supplementary.py`,
  否则只是规范不是工具
- Part B figure 规范需要在 Figure 1 / Figure 2 实际作图时**强制执行**,否则会退化
  回两 box 串联
- Part C 模板按 41_ 锁定术语写,**如 41_ 后续更新术语,本节句子要同步**
