# EvLink Paper Handoff, 2026-05-22

## 1. 当前论文入口

- 主论文入口：`paper/current_full_preview_acl.tex`
- 主论文 PDF：`paper/current_full_preview_acl.pdf`
- 方法章节：`paper/sections/04_method.tex`
- 实验章节：`paper/sections/05_experiments.tex`
- 附录：`paper/sections/A_appendix.tex`
- 参考文献：`paper/related_refs.bib`
- 编译命令：

```bash
cd /mnt/nvme/code/HippoRAG/paper
tectonic -X compile current_full_preview_acl.tex
```

最后一次编译已成功。当前仍有若干 LaTeX warning，主要是 `lineno.sty` 的 UTF-8 warning 和 Underfull box warning；没有 fatal error，也没有发现 undefined reference/citation 导致编译失败。

## 2. 当前论文主叙事

当前论文主张应保持为：

> 多跳检索的瓶颈未必是图状态不够细，而可能是文档状态之间缺少可靠的证据链接。

对应英文写法目前落在：

- documents/passages are retrieval states；
- extracted facts/source-grounded evidence records are evidence links or witnesses；
- retrieval first exposes graph-reachable evidence through local document traversal；
- final selection optimizes complementary coverage over reachable evidence.

不要把主叙事写成 reader budget 或 fixed-pool 之类的工程口径。现在的论文口径是：图是有效的检索对象，但关键不是把节点继续做细，而是让文档状态之间有可靠、可溯源的 evidence links。

## 3. 方法章节当前状态

文件：`paper/sections/04_method.tex`

### 3.1 总体结构

当前 §4 结构如下：

- `4 Method: EvLink`
- `4.1 Document Graph Construction`
  - `Knowledge Extraction`
  - `Graph Construction`
- `4.2 Local Graph Evidence Retrieval`
  - `Query-local subgraph construction`
  - `Evidence-link traversal`
- `4.3 Complementary Evidence Optimization`
  - `Evidence-need mining`
  - `Coverage-aware reranking`

用户明确要求过：

- §4.2 标题必须是 `Local Graph Evidence Retrieval`，不要改成 `Evidence Association Retrieval`。
- §4.3 现在用 `Complementary Evidence Optimization`，比带 retrieval 的标题更稳。
- 不要在论文里出现 low-level 的 fixed-pool、reader budget、prefix 等说法。
- 不要过度强调 fact，但事实抽取作为 evidence link 的来源可以正常解释。

### 3.2 方法 opening

当前 opening 已压缩到短段：

```tex
We propose \methodname{}, a graph-based retriever that keeps documents
as retrieval states and uses extracted facts as source-grounded links
between them ...
```

它明确了 document states + source-grounded links。若后续继续压缩，注意不要删掉 `document` 和 `passage` 等价说明，因为全文使用两者。

### 3.3 §4.1 Document Graph Construction

当前 §4.1 没有 subsection opening，直接用两个 paragraph。用户要求过删掉 `Degree controls prevent high-frequency entities...`，已经删掉。

当前 `Graph Construction` 段落已经解释：

- 图中 passage 是节点；
- evidence link 是边；
- edge 只有在 source-grounded evidence record `w` 支撑时才建立；
- OpenIE 的 argument、relation phrase、source span、provenance 分别用于 grounding endpoints、解释关系、记录来源；
- 斜体项是 edge families，不是 retrieval nodes。

这解决了用户说的“读者看到斜体会懵”的问题。

### 3.4 §4.2 Local Graph Evidence Retrieval

当前 `Query-local subgraph construction` 开头承接了 §4.1：

- §4.1 通过 OpenIE facts 和 provenance 构造 evidence links；
- query time 只激活 query-relevant region，不重新建图；
- link 可用的条件是 source passage 里有 fact/source span 能 ground target endpoint，因此 edge carries a textual witness；
- seed set 由 dense entry 和 question anchors 构成。

当前 traversal 公式已经写成 BFS：

```tex
C_q = \mathrm{BFS}_{h,L}\!\bigl(\mathcal{S}_q,\;\mathcal{G}\bigr),
\quad |C_q|\le L,
```

并且正文圆了为什么 BFS 足够：图本身已经把 document-to-document associations 编成 witnessed links，retrieval 阶段只是暴露通过这些 links 可达的 nearby evidence，而不是重新学习全局传播分数。

用户要求删除的实验设置句子已经删掉：

```tex
The same traversal and candidate-limit settings are used for all datasets ...
```

### 3.5 §4.3 Complementary Evidence Optimization

用户要求删掉 §4.3 开头的 reachability-vs-coverage 大段，已经删掉。

当前 `Evidence-need mining` 压缩为短段，讲：

- 从 question 中抽取 compact requirement set `B(q)`；
- 包括 entity slots、relation constraints、dependency bindings；
- 这是 question-only，不从 retrieved passages 生成 facts；
- 对 candidate `d` 和 requirement `b` 计算 bounded support value `\phi_{d,b}`。

当前 `Coverage-aware reranking` 引用了 SetR：

```tex
\citep{lee-etal-2025-shifting}
```

公式包括：

- noisy-OR coverage `\mathrm{cov}_q(b,R)`；
- set utility `\mathrm{Cov}_q(R)`；
- marginal gain `\Delta_q(d \mid R)`；
- greedy selection `d_t`。

注意：这里引用 SetR 只是作为 set-wise retrieval view 的启发，不要写成我们和 SetR 做同一件事。我们的区别是：选择空间来自 graph-reachable candidates `C_q`，不是普通 ranking list。

## 4. 实验章节当前状态

文件：`paper/sections/05_experiments.tex`

### 4.1 RQ opening

当前 RQ 已经分行，PDF 里显示正常：

```tex
\textbf{RQ1.} Main effectiveness.\\
\textbf{RQ2.} Component attribution.\\
\textbf{RQ3.} Evidence links vs.\ generic connectivity.\\
\textbf{RQ4.} Operational cost.\\
```

这比之前长句 RQ 更清楚，也不占太多篇幅。

### 4.2 Datasets 分类

用户指出 NQ 和 PopQA 是 simple QA，不应归为 open-domain。当前已改为：

```tex
We separate benchmarks by task type. The primary multi-hop axis uses
HotpotQA, 2WikiMultiHopQA, and MuSiQue; NQ and PopQA are reported only
as simple QA checks.
```

Table 1 caption 同步改为：

```tex
with NQ and PopQA reported separately as simple QA checks.
```

附录 `A_appendix.tex` 中残留的 `open-domain checks` 也已改为 `simple QA checks`。

我已经用 `rg` 查过当前主文和附录源文件：

```bash
rg -n "open-domain" paper/sections/05_experiments.tex paper/sections/A_appendix.tex paper/current_full_preview_acl.tex paper/generated
```

没有残留。PDF 里还能搜到 `open-domain`，但那是参考文献标题中的词，不是我们自己的数据集分类。

### 4.3 Main results 表

Table 1 当前分组：

- Classic RAG：BM25
- Large-model retriever：NV-Embed-v2 dense entry
- Hierarchical RAG：RAPTOR
- GraphRAG：HippoRAG 2、HGRAG、PropRAG
- Evidence-enhanced RAG：NeocorRAG
- Ours：EvLink

EvLink 当前主表结果：

| Dataset | R@5 | EM | F1 |
| --- | ---: | ---: | ---: |
| HotpotQA | 96.5 | 62.8 | 75.6 |
| 2WikiMultiHopQA | 96.2 | 65.9 | 74.7 |
| MuSiQue | 73.5 | 37.2 | 49.1 |
| NQ | 81.6 | 51.9 | 65.0 |
| PopQA | 65.0 | 49.4 | 62.2 |

Multi-hop average F1 当前是 66.5。

NeocorRAG 保留星号说明：它使用 native saved-context size，平均 3.3--3.6 passages per query on the multi-hop benchmarks。当前正文说它和其他方法共享同一 1,000-query split、corpus snapshot、GPT-4o-mini reader 和 answer normalization，但保留其 native saved-context size 作为唯一协议例外。

### 4.4 Ablation Study

Table 2 当前 ablation 行：

- Full EvLink
- w/o evidence-need mining
- w/o coverage optimization
- w/o evidence-linked transitions

当前 caption 已改掉过长名字，不再使用 `w/o Evidence-Coverage-Aware Retrieval Optimization`。

关键数值：

| Variant | Hotpot R@5/F1/All@5 | 2Wiki R@5/F1/All@5 | MuSiQue R@5/F1/All@5 |
| --- | --- | --- | --- |
| Full | 96.5 / 75.6 / 93.3 | 96.2 / 74.7 / 89.5 | 73.5 / 49.1 / 45.3 |
| w/o evidence-need mining | 95.2 / 74.8 / 90.9 | 92.8 / 71.1 / 81.1 | 73.0 / 48.9 / 43.9 |
| w/o coverage optimization | 95.0 / 74.8 / 90.5 | 93.5 / 72.8 / 82.6 | 71.8 / 48.3 / 42.2 |
| w/o evidence-linked transitions | 96.2 / 75.0 / 92.7 | 93.9 / 73.0 / 82.0 | 72.5 / 47.5 / 43.4 |

这张表服务于 RQ2：组件是否有贡献。它不是最强机制证明，机制证明主要在 RQ3。

### 4.5 Mechanism Controls

RQ3 当前核心是证明 evidence links 不是“边多/图大/泛连通”带来的收益，而是 source-grounded link semantics 有用。

Table 3 当前 control：

- Diagnostic full EvLink
- Dense-doc KNN
- Edge-count dense-doc KNN
- Degree-matched shuffled

关键数值：

| Variant | Hotpot R@5/All@5 | 2Wiki R@5/All@5 | MuSiQue R@5/All@5 | Avg R@5/All@5 |
| --- | --- | --- | --- | --- |
| Diagnostic full | 96.6 / 93.4 | 96.3 / 89.7 | 76.4 / 51.0 | 89.8 / 78.0 |
| Dense-doc KNN | 95.3 / 90.8 | 87.5 / 69.7 | 73.1 / 44.9 | 85.3 / 68.5 |
| Edge-count dense-doc KNN | 95.2 / 90.6 | 83.8 / 62.7 | 72.2 / 43.3 | 83.7 / 65.5 |
| Degree-matched shuffled | 94.8 / 89.8 | 77.5 / 51.9 | 71.1 / 41.1 | 81.1 / 60.9 |

这组实验应该这样解释：

- Dense-doc KNN 低于 full：不是随便连文档就行，embedding 相似边不能稳定替代 source-grounded evidence links。
- Edge-count dense-doc KNN 低于 full：不是因为 full 图边更多或候选更大。
- Degree-matched shuffled 低于 full：不是因为度分布或图结构模板，而是 link semantics 本身重要。
- 2Wiki 差距最大：符合它大量依赖显式 bridge relation 的结构。

### 4.6 FCRG

当前 FCRG 定位是 mechanism probe，不替代 R@5/F1。

Table 4 当前数值：

| Method | Hard FCRG | @5 recovered |
| --- | ---: | ---: |
| EvLink final | 0.300 | 67.6 |
| w/o evidence-linked transitions | 0.268 | 58.6 |
| Edge-count dense-doc KNN | 0.064 | 30.3 |
| Degree-matched shuffled | 0.006 | 15.7 |
| HippoRAG 2 pool | 0.130 | 32.1 |
| PropRAG pool | 0.252 | 58.2 |

FCRG 当前服务的问题：

> 对 dense retriever 漏掉、但能被另一个 gold support document 通过 source-grounded evidence link 指到的 gold support，方法能否把它推到更高排名？

它证明的是后半句：可靠文档间证据链接能把 dense-missed bridge evidence 找回来。它不能单独证明“图状态不必更细”，所以不要把 FCRG 吹成整个论文主张的全部证明。它应和 Table 3 的 strict edge-count control、主表 GraphRAG baseline 对比一起使用。

### 4.7 Cost

当前 cost 表拆成：

- operational statistics: docs/facts/local docs/local edges/graph iters/readout/index+ret time
- auxiliary token cost: offline avg / online avg / total avg

关键 token cost：

| Method | Offline Avg. | Online Avg. for 1,000q | Total Avg. |
| --- | ---: | ---: | ---: |
| RAPTOR | 1.50M | 0.00M | 1.50M |
| EvLink | 11.85M | 1.51M | 13.36M |
| HippoRAG 2 | 13.49M | 0.63M | 14.12M |
| PropRAG | 18.84M | 0.00M | 18.84M |
| NeocorRAG | 12.84M | 6.94M | 19.78M |

注意讲法要公平：不要把 PropRAG online cost 为 0 写成它便宜，因为它把成本放到 offline。正文当前按 offline/online 分开讲，比较稳。

## 5. 参考文献变更

文件：`paper/related_refs.bib`

已加入 SetR citation：

```bibtex
@inproceedings{lee-etal-2025-shifting,
  author    = {Lee, Dahyun and Jo, Yongrae and Park, Haeju and Lee, Moontae},
  title     = {Shifting from Ranking to Set Selection for Retrieval Augmented Generation},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {17606--17619},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.acl-long.861},
  url       = {https://aclanthology.org/2025.acl-long.861/}
}
```

用途：只在 §4.3 coverage-aware reranking 中说明 set-wise retrieval view 的启发。

## 6. PDF 检查方法

用户非常关注 PDF 实际换行，不只看源码。建议每次改完用：

```bash
cd /mnt/nvme/code/HippoRAG/paper
tectonic -X compile current_full_preview_acl.tex
pdftotext -layout current_full_preview_acl.pdf - | rg -n "Datasets\.|RQ1|RQ2|RQ3|RQ4|Local Graph Evidence Retrieval|Complementary Evidence Optimization" -C 4
```

最近一次 PDF 检查结果确认：

- Datasets 段显示为 “We separate benchmarks by task type”；
- NQ/PopQA 显示为 “simple QA checks”；
- Table 1 caption 显示 “reported separately as simple QA checks”。

## 7. 当前风险和待办

### 7.1 不要误删或误提交

当前 git 状态非常脏，存在大量非论文文件的 modified/untracked 项。`paper/` 在当前仓库状态中也显示为 untracked directory。后续如果要提交，必须只 stage 明确文件，不要使用：

```bash
git add .
```

建议只 stage：

```bash
git add paper/current_full_preview_acl.tex \
        paper/sections/04_method.tex \
        paper/sections/05_experiments.tex \
        paper/sections/A_appendix.tex \
        paper/related_refs.bib \
        docs/evidencelink_paper_handoff_20260522.md
```

是否提交 PDF 取决于项目习惯；当前 PDF 是 `paper/current_full_preview_acl.pdf`。

### 7.2 Figure 仍是占位式示意图

`paper/sections/04_method.tex` 里的 Figure 目前是 `\fbox` 文本示意，不是真正绘制好的架构图。用户之前说过“保留图片”，所以我没有删。但如果最终投稿需要视觉质量，最好补一张正式 method figure。

### 7.3 方法术语需要继续守住

当前术语主线是：

- evidence link
- source-grounded evidence record
- witness
- document/passages as retrieval states
- complementary evidence optimization

不要又改回：

- fact-certified edge 到处出现；
- reader budget；
- fixed-pool；
- prefix；
- projection loss 泛化攻击 HippoRAG2/PropRAG。

如果需要批评既有方法，建议用中性机制描述：

- phrase/proposition/summary salience 到 passage set 的 readout ambiguity；
- document-level transition 是否有 source-grounded witness；
- dense/generic connectivity 和 evidence-linked connectivity 的差异。

### 7.4 FCRG 的边界

FCRG 不要被写成“证明 document-state 一定优于 fact-state”的指标。它证明的是：

- dense-missed；
- fact-conditioned；
- gold support；
- 是否被方法提升到 final top-5 或可用 graph ranking top-5。

如果 reviewer 问 facts-as-nodes vs facts-as-links，当前最稳回答是：

- 主张不是“facts as nodes 一定错”；
- 主张是“多跳检索瓶颈未必来自节点不够细，而是文档状态之间缺少可靠证据链接”；
- Table 3 和 FCRG 表证明可靠 evidence links 比 generic document connectivity 更关键；
- HippoRAG2/PropRAG 是不同图状态范式下的 strong baselines，但不是严格同 OpenIE substrate 的 topology-only 对照。

### 7.5 Relink / PathRAG

当前主表没有 Relink 行。之前已决定 Relink 不放主表，原因是 aligned-protocol runs 不稳定/不完整，避免低质量复现伤害论文。PathRAG 状态也不作为当前主文结果依赖项。若后续要补，只建议进 appendix 或 discussion，不建议再临时塞进 Table 1。

## 8. 最近一次实际完成的改动

最近一轮完成的是数据集分类修正：

- 将 `Datasets` 段从泛泛说明改为按 task type 分类；
- 明确 HotpotQA、2WikiMultiHopQA、MuSiQue 是 primary multi-hop axis；
- 明确 NQ、PopQA 只是 simple QA checks；
- 同步 Table 1 caption；
- 同步 appendix 里的配置描述；
- 编译生成最新 PDF；
- 检查主文/附录源文件中不再残留作为分类口径的 `open-domain`。

## 9. 推荐下一步

如果继续打磨，建议优先级：

1. 快速通读 abstract 和 intro，确保“图状态不必更细，关键是可靠证据链接”的科学问题在开头足够明确。
2. 检查 §4 figure，占位图是否需要替换为正式图。
3. 再审一遍 §5.4 mechanism analysis 的文字，确保 Table 3 和 FCRG 的解释不越界。
4. 检查 appendix 中 hyperparameter 表和 method §4 是否一致，尤其是 `B(q)`、BFS、candidate limits、coverage optimization 命名。
5. 最终提交前用 `pdftotext -layout` 检查 RQ opening、Datasets、Table 1 caption、§4.2、§4.3 的实际断行。
