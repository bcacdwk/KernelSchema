# 框架问题维护入口

运行 `python3 schema_extract/extract.py candidates <run-uuid> --repo <当前仓库>` 查看本任务已接受回执中的候选和同一稳定 ID 的已有决定；`python3 schema_extract/extract.py issues --repo <当前仓库>` 读取全部待决项与历史决定。这两个命令不修改提取状态或台账。

六项真实提取后的[长期分类与处理方案](archive/TRIAGE.md)逐项区分继续处理、保留现有修复和来源限制；已归档不等于无需复核。发现决定与实际实现不符时，沿用原稳定 ID 补充证据并修正决定。框架修复与历史产物修订分别验收，部分采纳的剩余事项不能因归档而省略。

批次收口时，Supervisor 逐个读取本批各 run（含接续前任）的 `candidates`，按根因查重后登记或记录已有决定；未解决项保持待决。`ready` 和证据审计通过不代表框架候选已处理，也不代替算子结果的表达质量验收。

本目录只保存可复用的问题与决定。同一根因沿用原稳定 ID；证据定位当前文件或明确 Git 版本，不依赖临时运行目录，不记录建设过程和试跑流水账。

只有共享规则会导致错误、遗漏或重复劳动时，才登记能解决它的最小改动。算子自身的实现差异、缺失来源与证明缺口留在本任务回执和导入来源记录中；用户要求独立追踪的源实现缺陷，也可由 Supervisor 在 `kernel_wiki/` 登记，源码修复状态与 Schema 审批独立。依 [REVIEW 的源缺陷分流边界](../REVIEW.md#源实现缺陷与设计分流)，有证据确认不改变同一完整抽象 Design 的源缺陷，登记后可归档，提取与源修复分别处理；残稿仍须补齐并独立审阅。

维护者查重后，将新项放入 `schema_docs/`、`schema_template/`、`kernel_wiki/`、`extract_workflow/` 或 `extract_tools/`，目录按需创建。每个 `<id>.json` 保留 `id/category/problem/evidence/proposal/necessity`；ID 用小写字母数字及 `-` 或 `_`，不包含运行指纹。

决定后移入 `archive/<类别>/<id>.json`，保留原字段，增加 `decision.outcome` 和 `decision.reason`。结果为 `declined`、`resolved` 或 `partially_adopted`；部分采纳还需 `adopted` 与 `remaining`。不要为已有 ID 另写副本。候选提报不改变当前任务的审批结论。

`resolved`/`partially_adopted` 归档须填写 `decision.version`，记录当前已解决或已采纳改动所属的 Schema Extract 版本，同版可对应多项；部分采纳仍保留未解决的 `remaining`。后续实质修复才更新版本并在 `reason/evidence` 保留旧依据，纯归档整理不更新；无产品版本号的历史项使用 `git:<完整修复提交 SHA>` 并在 `evidence` 注明依据。
