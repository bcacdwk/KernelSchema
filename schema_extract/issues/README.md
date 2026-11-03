# 框架问题维护入口

运行 `python3 schema_extract/extract.py candidates <run-uuid> --repo <当前仓库>` 查看本任务已接受回执中的候选和同一稳定 ID 的已有决定；`python3 schema_extract/extract.py issues --repo <当前仓库>` 读取全部待决项与历史决定。这两个命令不修改提取状态或台账。

台账与已有决定集中保存在本目录，迁移保留原内容和稳定 ID；旧包不是新版执行依赖。此前的单文件记录保留在[历史原文](archive/ISSUES.md)。同一根因再次出现时沿用原 ID 和决定，不重复登记。

只有共享规则会导致错误、遗漏或重复劳动时，才登记能解决它的最小改动。算子自身的实现差异、缺失来源与证明缺口留在本任务回执和导入来源记录中。

维护者按稳定 ID 和根因查重后，将新项放入本目录的 `schema_docs/`、`schema_template/`、`kernel_wiki/`、`extract_workflow/` 或 `extract_tools/`。每个 `<id>.json` 保留 `id/category/problem/evidence/proposal/necessity`；ID 用小写字母数字及 `-` 或 `_`，不包含运行指纹。证据必须能复查。

决定后移入 `archive/<类别>/<id>.json`，保留原字段，增加 `decision.outcome` 和 `decision.reason`。结果为 `declined`、`resolved` 或 `partially_adopted`；部分采纳还需 `adopted` 与 `remaining`。不要为已有 ID 另写副本。候选提报不改变当前任务的审批结论。
