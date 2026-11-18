# 活跃文字精简

2026-11-18。以 `f48823b` 的用户精修文字为底，只调整未提交差异中的新增或事实更新句子。

- `schema_extract/README.md`：精简 acceptance 类型、128000 单响应请求、cancel/pause 与 monitor 说明。
- `prompt/operations/check.md`：合并 worker 预览退出码及机械修正说明，保留 raw 未批准、reviewer/P6 严格检查边界。
- `prompt/operations/response.md`：缩短必要原文范围的说明。
- `prompt/phases/01_contract.md`：合并 comparisons 覆盖要求，删除精确名称要求已表达的反面提示。
- `tools/README.md`：简化调用次数追加及检查统计的措辞。

没有新增功能事实或规则。次数、取消结清、显式恢复、单份投递、ack/already_acknowledged 后 scan、证据绑定等边界保留；P1 的全部 HEAD 原段落保持不变。其他活跃文字已足够简练，未再改写。

检查通过：本轮五个文件的修改均限于此前新增/改写的行；代码块、行内代码、链接和占位符前后一致；`git diff --check` 通过。未改代码或历史材料，未运行模型、提取、monitor 或代码测试，未提交、推送。没有待解决的文字事实矛盾。
