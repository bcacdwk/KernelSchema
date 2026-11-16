# 第二轮任务级规范化交接

实现：`schema_template/tools/normalization.py`，无写文件行为。

- `load(s)_for_normalization(..., skip_hardware=False)` → `LoadedDocument(document, source_sha256, corrections)`。
- `normalize_documents(files, documents, canonical_fields=None, skip_hardware=False)` →
  `NormalizedTask(files, documents, candidates, source_hashes, corrections)`；`documents` 输入可混用严格 Document/LoadedDocument。
- `documents` 返回全任务候选；`candidates` 仅变化文件；`source_hashes` 始终是原文件摘要。
- warning 前缀固定为 `field / identifier+reference / quote / period / layout`，字段位置指向最终候选。
- CLI 必须传全任务、写前核对原摘要；写入、写后严格重读和最终摘要核对处于同一恢复边界。恢复使用原始 bytes，只撤回仍为本次候选的已写文件；read-only 使用同一候选，不能称为已写回。

字段/文本类型/闭集值分别直接消费 `CANONICAL_FIELDS / STRING_FIELDS / CANONICAL_VALUES`；纯 int/bool 叶由同源 `LITERAL_TYPES` 标记。
`recover_scalar_quotes` 在独立源阶段核对全部用途后恢复引号：数值/bool 原 lexeme 可包成已知文本；纯 int/bool 叶的普通单行 quoted lexeme 可按同一 YAML core 规则还原类型。null、显式 tag、联合表达式类型、空白/转义/多行和混合 alias/merge 用途不猜转，范围规则仍独立检查。
只读字段视图用于绑定；最终一次 combined ScalarEdit 授权同时覆盖固定键、具名键及相关 alias 用途。
MAIN 的唯一非 case 编辑通过 `ScalarEdit.main_names` 明确授权完整身份 token，Dispatch.id/from 值仍禁止改名。

内存 smoke：声明与引用/公式/Index/Event/view/Waiting/Flow/Realization、MAIN、from 全消费者一致和分歧拒绝、
bool/number 原 lexeme 与 hash、多个缺引号、block/flow Event/Waiting、共享 alias、精确值、输入只读、幂等、skip_hardware。
pipeline 仅适配了 Info.Member、Storage.view、Launch.Step，未改控制、窗口或执行语义。
第三轮首次完整 suite 93 项中 92 通过；唯一测试诊断文案断言由第三轮调整，实际碰撞已正确拒绝。

复核边界：自由正文不抽取伪符号表；未知外部函数与属性成员不更名，已绑定的属性链根随同作用域声明同步；
公式形参覆盖可证明的单行定义与相邻运算符续行，下一定义及独立量词保持各自作用域。需要改名但不能唯一界定的多行形式或尾部使用报明确错误。
未通过严格解析且无法独立证明补引号的输入仍 error；显式 tag、null、数字字段及混合 alias 类型不被改成文本。
A/B/C 待决字段保持当前唯一拼写，arch 不作历史迁移。

R1/R2/R3/R4 审阅修正和最新验证见 [FIXES](FIXES.md)。
