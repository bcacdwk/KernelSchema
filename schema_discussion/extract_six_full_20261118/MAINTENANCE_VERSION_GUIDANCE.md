# 维护版本提示

已在[维护提示](../../schema_extract/prompt/operations/maintenance.md)的权限段落补齐 worker 的版本更新责任：随获准内容修改按已有规则更新 `META_DATA.Version`，属于本次写权限，Runner 仅校验。

依据是[公共版本规则](../../schema_docs/0.common.md#单份-plan-与元数据)及[现有产物校验](../../schema_extract/tools/artifacts.py)的 `_erase_fields`、`_versions`；其他现行维护说明无直接矛盾，无需重复修改。

文字差异、相对链接及占位符检查通过。新提示仅供后续新 run 使用；FMHA `fb8b3758-92d7-4093-aa18-c2e5028f0a43` 的当前维护已完成，沿用冻结提示，不重跑。
