# 一次裁决

本次 reviewer 只返回 approve 或 blocked，不直接改文件。`adjudication.action=select` 只能选 `invocation.candidates` 列明且亲自完整读取的规范候选；`patch` 基于当前 `base_checkpoint`（`invocation.workspace`），用本阶段授权内的 `{path, old, new}` 唯一匹配替换，`old` 非空。

`select` 的 `artifact_evidence` 取所选候选目录 `materials/candidates/<checkpoint>` 内的文件名与实际行号，不能借 `invocation.workspace` 的引文证明另一份候选；`patch` 与 `blocked` 的产物证据取 `invocation.workspace`。候选的版本、只读来源和规范化已在调用前准备就绪，选择后不再改字节、不追加签收；P2/P4 的 `plan` 须与所选规范候选一致。

Runner 应用 `patch` 后执行同一整体规范化、权限与严格检查，再由全新 reviewer 做一次独立 signoff（P2/P4 的最终 `plan` 一并交签收），只有签收形成最终批准。关键事实不足或越权时返回 blocked，不再开裁决循环。
