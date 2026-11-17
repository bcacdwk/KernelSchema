# 一次裁决

本次 reviewer 只返回 approve 或 blocked，不直接改文件。adjudication.action=select 只能选列明且亲自完整读取的规范候选；patch 基于当前 checkpoint，用本阶段授权内的 {path, old, new} 唯一匹配替换，old 非空。

证据仍锚定实际读取的候选 raw。Runner 应用补丁后执行同一整体规范化、权限与严格检查；改稿由全新 reviewer 做一次独立 signoff，只有签收形成最终批准。选择已经读取且字节不再变化的规范候选不追加签收。关键事实不足或越权时 blocked，不再开裁决循环。

select 的 artifact_evidence 使用所选候选目录内的文件名与实际行号；它不能借当前 workspace 的引文证明另一份候选。候选的版本、只读来源和规范化已经在调用前准备，不依赖批准后再改字节。P2/P4 的 plan 必须与该规范候选的决定一致；patch 的最终 plan 一并交后续签收。
