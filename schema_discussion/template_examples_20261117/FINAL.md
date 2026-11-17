# 四例最终验收

2026-11-17，接受当前九份完整设计及其目录、README 和同名 PNG；追加的 xhigh 工具审阅与最终验收均已完成。四个案例已分别完成源码写作与独立全文审阅，本次再核对集合、共享关系、来源、现行文档和交付文件；不重新扩张选型。公共入口为 [Template 案例](../../schema_template/examples/README.md)，简短变更见 [CHANGELOG.md](CHANGELOG.md)。

## 集合与来源

| 案例 | 旧→现行 | 保留的完整设计及价值 |
| --- | --- | --- |
| FA4 prefill | 11→3 | root：静态每 CTA pair 单工作；packed_persistent：跨工作连续环与提前输出释放；split_kv：五路 partial 与独立归并 |
| FA4 decode | 7→4 | root：Q4/global；q4_cluster：DSMEM 蝶形统计与 BF16 reduce-add；q4_direct：CTA 内输出；q1_global：双 softmax WG 的完整主循环 |
| QSA prefill | 1→1 | 行块 union 预处理、ticket 动态工作与跨工作 attention |
| GDN prefill | 2→1 | 多序列 M64、64-token 成对 chunk、片上状态递推；single 输入/通知变化在目录闭合 |

21→9 包括删除13份旧 YAML/PNG、新增 `split_kv`；九份文件名与 Dispatch 身份一致，各目录只有 root 定义 Contract。仅 prefill/packed_persistent 和 decode/q1_global 整区引用各自 root 的 Math_reshape，provider/chain 与消费端参数绑定一致；其余 Design 本地。当前公共入口、案例 README/dispatch 没有指向已删文件的链接。目录保留参数、局部和组合的实际基线与源码定位；CLC 如实标为未展开独立候选，未假称已交付。

来源采用 [SOURCES.md](SOURCES.md) 的四题快照，固定提交 `3d64ad35900a0e6a66d6f93da34108fbceba0daf`、tree `67de35d84d464abce36452a3cc6b85fefa07b937`。恢复阶段已逐项核对470文件、52,207,886字节和权限；最终另对四个核心源码实算 SHA-256 并与固定 Git blob 比较，全部相同。`catalog.py check` 返回 passed：44 operators、1266 source_files、3 derived_indexes、gpu_evaluation_rerun=false。两个批次没有混用。

## 最终实际检查

所有 Python 命令均用 `/opt/venv/python3.13.12/bin/python3 -B`。完整命令和输出见 [final_checks.log](final_checks.log)：四例各运行默认和只读一次，均为零变化；逐份采用当前案例 README 参数运行 `--read-only --graph check`，再检查六组代表边界。共23次 CLI 退出0，检查期间所有 YAML/PNG 字节与 mtime 不变，没有跳过硬件。

下表为最终九份主图参数下的 `actions / edges / boundary skips`，均无环：

| 文件 | Step 与结果 |
| --- | --- |
| fa4_prefill/root | MAIN 194 / 700 / 29 |
| fa4_prefill/packed_persistent | MAIN 217 / 782 / 40 |
| fa4_prefill/split_kv | ATTENTION 194 / 705 / 18；COMBINE 20 / 58 / 1 |
| fa4_decode/root | PARTIAL 351 / 1168 / 20；COMBINE 24 / 70 / 0 |
| fa4_decode/q1_global | PARTIAL 359 / 1196 / 25；COMBINE 38 / 112 / 0 |
| fa4_decode/q4_direct | MAIN 88 / 284 / 8 |
| fa4_decode/q4_cluster | MAIN 1730 / 5782 / 128 |
| qsa_prefill/root | PREPROCESS 2 / 4 / 0；ATTENTION 158 / 586 / 12 |
| gdn_prefill/root | INDEX_CAST 1 / 1 / 0；GDN 249 / 979 / 30 |

整合阶段六组代表检查为 prefill N1（71/225/14）、persistent 有效工作→padding（80/259/21）、split 空 ATTENTION（12/22/0，COMBINE 仍20/58/1）；decode cluster T_b70（5268/17684/320）；QSA 空 worker（ATTENTION 9/26/0）；GDN 最小2 chunks（68/242/19）。全部无环，其他审阅过的排列不再重跑。

追加工具修正完成后，独立 xhigh 按相同参数重查九份主图与上述六组边界，计数逐项一致；另查 decode cluster T_b1（590/1972/8）。最终16次只读检查全部退出0、零规范化变化、无环，见 [tools_polish_cases.log](tools_polish_cases.log)。签收采用该冻结后结果，不再重复探针；七组边界中的 T_b1 仍只作跨 when 的同机制诊断。

另外从当前 decode cluster 的实际 Operation.context 复核 T_b70：成员0..5各5轮、6..15各4轮，每个动作的 split 等于其成员，异构轨迹不折叠，见 [final_member_context.json](final_member_context.json)。当前 PNG 采用 when 内 B1/Q4/L4096/P64、T_b16，16成员各一轮；历史 T_b1 跨 when 诊断已被替换，未扩大 when 或删成员。QSA 两工作×四片低于真实至少32片下界，仅作局部协议示意；GDN 6+2 主图与最小2 chunks 的合法输入关系、prefill rescale 分支示意范围均保留在案例 README。

PNG 逐项核对 raw/source_manifest、provider/chain、effective identity、样本、步数与动作数，并验证 PNG chunk 校验和及像素数据、查看九张缩览。发现 QSA 图绑定最后文字修改前摘要后，仅按原 README 参数重绘，结果见 [final_qsa_png.log](final_qsa_png.log)；YAML 未改，另外八图未重绘。最终九图完整身份见 [final_artifacts.json](final_artifacts.json)，QSA 当前 raw 摘要为 `04483570534cd429c34fdb721a3612d421fd8f5f9a5b65588688935c0aee0cbd`。

工具最终冻结后再次逐项核对九份 YAML/PNG 的身份和样本，均与该记录一致；当前公共入口与交付记录的本地文件链接检查通过，`git diff --check` 无输出。

## 工具与文档

独立复核了 [TOOL_FIXES.md](TOOL_FIXES.md)、[MEMBER_GRAPH.md](MEMBER_GRAPH.md) 对应代码与回归，并签收 [TOOLS_POLISH.md](TOOLS_POLISH.md) 的最终修正：局部 count/last 需要绑定外层坐标，动态工作总长和完整 flatten 查询仍拒绝；固定参数只投影自身坐标且保留元组关系；假目标不能隐藏窗口内缺失前驱。成员数学坐标只由 Mapping.Member 已登记的单点区域取得，不读 Strategy 猜值；局部同名索引按词法作用域遮蔽，范围端点仍取外层值。域、peer 窗口和实际事件分别按成员处理，Waiting 地址按当前目标成员解释，Event 显式实参优先。有界 XOR/左移接受精确整数值，拒绝布尔、非整数与负位移，位移超过40或结果超过既有幅度限制报告未完成。无新 schema 字段、保留变量或一般条件求解器。

工具最后冻结时137项测试在30.294秒内全部通过，见 [tools_polish_tests.log](tools_polish_tests.log)；两份原始复现也完成。追加审阅修正两个遗漏并合并窗口投影/缓存和成员查找，精简重复遍历及历史性注释；九份现有主图结果不变，未重绘。QSA 保留等价 `tile_count(t)-1` 写法，其早期局部 last(i) 误拒已修复；工具阶段报告中的旧 PNG hash/计数只代表该阶段，以本最终记录和案例 REVIEW 为准。公共 examples README 已重写为当前导航，根/Template README 及执行器说明的案例状态同步；工具 README 的 Member API 与实现一致。历史讨论、历史 CHANGELOG、固定快照、44题库、schema_wiki、Extract 代码/状态及旧 text-limit demo 未改。

`EXECUTOR_ALIGNMENT.md` 在本轮只更新案例状态一句。工作区预先存在的 Claude/Qwen 说明追加及 `schema_extract/inputs/nvfp4_fused_moe_default_scope.md` 删除未由整合者改动，不计入本轮交付或 Extract 对接成果。

## 遗留与结论边界

四例 REVIEW 的关键限制均保留：prefill 的快速数学误差与空 PartialO 实际预取；decode 的 PageIndex 覆盖顺序和空 global O 跨布局复用；QSA 的未写 padding 投票、HalfMax 覆盖、缓存/捕获与近似误差；GDN 的 descriptor 在途读取/缓存互斥与快速数学误差。既有硬件事实保留，正式 Wiki 占位不等于资料绑定完成；不足信息没有编造为保证。详见各例 [prefill](fa4_prefill/REVIEW.md)、[decode](fa4_decode/REVIEW.md)、[QSA](qsa_prefill/REVIEW.md)、[GDN](gdn_prefill/REVIEW.md) 审阅。

有限图不证明全部输入、数值、竞态自由、跨 step 活性或性能；历史 GPU 记录只在其原输入与判据范围内有效。本轮没有 GPU/Extract Runner 重跑，没有修改源 kernel、流程或冻结批准，也没有 Git commit。当前交付无需用户补充决定；补齐资料、修正源实现或接入 Runner 属于后续独立工作。
