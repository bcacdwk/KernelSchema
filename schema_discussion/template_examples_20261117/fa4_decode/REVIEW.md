# FA4 decode 独立全文终审

2026-11-17，新 high reviewer 直接核对固定 v55 源码及四份有效全文，结论：保留 root、q4_cluster、q4_direct、q1_global 四份；当前提取可交主代理签收。下列两项源实现边界仍未证明安全，不将结构/有限图通过写成 GPU 或全域正确性结论。本轮只修改本案例和本讨论目录，未改 kernel、工具、公共文档或规则，未运行 GPU/Runner，未提交 Git。

## 选型与完整性

固定来源是 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution`，不是相近题目的实现。重新实算 paged SHA-256 为 `ec930c32758ea4f34e2df9fefa2de90f3ef345dc11fa66948fef74ae016fbe92`。从 host AST 仅抽出标量 selector/helper/常量执行，六个代表均返回预期模式，未导入 torch/cutlass，结果见 [review_source_checks.log](review_source_checks.log)。

| 内容 | 独立判断及依据 |
| --- | --- |
| 7→4 | root 的独立 CTA FP32 partial/global combine、cluster 的 DSMEM/BF16 reduce-add、direct 的 CTA 内 normalization 分别有完整输出组织价值；Q1 的双 softmax WG 则改变主循环的最大值互斥、S/P 游标与 L parity 生产关系，保留一个完整 global 代表。host `operator.py:78–224,269–290`。 |
| compact 局部闭合 | 名义 S/grid/workspace 不变；请求 worksplit 与空 split 条件一起改变，combine 同时采用 `Mpartial>-inf` guard 的 serial 分支。KV 条纹、主循环、归并拥有者均能直接定位已有基线，未发现另一个必须独立重贯通的机制。host `:730–747`、paged `:1078–1108`、combine `gqa_decode.py:531–555`。 |
| Q1 组合闭合 | 源码实际支持 Q1 cluster/direct；双 WG 生产的每列 max/sum/双 numerator parity 接口不变。querytile1、packedcolumns16、512线程与无 Q4 register redistribution 均已记入 dispatch，不把任意开关组合算作已验证实现。 |
| 共享区 | q1_global 只整区引用 root 的 Math_reshape；querytile、packedcolumns、splitcount、worksplit、bk 都在使用节点绑定，没有继承 root 的 Q4/S9 工作配置。其余设计三区本地。 |
| Contract | 逐项核对 shape_train、input、reference、metadata 与 harness：B1..17、Q{1,4}、Q≤L_b≤131072，固定 None ABI，全部 out 覆盖并返回 alias；out/workspace 以外输入不变。两项输出都用 FP32 有限比较 `abs(a-b)<=0.06+0.04*abs(b)`，不以 native probe 的相对范数阈值替换。 |
| 调用范围 | 四份 when 都是 selector 的真实固定子域；保留同对象/签名、线程和当前设备、非 capture 的 native 稳态命中条件。首次正常计算、clone、重复 ffi、同步比较和失败恢复已在 dispatch 简记，源据 host `:474–583,652–843`。 |
| 硬件 | validate.json 的实际环境支持 L20D/SM103、CuTe DSL 4.6.1、torch 2.9.0+cu130、cuda-python 13.1.1。既有事实保留，未知 Wiki 仍按规范占位；所有检查均未 skip-hardware。 |

## 数值、协议与实际修正

重新核对了两次 FP8 PV：leading 和 residual 在每个原生 V reduction slice 内顺次累加到同一 FP32 numerator；指数和分母不窄化。两个 O/L parity 的 correction 与最后合并保留。N1 的空 Y/L parity 实际初始化、独立通知、读取和释放均有动作；N0 仍完成 page/CTA/TMEM 准备，发布中性统计并释放 TMEM。源据 paged `:1510–1579,1664–1788,1834–2051`。

cluster 的16个成员由 Mapping.Member 单点 split 绑定；i 域独立为 `ceil_div(max(0,T_b-split),16)`，没有统一假 N。空成员仍参加4轮 max和4轮 sum，split0 才清 out并发布 zero_done；活跃输出在接收通知及 async-proxy fence 后执行 BF16 contribution reduce-add。数学保留分别 BF16 转换及运行时归约次序，不等同于 FP32 汇总后一次 cast。源据 paged `:796–827,1388–1398,1847–1917`、combine `:205–326`。

实际改动如下，四份版本为 root11、其余9：

1. 原 SOFTMAX.compute:maximum 误写“producer rendezvous 之前捕获”。现在 compute 发布 rendezvous，随后独立 load:maximum 捕获到 softmaxmaximum；Correction 与 Softmax 可在发布后分别读取。Q1 的下一 parity mutex 等待前一 WG 的捕获完成，和源码 `colmax=sM.load()` 后才 release 的次序一致（paged `:1726–1754`）。
2. SCORE 的纯 sum_ready rendezvous 从 compute 改为 await，依赖约束其 done；global 的四项分母归约和 lpartial 写回从 store 改为 compute。实际数据/通知不变，符合 DOC5 的 verb 含义。
3. direct 的 SCORE 原先声称读取 mlocal。LSE/sink 均关闭时源码只等待 final-max rendezvous，没有该读取，现改 await:maximum_ready，normalization 只读 finalsum（combine `:145–159`）。
4. cluster 的 REALIZATION.vslot 用本成员 `count(i)` 替换未绑定的 N；真实槽数、成员数及分片方法不变。
5. 更新 README/WRITE 的图和成员工具状态，撤回 SELECTION 中残留的 PageIndex 安全断言。四份同名 PNG 全部按最终来源重绘。

逐项复核 Storage 的单成员形状和 worker 总量，cluster 按16成员计数；Q4/Q1 的 KV 环分别8/10槽。SMEM 的最终 ostage 共享 KV 区且保持消费者退休条件，TMEM 由 Correction 捕获两尾槽后归还，Value 才 dealloc。issue-turn 仍在最后 operand acquire 后、copy/MMA 前释放，不代表结果完成。PDL 保留 launch-ready、partial 可读与最终输出完成的区别，combine 在首次 partial 读前等待。源码 helper 容量计算与实际分配粒度仍在 REALIZATION，未新增硬件容量保证。

## 未决源边界

- **PageIndex 覆盖顺序**：独立复核仍没有找到 Page4 覆盖槽0前 V0 必已捕获的保证。K3 开头预取在 V-turn 等待之前；K2 的 KV acquire 只等 K0 operand 退休，不能推出 V0 PageIndex 读取。容量断言和 cp.async wait_group 都不能替代另一 warp 的读者交接。详见 [pageindex_order.md](pageindex_order.md)。没有补同步或隐藏风险。
- **空 global O payload**：非 compact combine 对所有名义 split 读取 opartial，零权不能中和 NaN。空 split 不写 O；首次零 scratch 不证明跨 `(S,B,Q)` 复用后这些 FP32 位仍有限。已在数学、README、dispatch 和 REALIZATION 保留，不添加 per-call clear 或 guard。源据 host `:227–255`、paged `:1138–1145`、combine `:495–530`。

没有新的公共工具缺陷。有限图只检查声明的事件关系，不推断共享地址冲突、不模拟其他 worker 或跨 step 活性；历史30/30只支持那些输入。

## 最终命令与结果

均从仓库根目录使用 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py`，所有下表调用退出码0。最终两条普通命令为：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/root.yaml
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/root.yaml --read-only
```

分别为 `4 YAML; 0 changed`、`4 YAML; 0 would change`；结构、文本、from、关系、dispatch 均通过，日志见 [默认](review_final_default.log)、[只读](review_final_readonly.log)。

以下每行命令的文件前缀为 `schema_template/examples/fa4_decode/`，各次均加 `--worktiles 1`；`actions / edges / boundary skips` 为实际完整展开数：

| 文件及其余命令参数 | PARTIAL 或 MAIN | COMBINE | 日志 |
| --- | --- | --- | --- |
| root.yaml `--graph png --set N=5 --set splitcount=9` | 351 / 1168 / 20 | 24 / 70 / 0 | [PNG](review_root_png.log) |
| q1_global.yaml `--graph png --set N=5 --set splitcount=16` | 359 / 1196 / 25 | 38 / 112 / 0 | [PNG](review_q1_global_png.log) |
| q4_direct.yaml `--graph png --set N=1 --set splitcount=1` | 88 / 284 / 8 | — | [PNG](review_q4_direct_png.log) |
| q4_cluster.yaml `--graph png --set T_b=16 --set splitcount=16` | 1730 / 5782 / 128 | — | [PNG](review_q4_cluster_png.log) |
| root.yaml `--read-only --graph check --set N=14 --set splitcount=9` | 954 / 3229 / 20 | 24 / 70 / 0 | [检查](review_root_N14_check.log) |
| root.yaml `--read-only --graph check --set N=0 --set splitcount=9` | 13 / 32 / 0 | 24 / 70 / 0 | [检查](review_root_N0_check.log) |
| root.yaml `--read-only --graph check --set N=1 --set splitcount=9` | 88 / 281 / 8 | 24 / 70 / 0 | [检查](review_root_N1_check.log) |
| q1_global.yaml `--read-only --graph check --set N=8 --set splitcount=16` | 566 / 1904 / 25 | 38 / 112 / 0 | [检查](review_q1_global_N8_check.log) |
| q1_global.yaml `--read-only --graph check --set N=0 --set splitcount=16` | 14 / 34 / 0 | 38 / 112 / 0 | [检查](review_q1_global_N0_check.log) |
| q1_global.yaml `--read-only --graph check --set N=1 --set splitcount=16` | 91 / 290 / 8 | 38 / 112 / 0 | [检查](review_q1_global_N1_check.log) |
| q4_cluster.yaml `--read-only --graph check --set T_b=70 --set splitcount=16` | 5268 / 17684 / 320 | — | [检查](review_q4_cluster_T_b70_check.log) |
| q4_cluster.yaml `--read-only --graph check --set T_b=1 --set splitcount=16` | 590 / 1972 / 8 | — | [检查](review_q4_cluster_T_b1_check.log) |

当前 PNG 的实际输入见 [案例 README](../../../schema_template/examples/fa4_decode/README.md)。cluster 正式图采用 when 内合法 B1/Q4/L4096/P64、T_b16：16成员各一轮，展示 N1 空 parity 与完整统计协议。工具仅在显示时按现有规则合并相同轨迹并列出成员编号，完整事件数仍包含16成员；已查看缩略图。T_b70 仍是成员0..5各5轮、其余4轮的完整检查。T_b1 来自同机制 B2/Q4/seq_lens=[64,17844]/P280 的 request0，1活跃/15空成员，**不在文件 B1 的 when 内**，仅为额外边界检查。未扩大 when、改变 S 或删成员。

root N5 与 Q1 N5 图都取各自历史实际 batch 的 request0；最长请求 N14/N8另行检查。全局图各显示一个 partial worker 和一个 combine worker 的接口，不能看作全部 producer。主图和这些检查都不把 PageIndex 风险标成安全。

早期 T_b70/T_b17 的 PNG 渲染均超过60秒，旧 root N14/Q1 N8也有同类超时；这些历史失败仍留日志。本轮 T_b16 在修改前已成功一次（[日志](review_cluster_t16_png.log)，1714动作），最终修订后重新生成1730动作的图；旧跨 when 的 T_b1 PNG已被替换。没有把旧计数或旧摘要当成最终值。

使用 `tools.pipeline.output.metadata` 读取四张 PNG，并逐项实算 `source_sha256/source_manifest` 对应的 raw YAML，全部一致，见 [来源核对](review_png_provenance.log)。`git diff --check -- schema_template/examples/fa4_decode schema_discussion/template_examples_20261117/fa4_decode` 无输出。公共工具冻结后的135项测试结果沿用 [MEMBER_GRAPH.md](../MEMBER_GRAPH.md)，本 reviewer 未改工具、不重复该全套测试。

本案例写入已结束；共同入口和最终集合收尾由主代理接续。
