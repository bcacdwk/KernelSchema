# Sparse 专项人工审修完成

正式目录：[dual_cache_latent_bf16](../../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16)。**四份 YAML 与关联 dispatch 已限域修订，现行完整 checker 通过。** 本轮由 `/root/sparse_post_delivery_audit_xhigh`（gpt-6-astra/xhigh）同一代理承担 worker 和 reviewer 自审；没有另一个独立模型、没有新 P6，也没有启动/恢复 extract。

| 文件 | Version |
| --- | --- |
| `root.yaml` | 4 → 5 |
| `split_kv.yaml` | 2 → 3 |
| `tcgen05.yaml` | 1 → 2 |
| `tcgen05_packed.yaml` | 2 → 3 |

原始五份正文及 receipt 已按原字节备份到 [before/](schema_audits/sparse/before/)，清单见 [before.json](schema_audits/sparse/before.json)。精确变动见 [changes.patch](schema_audits/sparse/changes.patch)，当前五文件完整 SHA-256、版本和核对结果见 [after.json](schema_audits/sparse/after.json)。正式追加证据为 [schema-audit-20261119.json](../../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/.extraction/schema-audit-20261119.json)，SHA-256 `279d47efdfac307328ad964f749a398bc7fe3f0b374d10e9e6f0a6c3a9b70d17`。

## 逐标记结论

1. **P6 retained-scope 误拒及批准适用性：已审计，不需回退 schema。** 原失败 `aa4176a8…` 的 scope 合法命名四节点库存中的 packed 文件；冻结 gate 却只用局部解析的 root 校验。其 response=approve 未采纳、没有原 receipt。正常 `a0af816f…` 的有效批准 manifest、两次只读 workspace、`66ad…` checkpoint 与本轮 before 五文件一致。P1–P4/节点保留批准的 canonical receipt 身份已核实；packed 的旧 P5 批准不被当作后来 V2 字节的单独批准，正常 P6 才覆盖该稿。原 `continuation_audit` 的 pending 标记保留其历史时点，本记录完成用户随后授权的专项核对。详见 [historical-evidence.json](schema_audits/sparse/historical-evidence.json)。
2. **root 的 208 / without spilling：确有 schema 过强保证，已改。** 固定 cfg2 中 qreg/sacc/oacc 是每线程 `64+16+128=208` 个 32 位槽，这个来源成立；原三个 Storage 的字节账也完全保留。临时 fragment、地址和控制状态不在该小计内，冻结 NCU 十用例没有匹配 cfg2 实例，不能据此推出编译后总寄存器或无 spill。root 现在只记录实际 `__launch_bounds__(256,1)`、207,232 B 动态共享内存请求及运行时 opt-in 检查，明说数组账不决定编译分配/spill。

## 必要相关范围及实际修订

- **Storage 名称归属：** 在原 thread 摘要登记 thread-local register，在原 cta 摘要登记 smem/tmem。六个 Execution 键、Worker 与 Role 不变，不新增 register 执行者，不增加硬件容量保证。
- **split 寄存器：** ATTENTION 的 128/64 明确为数组 32 位槽；COMBINE 原“512 线程、两块 launch bound，即 64 registers”不成立。源码只有 `__launch_bounds__(512,2)`，冻结 case 4 的同一 `combine_kernel<32,512,1,false,0>` 实测分配为 48。正文保留 launch 与四组/四深预取结构，目录将 64 标为来源预算，不再当实测值。见 [现有 NCU 核对](schema_audits/sparse/ncu-existing-census.json)，本轮没有运行 profiler。
- **两份 tc 的 row_state：** 源码 SX2 让 128 个 softmax lane 的两个半 warp 各持一份 64 行 m/l。旧 512 B 漏算一份；新增 `lane_half: 2`，改为 `64×2×2×4=1024 B`，原 m/l 视图保留。这是设计存储计数。
- **两份 tc 的 output staging：** `64×1040=66,560 B` 跨过单个 65,536 B gather stage。原 out_block 只指 slot 0 的视图不足。现在同一 131,072 B 分配用 `depth:1`、`shape.stage:2` 和 kv_rows 的 stage region 表达，out_block 保守占整个已死分配；packed hash/index 仍分别在 stage `[0,1)`/`[1,2)`。未增容量或等待、未改两级环的重叠与 modulo-2 复用，关联目录定位已同步。
- **out_buffer / Python ABI：无需改。** `nullable:true`、`default:None`、finite/±inf/NaN 旧内容、None 时分配、给定缓冲时全写回并同对象返回已正确。固定 input.py 的 NaN 预填、reference 与 candidate 的分支已逐项对照，非 Tensor 标量与三个保留 None 对象符合现行规范；Algorithm_requirement 保持原字节语义。
- **最新 packed 与引用：** 两处空 union 的“各角色跨步份额”和最新 Prefer 均保留；源码零分支先退完拷贝、256 线程分工全量写零。四份三区均本地定义，`from` 边为 0；全部 Math、Work_schedule、Pipe、Waiting、Source、when 与名称不变。当前有效来源均含修订后的 root，见 [effective-provenance.json](schema_audits/sparse/effective-provenance.json)。

本轮直接读取 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 的对应源材料；算子下 25 个文件全部与该 commit 字节一致，见 [source-integrity.json](schema_audits/sparse/source-integrity.json)。重要 CUDA 锚：cfg2 6058；数组和槽推导 763–765、850–859、1162–1269；共享内存 guard 5308–5318；COMBINE launch 1558–1567；SX2 行映射/副本 2551–2554、3023–3039、3167–3220；输出 pitch/写回 1840–1847、3463–3555。

## 验证与边界

实际最终命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/root.yaml --read-only --skip-hardware
```

退出 **0**，4 YAML 的结构、文本预算、from 与关联、dispatch 全部通过，规范化预览 **0 would change**；[完整输出](schema_audits/sparse/check-after.json)。两个中间只读检查仅报新 meaning 超预算，已压短后通过，原日志保留。未运行 formatter 写回。另回读四份成品，核实全部 37 处 Storage 容量等式、33 个变更叶及版本、packed 最新稿、原 receipt 和本次读取的历史文件未变，见 [self-review.json](schema_audits/sparse/self-review.json)。`show-effective` 只记录展开 provenance，不被冒充独立校验。

原 receipt SHA-256 保持 `aee7bb8b19cf85ce7eb6ab1347665d6c423161adf9391e76a45a009eb1af53ee`；它对应原 run `8f0844f9-f279-4781-a19d-0216b7fc4ef5`、checkpoint `66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`，仅批准 before 字节。48 次调用及其中 6 次 P6 的历史保持不变，本轮新阶段调用数为 0；取消的 successor 未触碰。

**仍保留的来源限制：** 固定源码的 `bar_land` 与 `tmem_base_s` 同在 `mbar+4*STG`，packed union 又接在其后，控制区重叠仍未修。已复核既有容量、对象寿命、访问与交接足以支持同一完整抽象 Design，按现行源缺陷分流保留，不复制坏地址，不把 schema 审修写成 kernel 修复。另保留原 source-only comparator 缺口、硬件占位、cfg2 编译资源未知及历史时间记录缺口；本轮没有 GPU/性能/数值正确性测试，也不是对全部 48 次原始模型日志的新全面审计。
