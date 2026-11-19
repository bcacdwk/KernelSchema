# FMHA 最终专项审修（2026-11-19）

FMHA 授权审修已完成，正式目录中的实际缺陷已直接修正。当前版本为 **root 8 / mma_sync 4 / split_kv 4**，仍是三份完整设计，dispatch 已同步。现行完整目录 `--read-only --skip-hardware` 检查退出 0，规范化预览零改动，来源定位、结构、预算、`from` 与关联通过。

本次由 `/root/fmha_post_delivery_audit_xhigh`（用户指定 `gpt-6-astra / xhigh`）同一代理承担 worker 与 reviewer：全文阅读三份 YAML、dispatch、旧 receipt 与全部 continuation_audit，沿实际问题核查固定源码，再回读最终改动和有效设计。没有第二独立 reviewer，没有新的 Runner P6 批准；未运行 extract/resume/cancel/monitor、模型 CLI、GPU、commit/push，未改源码、框架、规范或旧 issue。

## 已落实的修订

- **调用与写回**：原 `scale_softmax` 错写为必填 FP32 且混同准备期标量。固定 `input.py` 实际传 None，candidate/reference 均忽略实参；现登记 nullable/default None，另以 `prepared_scale` 表达准备期值并同步消费者。确认 O 必传且覆写原对象，LSE 为 None 时不分配不写，forward 返回 None；修正“输出缓冲之外不得写”与合法 workspace scratch 的矛盾，明确旧输出值不参与计算。
- **有效位与存储**：E4M3×2^e 的有效位至多四位，改正 mma_sync 两处 five；BF16 精确性还需可表示条件，不由 scale 生成方式自动保证。mma_sync 每行由四个 lane 持有状态，`state_regs` 由 `[64,2] / 512 B` 修为 `[64,4,2] / 2048 B`，与 128 线程各 `mA/mB/denA/denB` 一致。root 执行摘要登记 CTA 的 smem/tmem 与 thread 的 register；设计槽位不再等同编译寄存器数或无 spill 保证。
- **数值声明**：源码 NO_KTINY 注释由 `x≤4ln2` 推出 `|x|≤4ln2`，推导错误；`x=-8` 足以反驳，未运行 GPU。撤去 root/split 全域约 1.1 ulp 上界及重结合、倒数归一“必在容差内”的断言。保留真实运算序列，将实验与跨构建位等同性限定为来源声明；split 明确局部累加器、逐组 readback 加法和 FP32 权重/归并的不同舍入。
- **设计适用范围**：split 的 GM 正文是流水体，实际只在每个非空 chunk 至少 8 组时成立；when 已补该条件，并明确非融合单遍 flat scan 和 narrow 必需的 `nsplit≤128`。短 chunk 顺序体仍由原变体记录承担，不扩建节点、不缩小共享 Contract。原 shape 9 的 GM chunk 为 33/33/33/30，仍在该范围。
- **资源与来源缺陷**：将两 CTA 写成容量上限，区分 host 估计与实际常驻，修正 scan 一律单 CTA 的说明；保留实际 PDL wait/Event 和三节点引用结构。mma.sync×split 的旧 GM 漏行缺陷仍记录为源码问题，同时精确补上 `qtiles128*qph<Sq` 条件，不把短 Q 也一律判为漏行。

## 旧 marker 的逐项结论

| 标记 | 本次结论 |
| --- | --- |
| P6 retained-scope inventory / 沿用前提 | 审修前四文件 manifest 与旧 receipt 完全匹配；29 个来源文件同时匹配原冻结快照与固定 commit；当前三设计与来源依赖已按记录范围人工复核。旧工具误拒及批准事实不改写。 |
| 外部比较器来源 | 固定 `fmha_common.cuh:242–251` 有作者实验 ratio，分母使用 `atol+rtol*abs(actual)`；正式验收器本体仍缺，不能将该注释认作 Golden comparator。两条 Tolerance 保留原阈值来源记录，dispatch 说明来源边界。 |
| mma_sync 五位措辞 | 已改为至多四位，并明确 BF16 可表示条件；未声称全动态域或跨构建数值通过。 |
| 原 P6 evidence[13] 528–573 | 审修前 split_kv 确为 572 行，既有独立勘误 528–572 正确；原 reopen/拒收保持原字节，不重放回执。 |
| QK 的 32 keys 轴词 | 四处此前小修正确，已对 `KB=D/32` 和 scale 消费循环复核；保留正确的 QK 通道轴以及 PV/den 键轴。 |

固定来源为 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`，原 snapshot 为 `a6e39e6f3a33bb2e40c8cc754837da6fb2a7b4abb62d5aca96a128f4647c839b`。本次直接依据见 [24 段源码证据](schema_audits/fmha/source_evidence.json)、[29 文件来源摘要](schema_audits/fmha/source_manifest.json) 和 [CPU 算术核对](schema_audits/fmha/arithmetic_checks.json)。这些记录没有声称逐行重审全部候选分支或复现旧性能实验。

## 验证与交付身份

检查命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/root.yaml --read-only --skip-hardware
```

结果：三节点全部通过，无规范化待写，无悬空的已声明引用，检查前后四文件摘要相同。曾按既有流程应用两处多余引号去除和一处 `V→v` 命名规范化；Version 仅随内容修订递增。见 [最终 check](schema_audits/fmha/check.json)、[规范化记录](schema_audits/fmha/normalization.json) 与 [完整 diff](schema_audits/fmha/changes.diff)。`--skip-hardware` 明确跳过硬件，检查不替代 GPU、性能或全动态域数值验证。

审修前全部 YAML/dispatch 和原 receipt 位于 [before](schema_audits/fmha/before.json)；当前交付 manifest 位于 [after](schema_audits/fmha/after.json)。正式 [本次附加记录](../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/schema-audit-20261119.json) SHA256 为 `769eb216b06ce2e1120cc682a58c26f7b7f1e41ff99a0e74f49040d8bfaf2e05`。

原 [receipt](../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/receipt.json) 保持原字节，SHA256 `218410acfc609552953684e47307025e9dd097bfb92d6d918364dc80d52c0c56`。原 run `fb8b3758-92d7-4093-aa18-c2e5028f0a43` 仍是 `invalid_receipt / P6 / published:false`、47 次登记调用；其 state SHA 与原 receipt 完全一致。本次新字节只由用户授权的直接审修记录说明，不能借旧回执称已获得新 P6 批准。

授权的 schema 工作已完成。正式 Golden 比较器本体、完整动态域的数值/性能、跨构建位等同性与实际编译资源仍无新增证据，相关未知项没有填成“已验证无风险”。
