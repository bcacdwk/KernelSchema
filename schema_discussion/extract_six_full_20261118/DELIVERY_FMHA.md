# FMHA 交付：用户授权小修完成

2026-11-19，FMHA 已交付到 [正式目录](../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md)。完成依据为 `user_authorized_minor_fix`。用户明确指令为：“这种小问题，找个xhigh修了就行。不要再打回去继续了。”本次由内部 `gpt-6-astra / xhigh` 子代理完成定位修正与交付，未新增 Schema Extract 模型调用。

- 父 run：`fb8b3758-92d7-4093-aa18-c2e5028f0a43`；父候选：`d3830a2c67c2298494a5d7961e5fde8373c60978cfaa510212cd622910e70252`。
- 改动仅六行：`root.yaml` 183/193、`split_kv.yaml` 18/27 的 QK scale 归约块从“32 keys / 32-key block”改为“32 channels / 32-channel block”；Version 分别 6→7、2→3。固定源码 `fmha_tcg.cuh:2893` 的 `KB=D/32` 及 kb 消费循环、root Contract 76–77 和 mma_sync 平行表述构成修正依据。公式及其他内容保留；`mma_sync.yaml`、`dispatch.md` 与父候选逐字节一致。见 [精确 diff](fmha_minor_fix/parent.diff)。
- 固定 Template 的只读全目录检查通过：三个节点齐全，无排除、无诊断，来源校验通过；检查未改变稿件，跳过硬件，未运行 GPU 或性能实验。
- 使用既有 manifest 校验、fsync 与 `rename_new` 原子不替换复制发布。正式目录字节及回执摘要已复核，原候选及原记录摘要保持。

原 P6 attempt `be584879-c7c2-4210-a68c-0073bf00cda2` 的真实 response 为 `reopen`；调用完成、exit 0、终止确认、session ready。工具因证据第 13 项引用 `split_kv.yaml:528–573`、而实际仅 572 行拒收。此次只在独立 [修正记录](fmha_minor_fix/corrections.json) 中登记坐标应为 528–572，原 response/result/rejected 未改写或重放。原自动 run 保持 `invalid_receipt`、`published=false`，47 次登记调用与原阶段批准保持。本次交付未产生新的正常 Runner 独立 P6 批准，不能将其计作正常 P6 通过。

[交付回执](../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/.extraction/receipt.json) 原样继承执行器续接的 `continuation_audit`，并追加本次坐标纠正、轴词小修及无新 P6 批准的说明，均保留 `pending_user_decision`。此前工具故障影响、外部比较器来源缺口、mma_sync 有效位数措辞等潜在审计点仍待用户在完成后决定是否额外检查或重来。

最终检查：三类在途集合为空，登记 Runner 和本次 P6 进程身份均已退出。完整核对见 [精简证据](DELIVERY_FMHA_EVIDENCE.json) 与 [最终字节/记录/进程核验](fmha_minor_fix/final_verification.json)。
