# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 默认行为的常规吞吐：多 kernel 准备、cfg97 融合 FC1、cfg81 WS/TSTORE FC2。 | 代表路径；保存共享 Contract 和已审完整设计。 | 有效或缓存 ABI 满足 root.when：默认行为、cfg97/cfg81、zflag=0、R>S；显式设置同等行为也包含。 |
| [bf16_tc](bf16_tc.yaml) | mono_fused_mono | 显式选择 BF16 tcgen05 梯级。 | 增加 hidden 解码，GEMM 内解码权重，FC1 融合激活/QDQ 后物化 BF16 act；FC2 写 BF16 route。 | 有效 cfg、cfg2 均在4..6；按两个实例各自 BM/BN/NSUB 描述，不补造配对校正。 |
| [bf16_warp](bf16_warp.yaml) | bf16_tc | 显式选择 BF16 warp-mma 梯级。 | 改为 warp-mma，物化 FP32 FC1 后独立 k_act 产生 BF16 act；两次投影的 tile 取各自 cfg。 | 有效 cfg、cfg2 均在0..3；run_pipeline 走 decode、k_gemm、k_act、k_gemm、finalize。 |
| [mono_arrival_flat](mono_arrival_flat.yaml) | mono_split_flat | 单体 last-arriver FC1；flat-stream FC2。 | 删除独立激活，由最后到达 CTA 归约部分和并量化；保留全局计数与可见性协议。 | zflag=0；实际 FC1 为 f4_body 的 G1、FUSE=true（cfg31/32），FC2 为 f4_flat_g2。 |
| [mono_arrival_mono](mono_arrival_mono.yaml) | mono_split_mono | 单体 last-arriver FC1；单体 FC2。 | 删除独立激活，由最后到达 CTA 归约部分和并量化；保留全局计数与可见性协议。 | zflag=0；实际 FC1 为 f4_body 的 G1、FUSE=true（cfg31/32），FC2 为 f4_body 的 G2、REDF=false。 |
| [mono_arrival_redf](mono_arrival_redf.yaml) | mono_split_redf | 单体 last-arriver FC1；REDF FC2。 | 删除独立激活，由最后到达 CTA 归约部分和并量化；保留全局计数与可见性协议。 | zflag=0；实际 FC1 为 f4_body 的 G1、FUSE=true（cfg31/32），FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [mono_arrival_w1](mono_arrival_w1.yaml) | mono_split_w1 | 单体 last-arriver FC1；单 warp FC2。 | 删除独立激活，由最后到达 CTA 归约部分和并量化；保留全局计数与可见性协议。 | zflag=0；实际 FC1 为 f4_body 的 G1、FUSE=true（cfg31/32），FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [mono_arrival_ws](mono_arrival_ws.yaml) | mono_split_ws | 单体 last-arriver FC1；WS FC2。 | 删除独立激活，由最后到达 CTA 归约部分和并量化；保留全局计数与可见性协议。 | zflag=0；实际 FC1 为 f4_body 的 G1、FUSE=true（cfg31/32），FC2 为 f4_body_ws 的 G2。 |
| [mono_fused_flat](mono_fused_flat.yaml) | ws_fused_flat | 单体融合 FC1；flat-stream FC2。 | FC1 改为四 warp 单体主循环及融合激活；复制方式和 PIPE 等取实际实参。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK=1，FC2 为 f4_flat_g2。 |
| [mono_fused_mono](mono_fused_mono.yaml) | ws_fused_mono | 单体融合 FC1；单体 FC2。 | FC1 改为四 warp 单体主循环及融合激活；复制方式和 PIPE 等取实际实参。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK=1，FC2 为 f4_body 的 G2、REDF=false。 |
| [mono_fused_redf](mono_fused_redf.yaml) | ws_fused_redf | 单体融合 FC1；REDF FC2。 | FC1 改为四 warp 单体主循环及融合激活；复制方式和 PIPE 等取实际实参。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK=1，FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [mono_fused_w1](mono_fused_w1.yaml) | ws_fused_w1 | 单体融合 FC1；单 warp FC2。 | FC1 改为四 warp 单体主循环及融合激活；复制方式和 PIPE 等取实际实参。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK=1，FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [mono_fused_ws](mono_fused_ws.yaml) | ws_fused_ws | 单体融合 FC1；WS FC2。 | FC1 改为四 warp 单体主循环及融合激活；复制方式和 PIPE 等取实际实参。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK=1，FC2 为 f4_body_ws 的 G2。 |
| [mono_split_flat](mono_split_flat.yaml) | mono_fused_flat | 单体分片 FC1；flat-stream FC2。 | FC1 改为单体 split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK>1、FUSE=false，FC2 为 f4_flat_g2。 |
| [mono_split_mono](mono_split_mono.yaml) | mono_fused_mono | 单体分片 FC1；单体 FC2。 | FC1 改为单体 split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK>1、FUSE=false，FC2 为 f4_body 的 G2、REDF=false。 |
| [mono_split_redf](mono_split_redf.yaml) | mono_fused_redf | 单体分片 FC1；REDF FC2。 | FC1 改为单体 split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK>1、FUSE=false，FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [mono_split_w1](mono_split_w1.yaml) | mono_fused_w1 | 单体分片 FC1；单 warp FC2。 | FC1 改为单体 split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK>1、FUSE=false，FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [mono_split_ws](mono_split_ws.yaml) | mono_fused_ws | 单体分片 FC1；WS FC2。 | FC1 改为单体 split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body 的 G1、SK>1、FUSE=false，FC2 为 f4_body_ws 的 G2。 |
| [route_fused](route_fused.yaml) | root | 空路由发布或显式启用的 MEGA 路由计算。 | 阶段链合为单 kernel，逐 route FP32 累加及逐 token 完成计数；cfg91 暂存权重，cfg92 直接读取，作为 STAGED 参数保留。 | 有效 cfg 为91或92；自动门见 Model.forward 1415–1426，显式 cfg 先经 rung_ok。回放以当前 ids 计算，不能将发布时 empty 写成永久清零。 |
| [selfprep_flat](selfprep_flat.yaml) | ws_split_flat | grid 内准备的 WS 分片 FC1；flat-stream FC2。 | 各 FC1 CTA 重算并发布路由映射；独立准备是否仍提交服从 PREP2、KEEP_PREP 和 R/S。 | zflag=0；实际 FC1 为 f4_body_ws 的 SELFPREP=true（cfg96，R<=256），FC2 为 f4_flat_g2。 |
| [selfprep_mono](selfprep_mono.yaml) | ws_split_mono | grid 内准备的 WS 分片 FC1；单体 FC2。 | 各 FC1 CTA 重算并发布路由映射；独立准备是否仍提交服从 PREP2、KEEP_PREP 和 R/S。 | zflag=0；实际 FC1 为 f4_body_ws 的 SELFPREP=true（cfg96，R<=256），FC2 为 f4_body 的 G2、REDF=false。 |
| [selfprep_redf](selfprep_redf.yaml) | ws_split_redf | grid 内准备的 WS 分片 FC1；REDF FC2。 | 各 FC1 CTA 重算并发布路由映射；独立准备是否仍提交服从 PREP2、KEEP_PREP 和 R/S。 | zflag=0；实际 FC1 为 f4_body_ws 的 SELFPREP=true（cfg96，R<=256），FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [selfprep_w1](selfprep_w1.yaml) | ws_split_w1 | grid 内准备的 WS 分片 FC1；单 warp FC2。 | 各 FC1 CTA 重算并发布路由映射；独立准备是否仍提交服从 PREP2、KEEP_PREP 和 R/S。 | zflag=0；实际 FC1 为 f4_body_ws 的 SELFPREP=true（cfg96，R<=256），FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [selfprep_ws](selfprep_ws.yaml) | ws_split_ws | grid 内准备的 WS 分片 FC1；WS FC2。 | 各 FC1 CTA 重算并发布路由映射；独立准备是否仍提交服从 PREP2、KEEP_PREP 和 R/S。 | zflag=0；实际 FC1 为 f4_body_ws 的 SELFPREP=true（cfg96，R<=256），FC2 为 f4_body_ws 的 G2。 |
| [w1_fused_flat](w1_fused_flat.yaml) | mono_fused_flat | 单 warp 融合 FC1；flat-stream FC2。 | FC1 改为单 warp 主循环、其余 warp 参与 epilogue，仍融合激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK=1（cfg33 的实际分支），FC2 为 f4_flat_g2。 |
| [w1_fused_mono](w1_fused_mono.yaml) | mono_fused_mono | 单 warp 融合 FC1；单体 FC2。 | FC1 改为单 warp 主循环、其余 warp 参与 epilogue，仍融合激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK=1（cfg33 的实际分支），FC2 为 f4_body 的 G2、REDF=false。 |
| [w1_fused_redf](w1_fused_redf.yaml) | mono_fused_redf | 单 warp 融合 FC1；REDF FC2。 | FC1 改为单 warp 主循环、其余 warp 参与 epilogue，仍融合激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK=1（cfg33 的实际分支），FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [w1_fused_w1](w1_fused_w1.yaml) | mono_fused_w1 | 单 warp 融合 FC1；单 warp FC2。 | FC1 改为单 warp 主循环、其余 warp 参与 epilogue，仍融合激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK=1（cfg33 的实际分支），FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [w1_fused_ws](w1_fused_ws.yaml) | mono_fused_ws | 单 warp 融合 FC1；WS FC2。 | FC1 改为单 warp 主循环、其余 warp 参与 epilogue，仍融合激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK=1（cfg33 的实际分支），FC2 为 f4_body_ws 的 G2。 |
| [w1_split_flat](w1_split_flat.yaml) | mono_split_flat | 单 warp 分片 FC1；flat-stream FC2。 | FC1 改为单 warp split-K 主循环，保留部分和及独立归约激活。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK>1（cfg34/35 的实际分支），FC2 为 f4_flat_g2。 |
| [w1_split_mono](w1_split_mono.yaml) | mono_split_mono | 单 warp 分片 FC1；单体 FC2。 | FC1 改为单 warp split-K 主循环，保留部分和及独立归约激活。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK>1（cfg34/35 的实际分支），FC2 为 f4_body 的 G2、REDF=false。 |
| [w1_split_redf](w1_split_redf.yaml) | mono_split_redf | 单 warp 分片 FC1；REDF FC2。 | FC1 改为单 warp split-K 主循环，保留部分和及独立归约激活。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK>1（cfg34/35 的实际分支），FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [w1_split_w1](w1_split_w1.yaml) | mono_split_w1 | 单 warp 分片 FC1；单 warp FC2。 | FC1 改为单 warp split-K 主循环，保留部分和及独立归约激活。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK>1（cfg34/35 的实际分支），FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [w1_split_ws](w1_split_ws.yaml) | mono_split_ws | 单 warp 分片 FC1；WS FC2。 | FC1 改为单 warp split-K 主循环，保留部分和及独立归约激活。 | zflag=0；实际 FC1 为 f4_body_w1 的 G1、SK>1（cfg34/35 的实际分支），FC2 为 f4_body_ws 的 G2。 |
| [ws12_flat](ws12_flat.yaml) | ws_fused_flat | 十二 warp 融合 FC1；flat-stream FC2。 | FC1 改为十二 warp 专用分工、双累加器及四槽绑定 payload/scale 环。 | zflag=0；实际 FC1 为 f4_ws_g1（cfg26 的实际分支），FC2 为 f4_flat_g2。 |
| [ws12_mono](ws12_mono.yaml) | ws_fused_mono | 十二 warp 融合 FC1；单体 FC2。 | FC1 改为十二 warp 专用分工、双累加器及四槽绑定 payload/scale 环。 | zflag=0；实际 FC1 为 f4_ws_g1（cfg26 的实际分支），FC2 为 f4_body 的 G2、REDF=false。 |
| [ws12_redf](ws12_redf.yaml) | ws_fused_redf | 十二 warp 融合 FC1；REDF FC2。 | FC1 改为十二 warp 专用分工、双累加器及四槽绑定 payload/scale 环。 | zflag=0；实际 FC1 为 f4_ws_g1（cfg26 的实际分支），FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [ws12_w1](ws12_w1.yaml) | ws_fused_w1 | 十二 warp 融合 FC1；单 warp FC2。 | FC1 改为十二 warp 专用分工、双累加器及四槽绑定 payload/scale 环。 | zflag=0；实际 FC1 为 f4_ws_g1（cfg26 的实际分支），FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [ws12_ws](ws12_ws.yaml) | ws_fused_ws | 十二 warp 融合 FC1；WS FC2。 | FC1 改为十二 warp 专用分工、双累加器及四槽绑定 payload/scale 环。 | zflag=0；实际 FC1 为 f4_ws_g1（cfg26 的实际分支），FC2 为 f4_body_ws 的 G2。 |
| [ws_arrival_flat](ws_arrival_flat.yaml) | ws_split_flat | WS last-arriver FC1；flat-stream FC2。 | 删除独立激活，由最后到达 CTA 的 epilogue 归约/量化，计数和部分和寿命随之改变。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、FUSE=true（cfg88），FC2 为 f4_flat_g2。 |
| [ws_arrival_mono](ws_arrival_mono.yaml) | ws_split_mono | WS last-arriver FC1；单体 FC2。 | 删除独立激活，由最后到达 CTA 的 epilogue 归约/量化，计数和部分和寿命随之改变。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、FUSE=true（cfg88），FC2 为 f4_body 的 G2、REDF=false。 |
| [ws_arrival_redf](ws_arrival_redf.yaml) | ws_split_redf | WS last-arriver FC1；REDF FC2。 | 删除独立激活，由最后到达 CTA 的 epilogue 归约/量化，计数和部分和寿命随之改变。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、FUSE=true（cfg88），FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [ws_arrival_w1](ws_arrival_w1.yaml) | ws_split_w1 | WS last-arriver FC1；单 warp FC2。 | 删除独立激活，由最后到达 CTA 的 epilogue 归约/量化，计数和部分和寿命随之改变。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、FUSE=true（cfg88），FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [ws_arrival_ws](ws_arrival_ws.yaml) | ws_split_ws | WS last-arriver FC1；WS FC2。 | 删除独立激活，由最后到达 CTA 的 epilogue 归约/量化，计数和部分和寿命随之改变。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、FUSE=true（cfg88），FC2 为 f4_body_ws 的 G2。 |
| [ws_fused_flat](ws_fused_flat.yaml) | ws_fused_mono | WS 融合 FC1；flat-stream FC2。 | FC2 改为跨 workitem 的连续 stage 流；NK1 由 I=256 决定，保留 route/finalize。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK=1、非 SELFPREP，FC2 为 f4_flat_g2。 |
| [ws_fused_mono](ws_fused_mono.yaml) | ws_fused_ws | WS 融合 FC1；单体 FC2。 | FC2 改为单体主循环及 BF16 route 写回，保留独立 finalize。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK=1、非 SELFPREP，FC2 为 f4_body 的 G2、REDF=false。 |
| [ws_fused_redf](ws_fused_redf.yaml) | ws_fused_mono | WS 融合 FC1；REDF FC2。 | FC2 将 BF16 舍入后的 route 加权原子累加至 FP32 acc32；侧流清零并在 FC2 前汇合，末尾 convert。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK=1、非 SELFPREP，FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [ws_fused_w1](ws_fused_w1.yaml) | ws_fused_mono | WS 融合 FC1；单 warp FC2。 | FC2 改为单 warp 主循环及 BF16 route 写回，保留独立 finalize。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK=1、非 SELFPREP，FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [ws_fused_ws](ws_fused_ws.yaml) | root | WS 融合 FC1；WS FC2。 | FC1 保持 WS 融合激活；扩展源实例参数及准备/gather 策略。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK=1、非 SELFPREP，FC2 为 f4_body_ws 的 G2。排除 root.when。 |
| [ws_split_flat](ws_split_flat.yaml) | ws_fused_flat | WS 分片 FC1；flat-stream FC2。 | FC1 改为 WS split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK>1、FUSE=false、SELFPREP=false，FC2 为 f4_flat_g2。 |
| [ws_split_mono](ws_split_mono.yaml) | ws_fused_mono | WS 分片 FC1；单体 FC2。 | FC1 改为 WS split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK>1、FUSE=false、SELFPREP=false，FC2 为 f4_body 的 G2、REDF=false。 |
| [ws_split_redf](ws_split_redf.yaml) | ws_fused_redf | WS 分片 FC1；REDF FC2。 | FC1 改为 WS split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK>1、FUSE=false、SELFPREP=false，FC2 为 f4_body 的 G2、REDF=true（cfg2=46/47）。 |
| [ws_split_w1](ws_split_w1.yaml) | ws_fused_w1 | WS 分片 FC1；单 warp FC2。 | FC1 改为 WS split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK>1、FUSE=false、SELFPREP=false，FC2 为 f4_body_w1 的 G2（cfg36 的实际分支）。 |
| [ws_split_ws](ws_split_ws.yaml) | ws_fused_ws | WS 分片 FC1；WS FC2。 | FC1 改为 WS split-K，物化 FP32 部分和，再独立归约激活/QDQ。 | zflag=0；实际 FC1 为 f4_body_ws 的 G1、SK>1、FUSE=false、SELFPREP=false，FC2 为 f4_body_ws 的 G2。 |
| [zero_fused_flat](zero_fused_flat.yaml) | ws_fused_flat | 零探测成功；融合 FC1；flat-stream FC2。 | 链首验证当前字节；clean 时清零并由 guards 跳过计算，dirty 执行该配对完整链；单图/双图及失败回落由实际 graph 状态参数化。 | 有效 zflag 非零，cfg=97，实际 FC2 为 f4_flat_g2（cfg2=23 且 map 成功）；host verdict 资格见1548–1553，提交见 graph_replay。 |
| [zero_fused_mono](zero_fused_mono.yaml) | ws_fused_mono | 零探测成功；融合 FC1；单体 FC2。 | 链首验证当前字节；clean 时清零并由 guards 跳过计算，dirty 执行该配对完整链；单图/双图及失败回落由实际 graph 状态参数化。 | 有效 zflag 非零，cfg=97，实际 FC2 为 f4_body 的 G2、REDF=false（含源码已有 map 回落）；host verdict 资格见1548–1553，提交见 graph_replay。 |
| [zero_fused_ws](zero_fused_ws.yaml) | ws_fused_ws | 零探测成功；融合 FC1；WS FC2。 | 链首验证当前字节；clean 时清零并由 guards 跳过计算，dirty 执行该配对完整链；单图/双图及失败回落由实际 graph 状态参数化。 | 有效 zflag 非零，cfg=97，实际 FC2 为 f4_body_ws 的 G2（cfg2=57/81）；host verdict 资格见1548–1553，提交见 graph_replay。 |
| [zero_split_flat](zero_split_flat.yaml) | ws_split_flat | 零探测成功；分片 FC1；flat-stream FC2。 | 链首验证当前字节；clean 时清零并由 guards 跳过计算，dirty 执行该配对完整链；单图/双图及失败回落由实际 graph 状态参数化。 | 有效 zflag 非零，cfg=99，实际 FC2 为 f4_flat_g2（cfg2=23 且 map 成功）；host verdict 资格见1548–1553，提交见 graph_replay。 |
| [zero_split_mono](zero_split_mono.yaml) | ws_split_mono | 零探测成功；分片 FC1；单体 FC2。 | 链首验证当前字节；clean 时清零并由 guards 跳过计算，dirty 执行该配对完整链；单图/双图及失败回落由实际 graph 状态参数化。 | 有效 zflag 非零，cfg=99，实际 FC2 为 f4_body 的 G2、REDF=false（含源码已有 map 回落）；host verdict 资格见1548–1553，提交见 graph_replay。 |
| [zero_split_ws](zero_split_ws.yaml) | ws_split_ws | 零探测成功；分片 FC1；WS FC2。 | 链首验证当前字节；clean 时清零并由 guards 跳过计算，dirty 执行该配对完整链；单图/双图及失败回落由实际 graph 状态参数化。 | 有效 zflag 非零，cfg=99，实际 FC2 为 f4_body_ws 的 G2（cfg2=57/81）；host verdict 资格见1548–1553，提交见 graph_replay。 |

## 覆盖缺口

本表按实际 FC1 执行族、激活物化边界及 FC2 执行族划分；独立 cfg/cfg2 覆盖入口使表中配对可达。模板内的 tile、环深、SK 数、PIPE/DEEP、PP、EPI、TSTORE、预取及数值分支保留为参数，不按 rung 编号复制节点。ws_fused_ws 承接该族除已审 root.when 外的区间，包括单 CTA 准备和显式调优。父节点只作为完整设计的复制起点；每个 when 在共享 Contract 全域独立求值，精确条件和版本以各 YAML 为准。

表中“实际 FC1/FC2”为 Model.forward 完成形状/表示校正、MEGA/零探测门后，launch_gemm 的真实分支和模板实参；发生源码已有 map fallback 时取回落后的入口。不能依据 CFGS 注释替代实际代码。选择量包括有效或缓存 ABI、R=T*K、S（正的 small_thresh，否则 native getter）、实际 map/graph 状态及已捕获的提交策略。常规节点统一取 zflag=0，零验证节点取 zflag 非零；MEGA 提前返回，独占 cfg91/92。默认 _pick_cfg 在四组契约几何只返回97/99，cfg43 的旧自动叶及要求 cfg78 的 SELFPREP 自动门不可达，显式选择仍须按实际入口处理。

各常规派生节点完整保留 run_pipeline 的可参数化准备前缀：R<=S 时选 PREP2 或普通单 CTA 准备，否则 memset→count→scan_tiles→scatter（aggregated 或逐 route）。这些前缀产生同一套路由接口，不为与两个 GEMM 的每种组合再复制一套节点。需预 gather 的 cfg 列表及 flat/token-major/route-major 选择保留实际物化和提交边界；其余由 GEMM 内 gather。SELFPREP 仍可能因 PREP2、KEEP_PREP 或大准备选择而同时运行独立准备。参数化不能把缺席 kernel 写成必有，也不能消除数据就绪或复用依赖。

零验证节点保留相应融合/分片 FC1 和 WS/单体/flat FC2 的完整脏输入计算链。ZERO2 的 host word 只在控制值为2且 T*H>=2^20 时存在；有 clean graph 才先提交验证图，并在 dirty、超时或外层 capture 时继续完整图；创建失败回单图。MEGA 的 STAGED 参数分别对应权重暂存和 direct 读取；发布时 empty 不保证回放 empty。缓存不因输入原地变化重新选路，eager/graph 不另拆节点；不补造缓存锁或跨调用、跨 stream 的完成协议。

已知覆盖边界：host 拒绝缺失 output、E>256、未对齐或不一致几何；契约只取四组既定几何。REDF、WS 和部分 split-K 的 map 失败会返回错误，TSTORE 还要求 drain map；只有源码明确列出的单体路径有 cp.async 回落。过大的 SMALL_THRESH 可使单 CTA 动态共享内存申请超过已设置上限，提交失败后没有大准备重试。表格不把这些拒绝路径当成输出计算 fallback。

显式控制存在源码自身的不完整支持：cfg2=82 在 G2 分支提交 GEMM1=true 的实例，不能归入正常 FC2；cfg=82 在 G1 则实际落到单体分支，已由 mono_fused 系列接手。84..87 不在 packed 入口列表，却经 Python >=7 家族判断，可能落到不匹配的 warp 路径；非 MEGA 的 cfg 配上 cfg2=91/92 也不能推定为合法 FC2。BF16 两梯级可由两个配置独立选 BM，host 没有一般的 BM 相等校正；派生设计须忠实保留这一事实，不保证任意强制配对正确。越界配置索引、错侧拒绝及上述不匹配组合没有补造修复或成功输出保证。

共享 Contract 的验收 source_gap 沿用 root.tolerance：已知配置判据不改写；外部检查路由、相对基准、rel-L2 归一化及特殊值处理缺少实现来源。本缺口不阻塞路径设计，也不支持任何全域误差或逐位一致承诺。表内计划文件由后续 P5 生成，当前缺文件不代表已有完整 Design。
