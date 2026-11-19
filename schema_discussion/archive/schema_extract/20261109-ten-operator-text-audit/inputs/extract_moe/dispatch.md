# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 复用 ABI 的大路由融合主干 | count/scan/scatter、flat gather、融合 FC1、分工 bulk-store FC2、finalize；七次计算提交。 | 普通复用；无零验证；cfg/cfg2=97/81；大路由准备；PDL 自动开启。 |
| [direct](direct.yaml) | root | 建立 ABI 时空路由、后续内容可变 | 以单个 k_mega_direct 取代完整主干；保留当前路由的 census、零输出和非空 direct 计算分支。 | 普通复用；cfg/cfg2=92/92；保存的默认 MEGA 空路由选择仍有效。 |
| [double_verify_fused_big_cp](double_verify_fused_big_cp.yaml) | verify_fused_big_cp | 独立验证图后接大路由融合 FC1，FC2 描述符回退 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg=97；大路由准备；FC2 的 23→12 或 44 描述符请求被拒绝并实际选中 fc=8。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_big_flat](double_verify_fused_big_flat.yaml) | verify_fused_big_flat | 独立验证图后接大路由融合 FC1，单阶段 FC2 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/23；大路由准备；FC2 flat 入口成功选中。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_big_mono](double_verify_fused_big_mono.yaml) | verify_fused_big_mono | 独立验证图后接大路由融合 FC1，低 FC2 工作量 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/44；大路由准备；FC2 TMA 入口成功选中。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_serial](double_verify_fused_serial.yaml) | verify_fused_serial | 独立验证图后接最大路由规模的融合主干 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/81；R≥49152，命中默认 PDL 关闭门。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_small](double_verify_fused_small.yaml) | verify_fused_small | 独立验证图后接单 CTA 准备的融合主干 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/81；小路由准备。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_small_cp](double_verify_fused_small_cp.yaml) | verify_fused_small_cp | 独立验证图后接单 CTA 准备，FC2 描述符回退 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg=97；小路由准备；FC2 实际选中 fc=8 回退。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_small_flat](double_verify_fused_small_flat.yaml) | verify_fused_small_flat | 独立验证图后接单 CTA 准备，单阶段 FC2 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/23；小路由准备；FC2 flat 入口成功选中。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_fused_small_mono](double_verify_fused_small_mono.yaml) | verify_fused_small_mono | 独立验证图后接单 CTA 准备，低 FC2 工作量 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/44；小路由准备；FC2 TMA 入口成功选中。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_root](double_verify_root.yaml) | verify_root | 独立验证图后接大路由融合主干 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=97/81；大路由准备；PDL 自动开启。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_split_cp](double_verify_split_cp.yaml) | verify_split_cp | 独立验证图后接低 R split-K，FC2 描述符回退 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg=99；FC2 实际选中 fc=8 回退。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_split_flat](double_verify_split_flat.yaml) | verify_split_flat | 独立验证图后接低 R split-K，单阶段 FC2 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=99/23；FC2 flat 入口成功选中。 dirty、轮询超时或外层图的已录联合执行。 |
| [double_verify_split_store](double_verify_split_store.yaml) | verify_split_store | 独立验证图后接低 R、两片 FC1 部分和 | 先执行独立验证图，再执行含第二次验证的完整主干；两次验证使用同一输出和标志存储，分别保留提交身份。 | 联合复用；zflag_host 非空；cfg/cfg2=99/81；默认低 R 分支与 WS/TSTORE 门同时选中。 dirty、轮询超时或外层图的已录联合执行。 |
| [fused_big_cp](fused_big_cp.yaml) | fused_big_mono | 大路由融合 FC1，FC2 描述符回退 | FC2 改用 k_gemm_f4<128,false,3> 的 cp.async 主体；来源 cfg2=23 和 44 汇合到同一组织。 | 普通复用；无零验证；cfg=97；大路由准备；FC2 的 23→12 或 44 描述符请求被拒绝并实际选中 fc=8。 |
| [fused_big_flat](fused_big_flat.yaml) | fused_big_mono | 大路由融合 FC1，单阶段 FC2 | FC2 改用 I=256 的 flat-stream，将各 item 的 stage 串成连续流；仍物化 BF16 路由结果。 | 普通复用；无零验证；cfg/cfg2=97/23；大路由准备；FC2 flat 入口成功选中。 |
| [fused_big_mono](fused_big_mono.yaml) | root | 大路由融合 FC1，低 FC2 工作量 | FC2 改用四 warp 单体 TMA/SFCP 组织和直接 BF16 写回，取消分工 bulk-store 收尾。 | 普通复用；无零验证；cfg/cfg2=97/44；大路由准备；FC2 TMA 入口成功选中。 |
| [fused_serial](fused_serial.yaml) | root | 最大路由规模的融合主干 | 计算提交集合不变；自动关闭 PDL，提交间改用普通 stream 完成接续。 | 普通复用；无零验证；cfg/cfg2=97/81；R≥49152，命中默认 PDL 关闭门。 |
| [fused_small](fused_small.yaml) | root | 单 CTA 准备的融合主干 | 一个 k_prepare_small 替换 memset 加 count/scan/scatter；其余四次计算提交保留。 | 普通复用；无零验证；cfg/cfg2=97/81；小路由准备。 |
| [fused_small_cp](fused_small_cp.yaml) | fused_small_mono | 单 CTA 准备，FC2 描述符回退 | FC2 改用 k_gemm_f4<128,false,3> 的 cp.async 主体；23→12 和 44 的回退合并。 | 普通复用；无零验证；cfg=97；小路由准备；FC2 实际选中 fc=8 回退。 |
| [fused_small_flat](fused_small_flat.yaml) | fused_small_mono | 单 CTA 准备，单阶段 FC2 | FC2 改用 I=256 的 flat-stream；计算提交与 BF16 路由物化边界保留。 | 普通复用；无零验证；cfg/cfg2=97/23；小路由准备；FC2 flat 入口成功选中。 |
| [fused_small_mono](fused_small_mono.yaml) | fused_small | 单 CTA 准备，低 FC2 工作量 | FC2 改用四 warp 单体 TMA/SFCP 和直接 BF16 写回。 | 普通复用；无零验证；cfg/cfg2=97/44；小路由准备；FC2 TMA 入口成功选中。 |
| [init_no_scan_direct](init_no_scan_direct.yaml) | no_scan_direct | 查表、无scan重建：direct | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=92/92；默认 MEGA。 |
| [init_no_scan_fused_big](init_no_scan_fused_big.yaml) | no_scan_fused_big | 查表、无scan重建：大路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/81；大准备；PDL开。 |
| [init_no_scan_fused_big_cp](init_no_scan_fused_big_cp.yaml) | no_scan_fused_big_cp | 查表、无scan重建：大路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [init_no_scan_fused_big_flat](init_no_scan_fused_big_flat.yaml) | no_scan_fused_big_flat | 查表、无scan重建：大路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/23；大准备；实际flat。 |
| [init_no_scan_fused_big_mono](init_no_scan_fused_big_mono.yaml) | no_scan_fused_big_mono | 查表、无scan重建：大路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [init_no_scan_fused_serial](init_no_scan_fused_serial.yaml) | no_scan_fused_serial | 查表、无scan重建：大R串行融合 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [init_no_scan_fused_small](init_no_scan_fused_small.yaml) | no_scan_fused_small | 查表、无scan重建：小路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/81；小准备。 |
| [init_no_scan_fused_small_cp](init_no_scan_fused_small_cp.yaml) | no_scan_fused_small_cp | 查表、无scan重建：小路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [init_no_scan_fused_small_flat](init_no_scan_fused_small_flat.yaml) | no_scan_fused_small_flat | 查表、无scan重建：小路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/23；小准备；实际flat。 |
| [init_no_scan_fused_small_mono](init_no_scan_fused_small_mono.yaml) | no_scan_fused_small_mono | 查表、无scan重建：小路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [init_no_scan_split_cp](init_no_scan_split_cp.yaml) | no_scan_split_cp | 查表、无scan重建：低R split-K、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg=99，cfg2=23或44；实际fc8。 |
| [init_no_scan_split_flat](init_no_scan_split_flat.yaml) | no_scan_split_flat | 查表、无scan重建：低R split-K、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=99/23；实际flat。 |
| [init_no_scan_split_mono](init_no_scan_split_mono.yaml) | no_scan_split_mono | 查表、无scan重建：低R split-K、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=99/44；实际单体TMA。 |
| [init_no_scan_split_store](init_no_scan_split_store.yaml) | no_scan_split_store | 查表、无scan重建：低R split-K、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=99/81；实际bulk-store。 |
| [init_no_scan_split_ws](init_no_scan_split_ws.yaml) | no_scan_split_ws | 查表、无scan重建：低R split-K、直接写回 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、无scan；zero_ok=false；cfg/cfg2=99/57；R<32。 |
| [init_probe_double_verify_fused_big](init_probe_double_verify_fused_big.yaml) | probe_double_verify_fused_big | 查表、有scan重建：双验证、大路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/81；大准备；PDL开。 |
| [init_probe_double_verify_fused_big_cp](init_probe_double_verify_fused_big_cp.yaml) | probe_double_verify_fused_big_cp | 查表、有scan重建：双验证、大路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [init_probe_double_verify_fused_big_flat](init_probe_double_verify_fused_big_flat.yaml) | probe_double_verify_fused_big_flat | 查表、有scan重建：双验证、大路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/23；大准备；实际flat。 |
| [init_probe_double_verify_fused_big_mono](init_probe_double_verify_fused_big_mono.yaml) | probe_double_verify_fused_big_mono | 查表、有scan重建：双验证、大路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [init_probe_double_verify_fused_serial](init_probe_double_verify_fused_serial.yaml) | probe_double_verify_fused_serial | 查表、有scan重建：双验证、大R串行融合 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [init_probe_double_verify_fused_small](init_probe_double_verify_fused_small.yaml) | probe_double_verify_fused_small | 查表、有scan重建：双验证、小路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/81；小准备。 |
| [init_probe_double_verify_fused_small_cp](init_probe_double_verify_fused_small_cp.yaml) | probe_double_verify_fused_small_cp | 查表、有scan重建：双验证、小路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [init_probe_double_verify_fused_small_flat](init_probe_double_verify_fused_small_flat.yaml) | probe_double_verify_fused_small_flat | 查表、有scan重建：双验证、小路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/23；小准备；实际flat。 |
| [init_probe_double_verify_fused_small_mono](init_probe_double_verify_fused_small_mono.yaml) | probe_double_verify_fused_small_mono | 查表、有scan重建：双验证、小路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [init_probe_double_verify_split_cp](init_probe_double_verify_split_cp.yaml) | probe_double_verify_split_cp | 查表、有scan重建：双验证、低R split-K、cp.async FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg=99，cfg2=23或44；实际fc8。 |
| [init_probe_double_verify_split_flat](init_probe_double_verify_split_flat.yaml) | probe_double_verify_split_flat | 查表、有scan重建：双验证、低R split-K、flat FC2 | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=99/23；实际flat。 |
| [init_probe_double_verify_split_store](init_probe_double_verify_split_store.yaml) | probe_double_verify_split_store | 查表、有scan重建：双验证、低R split-K、bulk-store | 增加查表 kernel 与设备同步。 | 联合重建；有查表、有scan；zero_ok=true；cfg/cfg2=99/81；实际bulk-store。 |
| [init_probe_fused_big](init_probe_fused_big.yaml) | probe_fused_big | 查表、有scan重建：大路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/81；大准备；PDL开。 |
| [init_probe_fused_big_cp](init_probe_fused_big_cp.yaml) | probe_fused_big_cp | 查表、有scan重建：大路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [init_probe_fused_big_flat](init_probe_fused_big_flat.yaml) | probe_fused_big_flat | 查表、有scan重建：大路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/23；大准备；实际flat。 |
| [init_probe_fused_big_mono](init_probe_fused_big_mono.yaml) | probe_fused_big_mono | 查表、有scan重建：大路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [init_probe_fused_serial](init_probe_fused_serial.yaml) | probe_fused_serial | 查表、有scan重建：大R串行融合 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [init_probe_fused_small](init_probe_fused_small.yaml) | probe_fused_small | 查表、有scan重建：小路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/81；小准备。 |
| [init_probe_fused_small_cp](init_probe_fused_small_cp.yaml) | probe_fused_small_cp | 查表、有scan重建：小路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [init_probe_fused_small_flat](init_probe_fused_small_flat.yaml) | probe_fused_small_flat | 查表、有scan重建：小路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/23；小准备；实际flat。 |
| [init_probe_fused_small_mono](init_probe_fused_small_mono.yaml) | probe_fused_small_mono | 查表、有scan重建：小路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [init_probe_split_cp](init_probe_split_cp.yaml) | probe_split_cp | 查表、有scan重建：低R split-K、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg=99，cfg2=23或44；实际fc8。 |
| [init_probe_split_flat](init_probe_split_flat.yaml) | probe_split_flat | 查表、有scan重建：低R split-K、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=99/23；实际flat。 |
| [init_probe_split_mono](init_probe_split_mono.yaml) | probe_split_mono | 查表、有scan重建：低R split-K、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=99/44；实际单体TMA。 |
| [init_probe_split_store](init_probe_split_store.yaml) | probe_split_store | 查表、有scan重建：低R split-K、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=99/81；实际bulk-store。 |
| [init_probe_split_ws](init_probe_split_ws.yaml) | probe_split_ws | 查表、有scan重建：低R split-K、直接写回 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=false；cfg/cfg2=99/57；R<32。 |
| [init_probe_verify_fused_big](init_probe_verify_fused_big.yaml) | probe_verify_fused_big | 查表、有scan重建：完整验证、大路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/81；大准备；PDL开。 |
| [init_probe_verify_fused_big_cp](init_probe_verify_fused_big_cp.yaml) | probe_verify_fused_big_cp | 查表、有scan重建：完整验证、大路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [init_probe_verify_fused_big_flat](init_probe_verify_fused_big_flat.yaml) | probe_verify_fused_big_flat | 查表、有scan重建：完整验证、大路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/23；大准备；实际flat。 |
| [init_probe_verify_fused_big_mono](init_probe_verify_fused_big_mono.yaml) | probe_verify_fused_big_mono | 查表、有scan重建：完整验证、大路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [init_probe_verify_fused_serial](init_probe_verify_fused_serial.yaml) | probe_verify_fused_serial | 查表、有scan重建：完整验证、大R串行融合 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [init_probe_verify_fused_small](init_probe_verify_fused_small.yaml) | probe_verify_fused_small | 查表、有scan重建：完整验证、小路由融合、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/81；小准备。 |
| [init_probe_verify_fused_small_cp](init_probe_verify_fused_small_cp.yaml) | probe_verify_fused_small_cp | 查表、有scan重建：完整验证、小路由融合、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [init_probe_verify_fused_small_flat](init_probe_verify_fused_small_flat.yaml) | probe_verify_fused_small_flat | 查表、有scan重建：完整验证、小路由融合、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/23；小准备；实际flat。 |
| [init_probe_verify_fused_small_mono](init_probe_verify_fused_small_mono.yaml) | probe_verify_fused_small_mono | 查表、有scan重建：完整验证、小路由融合、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [init_probe_verify_only](init_probe_verify_only.yaml) | probe_verify_only | 查表、有scan重建：仅验证 | 增加查表 kernel 与设备同步。 | 仅验证重建；有查表、有scan；zero_ok=true；exec_clean可用、zflag_host非空；非外层capture，host观察clean且无完整图提交。 |
| [init_probe_verify_split_cp](init_probe_verify_split_cp.yaml) | probe_verify_split_cp | 查表、有scan重建：完整验证、低R split-K、cp.async FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg=99，cfg2=23或44；实际fc8。 |
| [init_probe_verify_split_flat](init_probe_verify_split_flat.yaml) | probe_verify_split_flat | 查表、有scan重建：完整验证、低R split-K、flat FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=99/23；实际flat。 |
| [init_probe_verify_split_mono](init_probe_verify_split_mono.yaml) | probe_verify_split_mono | 查表、有scan重建：完整验证、低R split-K、单体FC2 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=99/44；实际单体TMA。 |
| [init_probe_verify_split_store](init_probe_verify_split_store.yaml) | probe_verify_split_store | 查表、有scan重建：完整验证、低R split-K、bulk-store | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=99/81；实际bulk-store。 |
| [init_probe_verify_split_ws](init_probe_verify_split_ws.yaml) | probe_verify_split_ws | 查表、有scan重建：完整验证、低R split-K、直接写回 | 增加查表 kernel 与设备同步。 | 完整重建；有查表、有scan；zero_ok=true；cfg/cfg2=99/57；R<32。 |
| [no_scan_direct](no_scan_direct.yaml) | direct | 无scan重建：direct | 保留重建服务初始化；不调用 probe。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=92/92；默认 MEGA。 |
| [no_scan_fused_big](no_scan_fused_big.yaml) | root | 无scan重建：大路由融合、bulk-store | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/81；大准备；PDL开。 |
| [no_scan_fused_big_cp](no_scan_fused_big_cp.yaml) | fused_big_cp | 无scan重建：大路由融合、cp.async FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [no_scan_fused_big_flat](no_scan_fused_big_flat.yaml) | fused_big_flat | 无scan重建：大路由融合、flat FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/23；大准备；实际flat。 |
| [no_scan_fused_big_mono](no_scan_fused_big_mono.yaml) | fused_big_mono | 无scan重建：大路由融合、单体FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [no_scan_fused_serial](no_scan_fused_serial.yaml) | fused_serial | 无scan重建：大R串行融合 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [no_scan_fused_small](no_scan_fused_small.yaml) | fused_small | 无scan重建：小路由融合、bulk-store | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/81；小准备。 |
| [no_scan_fused_small_cp](no_scan_fused_small_cp.yaml) | fused_small_cp | 无scan重建：小路由融合、cp.async FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [no_scan_fused_small_flat](no_scan_fused_small_flat.yaml) | fused_small_flat | 无scan重建：小路由融合、flat FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/23；小准备；实际flat。 |
| [no_scan_fused_small_mono](no_scan_fused_small_mono.yaml) | fused_small_mono | 无scan重建：小路由融合、单体FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [no_scan_split_cp](no_scan_split_cp.yaml) | split_cp | 无scan重建：低R split-K、cp.async FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg=99，cfg2=23或44；实际fc8。 |
| [no_scan_split_flat](no_scan_split_flat.yaml) | split_flat | 无scan重建：低R split-K、flat FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=99/23；实际flat。 |
| [no_scan_split_mono](no_scan_split_mono.yaml) | split_mono | 无scan重建：低R split-K、单体FC2 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=99/44；实际单体TMA。 |
| [no_scan_split_store](no_scan_split_store.yaml) | split_store | 无scan重建：低R split-K、bulk-store | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=99/81；实际bulk-store。 |
| [no_scan_split_ws](no_scan_split_ws.yaml) | split_ws | 无scan重建：低R split-K、直接写回 | 保留失败前服务操作，不提交 scan。 | 完整重建；无查表、无scan；zero_ok=false；cfg/cfg2=99/57；R<32。 |
| [probe_double_verify_fused_big](probe_double_verify_fused_big.yaml) | double_verify_root | 有scan重建：双验证、大路由融合、bulk-store | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/81；大准备；PDL开。 |
| [probe_double_verify_fused_big_cp](probe_double_verify_fused_big_cp.yaml) | double_verify_fused_big_cp | 有scan重建：双验证、大路由融合、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [probe_double_verify_fused_big_flat](probe_double_verify_fused_big_flat.yaml) | double_verify_fused_big_flat | 有scan重建：双验证、大路由融合、flat FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/23；大准备；实际flat。 |
| [probe_double_verify_fused_big_mono](probe_double_verify_fused_big_mono.yaml) | double_verify_fused_big_mono | 有scan重建：双验证、大路由融合、单体FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [probe_double_verify_fused_serial](probe_double_verify_fused_serial.yaml) | double_verify_fused_serial | 有scan重建：双验证、大R串行融合 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [probe_double_verify_fused_small](probe_double_verify_fused_small.yaml) | double_verify_fused_small | 有scan重建：双验证、小路由融合、bulk-store | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/81；小准备。 |
| [probe_double_verify_fused_small_cp](probe_double_verify_fused_small_cp.yaml) | double_verify_fused_small_cp | 有scan重建：双验证、小路由融合、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [probe_double_verify_fused_small_flat](probe_double_verify_fused_small_flat.yaml) | double_verify_fused_small_flat | 有scan重建：双验证、小路由融合、flat FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/23；小准备；实际flat。 |
| [probe_double_verify_fused_small_mono](probe_double_verify_fused_small_mono.yaml) | double_verify_fused_small_mono | 有scan重建：双验证、小路由融合、单体FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [probe_double_verify_split_cp](probe_double_verify_split_cp.yaml) | double_verify_split_cp | 有scan重建：双验证、低R split-K、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg=99，cfg2=23或44；实际fc8。 |
| [probe_double_verify_split_flat](probe_double_verify_split_flat.yaml) | double_verify_split_flat | 有scan重建：双验证、低R split-K、flat FC2 | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=99/23；实际flat。 |
| [probe_double_verify_split_store](probe_double_verify_split_store.yaml) | double_verify_split_store | 有scan重建：双验证、低R split-K、bulk-store | 增加一次 scan 及实际服务前缀。 | 联合重建；无查表、有scan；zero_ok=true；cfg/cfg2=99/81；实际bulk-store。 |
| [probe_fused_big](probe_fused_big.yaml) | root | 有scan重建：大路由融合、bulk-store | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/81；大准备；PDL开。 |
| [probe_fused_big_cp](probe_fused_big_cp.yaml) | fused_big_cp | 有scan重建：大路由融合、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [probe_fused_big_flat](probe_fused_big_flat.yaml) | fused_big_flat | 有scan重建：大路由融合、flat FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/23；大准备；实际flat。 |
| [probe_fused_big_mono](probe_fused_big_mono.yaml) | fused_big_mono | 有scan重建：大路由融合、单体FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [probe_fused_serial](probe_fused_serial.yaml) | fused_serial | 有scan重建：大R串行融合 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [probe_fused_small](probe_fused_small.yaml) | fused_small | 有scan重建：小路由融合、bulk-store | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/81；小准备。 |
| [probe_fused_small_cp](probe_fused_small_cp.yaml) | fused_small_cp | 有scan重建：小路由融合、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [probe_fused_small_flat](probe_fused_small_flat.yaml) | fused_small_flat | 有scan重建：小路由融合、flat FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/23；小准备；实际flat。 |
| [probe_fused_small_mono](probe_fused_small_mono.yaml) | fused_small_mono | 有scan重建：小路由融合、单体FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [probe_split_cp](probe_split_cp.yaml) | split_cp | 有scan重建：低R split-K、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg=99，cfg2=23或44；实际fc8。 |
| [probe_split_flat](probe_split_flat.yaml) | split_flat | 有scan重建：低R split-K、flat FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=99/23；实际flat。 |
| [probe_split_mono](probe_split_mono.yaml) | split_mono | 有scan重建：低R split-K、单体FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=99/44；实际单体TMA。 |
| [probe_split_store](probe_split_store.yaml) | split_store | 有scan重建：低R split-K、bulk-store | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=99/81；实际bulk-store。 |
| [probe_split_ws](probe_split_ws.yaml) | split_ws | 有scan重建：低R split-K、直接写回 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=false；cfg/cfg2=99/57；R<32。 |
| [probe_verify_fused_big](probe_verify_fused_big.yaml) | verify_root | 有scan重建：完整验证、大路由融合、bulk-store | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/81；大准备；PDL开。 |
| [probe_verify_fused_big_cp](probe_verify_fused_big_cp.yaml) | verify_fused_big_cp | 有scan重建：完整验证、大路由融合、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；大准备；实际fc8。 |
| [probe_verify_fused_big_flat](probe_verify_fused_big_flat.yaml) | verify_fused_big_flat | 有scan重建：完整验证、大路由融合、flat FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/23；大准备；实际flat。 |
| [probe_verify_fused_big_mono](probe_verify_fused_big_mono.yaml) | verify_fused_big_mono | 有scan重建：完整验证、大路由融合、单体FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/44；大准备；实际单体TMA。 |
| [probe_verify_fused_serial](probe_verify_fused_serial.yaml) | verify_fused_serial | 有scan重建：完整验证、大R串行融合 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/81；R≥49152；PDL关。 |
| [probe_verify_fused_small](probe_verify_fused_small.yaml) | verify_fused_small | 有scan重建：完整验证、小路由融合、bulk-store | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/81；小准备。 |
| [probe_verify_fused_small_cp](probe_verify_fused_small_cp.yaml) | verify_fused_small_cp | 有scan重建：完整验证、小路由融合、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg=97，cfg2=23或44；小准备；实际fc8。 |
| [probe_verify_fused_small_flat](probe_verify_fused_small_flat.yaml) | verify_fused_small_flat | 有scan重建：完整验证、小路由融合、flat FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/23；小准备；实际flat。 |
| [probe_verify_fused_small_mono](probe_verify_fused_small_mono.yaml) | verify_fused_small_mono | 有scan重建：完整验证、小路由融合、单体FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=97/44；小准备；实际单体TMA。 |
| [probe_verify_only](probe_verify_only.yaml) | verify_only | 有scan重建：仅验证 | 增加一次 scan 及实际服务前缀。 | 仅验证重建；无查表、有scan；zero_ok=true；exec_clean可用、zflag_host非空；非外层capture，host观察clean且无完整图提交。 |
| [probe_verify_split_cp](probe_verify_split_cp.yaml) | verify_split_cp | 有scan重建：完整验证、低R split-K、cp.async FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg=99，cfg2=23或44；实际fc8。 |
| [probe_verify_split_flat](probe_verify_split_flat.yaml) | verify_split_flat | 有scan重建：完整验证、低R split-K、flat FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=99/23；实际flat。 |
| [probe_verify_split_mono](probe_verify_split_mono.yaml) | verify_split_mono | 有scan重建：完整验证、低R split-K、单体FC2 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=99/44；实际单体TMA。 |
| [probe_verify_split_store](probe_verify_split_store.yaml) | verify_split_store | 有scan重建：完整验证、低R split-K、bulk-store | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=99/81；实际bulk-store。 |
| [probe_verify_split_ws](probe_verify_split_ws.yaml) | verify_split_ws | 有scan重建：完整验证、低R split-K、直接写回 | 增加一次 scan 及实际服务前缀。 | 完整重建；无查表、有scan；zero_ok=true；cfg/cfg2=99/57；R<32。 |
| [split_cp](split_cp.yaml) | split_mono | 低 R split-K，FC2 描述符回退 | FC2 改用 k_gemm_f4<128,false,3> 的 cp.async 主体；23→12 和 44 的回退合并。 | 普通复用；无零验证；cfg=99；FC2 实际选中 fc=8 回退。 |
| [split_flat](split_flat.yaml) | split_mono | 低 R split-K，单阶段 FC2 | FC2 改用 I=256 的 flat-stream；FC1 部分和及独立激活提交保留。 | 普通复用；无零验证；cfg/cfg2=99/23；FC2 flat 入口成功选中。 |
| [split_mono](split_mono.yaml) | split_store | 低 R split-K，低 FC2 工作量 | FC2 改用四 warp 单体 TMA/SFCP 和直接 BF16 写回。 | 普通复用；无零验证；cfg/cfg2=99/44；FC2 TMA 入口成功选中。 |
| [split_store](split_store.yaml) | fused_small | 低 R、两片 FC1 部分和 | 去掉 flat gather；FC1 改用 clipped gather4、两片 FP32 部分和及独立 k_act_packed 合并/激活/量化；不沿用融合 FC1 的 hoist 和提前 accumulator release。 | 普通复用；无零验证；cfg/cfg2=99/81；默认低 R 分支与 WS/TSTORE 门同时选中。 |
| [split_ws](split_ws.yaml) | split_store | 极小 R，但 FC2 WS 工作量达门限 | FC2 改用 cfg57：直接写回，无 Sink/bulk-store；payload 与 scale 等待恢复耦合，无 hoist，完整 epilogue 后才释放 accumulator。 | 普通复用；无零验证；cfg/cfg2=99/57；WS 门成立而 R<32。 |
| [verify_fused_big_cp](verify_fused_big_cp.yaml) | fused_big_cp | 设备零验证后的大路由融合 FC1，FC2 描述符回退 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg=97；大路由准备；FC2 的 23→12 或 44 描述符请求被拒绝并实际选中 fc=8。 |
| [verify_fused_big_flat](verify_fused_big_flat.yaml) | fused_big_flat | 设备零验证后的大路由融合 FC1，单阶段 FC2 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/23；大路由准备；FC2 flat 入口成功选中。 |
| [verify_fused_big_mono](verify_fused_big_mono.yaml) | fused_big_mono | 设备零验证后的大路由融合 FC1，低 FC2 工作量 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/44；大路由准备；FC2 TMA 入口成功选中。 |
| [verify_fused_serial](verify_fused_serial.yaml) | fused_serial | 设备零验证后的最大路由规模的融合主干 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/81；R≥49152，命中默认 PDL 关闭门。 |
| [verify_fused_small](verify_fused_small.yaml) | fused_small | 设备零验证后的单 CTA 准备的融合主干 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/81；小路由准备。 |
| [verify_fused_small_cp](verify_fused_small_cp.yaml) | fused_small_cp | 设备零验证后的单 CTA 准备，FC2 描述符回退 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg=97；小路由准备；FC2 实际选中 fc=8 回退。 |
| [verify_fused_small_flat](verify_fused_small_flat.yaml) | fused_small_flat | 设备零验证后的单 CTA 准备，单阶段 FC2 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/23；小路由准备；FC2 flat 入口成功选中。 |
| [verify_fused_small_mono](verify_fused_small_mono.yaml) | fused_small_mono | 设备零验证后的单 CTA 准备，低 FC2 工作量 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/44；小路由准备；FC2 TMA 入口成功选中。 |
| [verify_only](verify_only.yaml) | verify_root | ZERO2 host 轮询观察到 clean | 只执行独立 k_zero_verify 图并返回输出别名；完整主干及其第二次验证均未提交。 | 复用 ABI；exec_clean 存在且 zflag_host 非空；非外层 capture；独立图提交成功，轮询最终观察值 1，未提交完整图。 |
| [verify_root](verify_root.yaml) | root | 设备零验证后的大路由融合主干 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=97/81；大路由准备；PDL 自动开启。 |
| [verify_split_cp](verify_split_cp.yaml) | split_cp | 设备零验证后的低 R split-K，FC2 描述符回退 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg=99；FC2 实际选中 fc=8 回退。 |
| [verify_split_flat](verify_split_flat.yaml) | split_flat | 设备零验证后的低 R split-K，单阶段 FC2 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=99/23；FC2 flat 入口成功选中。 |
| [verify_split_mono](verify_split_mono.yaml) | split_mono | 设备零验证后的低 R split-K，低 FC2 工作量 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=99/44；FC2 TMA 入口成功选中。 |
| [verify_split_store](verify_split_store.yaml) | split_store | 设备零验证后的低 R、两片 FC1 部分和 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=99/81；默认低 R 分支与 WS/TSTORE 门同时选中。 |
| [verify_split_ws](verify_split_ws.yaml) | split_ws | 设备零验证后的极小 R，但 FC2 WS 工作量达门限 | 完整主干前增加 k_zero_verify；各已提交 kernel 保留并按 ZGUARD 早退或计算，clean 时输出由验证产生。 | 普通复用；zflag 非空；cfg/cfg2=99/57；WS 门成立而 R<32。 |

“普通复用”指 Model.forward 的 identity/key 缓存或默认 C 包装复用已发布 ABI，moe_run 返回零，且主干计算均已实际接受提交，只执行一次 run_pipeline 主体：可以是 eager、成功录图后的首次回放或已有完整图回放；不额外执行独立验证图，也没有私有流上的未捕获流水线提交。“联合复用”同样复用 ABI，但实际执行独立验证图和完整图各一次，且无私有流上的未捕获流水线提交。纯录图不产生这些计算的完成事件；外层图中的请求以真正回放为计算执行实例。

小路由准备指 R=T*K 不超过 ABI 保存的 small_thresh，大路由准备指超过。默认建立 ABI 时，3000<R≤6500 且路由直方图非空取 3000，其余取 6500。FC1 默认在 R≤1536 选 cfg=99，否则选 cfg97；满足 MEGA 门且建立时直方图为空则改选 cfg92。FC2 的 WS 门比较路由 tile 数乘 (H/128)*max(1,I/256) 与 3072：仅上下界跨门限时读取直方图；WS 分支按 R≥32 选 81，否则 57，非 WS 分支在 I=256 选 23，否则选 44。

表中的 cfg、cfg2、small_thresh、zflag 和 zflag_host 均指实际 ABI 的保存值；图中的实际 kernel 与按值参数取录制绑定，不能按回放时的新直方图或描述符缓存重新选择。设备读取的张量内容仍取本次执行。直接父只表示完整 Design 的复制来源，各节点条件在共享 Contract 全域独立求值，精确条件及版本以各 YAML 为准。

线程处理宽度、网格数量、实际专家 tile 数、空工作、完整/部分 quadrant 和合法的 hoist 门保留参数化。cfg23 描述符拒绝后先改为 cfg12；相同 key 的拒绝被缓存，最终与 cfg44 的拒绝分支汇合到相同的 wide cp.async FC2，故不按来源 cfg 再复制回退节点。split_ws 的微小 R 分支只可能在本契约的 H=4096、I=1536 几何达到 WS 门；它和 split_mono 均达不到 ZERO2 的 T*H 门，不派生联合验证节点。

“完整重建”指本次没有复用 ABI、moe_run 正常返回零，实际计算提交恰为条目所列初始化前缀和一次完整主干，不含独立验证图，也无额外未捕获私有流流水线提交。“联合重建”的提交恰为此前缀、一次独立验证图和一次完整图；“仅验证重建”的提交恰为此前缀和一次独立验证图。后两者同样要求正常返回并排除额外未捕获私有流流水线提交。它们按已经发生的提交选择，不由 probe 成功或失败推断后继必然提交；返回错误及未提交后继按实际出口保留。纯录图仍不产生计算执行实例。

实际flat、单体TMA、fc8 和 bulk-store 分别指固定源码选择的 FC2 实例。所有重建条目均保留实际 route_hist 读取、workspace 缓存/服务初始化及 zflag 初始化；这些服务不暗中增加或删除表列计算提交。

查表存在性取 ensure_tables 本次是否实际提交 k_init_tables；scan 存在性取本次是否实际提交 k_zero_scan。probe 的 -1/-2 出口位于 scan 调用之前；-3/-4 出口及捕获异常仍须区分 scan 是否已经实际提交，不把错误码当作删除既有提交的依据。zero_ok=false 包括正常拒绝和失败，zero_ok=true 仅来自 Python 的返回值等于 1 判定。两者分别决定是否建立 zflag，不能将重建解释为 ABI 复用。

例如 T=1、K=8、H=2048、I=768、E=128 的非空新请求选择 cfg/cfg2=99/44。zero_ok=false 时，scan 已提交对应 probe_split_mono，未提交对应 no_scan_split_mono；zero_ok=true 时对应 probe_verify_split_mono，查表发生时使用各自的 init 条目。它们只描述正常返回且主体完整提交的情况；终止错误及其实际前缀见覆盖缺口。各条件均在共享 Contract 全域独立求值。

ABI 建立时，ensure_tables 按每设备标志决定是否提交 k_init_tables 并同步；非 direct 默认家族还调用 moe_zero_probe，其 malloc、memset、scan、同步和读回有不同错误出口。新 workspace 的清零依实际缓存命中情况发生，ABI 重建本身不等于所有初始化都发生。这些实际执行前缀属于同一次算子请求。probe 的负返回值使 Python 的 zero_ok 为 false，不由这一步直接终止 forward；后续不建立 zflag，不能把探测失败统一视为算子错误出口。probe 的 D2H cudaMemcpy 返回码未检查，返回判定只读取初值为 1 的 h，不能补造读回成功保证。

## 覆盖缺口

共享 Contract 不因下列拒绝、终止错误或源码缺陷而收窄。它们不是表列完整设计的正常输出保证，也没有被补成可恢复的 fallback。

host 在进入计算前拒绝缺失 output、E>256、未满足 H/128、I/64 或关联形状的输入。查表初始化失败可在 k_init_tables 已提交后抛错；零探测的 -1/-2 出口位于 scan 调用前，-3/-4 位于调用后，且 Python 仍以 zero_ok=false 继续。已经发生的查表、scan、内存服务和读回分别保留；探测失败本身不表示整个请求终止。

每次实际 run_pipeline 的计算调用顺序由默认分支确定：cfg92 只有 direct；cfg97 是路由准备、flat gather、融合 FC1、FC2、finalize；cfg99 是小路由准备、split-K FC1、独立 activation/量化合并、FC2、finalize。大路由准备分别提交 count、scan、scatter，小路由准备只有 k_prepare_small；zflag 非空时在这些主干之前另提交一次 k_zero_verify。失败时保留该次实际接受的全部前缀，不撤销已经排队的工作，也不将未调用的后继计入提交。查表/scan 等 ABI 建立前缀仍属于同一请求。

具体退出边界：小准备属性设置失败返回 91/92；准备 launch 返回 31，大路由 memset 错误直接返回其代码，count/scan/scatter 的 launch 错误分别返回 32/33/34；flat gather 的 launch 错误返回 43；FC1 入口错误返回 100+gerr，此后不调用 activation、FC2 或 finalize；split-K activation 的 launch 错误返回 35；FC2 入口错误返回 200+gerr，此后不调用 finalize；finalize 的 launch 错误返回 36。最后的 cudaGetLastError 也可能在整条调用链已提交后才返回非零，不能将该情况倒推成零提交。cfg23 的 flat launch 未检查 launch_pdl 返回值，只查询 cudaGetLastError；因此返回零本身也不能替代每次提交已被接受的事实。发生错误的那次 launch 是否已接受按实际结果记录，不用错误码自行删除其工作。

descriptor 拒绝须区别对待：cfg97、99、57、81 没有 cp.async 回退，operand map 拒绝返回 733；cfg81 的输出 map 拒绝返回 738。cfg23/44 才有表列 fc8 回退。没有验证、direct 或 finalize 写者的前缀不写调用者 output；k_zero_verify 会先写零，即使随后主干失败也不能省略该副作用。错误前已提交的 direct/finalize 仍可能写出。源码没有在所有错误出口等待 GPU 完成，故 host 抛错时不能保证 output 的完整性或写入已结束。

BeginCapture 失败后源码仍调用私有非阻塞流上的 run_pipeline；随后还调用 caller-stream run_pipeline。此类请求包含两组实际接受的前缀，分别按上一段的阶段顺序和各自退出边界计数，不能把它们算成两次独立逻辑调用，也不能把第一组当作录图而抹去。两组使用同一个 Params：counts/fill、路由表、activation、rout 和 output 均共用地址；源码没有建立隔离这两组工作所需的跨流等待。即使第二组返回零，也不据此保证共享状态未被干扰或输出正确。这是已知实现缺陷范围，完整设计节点不覆盖该保证。

成功 BeginCapture 所录但未回放的工作没有计算完成事件；EndCapture/Instantiate 失败后，仍须保留随后实际发生的 eager 前缀。独立验证图构建失败可保留完整图；其 launch 报错直接返回。独立验证图已提交而完整图 launch 失败时，保留前一验证提交及写零副作用，不补造完整图或等待。

外部验收公式、相对误差基准、relative-L2 归一化及返回别名/原地输出检查路由的来源缺口，按 root 的 tolerance 原样保留；它们不提供设计或数值保证。
