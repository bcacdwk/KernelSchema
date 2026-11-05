# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 非分段、无请求排序、缓存 Q/K 的 packed prefill | 代表路径：prep 物化系数和 Q/K 图像，再由一次 MODE0 chain 完成请求递推。 | gdn_forward 3220–3250 不分段；3320–3321 关闭 kqtma；3341–3347 不分配 perm。 |
| [cached_ordered](cached_ordered.yaml) | root | 非分段、请求排序、缓存 Q/K | prep 增加按 chunk 数稳定降序的 perm；chain 等待后解析请求，关闭 PRE0，保留单链和 Q/K 缓存。 | gdn_forward 3220–3250 不分段；3320–3321 关闭 kqtma；3341–3347 分配 perm。 |
| [direct_qk](direct_qk.yaml) | root | 非分段、无请求排序、直接读取 Q/K | prep 不写全局 Q/K 图像；MODE0 chain 用输入 Q/K 的 TMA boxes，移除该缓存交接，保留系数交接。 | gdn_forward 3320–3321 启用 kqtma（必为非分段且 VTT≥32）；3341–3347 不分配 perm。 |
| [direct_qk_ordered](direct_qk_ordered.yaml) | direct_qk | 非分段、请求排序、直接读取 Q/K | 在直接读取 Q/K 的单链上增加 prep 生成的 perm；chain 等待后按排序结果解析请求。 | gdn_forward 3320–3321 启用 kqtma；3341–3347 分配 perm。 |
| [segmented](segmented.yaml) | root | 分段、无请求排序、缓存 Q/K | 改为 prep→MODE1 pass A→MODE2 pass C；prep 用 slot-fastest 排列，A 省去 Q/Qd 载荷并发布 FP32 分段传递量，C 组合入口状态并重跑输出。 | gdn_forward 3220–3250 选择 Gseg>1；3341–3347 不分配 perm；3320–3321 排除 kqtma。 |
| [segmented_ordered](segmented_ordered.yaml) | segmented | 分段、请求排序、缓存 Q/K | 在分段双 pass 上增加 prep 生成的 perm；A/C 等待后按排序结果解析请求，保留分段传递和重跑。 | gdn_forward 3220–3250 选择 Gseg>1；3341–3347 分配 perm，依据重分块后的 chain_blocks；3320–3321 排除 kqtma。 |

入口为 `candidate/solution.py::Model.forward` → `candidate/gdn_kernel.cu::gdn_forward`；表内行号均指该固定 C++ 来源。`parent` 只指定设计复制来源，各节点在共享 Contract 全域独立选择；精确条件和版本以各 YAML 为准。

选择依次使用 host 的 VTT/分段决策、kqtma 和 perm 决策。分段重选 VTT 后再计算排序依据；L2 判定采用实际进程缓存值和本次请求的 workspace 字节数，不采用旧分配容量。六类覆盖这三项决策的合法组合，分段与 direct-Q/K 不并存。

VTT、NSTAGE、GQA 分组及 PRE0 保留为参数或局部索引域。PRE0 只在无排序的缓存 vt16 单链启用，prep 仍写首块 Q/K 图像。分段路径的弱衰减判定发生在已提交的 MODE2 内：segment 0 重跑整条请求，其他 segment CTA 提前退出；MODE1/2 两次提交仍然存在。

## 覆盖缺口

- 路径覆盖限于共享 Contract：已提供 packed 元数据、BF16 初态且不再做 Q/K 归一化。入口的缺省元数据、额外归一化及 FP32 初态支持不扩张本任务范围。
- `get_vmap/get_kqmap` 编码失败直接报错，无备用路径；已有连续偏移视图不会因连续化而重新分配，不能据此承诺其对齐可用性。非连续 `core_attn_out` 被写入临时连续副本且不回写，未满足调用者可见前缀的 Contract 要求。
- root 的最终 mini-chunk 在 VTT=64、请求最终状态且末块占 stage 0 时，最终状态暂存可能覆盖其他 warp 尚在读取的 K 行；保留源码缺少读取退休保证的事实。进程共享 workspace 未提供跨 stream 事件或重入协议，不据同 stream 顺序推导并发安全；L2 缓存也没有设备键检查。
- 共享 Contract 的 tolerance 保留已知阈值及比较对象，但外部比较公式、相对基准、算术精度、NaN/Inf 策略和 core 检查路由仍缺失。root 已记录 BF16 窄化、近似指数和归约重排，无全域误差界；本索引不把路径覆盖当作数值保证。
