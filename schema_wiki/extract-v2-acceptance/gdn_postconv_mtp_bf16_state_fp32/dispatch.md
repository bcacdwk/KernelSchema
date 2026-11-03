# 设计索引

- [root](root.yaml)：融合 post-convolution GDN decode/MTP 设计，处理 packed BF16 QKV、FP32 状态检查点和 sigmoid 门控输出；无父节点、无派生节点。

入口为 `Model.forward` → `fused_gdn_decode_post_conv_mtp` → `gdn_post_conv_mtp<float, R, true>`，其中 R 为 value/key 头数比。每个请求与 value head 由一个 CTA 负责，在当前流上统一启动。

核心设计先准备实际 token 的归一化 Q/K 和门控量，再以四个 32 行块推进 FP32 状态递推与检查点写回；双槽预取配合复制完成等待和 CTA 屏障。状态读出先物化为 BF16，全部行完成后再做 RMSNorm、权重和 sigmoid 门控，写入调用方 out。

声明 workload 中的头比例模板实例共用数学关系、工作划分和缓冲组织；TP、请求数、请求长度、accepted 位置及槽布局变化均由同一套参数和索引表达。空请求返回、零源槽请求输出归零、零目标槽跳过检查点写回，属于同一 kernel 的边界行为，不构成派生设计。

host 暴露的 BF16 state、SiLU、其他头比例及省略 out 分支不属于共享 Contract 的输入变体；dt_bias 和 norm_weight 的类型在本任务中固定，不形成额外节点。

保留既有覆盖缺口：FP32 状态基址须满足 16 字节对齐，槽 stride 须可被 4 整除；部分合法 padding 会被 host 直接拒绝，且没有 fallback。该拒绝不构成另一条计算设计，单 root 也不表示候选全域可执行。共享 Contract 不因该限制收窄，RMSNorm 前的 BF16 读出舍入也不改变其参考标准与误差判据。

精确选择条件与版本以 YAML 为准。`parent` 仅表示直接设计复制来源，不叠加父节点条件；节点条件在共享 Contract 全域独立求值。
