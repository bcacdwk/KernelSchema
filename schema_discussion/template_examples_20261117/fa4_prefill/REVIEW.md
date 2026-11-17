# FA4 prefill 独立全文审阅

本轮是新的 high reviewer，重新阅读三份有效全文、dispatch、当前 DOC0–6 与固定源码；没有继承作者的通过结论。保留 **root / packed_persistent / split_kv 三份**：静态每 pair 单工作、跨工作常驻的连续环与提前输出释放、split partial/独立 combine 的完整价值均成立。矩形网格、请求映射、joint/dual、shared alpha 和已知组合已在 dispatch 对应完整基线上闭合；没有需要重开已定选型的新事实。CLC 保持“未展开独立候选”，没有伪装成局部参数。

## 实际修正

- root 的 `requestmap` 实际借用 `SharedStorage.clc_response: MemRange[Int32,4]`。先前按使用的头两个元素计容量，少计分配；改为每 CTA 四个元素，pair 总计32字节。源码 `kernel:952,1677–1809`，DOC5 要求容量含有效分配粒度。persistent 的36元素请求表、S/P/Stats 的别名区域和独立 Y 分配原本正确。
- root 与 split 的页读取改为 `logicalpage / physicalpage / kc / vc` 公式，保留尾半页重放首地址；输出改为显式坐标等式。未改在线 FP8 P 与未窄化 E 分母路径、offset=4、max-lag=4、FMA 舍入点或 sum 的零/NaN guard。split 的 partials 步骤补显式引用 native log/division 近似缺口，避免依赖其他步骤的隐式精度说明。
- 三份 `CORRECTION.load:stats` 现在明确区分首项只确认并归还 token 与后续读取 old/base、派生 alpha/warp vote。避免 role 总述称“不读”，动作契约又无条件称“读取”的矛盾。源码 `kernel:3138–3185,3689–3762`。
- initial free slots 也显式执行源码已有 acquire；其旧读者等待落域外时跳过。删除 LOAD/SOFTMAX 的 `n>=2 / i>=2 / e>=4` guards，采用等价正向复用等待。joint MMA 保留连续 `e=2*p+d` 流和首 QK、稳态 QK/PV、末 PV 的源码顺序，以 `initialscore / initialy` 明确初次自由槽获取，以较短 e%4 条件确定后续复用；四个协议槽、全部载荷与独立 K/V 数组不变。
- combine 的 `lastvalidsplit` 改称 “LSE != -inf 的最后 split”，与源码判断一致，不再加上源码未作的 finite 判断。没有改成“零权重不读取 partial”：无条件初始 prefetch、空 split 未写 PartialO 和只对正权重作 FP32 累加均保留。
- 本地版本改为 root 8、packed_persistent 7、split_kv 2；保留 Contract 唯一定义、persistent 仅 `Math_reshape: {from: root}`，其使用上下文的维度、数值与状态重置仍成立。

## 重新核对的源据

所有路径固定为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`；下列 kernel 指 `fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`。

- `metadata.json:benchmark_contract`、`reference.py:Model.forward`、`input.py:_make_inputs`、`shape_train.json:shape_domain/invariants`：完整16参数 ABI，page64 HND、16/1 heads、256 channels、固定两种 scale、tablecolumns=16434、workspacebytes=413138944；正 query/KV 长度及 query≤KV；输出原地更新与返回同一个 out，两个输出均 atol=0.06、rtol=0.04。B≤16 来自 seq_lens 列表长度的显式上界，总 Q≤16384 与 KV≤131072 也有明示 domain/invariant，未按样例最大值推断。
- `solution.py:Model.forward`、`interface.py:935–1020,1146–1299`：本 ABI 直接调用 `_flash_attn_fwd`，不是公有入口的 replay。三份 when 同时约束已选调度和数值/加载分工，不把全部相同 B 的输入声称为同一 selector 分支。
- `kernel:650–672,1170–1263,1577–1810`：compact/raster 与 rectangular 的局部边界，静态一 pair 一候选，persistent 的148-SM/74-worker stride，请求表初始化一次、四步二分、逻辑 CTA extent 补偶数。padding 只在有效 compact 前缀之后；没有 payload 时仍推进和退出。
- `kernel:1327–1450,2116–2261,2598–2812`：Q/K/V 两成员准备，joint token 与 K/V 独立 SMEM，QK 领先 PV，N=1 的实际 Q 释放，S/P 两槽和 Y 单 ownership token，worker 循环之后才 producer tail。
- `kernel:3431–3674,3694–3816`：两次128-channel QK、原始最大值滞后、FP8 P narrowing 与 FP32 未窄化 E 的分母、四组 packed-FP32 分母累加树；score 捕获后归还、P 与 Stats 各自旧消费者归还；skip correction 仍取得/释放 Y。TMEM 原语的累积完成与先读旧操作数后覆盖关系留在 Realization，不从普通异步模式额外推导。
- `kernel:3891–3984,3987–4033,3239–3245`：末 half 已从 TMEM 到 SMEM 并 fence 后先归还 Y，随后才 global store；stage group barrier/复用和最终 cluster rendezvous/free 保留。
- `kernel:3136–3147,3250–3289`、`interface.py:1758–1769`、`flash_fwd_combine.py:398–711`：split span 与 causal trip 相交；空 work 仅写 -inf LSE；非空 normalized FP32 partial 与 natural-log LSE；transpose 仅视图；独立提交 handoff；head-fast attention 与 head-major combine 的坐标衔接；四槽 cp.async 与前三组预取、thread-owned SMEM 读取、正权重累加和真实尾行为。
- 历史 `validation-20261021/results/fa4_prefill/validate.json:environment` 核实 L20D、SM103、148SM、CuTe DSL4.6.1、torch2.9.0+cu130、cuda-python13.1.1。现有硬件事实未清空，未知 Wiki 保持正式占位，没有 skip-hardware。

主 kernel 的本轮 `sha256sum` 仍为 `64d9715dd199f4c843533f365b5a702305805702900f40bf541035fa64d885f1`。AST 单独执行原 `_hd256_split_heuristic`、不 import GPU 库：README 的 root/persistent/split/N1 缩小输入 split 结果分别为1/1/5/1，cpad分别8/296/13/6。

## 检查记录

默认命令使用 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml`。修正后结果：3 YAML、0 changed，structure/text/from/relations/dispatch 通过。首次检查发现两条过长 meaning 与 inline `for(i=...)` 遮蔽公共 i domain，已分别精简文字和改为 `for(n[t,1:])`；这些是本次局部写作错误，不报告成工具缺陷。

最终公共工具冻结（[MEMBER_GRAPH.md](../MEMBER_GRAPH.md)，135项测试由工具作者完成）后，只读规范化每次均为 `3 YAML; 0 would change`；本 reviewer 没有重跑公共测试。三个主样本先用 README 命令的 `--read-only --graph check` 检查，再按 README 的确切 `--graph png` 命令重绘同名 PNG。

| 样本 | actions | edges | boundary skips |
| --- | ---: | ---: | ---: |
| root 主图：N=4，rescale=1 | 194 | 700 | 29 |
| persistent 主图：N=1→3，rescale=1 | 217 | 782 | 40 |
| split 主图 ATTENTION：N=4 | 194 | 705 | 18 |
| split 主图 COMBINE：lastvalid=3 | 20 | 58 | 1 |
| root N=1，rescale=0 | 71 | 225 | 14 |
| root 无效候选：N=0，valid=0 | 16 | 37 | 0 |
| persistent N=1→padding N=0 | 80 | 259 | 21 |
| persistent N=1→3，rescale=0 | 213 | 770 | 40 |
| split 空 ATTENTION：N=0 | 12 | 22 | 0 |
| split 空样本 COMBINE：lastvalid=3 | 20 | 58 | 1 |
| joint N=1，rescale=0 | 68 | 213 | 12 |
| combine 全零协议样本：lastvalid=-1 | 11 | 28 | 1 |

全部完成且无环。已打开三张最终 PNG 的缩略视图核对布局、成员和主要交接；图片不表示时间比例。PNG 后再运行 `root.yaml --read-only`，仍是3 YAML、零待规范化变化；`git diff --check` 在案例与本次讨论目录无输出。边界命令均为 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/<id>.yaml --read-only --graph check`，后接：

```text
root: --set 'tile_count(t)=1' --set 'valid(t)=1' --set 'rescale(n)=0' --worktiles 1
root: --set 'tile_count(t)=0' --set 'valid(t)=0' --set 'rescale(n)=0' --worktiles 1
packed_persistent: --set 'tile_count(0)=1' --set 'tile_count(1)=0' --set 'valid(0)=1' --set 'valid(1)=0' --set 'rescale(n)=0' --worktiles 2
packed_persistent: --set 'tile_count(0)=1' --set 'tile_count(1)=3' --set 'valid(t)=1' --set 'rescale(n)=0' --worktiles 2
split_kv: --set 'tile_count(t)=0' --set 'valid(t)=1' --set 'rescale(n)=0' --set lastvalid=3 --worktiles 1
split_kv: --set 'tile_count(t)=1' --set 'valid(t)=1' --set 'rescale(n)=0' --set lastvalid=-1 --worktiles 1
```

padding 的合法缩小输入为 B=9、每请求 Q=KV=128：cpad=80、有效候选72、74 workers，worker0 的候选0为单轮有效工作，随后候选74无效。root/persistent/split 主图与 N1 的合法缩小输入关系见 README；强制 rescale=1 是分支示意，joint N1+combine all-zero 是协议边界，不声称它们复现真实 host 五路输入的数值轨迹。

首次冻结后主图调用没有全部通过：反向 n-2 依赖与 joint 复合动作索引触发名义窗口/反求边界。改为正向复用依赖、连续 e 动作族并区分初次与复用 acquire 后完成，细节见 [TOOLS.md](TOOLS.md)。未把失败命令列为通过，未修改工具来迎合案例。

## 保留的真实边界

独立 native exp2/log/division 误差界仍无材料；Wiki 入口仍有正式占位；完整默认分支集合之外的 CLC/开发 override 不作已交付或已验证结论。有限图只证明指定参数事件图能展开且无环，不能据此证明全部输入数值或性能。未运行 Runner/GPU，未改 kernel、公共工具、其他案例或 schema_wiki，未提交 Git。
