# GEMM 直接 Schema 审修

当前正式目录保留 2 份完整设计：root Version 5、wide Version 2；两份 YAML 与 dispatch 已修订并完成回读。工作由 `/root/gemm_schema_review_xhigh` 同一代理承担 worker 与 reviewer，自核不冒充独立 Runner 审批。原 run `06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31` 的回执和历史记录没有改动，也没有新增 extract、P3/P5/P6、GPU、模型 CLI 或运行状态操作。

## 保全与核准

- `before/` 保全正式 root、wide、dispatch 和全部现有 `.extraction` 原字节，摘要见 `before.json`；原 root/ wide 版本为 4/1。`after/` 保存当前正文副本，差异另存三个 `.diff`。
- 对固定算子目录的全部 27 个来源文件逐字核对 `git show 24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2:<path>`，全部一致，见 `source_manifest.json`。`SELECTION.json` 的 candidate 与 contract 摘要也全部相符。
- 实际阅读核准了 problem input、shape range/cases、metadata、reference，candidate Python ABI、C weakref binding，以及 CUDA 中两个具体实例、host selector、tensor map 和 graph cache。参考入口与实现入口的 `Model.forward` 均由 AST 确认存在，未 import 或执行 CUDA 源程序。
- 使用现行 `schema_template/README.md`、DOC0–6 中相关字段、存储、索引、精度、范围及版本规定；未改 checker、框架或规范。旧批准仍只覆盖原批准字节；本记录是当前字节的人工审修证据。

## 已修正

| 项目 | 源码核对及当前结果 |
| --- | --- |
| 寄存器副本 | root `sb_regs` 原只计逻辑 scale 对，16 B 改为 128 个线程当前/下一对的 2048 B。wide `sa_regs` 原遗漏列组复制，1024→2048 B；`sb_regs` 原遗漏每半块 128 个线程复制，16→2048 B。 |
| 旋转排空和槽释放 | 两实例源码在最后一次 TMEM 读取退休后先 arrive，再做末批 32 列 FFMA。原 `promote` serial 后再 `acc_free` 延后了槽复用。现拆成 `promote → acc_free → promote_tail`，并登记两批 32 值的 `drain_regs`：root 32768 B、wide 65536 B；tail 只读寄存器。root 另明确 `scale_ready` 位于下一对 B scale 预取之前，对应源码 `cp_async_wait_g1` 的真实位置。 |
| 持久动作身份 | 原 scale `load(...j)` 在多个 t 上重复生成同名事件。现两设计的 scale 动作、Waiting 和对应 view 均带 `(t,j)`；槽仍按 j%2 复用，未改环深度或全局 K 序号。 |
| 精度与账本 | 将“每输出元素 m*n*k”改为每调用 MAC 数，区分 e4m3 输入 tensor-core contraction 与标量 FP32 promotion，删除“恰好多一次舍入”。保留四个 k32 指令和 FP32 accumulator 事实，明确源码没有内部归约舍入/subnormal 定义；后置 scale 的溢出、下溢及误差没有全域保证。 |
| host 与证据 | 调度枚举无需全局设备 workspace，但 host selector 在 miss 上确实 D2H；weakref 身份不证明张量 storage/metadata 永久不变。合法形状不能排除资源、tensormap 或 CUDA API 失败。修正 NCU SM 数“无材料”说法：固定 CSV 直接记录 148；实例归属由 NCU kernel 名直接给出，dual/bstream 再按源码和固定输入推导。 |
| 边界说明 | wide 无效 padded row 会消费补零 scale，只是不写输出；A 紧缩 box 之外保持未写入，不假定已初始化；staging 读完可能先于全局完成，并非严格提前。Execution 摘要区分 wide 的 320 kernel threads 与 256 epilogue threads，以及两实例的 tile/commit 粒度。 |

## 无需改动的内容

共享 Contract 的 Constant、Input、Output、Compute 和 Tolerance 与 before 完全相同。五组 (n,k)、group_count 集合、物理组行数范围及 4 对齐来自 shape_range；logical_group_sizes/mode 不参与计算，空组无输出。所有输出位置由物理分段覆盖，caller-owned `out_buffer` 与返回 `out` 的别名、初值忽略及 view 外保护保持清楚。

三整数 tuple 的元素轴加 python_int 与标量 `torch.dtype` 对象均自洽；固定 task 的 input 构造显式给出全部 selector 和 out，不把 Python API 较宽的默认参数/None 分支移入冻结 Contract，也不因“统一 tuple 写法”修改它们。reference 的 FP32 dequantization、关闭 TF32、FP32 einsum、BF16 收窄及 `out.copy_` 已核对。

root 与 wide 的 128/256 列分解、pair/block 同步轮、3×64 KiB/4×48 KiB operand ring、128/256 epilogue 线程和全块/半块 staging 组织确有不同。保留现有两设计集合；三区各自本地定义，`from` 数量为 0，所有共享 Contract 引用解析到 root。w1g 仍为 wide 的局部写回及环深变化，未增删设计。

## 存储与生命周期核算

| 一 CTA 的区域 | root | wide |
| --- | ---: | ---: |
| A+B operand ring | 196608 B | 196608 B |
| staging | 32768 B | 32768 B |
| shared a_scale ring | 2048 B | 无 |
| sched_table，G=64 | 520 B | 520 B |
| 控制尾区全部有效字节，含 sched_table | 588 B | 604 B |
| dynamic shared memory，含 1 KiB tail | 232448 B | 230400 B |
| 512 列 TMEM | 262144 B | 262144 B |
| row_acc | 65536 B | 131072 B |
| source-level scale+drain+row_acc 合计 | 100352 B | 200704 B |

各 `total_size` 均按 depth×单槽元素数×元素字节数×一 CTA 核准，详细表在 `verification.json`。源控制尾区分别为 `68+8*(G+1)` / `84+8*(G+1)`；在 G≤64 时无重叠，1 KiB 内尚余 436/420 B。未将 driver 的额外 1024 B 预留重复计入 dynamic smem。prefetch ring、TMEM ping-pong 和 staging 复用分别由原等待协议约束，最后 TMEM deallocation 在 CTA exit rendezvous 后。

寄存器表描述源码具名缓冲的容量，不能相加成为已证明的同时活跃量或编译后分配；NCU 中 root/wide 的 224/168 registers/thread 只属于采样构建。没有根据局部数组、源码注释或一次 NCU 结果作无 spill 保证。

## 检查结果与准确范围

完整命令：

```text
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/root.yaml --read-only --skip-hardware
```

最终退出 0，2 YAML、0 规范化改动，结构、文本预算、引用、关联及 dispatch 通过，见 `check-final.log`。硬件区及依赖硬件的规则明确跳过，没有改宽检查规则。

有限事件样本使用同一命令加 `--graph check --all --set pair_count=6 --set kb_count=12 --set 'full_rows(0)=1' --set 'full_rows(1)=0' --worktiles 2`。原件 `before/root.yaml` 的两设计均报 duplicate `load:a_scale(0)`，见 `graph-before.log`。当前字节 root 为 188 actions/554 edges、wide 为 359 actions/1065 edges，均无环，见 `graph-final.log`。另以 `--worktiles 0`、相同 pair_count/kb_count 检查无工作 worker，两设计均为 12 actions/33 edges、无环。样本覆盖局部首次等待、跨工作索引和整块/残块/无工作分支，不验证完整输入几何、GPU 活性、全部数值或硬件存储安全。

修稿期间的 YAML 引号及文本预算失败、较早通过日志与最后一次 scale-wait 顺序修订均如实保留在 `checks.json`，未放回正式正文或 `.extraction`。最终检查条目绑定当前 manifest。

## 来源缺口及源实现限制

1. metadata 给出 atol=.01、rtol=.05、max_mismatch_fraction=0；外部检查器只有 hash/来源记录，没有随本算子材料交付完整公式、相对基准和非有限值检查路由。保留 `Source only ... no defaults`，未声称复刻比较器或以采样 pass 证明全域通过。
2. 后置 scale 存在极值边界：在本 Schema 显式的正有限 scale 值域内，a/b 全零而两 scale 均为 2^100 时，参考先反量化为零，后置 FP32 scale 乘积却溢出，后续与零相乘可成 NaN。这是对现有公式的解析反例，未声称该值由固定生成器生成或已 GPU 复现；未收窄任务、改变比较器或修 kernel。
3. 源 weakref memo 缺 storage/metadata 失效检测；`set_`、`resize_` 后仍可能命中旧指针/尺寸。indptr 内容改变还可能使 host engine/pp 缓存陈旧，但 device 会重读当前 indptr，不是结果缓存。设备 0 和进程全局缓存的前提也不能升级为任意多设备保证。
4. NCU 的 148 SM、shared-memory 上限、occupancy limit、寄存器和 kernel name 均来自固定 10 shapes×5 captures。只支持该来源范围，不能替代跨构建、全域或本次 GPU 测试。没有进行 GPU、编译或数值性能验证。

## 统一清理交接

本轮 `.extraction/receipt.json` 与 `issue-review-20261119.json` 未改未移，原字节在 before 中。它们的过程字段、旧批准适用范围，以及 dispatch 中既有的 P1/旧收录措辞交由后续统一清理；不得把旧 receipt 当作当前字节的新批准。当前设计使用条件、精度/host 缺陷、资源采样范围和外部比较器缺口必须保留。`after.json` 指向本次 `review.json`，完整证据只在此 discussion 目录。
