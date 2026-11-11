**extract_gemm：pipe-control-v0.2 独立试写**

可以完整表达 root 的 main 三个角色，且比原写法更容易辨认 worker 首尾、真实嵌套和条件动作。层次来自工作、产品与分片本身；异步完成、成员汇合和资源归还仍须结合继承的契约与依赖阅读。交付见 [YAML](extract_gemm.yaml)。

| 覆盖 | for | if | while | 静态动作 | 最深控制层 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main：Producer、Issuer、Epilogue | 6 | 2 | 0 | 23 | 3 |

- **范围与证据。** 从原始 root、冻结规则及源码重新核对，限定正常编译的 UMMA pair 路径 `umma_gemm<0>`：四个 epilogue warp、`ECOLS=128`、`V0_X16=true`；继承原 dispatch 的单次提交边界。四份源码的字节数、SHA-256 均与 manifest 及输入锁 `source.files` 相符，锁的 `source.ref` 为 `497a8077…`。该提交对象本地不可用，字节核对不等同于读取该 Git 对象。

关键 CUDA 行号均指固定 [fp8gg_umma.cu](sources/extract_gemm/fp8gg_umma.cu)；入口另列：

| 源码位置 | 核对结论 |
| --- | --- |
| [solution.py](sources/extract_gemm/solution.py) 143–162、253–287、327–331；[fp8gg_bind.cpp](sources/extract_gemm/fp8gg_bind.cpp) 79–83、163–167；CUDA 1582、1860、1884 | Python 经 binding 或 ctypes 到 UMMA；正常 sareg=0，图节点与直接提交均可选择本 root |
| CUDA 327–348、363–412、422–519 | 一次 Schedule；输入环与产品对序号跨工作接续；单／双 issuer 的完成仍按原动作汇合 |
| CUDA 798–849、854–924、985–995 | 按工作初始化，两组 scale 准备，读取与计算交错；末片读取完成后提前归还；替换组始终提交 |
| CUDA 1015–1055、1059–1061 | full/tail 写回、历史源读等待、最后全局完成等待及 CTA 退役 |

- **顺序与层次。** Producer 一次 `initialize:Schedule` 后沿连续 n 载入，Issuer 同样沿 n 发起。Epilogue 先等 Schedule，外层 t 内准备 F，再按 `n[t]` 消费当前工作的产品；仅分片层遍历 `d[0:7]`。这三层各有真实数据身份，不能压成单层批次；n 的环槽与完成世代也不能随 t 重置。控制边界不增加同步。

- **读写交错。** 保留 read0，随后逐片 await 当前、read 下一片、compute 当前。末尾直接列出 await7→release DotPair→compute7，再补 activation scale。release 仍要求全部 epilogue 行的 TMEM 读完成，因此产品槽可在末次寄存器累加前复用；没有把读取、等待或归还并入新动作。

- **空组仍发生。** 每个工作先提交 p=0、1，两者顺序早于第一次 capture；每消费一对再提交 q+2。共 Q+2 个 activation group，其中末尾 Q、Q+1 无 payload，仍有动作与提交。Q=1 时第二个准备组已经为空，仍须保留。BS(t,Q) 赋零也保留事件。工作边界没有额外排空或重置该组流。

- **两个条件。** Full(t) 来自当前 V，在整个工作内不变，所以两个 if 各判断一次仍互斥且完备。完整块依次 release Staging、publish、异步 store；首次完整块仍有 release，只是没有旧 store 可等。尾块只直接写有效行，不清空 bulk store 历史；full→tail→full 时后一个 release 仍等之前完整块的源读取结束，worker 最后的 drain 则等目的写完成。

- **域与依赖。** 完整保留 t、q、n=flatten(t,q)、p、d 五个域。删除的 f/z 是原内联门控索引，实际唯一值均为 0，相关动作显式写常数；原资源视图中的形式参数仍可接受这些实例。全部 waits_for 均未引用 f/z，七条目标声明、八个前驱引用原样继承，无需替换。移位 n+3、n+2 仍按连续 n 的原窗口判断。

- **有限检查。** PyYAML 与递归结构检查通过；枚举 Q=1、2、3 的合成短流水及合法 Q=6、8、16、28，配合 0–4 个工作全部 full/tail 组合，共 217 例。199199 次动作与原 root 展开及源码顺序模型一致；86987 个显式等待实例的前驱均存在，未用事件交集删除缺失边。包含源读里程碑的有限 issue/done 图无环，动作身份无重复。空 worker 仍初始化、等 Schedule、drain；Issuer 无正文动作。Q=0 不属于继承契约，合法 KB 均为偶数。

- **计数核对。** 若 worker 获配 T 个工作，其中 F 个完整块，Producer 有 `1+TQ` 次动作，Issuer 有 `TQ` 次，Epilogue 有 `2+T(29Q+5)+2F` 次。这把空 scale 提交计入，同时排除条件为假的分支动作；完整块比尾块多两个动作。逐对还检查八次读取、八次等待、八次计算与一次产品归还，避免只比较总数而漏掉局部错序。

- **空 worker 的来源。** 合法输入行数为正，组边界覆盖全部行，因此全局工作总数非零。CUDA 1825–1827 用保守工作上界确定提交的 CTA 数，仍可能有 CTA 的起始工作号超出实际工作范围。零工作检查针对这一局部空序列；源代码的空尺寸提前返回不属于本 root 的继承契约。初始化和退役不能因局部循环为空而消失。

- **while 与局限。** 未使用 while。源码的组扫描属于既有工作坐标解码，barrier 轮询属于等待原语内部；本轮无需提升为新控制层，也不将有界遍历改成反馈循环。保留原稿已指出的源偏差：CUDA 787 行形成行所有权，864 行 TMEM 起址缺少行组地址贡献，890–924 行的后半对继续沿该地址；设计仍描述完整产品与独立行所有权。本稿没有用语法改写修复这一源码与设计的偏差，有限轨迹通过也不证明硬件或数值正确；未运行 GPU。
