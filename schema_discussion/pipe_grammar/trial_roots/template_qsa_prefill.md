可以完整表达旧 root 的控制次序，首尾也更容易读：worker 首票握手、每工作两次 QK 预填与两次 PV 收尾、worker 最后 K/V drain 都直接列出。覆盖 2 phase、7 role、40 个静态动作；使用 11 个 `for`，没有 `if` 或 `while`，最大循环嵌套为 2。

| phase／role | for | 动作 | 实际源码函数与行号 |
| --- | ---: | ---: | --- |
| preprocess／Main | 0 | 2 | `qsa_preprocess.py::QSAPreprocessUnion.kernel:76,109–291` |
| attention／Scheduler | 1 | 4 | `QSAForward.kernel:161,319–334` |
| attention／GatherK | 2 | 4 | `_gather_role:735,758–840` |
| attention／GatherV | 2 | 4 | `_gather_role:735,758–840` |
| attention／MMA | 2 | 9 | `_mma_role:650,666–731` |
| attention／SoftmaxA | 2 | 10 | `_softmax_role:349,379–646` |
| attention／SoftmaxB | 2 | 7 | `_softmax_role:349,379–646` |

- 已实际读取固定版本 `01fff90d714abc6e84904ea9c44523ba66d34a0e` 的四个入口／实现文件。`solution.py::Model.forward:5–8` 调用 `operator.py::run:42`，后者在 153–160 行先发 preprocess 再发 attention。表中未标文件的函数均在 `qsa_kernel.py`。baseline 与 manifest 五份源码的 SHA-256 全部匹配。
- **预处理边界：** build 包含小输入缓存、哈希清零、提前一组的向量加载、原子插入与成员位合并，末尾会合；compact 才做占用扫描、每 warp 保留连续输出区间、写 Union 及最后发布 Count。虽然源码注释称压紧顺序确定，275 行原子分配使 warp 间次序不确定，旧契约已正确保留这一点。每 CTA 预分配一份工作，不需要为了统一写法添加工作循环。
- **为何不用 while：** `t` 是调度层已有的运行时工作序列，首票、每份工作后的 Next 发布和五个消费者的确认仍是明确动作，未知工作总数不要求增加循环状态。源码的 ticket while 据此投影为 `for(t)`；hash 插入的 CAS 反馈属于现有 `build:Hash` 内部，本轮不拆动作，也不虚构“直到插入成功”的域。终止票无 Q 载荷仍保留动作，不用 if 删除事件。
- **域与依赖：** preprocess 显式写 `domain: {}`；attention 完整保留 `t: worktile`、`i: [0,N(b)]`、`n: {flatten: [t,i]}`。未增删域，未改索引身份；所有角色继承原 `waits_for`，共 28 个目标、38 项前驱表达式。执行模式、资源、契约及默认操作数等待也继承。
- **顺序解释：** 列表只约束发起次序，for 进出不提供同步、排空或资源归还。Gather 保留异步搬运，收票只是确认本票副本，不代表旧工作算完。MMA 的累积完成包括操作数读完，保障别名 S/P 复用；softmax 串行链中的 Q 预取仍是异步。窗口外移位边合法消失，窗口内缺失前驱则报错；本次核对未用事件交集删除等待边。
- **工作内次序：** MMA 先 QK(0)、QK(1)，随后每轮先 QK(n+2) 再 PV(n)，最后 PV(N−2)、PV(N−1)，再收 Next。源码 683–699 行的真实分支与此一致；658 行注释把 PV(0) 写到 QK(2) 前，不能据此改列表。`N(b)>=32` 来自合法选择至少 512 个不同四元组，保证所有首尾索引有效；没有把 N=0/1 扩入契约。尾行块只有六行，Q 与尾 store 仍按真实行过滤；行与通道条件留在原动作契约，未把线程层判断改成角色层 if。
- **跨工作边界：** 变长工作更新连续 n，双槽 KV 与三槽 S/P 不在 `for(t)` 边界复位。`load(n)@done` 等 `load(n+1)@issue` 可跨工作，只有整个 worker 尾部 drain 发布最后一份；地址预取的钳制仍限当前工作。SoftmaxA 在收 Next 后发 Q 预取，归一化、读完 Y_A 后发布 Qnext，再存尾片；B 也先释放 Y_B 再存尾片。下一工作首 PV 仍等待两半 drain，Q 预取可与输出重叠。
- **空 worker 与终止票：** 本 CTA 首票即可终止，但 launch 仍有有效工作。此时没有工作主体，首票发布／确认、A 的初始 Qblk0 发布、MMA 的 Qblk0 等待及两个无数据 drain 都存在。源码 386–398 行仍按钳制票号加载 Q 并发 qready，669 行仍等它；空 drain 不寻找不存在的末 tile。
- **真实契约差异：** 旧稿称 terminal `Qnext` 为 no-op。源码 584–588 行确实跳过 Q 拷贝，621–623 行跳过拷贝等待，但 624 行仍发 `qready`，且发生在 Y_A 读完之后、最后 gate/store 之前。本稿保留 `prefetch:Qnext(t)`、`publish:Qnext(t)` 的身份并记录差异，未暗改继承契约；末票后的信号没有下一工作消费者。
- **验证与限度：** PyYAML 及结构检查通过。枚举工作数 0–3、N 取 32/33/35 的 40 组，再加四组含 321 的序列；40,726 次动作与旧稿逐角色逐项一致。62,309 条显式依赖实例端点有效，加入顺序及既有操作数／发布契约后的有限事件图无环。未运行 GPU。动作内仍隐藏 hash 探测、mask／地址预取、条件 Y 修正、线程会合与数值实现；HalfMax 读完再覆盖、padding 对修正投票及近似误差界仍未证明。
