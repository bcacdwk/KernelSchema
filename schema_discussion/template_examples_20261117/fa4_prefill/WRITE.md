# FA4 prefill 写作交接

## 选型复核

独立阅读 AGENTS、COMMON/SOURCES/SELECTION、DOC0–6、Template 工具说明，以及固定 host/kernel 后，审定保留三份。Compact/static 与 rectangular/static 的差异封闭在 grid candidate、前置映射、raster 和无效工作过滤；有效 `(b,m)` 的 pair、主角色和 accumulator 寿命一致。joint/dual load、shared alpha、persistent 请求映射可以分别闭合记录；CLC 改变领取/退休，是未展开独立候选。

直接从固定 `interface.py` AST 提取 `_hd256_split_heuristic` 并以源码四个默认常量执行，未 import GPU 库；独立代入 persistence gate：

| 代表 | 结果 |
| --- | --- |
| shape_valid[6] | B=3，Q=[2019,584,4200]，KV=[6243,6067,4200]，T=6803，rounded K=6272，split=1，dual=true，persistent=false；cpad=428，D_bound≈0.963，D_tight≈0.975 |
| shape_valid[3] | B=9，T=16384，rounded K=7808，split=1，dual=true，persistent=true；cpad=1032，74 workers |
| shape_valid[0] | B=1，Q=200，KV=8648，rounded K=8704，split=5，dual=false，persistent=false |

公开 B≤16、总 Q≤16384、KV≤131072 来自 `shape_train.json` 的显式 domain/invariants；修正旧 root 的 B>16 域外条件，补固定 TableColumns=16434、WorkspaceBytes=413138944。保留全部 ABI、同对象 out 返回和 in-place 结果。

## 实际文件

- `root.yaml`：compact static，Version 7；Contract 唯一定义，三区本地。
- `packed_persistent.yaml`：compact plain stride，Version 6；仅 `Math_reshape: {from: root}`，其余本地。
- `split_kv.yaml`：新稳定 ID，Version 1；固定 five split/joint load/32容量 combine，三区本地。
- 三份均有新同名 PNG；新增 case README，重写正式 dispatch 为 DOC0 中文表格及分组记录。

删除旧 root 以外的被替代节点及同名 PNG：`small_static_lpt`→root；旧 root 矩形机制→局部记录；`rect_joint_load`→矩形+joint组合；`single_persistent_dual_load`、`small_persistent_lpt`→请求映射局部；`single_persistent_joint_load`→单请求+joint组合；`small_persistent_shared_rescale`、`small_static_shared_rescale`→shared alpha及已有组合；`splitkv_joint_load`→split_kv；`splitkv_dual_load`→split加载分工局部。没有备份 YAML 或额外交付设计。

## 核对的源码事实

固定主 kernel SHA-256 仍为 `64d9715dd199f4c843533f365b5a702305805702900f40bf541035fa64d885f1`，与 SOURCES 一致；使用 v37/v43 包，未使用 v53 或 44题近似来源。

- `interface.py:935–1020,1146–1299` 与 kernel `:650–672,1669–1810`：selector、compact/rectangular 边界、small-request 排序、请求映射生产/消费。Persistent 是 Preassigned start/stride、Refill=none；每 CTA 一次 req_map，逻辑 CTA 数补偶数。
- kernel `:1327–1450,1511–1523,2116–2261`：2CTA pair transaction bytes，page64；K每成员一页、V两页，缺第二页重用首地址后 mask；joint 共享四个 token，K/V SMEM 独立。
- `:2598–2812,3431–3674`：分两 channel half 的 FP8 QK、未窄化 E 分母、FP8 P numerator、offset/max-lag=4、native exp2、packed FP32 sum tree。首 PV 不读旧 numerator；每份工作明确重置 mhat/ell，未添加 numerator 清零。
- `:3476–3478,3523–3587,3694–3816`：S/P/Stats 别名、捕获与两种消费者返回。CORRECTION 首 Stats 仅确认，后续捕获/derive alpha，再取得旧 Y；warp vote 跳过乘法仍 acquire/release。
- `:2511–2519,2808–2812,3239–3245,3891–3984`：连续 Q/KV/S/P participant 索引跨工作；Q最后QK后可覆盖；LOAD/MMA tail 在整个worker之后；末半输出到SMEM并fence后先释放Y，再global store；最终cluster同步/free留在Realization。
- `:3136–3147,3250–3289,3987–4033`：split 按完整 used-KV block span与causal trip相交；空split只写LSE=-inf，不产PartialO、不等不存在的token；非空partial为FP32 normalized O和自然log LSE。
- `flash_fwd_combine.py:398–711`：先LSE及前三partial prefetch，weights/lastvalid，四槽commit/wait；prefetch不由权重保护，仅正weight执行累加；所有权重消失仍由真实 fill 初始化Facc。未增加最终copy-group drain。归并另一次提交、LSE transpose仅视图，head-major与attention head-fast转接明确。
- 历史环境 `validation-20261021/results/fa4_prefill/validate.json` 明记L20D/SM103、148SM、CuTe DSL4.6.1、torch2.9.0+cu130、cuda-python13.1.1；转换保留匹配硬件/栈/执行特征，Wiki使用DOC2尖括号占位，没有skip-hardware或新调查。

## 实际检查

所有命令从仓库根目录使用 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py`；默认检查规范化全目录后，最终 `root.yaml --read-only` 报 `3 YAML; 0 would change`，结构、text、from、relations、dispatch通过，没有 `--skip-hardware`。

三份最终 PNG 的完整命令写在案例 README。结果：root 168 actions/622 edges；persistent N=1→3 共191/704；split ATTENTION175/648、COMBINE20/58；均无环。已打开三张图核对布局及主要交接，图中无时间比例；Flow跨kernel只作静态校验。

额外只读有限检查（其余参数沿用 README）：

| 路径 | 参数 | 最终结果 |
| --- | --- | --- |
| root 无效候选 | tile_count(t)=0, valid(t)=0, rescale(n)=0, worktiles=1 | 16 actions/37 edges，无payload、无环 |
| root 单轮/尾部 | tile_count(t)=1, valid(t)=1, rescale(n)=0, worktiles=1 | 单轮通过；例如B=3、各Q=17/KV=64的m=1，query尾与缺第二page由源mask处理 |
| persistent 无rescale | tile_count(0)=1, tile_count(1)=3, valid(t)=1, rescale(n)=0, worktiles=2 | 187/692，无环；ownership握手保留 |
| split 空split | tile_count(t)=0, valid(t)=1, rescale(n)=0,lastvalid=3,worktiles=1 | ATTENTION12/22、COMBINE20/58；无环，仅empty LSE |
| split 单轮/all-zero combine协议边界 | tile_count(t)=1, valid(t)=1,rescale(n)=0,lastvalid=-1,worktiles=1 | ATTENTION54/171、COMBINE11/28，无环；不作为真实host五路评分输入复现 |

## 留给独立 review

当前三份内容与主图可开始全文独立审阅。需重点再核对：数学符号value的函数含义与调度绑定、联合token公式/独立SMEM、source tail与提前Y release、Stats别名覆盖的间接有序保证、combine初始无条件prefetch与正weight保护。单原语exp2/log/division误差界仍未知，没有GPU重跑结论。

工具问题已在 [TOOLS.md](TOOLS.md) 和可执行最小 [tools_repro.py](tools_repro.py) 中复现并发给主代理。源码已有reuse acquire被显式拆出以保留有界初始窗口；未添加新同步或虚假初始化。有效persistent工作后接无效padding的有限图仍被false-branch末项求值误拒；已静态确认源无payload路径，不声称该图通过。公共工具修复后按TOOLS中确切命令重跑。未修改公共工具/规则、其他case或schema_wiki，未commit。
