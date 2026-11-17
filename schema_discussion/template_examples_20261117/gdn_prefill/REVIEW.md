# GDN 独立全文审阅

结论：接受单份 `root.yaml` 的完整设计集合，当前 Version 9。独立读取了 root 全文、目录、公开合同/输入/reference、固定单文件入口和 selector、构建脚本及相关拆分 kernel/scheduler。未发现需要重新展开单序列完整 YAML 的源码事实。仅改本案例和本讨论目录，未改源码、公共工具或规则；未运行 GPU/Runner，未提交 Git。

## 选型及局部闭合

固定 `solution.py` SHA-256 实核为 `e216c76d8c434233cba9042be79500438a522da13cc42090281cc138ae427c69`；其 Model AST 保留 `cu_seqlens.to(torch.int32)` 和 `value_rows_override=64`。`build_solution.py:build` 给出 scheduler、kernel、adapter 拼接及 minifier 的关系。M64/M128 是 value/state 行数，token chunk 都为 64；公开 Model 不自动选择 M128。

single 与 multi 都由同一 `(sequence, head, value-half)` 工作承担 inverse、KS/QS/U/Y/S、FP32 状态和同 chunk 输出。single 的 static descriptor、raw scalar 两槽、ready-only 五槽及 compact CG1 通知可以在 root 已有输入/通知对象周围闭合：

- raw 槽 c 的 TMA 完成后，当前 head 列先进入寄存器，warp 同步和 async.shared fence 位于 c+2 的重新提交之前。
- ready 槽 c+5 的写入在 O(c) store 完成后。O(c) 的生产已依赖 tinv(c) 和 U/Y，故 beta(c) 的列缩放读取、INVERSE 的 g(c) 捕获、RECURRENCE 的 g/gamma(c) 捕获均已结束；不需要伪造 scalar empty-stage return。
- 单 CG1 barrier 的 stateoperand→r→ubar/udec 三次 arrival，分别由 KS、U、S 结果限制下一次发布；消费者 phase 为 `c&1`、`(c+1)&1`、`c&1`。通知载荷、状态归属、输出生产者均没有迁移。

M128 记录保留 K 槽、state 搬运和 QK 提前归还的联动；不把它与 single 的 M64 专属配置直接组合。内部 None/checkpoint/state-pool 路径不扩张公开合同，zero 模式仍加载显式 FP32 初态。

## 实际修正

- root Version 8→9：alpha/beta 的范围补出 `torch.rand` 均匀采样事实。
- inverse 的局部 `p/r/j` 改为 `pivot/row/col`，明确 8 行范围，避免行下标 `r` 与后续残差 `r` 同名；公式及舍入保持原义。
- 从 Contract Math 移出 oracle 的实现说明，放在 README 的参考/容差说明中，保留 FP32、32-token block inverse、log-space decay 的事实。
- dispatch 补出 raw 两槽和 ready-only 五槽的 stage/phase，以及 gate/beta 写入后 warp 同步、elected-lane ready arrival 的真实发布顺序。
- 重画 root.png；README 更新独立验证边界。未新增同步边、配置字段或完整变体。

## 来源核对

以下路径相对 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`。

| 核对项 | 来源及结论 |
| --- | --- |
| 公开域与输入 | `gdn_prefill/shape_train.json:25-74`、`input.py:_make_inputs`：B=1..16、总 T≤16384、各序列非空；连续张量；只有两个输出 buffer 可变。q/k/v 的行中心和元素分布、BF16 K normalize、FP32 uniform alpha/beta 均对应输入构造；Q 不归一化。 |
| 输出/初终态/None | `gdn_prefill/reference.py:Model.forward`、`metadata.json:benchmark_contract`：必传初终态，返回 supplied buffer alias；state 外部为 [B,H,V,K]，reference 内部转置为 [B,H,K,V]。continuation 来自每序列独立 32-token 参考前缀。 |
| scale 与容差 | `metadata.json` 给 .845，`input.py` 实际 `1/math.sqrt(128)` 给 .843；实核 Python binary64 不等、FP32 相同。`gdn_prefill.py:161-168` 的 Python equality 不被实际输入触发。公开四路逐元素阈值为 O=0.01/0.01、state=0.005/0.001；`validate.json` 实核 30 几何/180 case 的 aka passed。`gdn_agate_eval/summary.json` 的 runner rtol=.05 不覆盖更严格合同。 |
| reference 身份 | 实核 `gdn_prefill/reference.py` 与 `gdn_prefill/solution/agate_reference/reference.py` 字节相同；前者 Model 将 q/k/v 加宽到 FP32，默认 block_size=32。 |
| 数学与 inverse | `aka_solution/gated_delta_net_chunked.py:2643` 扫描两半 gate；`:3595` 先捕获 pair decay/row beta，再读两个 KK，完成 inverse/column beta，最后完成两个 QK 权重；`:4207` 的单位下三角 8x8 递推和三个合并级保持两次 BF16 乘积舍入。 |
| 状态与输出 | 同文件 `:2891`、`:4969`：sbar 在衰减前取 BF16；残差存在 KS 窄化和相减后窄化；ubar/udec 都来自未舍入 U，udec 不从 ubar 派生；QS 就地缩放后累加 Y；同 chunk Y→o，最终 s_final 独立于输出尾 predicate。 |
| 协作与寿命 | 同文件 `:1276-1510`、`:1519-1718`、`:4742-4966`：TMEM 由 CG1 分配，其他参与者先进入 retrieve rendezvous 合法；初态有 group rendezvous/arrival；最终 state 完成后读取并归还，随后释放 TMEM，O 的 SMEM store 可仍在途。root 保留 done←done allocation 边和 RECURRENCE o→SCALAROUTPUT store:o 准备边。 |
| single 局部 | 同文件 `:2316-2475`、`:2970-3080`、`:5244-5510`：raw phase/复用 fence、五槽共同发布/两消费者、三次 compact CG1 phase 均由源码核对；T%64==0 不足以关闭尾 predicate。 |
| descriptor 与提交 | 同文件 `:1880-2050`、`:2052-2275`；`gdn_prefill.py:443-476`：Q/K/V 每工作更新 descriptor，O 更新前完成旧 stores；descriptor workspace 跨调用缓存。`evidence/gdn_prefill/gdn_kernels.csv` 记录 conversion block128/grid1，支持真实 INDEX_CAST step。 |
| Hardware | `validation-20261021/results/gdn_prefill/{validate,profile}.json` 与 `gdn_prefill/aka_solution/AGATE_VALIDATION.json` 支持 L20D/SM103、148 SM、CuTe DSL4.6.1、torch2.9.0+cu130、cuda-python13.1.1、runtime13.0、driver580.105.08。现行 Wiki 尖括号链接仍是占位。 |

## 最终验证

从仓库根目录执行，均未 skip hardware：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --read-only
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --graph png --set 'chunk_count(0)=6' --set 'chunk_count(1)=2' --worktiles 2
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --read-only --graph check --set 'chunk_count(0)=2' --worktiles 1
```

默认检查通过，仅规范两个 gate range 的无必要引号；只读为 1 YAML / 0 would change。主图为 GDN 249 actions / 979 edges / 30 boundary skips，最小样本为 68 / 242 / 19；INDEX_CAST 均为 1 / 1 / 0，均无环。日志为 `review-check-default.log`、`review-check-read-only.log`、`review-graph-main.log`、`review-graph-minimum.log`。root.png 已目视检查，包含两 step、六个 GDN 角色和末端释放。

本范围最终 `git diff --check -- schema_template/examples/gdn_prefill schema_discussion/template_examples_20261117/gdn_prefill` 无输出。

6+2 主图可对应 B=3、lengths=[257,1,1]、num_sm=148 的 CTA0，覆盖 partial chunk、全 padding chunk、环复用和跨工作接续；2 chunks 是非空、成对补齐的最小合法数。另以张量 shape/dtype 与 capability 替身执行实际单文件和拆分源码中抽出的 selector，核对 `[1,1]`、`[257,1,1]` 以及 single T=1/64/65/128/129/257，共 8 例完全一致，见 `review-source-boundaries.log`。这是 host 元数据审计，不执行张量、kernel 或 GPU，也不声称是 single 完整 YAML 的图验证。

## 遗留与验收边界

没有新公共工具问题。源实现仍缺少 Q/K/V descriptor 改写相对于旧在途 descriptor 读取的充分顺序依据，以及共享缓存 workspace 的跨调用互斥；root 没有补造 drain/私有 workspace。fast log2/exp2 无已建立误差界，Wiki 仍未形成完整资料绑定。历史 30/180 通过只按原记录保留，有限图和本次静态审阅不替代 GPU 数值验证或证明这些来源限制已解决。
