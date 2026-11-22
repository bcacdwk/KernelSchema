# B01 / K7.materials：FA4 F0/F1 材料交付

日期：2026-11-20。工作树：`/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`，实施基线 `484902855fe7be82a74fe7c394503d1d7d222dbb`。适用 AGENTS 已读；按用户 xhigh 要求工作，未派生代理。

**签收建议：K7.materials 完成（附真实缺项）；K8.engineering 可消费材料开工；K8.real/KS-P2 未通过。** 本批是实际新增的案例准入材料，不是后续工作泛泛计划；没有 runtime、GPU、模型或生成代码资格。

## 实际新增文件

均在上述隔离工作树，未提交、未推送：

| 路径 | 实际内容 |
| --- | --- |
| `docs/workflows/cases/fa4_decode-admission.md` | 可独立阅读的版本/来源、完整 13 参数 ABI、比较/副作用、F1/when/effective、原栈/冷启动、源码格式与异步局限审查 |
| `examples/workflows/fa4_decode/materials.json` | 27 个原件的 commit/path/Git blob/SHA-256/size/mode；完整 aka_solution 12 文件；六个 runtime Python 文件；历史真实 comparator harness；原源码与 Schema 分别固定；旧/新 Schema 差异摘要 |
| `examples/workflows/fa4_decode/f1-parameters.json` | 一个固定 F1、原 helper 十个 kwargs、seed=20261120、全部 ABI 对照、完整比较/副作用/out poison、实际 host dispatch、runtime when 未测状态 |
| `examples/workflows/fa4_decode/preflight-checks.json` | 9 项真实 CPU 检查及原输出、首次 PyYAML 缺依赖诊断、实际 Python/PyYAML/Template 身份；没有 GPU 收据 |
| `examples/workflows/fa4_decode/licenses-and-gaps.md` | available/missing/incompatible，许可/公开取得、完整 evaluator/source 格式/计时/profile/fidelity 与历史证据边界 |

JSON 明确是案例内材料记录，尚非 `ks.workflow.v1` 可执行 Task/Materials/Source/request；K2/K8 需读取清单中实际字节、核 hash 并冻结正式 Ref。没有用占位 hash 或 effective 替换 raw。

## 版本关系：需要精确澄清 V 的一句话

建议把 V #fa4 来源句精确表述为：

> 原始 kernel/reference/input/shapes 固定为 KS@3d64ad35900a0e6a66d6f93da34108fbceba0daf 的四题快照 v55；F1 消费的四节点 Schema/dispatch 则另固定为 KS@484902855fe7be82a74fe7c394503d1d7d222dbb 的 schema_template/examples/fa4_decode/，其 Source/Golden_reference 指向前述原件，不能将源码提交当作这份后续 Schema 的版本。

依据：当前 `root.yaml:4,64`、`dispatch.md:5` 都指向固定原码；`git diff --name-status 3d64ad… 4849028… -- kernel_wiki/aka-four-problems-20261019/fa4_decode` 仅有 `.DS_Store` 删除。所选全部原源码/参考/input/shapes/metadata/原记录逐字节一致。

`3d64ad…` 里原有七节点 Schema；当前整理为 root/q4_direct/q4_cluster/q1_global 四节点。材料清单记录两版 raw 摘要与三个节点删除，没有把旧七节点设为新实施目标，也未修改 V 正文。

## 原件与 K8 可直接消费的边界

源码为 `aka_solution/solution.py::Model.forward`，必须与 `qoder_greenfield_decode_attention/` 同目录。六个 Python 文件的 hash 全部与原 `PROVENANCE.json` 一致；即使 direct 不执行 combine，paged 模块也导入 gqa_decode.py，不能删掉它。完整源格式/原子安装/删除语义由 K8 实现，本批只固定真实所需文件和不可 overlay 的语义。

关键 SHA-256（完整清单见 materials.json）：

| 原件 | SHA-256 |
| --- | --- |
| v55 solution.py | `4f9859587e4b3f3736bc77c0555767967ffd83f867069f277dd65a6ef78d65da` |
| v55 gqa_decode_paged.py | `ec930c32758ea4f34e2df9fefa2de90f3ef345dc11fa66948fef74ae016fbe92` |
| input.py | `8958c3b642719c2af51bc77f93d5eb15417aea28df5f7c03d64096f90292554c` |
| reference.py | `8784ff4064dd7ebaa36313af7e762398bd5e1d8783a931eb7f86614959b6e419` |
| 当前 root.yaml | `36e632b74625e7b52b7d027f38efb6a0f29aac555d30d4006ce09bdefae3a864` |
| 当前 q4_direct.yaml | `912f5bd29d36b49b29274cf1397bc782434391be75a7d1a3f037a1a0426c6ec0` |
| 当前 dispatch.md | `d51d7ee33863cee08cb31aaa51248774d15c9cd2e774fa6dd1afcea49a402473` |

F1 固定 B1/Q4/seq_lens[64]/P1/有效 block_table[0]/N1/S1，headcount16、kvheadcount1、D256、page64、tablewidth16434、workspace413138944、两个 scale 和两个 None 全部保持 root。原 `_make_inputs` 的 Q4 `randn(BF16)*1.22/0.96→FP8` 顺序、scratch 初零、out empty 语义保留；新 seed 明示未测。out poison 为同一输入点的完整 evaluator 要求，不新增域或批次。

真实 host 的纯算术 `_select_policy(1,4,64,1)` 返回 `(1,"none")`，tile `(16,2)`；完整 F1 为两个 query tile/CTA worktile。`q4_direct` 材料 when 命中，但 native launcher/对象/签名/线程/current device/非 capture 条件未跑。root S9 的 when 拒绝 F1，也拒绝 Q1。Template effective 使用 root Contract + direct 三个本地 Design 区和 Realization；raw direct 无 Contract；q1_global 的 Math_reshape 实际来源仍为 root。

F1 是原 Schema README 第 17 行代表点，**不是原 30 个 scored shape 的一项**；`shape_valid.json==shapes.json` 各 30 项，shape_train 是域规范且 development_cases 为空。本批没有重做 extract 或修改原 shapes。

完整判据原件：`root.yaml:63–67` 的 FP32 两输出有限/shape/dtype/逐元素 `abs(a-r)≤0.06+0.04*abs(r)`；`root.yaml:38–43` 与 `metadata.json:11–37` 的 alias/只许 out+workspace 变更；`reference.py:45–88` 数值实现。K8 仍须实现原 out storage/view、完整覆写、poison、只读输入不变等独立 harness，原 reference 本身不是完整 evaluator。Scope 只选 F1，task_uncovered 不为空，coverage/fidelity 分判。

补充实际 comparator 来源：同一 `3d64ad…` 的 `kernel_wiki/aka-four-problems-20261019/validation-20261021/bundles/fa4_decode/harness.py:44–96` 已有 FP32 finite/逐元素、poison、未变输入、返回/原位输出及 data_ptr alias 检查，`120–121` 行禁用 TF32，已作为第27项原件纳入；没有导入/运行。旧 alias 只核调用后 data_ptr，未绑定调用前原 out storage/view；`157–167` 行 post-timing 仅复查返回值。其 main 还加载 Production/原shape/旧计时，不能直接代签 F1。因此真实缺项是 K8 固定接入、原 out 身份边界与完整复查，不是 comparator 源码完全缺失。

## CPU 检查与结果

1. 27 个原件 Git 字节/hash/size/mode 与工作树核对，含历史 comparator harness 的 AST 函数位置。
2. 完整六文件 Python 源 hash 与 PROVENANCE 核对。
3. AST 核对 Model/reference/run 的同序 13 参数。
4. helper 十个 kwargs、固定常量、浮点类型、原 shape 别名/计数与 F1 来源核对。
5. 既有 Template 的 effective_document/raw/origins 检查。
6. 仅提取四个无 import 的纯整数 host 函数核 F1 policy；依据 root raw 的 Q/B/L 门作绑定反例。
7. 既有 Template CLI 只读结构/dispatch/有限图检查：四份 YAML、0 文件拟改；N1/S1/worktiles2 为 88 actions/284 edges/8 boundary skips/acyclic。
8. 进程未导入 torch/cutlass/cuda，也未执行 Model、run 或原 kernel。
9. 三个案例 JSON 可解析；另核两个 Markdown 文件共 10 个本地链接存在。

Template 命令：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/q4_direct.yaml --read-only --graph check --set N=1 --set splitcount=1 --worktiles 2
```

实际 CPU 为 Python3.13.12；默认缺 PyYAML 首次返回2，在 `/tmp/ks-b01-k7-pyyaml` 隔离安装 PyYAML6.0.3 后返回0，未改仓库依赖。此环境不是计划中 Python3.11/CuTe runtime 资格。图是每 step 一个代表 worker 的有限声明动作图，不是共享地址竞态检测或两 CTA GPU 执行。

## 真实门缺项

- 许可：四个实现文件标 NVIDIA/BSD-3-Clause，旁置 LICENSE 却以 Apache-2.0 开头并引用缺失 licenses/；包装、input/reference/shapes/Schema 的逐件授权/notice 也未闭合，根无 LICENSE。原件可读不等于全库再分发授权。
- 公开取得：历史有 GitHub 定位，本次未做两个固定 commit 的匿名取得/独立 checkout 证明。metadata 的部分旧生产 probe/raw-tensor 证据在私有来源，仅登记，不追取；不冒充原证据完整公开闭包。
- K8 工程：完整 evaluator、多文件 codec/adapter/删除与执行源身份、F1 公共性能 baseline 和固定 timing recipe 未实现。
- 原栈：CuTe DSL4.6.1、L20D/实际SM103、torch2.9.0+cu130、cuda-python13.1.1 是材料记录；实际驱动/编译器/Python/TVM FFI/profile 与硬件 Wiki 仍未闭合。旧 production torch2.11、roofline B300/SM100 或 H20/Triton 不能代签。
- 异步：F1 避开 global 空 split 旧 opartial 和 N≥5 PageIndex 四槽覆写路径，但 N1 空奇偶槽通知/读回、TMA padded 页/页尾、TMEM 和输出 staging lifetime 仍需 fidelity 与执行证据，不能因图无环签全部异步安全。
- 冷/稳态：未建立 native launcher、未记录首次 fmha+FFI probe/同步/恢复，也未量完整 forward。launcher `1e-2` 范数探测不是 `0.06/0.04` Task 容差。历史30/30、1.1126× kernel-sum 和 stress 叙述只作旧原码材料。

未访问/修改/运行 AKA_experiment；未修改 AGENTS、根 README、计划正文、pyproject、kernelschema、row_softmax、Git 配置/分支/提交/远端；未动原 YAML、源包或 PNG。原 main 本批唯一写入为本报告。
