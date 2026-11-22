# B01 / K1.materials 实施交付

日期：2026-11-20。源码工作树：`/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`；分支 `feat/schema-workflows-v1`。开工基线 `484902855fe7be82a74fe7c394503d1d7d222dbb` 已含此前六算子成果。本轮未访问、修改或运行 AKA_experiment，未触碰原 main 的提取源码/YAML/产物；未运行模型 CLI、GPU 或 Agate。

**结论：K1.materials 的 E0 材料交付可签收，K2.interface 可直接消费真实文件开展 CPU 开发。R0 仅完成材料层结构/关系校验及手审，未具备完整 DOC2 环境绑定和执行准入；完整 evaluator / 真实资格不可签收。** 具体设备、架构和 torch/Triton/CUDA 等版本尚未选定，不能把候选 H20 当作实际固定环境。该缺项显式列出，不以结构检查代签硬件语义或 GPU 支持。

## 实际文件（26 个）

全部源码位于 [row_softmax 材料目录](../../../.worktrees/schema-workflows-v1/examples/workflows/row_softmax/task/README.md)，没有用本报告代替文件。

| 子目录 | 文件 | 作用 |
| --- | --- | --- |
| `task/` | `README.md`、`task.json` | K2 交接入口；Task/source/R0 原件映射，明确非执行草案 |
| `task/` | `contract.yaml`、`acceptance.md` | ABI、域、CPU FP64 golden、原 out backing/view、x 字节及 canary 等独立硬项 |
| `task/` | `inputs.json`、`workload.json` | 四 shape × 九实例 = 36 case，双 out 初值 = 72 次；seed23 的 offset3/5 在其中；另列同 Model 两次复用轨迹 |
| `task/` | `timing.md`、`environment.md` | 20 warmup、100 完整 forward/批、10 对 AB/BA、完整样本与等权几何均值；实际环境缺项 |
| `task/` | `scope.json`、`materials.json`、`workflow_objective.md` | 四输入分区及 R0 绑定草案；唯一 workflow_objective；21 条显式角色材料清单 |
| `task/` | `check_materials.py` | 可从任意 cwd 运行的有限材料一致性与 CPU oracle 检查，不是 evaluator |
| `schema/` | `root.yaml`、`dispatch.md`、`r0_review.md` | 手工 R0 的 FP32 稳定路径、单 program/CTA 独占整行、单自有 Triton kernel 与当前义务手审 fixture |
| `starter/` | `kernel.py` | ABI 正确、故意抛 NotImplementedError 的 C0；不冒充 V0 |
| `evaluator/` | `golden.py` | 读取调用前 CPU FP32 快照值的独立 CPU binary64 stable softmax；不 import torch/CUDA |
| `evaluator/` | `baseline.py` | 可执行 `out.copy_(torch.softmax(x,dim=-1)); return out` FP32 baseline，同 ABI；不是 golden 或 R0 合格实现 |
| `evaluator/` | `interface.md`、`manifest.json` | 完整 evaluator 的输入/输出/加载/副作用/计量交接；manifest 明确 interface_only |
| `references/` | `README.md`、`sources.json`、`catalog.md`、`missing.md` | 固定来源、真实摘要/许可、邻近机制差异、权限及后续缺项 |
| `references/triton/` | `02-fused-softmax.py`、`LICENSE` | 未修改的官方教程与 MIT 原许可 |

## 固定来源与权限

- 官方 Triton `v3.5.1` 经 GitHub API 解析到 commit `0add68262ab0a2e33b84524346cb27cbb2787356`。教程 10075 bytes，SHA-256 `8a65b696d5b02a3900380395ecd883671afeb66328db108f55622ed8853a05da`；LICENSE 1132 bytes，SHA-256 `92640fb97222fd0a698ff28ce0c3782c172623f8d6c609b557636a80f28fb946`。真实 contents API URL、Git blob SHA、日期均在 sources.json。git/raw 连接失败后改用成功的 API；没有伪造 URL、版本、摘要或资格。
- 教程是参考，不是与当前 R0 相同的完整实现：其 persistent 多行循环、分配输出、占用率选择和 benchmark 不属于 R0。v3.5.1 是已固定参考版本，尚不是已资格运行栈。文件仅作只读参考，不能被资源加载器 import 后触发 GPU。
- 新 golden 原件先经 CPU 检查，由主代理**只提交该文件**到 `2efc92479db497e0ecf681e276ecddc72cad0de6`，并按相同 blob/tree/commit SHA 发布到 `bcacdwk/KernelSchema`；主代理回读验证后回传。R0 Golden_reference.Source 使用真实固定 commit。子代理未修改 Git 配置/分支或自行提交。
- 45 个既有 DOC0–6、根角色方法、Template/工具源文件以真实字节摘要列入 sources.json；没有复制 parser 或改动共用工具。KS 自有源码的项目级再分发许可未识别，保留 K6.release 缺项；MIT 教程许可已留存。
- Design 只收显式公共材料，不收 R0/手审表/starter/私有 evaluator。Optimize 只收其请求选定的当前 Schema；R0 义务不套用所有 Rd/R′。Coder 收准确 seed + 当前 Schema，只写授权源码；摘要/路径可见不自动授予读取。声明的 allowlist 尚需后续运行时实现并验证。

## 真实验证结果

系统 `python3` 为 3.13.12，最初无 PyYAML/torch/Triton。仅把 `PyYAML==6.0.3` 安装到临时 `/tmp/ks-b01-k1-python-deps`，没有改项目或共享 Python。此 Python 版本的 CPU 检查不代表计划中 Python3.11 运行环境已资格。

在源码 worktree 中执行：

```sh
PYTHONPATH=/tmp/ks-b01-k1-python-deps python3 -B schema_template/check_yaml.py examples/workflows/row_softmax/schema/root.yaml
PYTHONPATH=/tmp/ks-b01-k1-python-deps python3 -B schema_template/check_yaml.py examples/workflows/row_softmax/schema/root.yaml --read-only
PYTHONPATH=/tmp/ks-b01-k1-python-deps python3 -B schema_template/check_yaml.py examples/workflows/row_softmax/schema/root.yaml --read-only --graph check --set m=32 --set n=1024 --worktiles 1
python3 -B examples/workflows/row_softmax/task/check_materials.py
```

最终各退出码均为 **0**。Template 最终为 `1 YAML; 0 changed` / `0 would change`；structure、text budgets、from、relations 与 dispatch 检查通过。有限图样本为 1 action、1 edge、0 boundary skips、acyclic，只覆盖所列参数和代表 worker。

材料检查输出：

```text
PASS: 36 cases / 72 basic calls; 1 separate two-call state trace; 21 role materials
PASS: byte inventories, input/scope/timing consistency, source syntax/ABI; 28 deterministic CPU golden cases
NOT RUN: locked torch random bytes, baseline CUDA execution, full evaluator, model, GPU, timing or service qualification
```

另外从 `/tmp` 通过 checker 的绝对路径运行，同样退出 0。golden 冻结前还运行 20 项 CPU 检查，覆盖四 shape 三常值、单峰解析比值/总和/输入不变与非法输入拒绝；`sys.float_info.mant_dig=53`。这些是 CPU 原件/参数检查，不是 V01 GPU pass。baseline/starter 仅完成语法/ABI 检查，没有谎称 torch/GPU 执行通过。

## K2 可直接消费的接口与余项

以 `task/README.md` 和 `task/task.json` 进入；task.json 给实际 Task 原件、baseline/starter logical `kernel.py` source 映射及 raw root/dispatch、`design_revision=0`/`change=new_task`。输入/分区/双毒化/复用轨迹分别在 inputs.json/workload.json/scope.json；角色清单与真实源摘要在 materials.json/sources.json；唯一 evaluator I/O 交接在 evaluator/interface.md。

所有本轮自有 JSON 标为 `material_format=kernelschema.row_softmax.materials.v1` / `publication_status=unbound_materials`，**不是 `ks.workflow.v1` 可执行对象**。没有 request.json、假 Ref、占位 Q/C 或伪造收据。K2 将真实字节快照到稳定 `artifacts/<publication_id>/`，生成唯一 JCS/Ref/Source/选定闭包；所有规范 Ref 均以 request bundle 根解析，不按嵌套文件目录解析。

K1.evaluator 后续实现 `input.py/check.py/timing.py/entry.py`、完整 canonical manifest 和实际 adapter；当前环境/profile 缺项仍需固定。实际 CPU torch 随机输入摘要、GPU 原 backing/canary/72 次检查、同 Model 复用、V0/native hook、计时/恢复/费用、独立 fidelity 与模型资格均未执行。partial seed、五类请求模板、有限 driver 和 evidence 归后续 owner。本次不改变这些门，也不让任何缺项阻塞 K2 的独立 CPU 合同开发。
