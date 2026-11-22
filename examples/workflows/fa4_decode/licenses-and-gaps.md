# FA4 材料可用性、许可和缺项

本文件与 [materials.json](materials.json) 的两个固定提交配套；没有修改原材料、下载模型/GPU 依赖或假定授权。清单中的 `available` 表示原字节可由本地 Git 取得，不能解释为已经公开再分发或具备运行资格。

| 状态 / ID | 核实结果与实际证据 | 影响 |
| --- | --- | --- |
| available / A1 | `3d64ad…:fa4_decode/aka_solution/` 全部 12 文件；入口和包六个 Python 文件 SHA 与 PROVENANCE 完全相符 | F0 原码可定位，K8 可准备准确 seed/verify 源闭包 |
| available / A2 | 同提交 reference.py、input.py、metadata.json、shape_train.json、shape_valid.json、shapes.json；shape_valid/shapes 同字节，各 30 项 | 原 ABI、输入语义、数值参考与原域可核 |
| available / A3 | `4849028…:schema_template/examples/fa4_decode/` 四节点、dispatch、README；Source/Golden_reference 指向 3d64ad；原题目录仅移除 .DS_Store | root Contract + direct Design 的 F1 材料对应关系闭合 |
| available / A4 | N1/S1 有限图、effective 来源和原 host 纯算术 dispatch 通过 CPU 检查 | 仅材料/结构结论，见 preflight-checks.json |
| available / A5 | 同一 3d64ad 的 `validation-20261021/bundles/fa4_decode/harness.py:44–96` 已有 FP32 有限/逐元素比较、poison、输入不变和 data_ptr alias 源码，未执行 | K8 可复用真实 comparator/副作用检查来源；不必假设全部 harness 从零缺失 |
| missing / L1 | aka_solution/LICENSE 的 `1–201` 行是 Apache-2.0 文本，`203–230` 行列第三方并指向未随目录提供的 `licenses/`；`operator.py/gqa_decode.py/gqa_decode_paged.py/mask.py:1–2` 标 NVIDIA copyright + BSD-3-Clause | 有许可标识，但所选文件的完整授权/notice 对应关系及 BSD 正文未闭合；公开发行前须取得适用原件与许可结论，不能擅自改 header 或重授 Apache |
| missing / L2 | `solution.py`、`__init__.py`、input/reference/shapes/metadata、历史 harness 与当前 Schema 没有本次能确认的逐件授权；两固定提交根目录无 LICENSE/COPYING/NOTICE | 不能从另一个 solution/ 的 BSD 声明或旁置 LICENSE 推断整个案例/KS 全库可再分发；K6/K8 需按选定公开字节补齐 |
| missing / L3 | requirements 为 torch>=2.9、cuda-python、nvidia-cutlass-dsl==4.6.1；实际 TVM FFI/torch/cuda-python/CUTLASS 依赖闭包与分发许可未冻结 | 真实 profile 和公开依赖说明缺项；CPU 核心不因本案例引入这些依赖 |
| missing / P1 | 历史 Schema 有 bcacdwk/KernelSchema 仓库定位；本次未做两个固定提交的匿名取得、独立目录 checkout 及必要材料下载证明 | 本地可取得不等于公开取得；不把作者缓存当交付，不要求用户提供私有凭证以掩盖缺项 |
| missing / E1 | 原 comparator/harness 已定位，但旧 alias 仅检查调用后 data_ptr，未固定调用前 out storage/view；post-timing `:157–167` 仅比较返回数值，旧 main 加载 Production 和原 30 shapes | K8 仍需接入固定 F1 Task/evaluator/source adapter、补全原 out 身份/完整复查并核坏候选；原 AGATE_VALIDATION 不是新 evaluator 收据 |
| missing / E2 | 六文件源码需求已确认；本批未实现 F1 多文件 codec/删除/原子安装/执行身份 adapter | 阻止真实 source 准入；不能声称首题 single_file 自动兼容 |
| missing / E3 | 本批仅固定完整调用的计量边界；F1 的可执行公共性能 baseline、warmup/重复/统计 recipe 和 samples 尚未冻结 | K8 必须给出当次固定计时判据与实际原始记录；reference 和旧 Production NCU 数值不自动成为合格 baseline |
| missing / R1 | 当前 CuTe/实际 SM103 设备、驱动、CUDA 编译器、Python、TVM FFI 与完整原栈 profile 未检查/冻结；Schema Wiki 为 @pending | K8.real 未获资格，保留 blocked；不影响小题独立材料/工程门 |
| missing / R2 | native launcher 建立、签名/对象/线程/current device、非 capture 条件未实际观察；冷启动、probe、稳态完整调用与 samples 未运行 | F1 仅材料 when 通过；不能把完整 when 或性能写 pass |
| missing / F1 | 空 parity、TMA padded 页、TMEM/SMEM lifetime 与源到当前 Schema fidelity 仍需独立审阅/实际证据 | N1 不豁免适用异步义务；报告必须区分 coverage/fidelity，未明为 unknown |
| missing / H1 | metadata.json `1–8,38–58` 引用私有 production probe/replacement/eval 原件与 OSS raw tensors，未包含在选中公开材料内 | 不能声明历史生产来源证据完整可导入；F1 使用原合成 helper，不把这些私有记录默认为本次生成的依赖或继续向外追取 |
| incompatible / I1 | 3d64ad 的七节点旧 Schema 与 4849028 四节点 Schema 内容不同 | 源码提交、Schema 提交必须分开固定；不能静默替换 |
| incompatible / I2 | root 的 when 是 Q4/B15-or-16/长 L/S9；q4_direct raw 没有 Contract | root 绑定 F1/Q1、只拿 direct raw、缺必要 dispatch 都不准入 |
| incompatible / I3 | FA4 原栈为 CuTe + 实际 SM103；首题 H20/Triton、E5M2、BF16/page128 桥和其它 solution/ 都不是选中实现/域 | 这些结果不能代签 F1；不因此扩大输入域或改原 Schema |
| incompatible / I4 | root `:167–185` 空 split 旧 opartial；pageindex_order.md `:10–18` 的 N≥5 未决覆盖顺序 | F1 避开这两条具体路径，仅允许子域结论；不宣称全异步安全 |
| incompatible / I5 | 历史 30/30 与 1.1126× 来自旧原码；四题 README `:44–49` 明确 kernel-sum 不含 host | 不代替新重建、副作用、fidelity、原栈或完整调用性能资格 |

已记录的环境标签必须保留出处：Schema `root.yaml:69–82` 是 L20D/SM103、CuTe DSL 4.6.1、torch 2.9.0+cu130、cuda-python 13.1.1；`aka_solution/PROVENANCE.json:5` 也记 sm103。`metadata.json:86` 的 production torch 2.11 开发版和 `roofline.json:5` 的 B300/SM100 模型标签不属于同一个 F1 profile，不能混合成已验证环境。

本批没有发现六文件优化源码缺失；缺失的是上述授权/公开取得/完整 evaluator/适配/真实资格。K7.materials 可以完成，K8 依赖的真实门仍保持未通过。若后续选择纯 Schema 重建，只投递允许的 Contract/Schema/公开说明，不向 coder 投递优化源码、annotated、旧会话或二进制；选择原码复验/原码 seed 续改时则显式固定准确 seed 和相应消费模式。
