# GDN 第一轮写作交接

已停止使用旧字段并完成单份 `root.yaml`（Version 8）；删除旧 `m64_single_sequence.yaml/png`，重画 root.png，更新 dispatch 与本案例 README。只改本案例及本讨论目录，未改源码、工具、公共文档或其他案例，未运行 GPU/Runner，未提交 Git。下一步由 fresh high 独立审阅。

## 选择与删减

选择依据见 [SELECTION.md](SELECTION.md)。已验证 Model 强制 64 value 行；token chunk 恒为 64，成对补齐。单序列 descriptor/scalar/通知变化可以围绕已存在输入和通知接口闭合，未改变七个乘积、状态归属、输出生产者或主工作组织，降为 dispatch 局部条目。M128 是内部 override 参数及其布局、缓冲、提前归还联动，未新建文件。单份 root 的 Design 为 B>1；Contract 仍覆盖 B=1..16。保留局部变化信息不表示 root YAML 暗含 selector。

## 已核对的来源事实

所有路径相对固定 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`。

- `gdn_prefill/aka_solution/solution.py` SHA-256 为 `e216c76d8c434233cba9042be79500438a522da13cc42090281cc138ae427c69`。`build_solution.py:build` 拼接 scheduler/kernel/adapter 并追加 Model 后压缩；实际 forward 保留 `cu_seqlens.to(torch.int32)`，固定 `value_rows_override=64`。
- `gdn_prefill/shape_train.json` 的 shape_domain/invariants 明示 B≤16、sum(lengths)≤16384。不是从30个shape的 B15 样例外推。`metadata.json`、`input.py:_make_inputs`、`reference.py:Model.forward` 定义必传初态、必传终态、输出原 buffer alias、仅两个 buffer 可变、连续布局；zero 是 tensor，不走内部 None 首块捷径。
- `input.py` 对 BF16 K 做 normalize，不归一化 Q；alpha/beta 已是 FP32 线性 gate。continuation 是每序列独立32-token参考前缀的 FP32 终态，准备不计时。scale 元数据为 .845，实际 `1/math.sqrt(128)` 为 .843；两者 Python binary64 不等、FP32相同。已在 Contract/dispatch/README 留明，未宣称实际输入触发 static_scale。
- `reference.py` 与 `solution/agate_reference/reference.py` 字节相同。`metadata.json` 与 `validation-20261021/results/gdn_prefill/validate.json` 使用逐元素 `abs(c-r)<=atol+rtol*abs(r)`，O=0.01/0.01、state=0.005/0.001；检查返回与被改buffer。后者原件记录30几何/180种子，aka全部通过；`gdn_agate_eval/summary.json` 的通用 runner rtol=.05 不是合同替代。未把 relative-L2 最大值写成判据或界。
- 可读 `gated_delta_net_chunked.py:2793` 的 MMA0 是 KK0/KK1/QK0/QK1；`:3595` 先捕获门因子和row beta、读完两个KK、分层逆、列beta乘、再完成两个QK权重。`:4207` 从单位下三角8x8开始，BF16结果分层合并；Math显式写两次乘积的BF16舍入。
- `:2960` 与 `:5190` 的 KS/QS/U/Y/S 与递推交织；FP32初态、BF16快照、两次残差舍入、从同一FP32 U分别生成Ubar/Udec、QS就地缩放、同chunk Y→O、终态全量写回均保留。`:4742` 的初态写TMEM有group rendezvous与arrival，`:4861` 终态等待最后S完成；分配不当作清零。
- `:1880` Load 更新每work的 Q/K/V sequence-end descriptor，后续请求fence；旧在途descriptor读取顺序仍未确立。`:2052` O descriptor更新前有前work全部O store完成。尾部Q/K/V=0、alpha=1、beta=0；T%64==0仍可能有synthetic chunk，full标量捷径只允许B=1且T%128==0。
- `:2316` 单序列raw scalar `[64,4]` 两槽：读当前head列后warp同步与async.shared fence，再发c+2；五槽ready-only依赖四块prefetch及O推进，复用条件写在dispatch。compact CG1的三次barrier通知与phase也写明。
- 独立验证 NCU 的 conversion 实际 block=(128,1,1)、grid=(1,1,1)，因此 INDEX_CAST 保留四warp单CTA；不是虚构省去该提交。固定环境仍是 L20D/SM103、CuTe DSL4.6.1、torch2.9.0+cu130、cuda-python13.1.1；driver580.105.08和runtime13.0来自既有Agate原件。

## 工具结果

确切命令见案例 [README](../../../schema_template/examples/gdn_prefill/README.md)，Python固定 `/opt/venv/python3.13.12/bin/python3 -B`。最终默认检查通过；后续只读预览 1 YAML / 0 would change；均未 skip hardware。日志为 `check-default.log`、`check-read-only.log`、`graph-main.log`、`graph-minimum.log`。

| 图样本 | INDEX_CAST actions/edges/skips | GDN actions/edges/skips | 结果 |
| --- | --- | --- | --- |
| chunk_count(0)=6, chunk_count(1)=2, worktiles=2 | 1/1/0 | 249/979/30 | 无环，root.png已更新 |
| chunk_count(0)=2, worktiles=1 | 1/1/0 | 68/242/19 | 无环 |

主图可对应 B3、lengths=[257,1,1]、num_sm148 时worker0的两个工作，覆盖部分尾块、全padding chunk、三/四/五槽环和跨工作接续。0/1 padded chunk不在非空且成对补齐的真实域中，未假作合法分支。仅绘一个代表worker；图不是全部输入或GPU正确性证明。root.png为2386×9754，已做缩略视觉检查，保留两step和六角色。

首轮工具运行按反馈处理了旧字段Preassigned.step、大小写名歧义和文本长度；数学随后改为明确的batch_count/query_heads等名称，避免B/b、HQ/hq混淆。一次 `--read-only --graph png` 因CLI互斥被拒，随后按正式PNG命令成功生成；未计作通过。未遇到需要公共工具修改的新缺陷。最终本范围 `git diff --check` 无输出。

## fresh reviewer 重点

1. 独立核对单序列是否确属局部闭合，并逐项检查dispatch对ready-only标量和compact CG1的联动完整性。
2. 核对inverse BF16层级、残差两次舍入和Ubar/Udec的不同来源；数学中的s_final独立于尾块有效输出predicate。
3. 核对提前归还与实际等待：KK/QK寄存器捕获、tring三槽、bring两槽、Q/K双消费者、KS/U及QS/Y别名、V延后归还、SCALAROUTPUT store明确等待RECURRENCE O。原旧稿缺该O准备边，现已写入。
4. TMEM分配rendezvous的消费者等待为done←done，允许参与者先进入；释放通过最终state链传递保证，O的SMEM store可仍在途。Storage含关键寄存器捕获和TMEM物理容量，不靠分配假定初值。
5. 两项来源限制未补造修复：每work Q/K/V descriptor更新的旧在途读取；`gdn_prefill.py:443` 缓存workspace并用于当前stream，跨共享缓存调用无互斥。固定代码没有被修改。fast log2/exp2误差界、Wiki入口仍未齐备；只按已知边界陈述。
