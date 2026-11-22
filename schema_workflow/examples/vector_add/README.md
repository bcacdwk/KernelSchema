# FP32 vector_add

此示例包含手工固定的 [Task](task.md)、[eval 包](eval/) 和 [两分支 Schema](schema/dispatch.md)。材料在产品 coder 调用前确定，参考入口固定到真实提交 `2c9c346cdd0373821b633012bdfece446c766d65`。这不是运行时自动 oracle 业务；没有搬入旧 AKA 成绩，也没有人为增加 helper。

合法长度为 `1..65536`；root 的域为 `1..1024`，tiled 为 `513..65536`，重叠区明确选择 root。root 使用一CTA/1024元素，tiled 使用每CTA256元素；两者尾块均掩码。tiled 的数学区和流水线区真实 `from: root`，工作调度本地完整定义。固定七例为 `1,512,513,1024,1025,4097,65536`，case ID 为字符串 `0..6`。

参考为 PyTorch FP32相加；误差、计时和固定输入来源见 Task 与 eval.json。Task 选择 Agate L20D；手写 Schema 的 NVIDIA L20D/SM103 身份与 [environment.json](environment.json) 中服务报告的 `NVIDIA L20D / sm_103` 相符，架构依据返回字段，不由型号名称推断。

环境材料固定到 `KernelSchema@51f6b2399d29bafbeacf401de45afa7011160dc8:schema_workflow/examples/vector_add/environment.json`。它来自 SDK 0.14.37 的只读 `get_env('L20D')`，2026-11-22 12:29:54 UTC 获取，`source=cache`、`probed_at=2026-11-22T11:48:31.084555+00:00`；该查询没有创建 GPU job。服务缓存报告 PyTorch 2.9.0+cu130、CUDA 13.0、Triton 3.7.1、nvcc/ptxas 13.1.115。root Version 2 据此更新实现栈，未报告的 Python 版本明确留缺项。

此记录不是候选实测，也不保证未来分配保持相同环境；它不提供完整硬件或编程手册。Wiki 未绑定资料保留尖括号占位，不能当成已就绪的能力依据。实际候选的设备、工具链、正确性与计时以本次 Implement 返回为准。

[triton_3_7_1_language.json](triton_3_7_1_language.json) 补充所选 Triton 版本的官方语言资料：`v3.7.1` 对应提交 `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`，同提交的版本声明、load/store 原文、摘要、获取时间和 MIT 许可一并保留。它明确 false mask 不加载或存储对应地址；不声称周围算术也被屏蔽，不提供 GPU 实测或具体生成指令的证据。

真实实施从 Task、环境和语言资料独立 Design，使用新交付中正式绑定资料的 Schema。手写 Schema 保留为结构开发示例，仍有资料占位；不能把它视为已具备全部实现依据，也不把它作为以下 Design 的目标答案。

```sh
python3 schema_template/check_yaml.py schema_workflow/examples/vector_add/schema/root.yaml --read-only
python3 schema_design/design_schema.py --task schema_workflow/examples/vector_add/task.md --reference schema_workflow/examples/vector_add/environment.json --reference schema_workflow/examples/vector_add/triton_3_7_1_language.json --out .schema_workflow/runs/vector-add-design
python3 schema_implement/implement_kernel.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --schema .schema_workflow/runs/vector-add-design/schema --reference schema_workflow/examples/vector_add/environment.json --reference schema_workflow/examples/vector_add/triton_3_7_1_language.json --gpu L20D --out .schema_workflow/runs/vector-add-implement
python3 schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --reference schema_workflow/examples/vector_add/environment.json --reference schema_workflow/examples/vector_add/triton_3_7_1_language.json --eval-task schema_workflow/examples/vector_add/eval --gpu L20D --out .schema_workflow/runs/vector-add
```

第二、三条分别展示独立 Design 与 Implement；确认 Design complete 后，Implement 使用其实际交付目录。第四条由 Workflow 串联同样的入口，普通资料会继续交给 Implement 和 Optimize。每次使用全新 out；恢复采用原目录的 `--resume`。没有已有成功结果或速度承诺，必须查看本次固定全案例的真实反馈。
