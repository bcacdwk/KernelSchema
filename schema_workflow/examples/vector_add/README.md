# FP32 vector_add

此示例包含手工固定的 [Task](task.md)、[eval 包](eval/) 和 [两分支 Schema](schema/dispatch.md)。材料在产品 coder 调用前确定，参考入口固定到真实提交 `2c9c346cdd0373821b633012bdfece446c766d65`。这不是运行时自动 oracle 业务；没有搬入旧 AKA 成绩，也没有人为增加 helper。

合法长度为 `1..65536`；root 的域为 `1..1024`，tiled 为 `513..65536`，重叠区明确选择 root。root 使用一CTA/1024元素，tiled 使用每CTA256元素；两者尾块均掩码。tiled 的数学区和流水线区真实 `from: root`，工作调度本地完整定义。固定七例为 `1,512,513,1024,1025,4097,65536`，case ID 为字符串 `0..6`。

参考为 PyTorch FP32相加；误差、计时和固定输入来源见 Task 与 eval.json。任务声明 NVIDIA L20D/SM103，实际工具链与设备环境以此次 Agate 返回为准；Wiki 未绑定资料保留规范允许的尖括号占位，不当成能力保证或成绩。

```sh
python3 schema_template/check_yaml.py schema_workflow/examples/vector_add/schema/root.yaml --read-only
python3 schema_implement/implement_kernel.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --schema schema_workflow/examples/vector_add/schema --gpu L20D --out .schema_workflow/runs/vector-add-implement
python3 schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --gpu L20D --out .schema_workflow/runs/vector-add
```

第二条使用固定示例 Schema，第三条从 Task 开始让 Design 选材写作。每次使用全新 out；恢复采用原目录的 `--resume`。没有已有成功结果或速度承诺，必须查看本次固定全案例的真实反馈。
