[root.yaml](root.yaml) 是三步、两次 launch 的固定结构满额展示，使用明确的 placeholder，不是语义正确性样例；全文没有绝对最长

规模：354 个物理行（原 380 行），28,939 个 Unicode 字符（含换行），28,953 个 UTF-8 字节；最长物理行 407 字符，字段最多 10 个逻辑行

4 个 Storage 条目均固定两行：首行为条目名及 meaning/depth/location/shape，次行为 dtype/total_size/member/view；可选字段缺失时略去，内容长度不改变分组

按当前 `text_policy.BUDGETS` 统计 YAML 解码值：受限说明文本 **100/100 满额**；计入名称、数学等后共 **116/132 个受限字符串满额**。Math 为 1000 字符/10 行，when 为 500/5，Thesis 及 Require/Prefer/Forbid 的 Choice、Reason 为 300/2；其余 16 个为合法短引用或 retention，未用长对象名填充，无文本超限

实际使用普通入口自动写回：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_discussion/template_tooling/auto_normalization/display_demo/root.yaml
```

退出码 **0**，Launch 条目使用 `step / order / start_after`，Thesis、Approx/Accumulate、Baseline/Gain/Cost 及 Storage 两行布局均符合当前规范；追加 `--read-only` 复查退出 **0**、0 项修改、无 warning/error，文件字节与 mtime 不变。占位数学、硬件、testing.invalid 引用与证据没有经过真实性验证；未发现工具问题

root 的三个 Design 区均本地定义，`from` 不适用；互斥的 count/stride、value/binding 分属不同记录。数字、枚举、类型和形状保持小值，源码与路径适度占位；保留 for/if、三类 Waiting、Storage/view、Index/Event 和八种 verb
