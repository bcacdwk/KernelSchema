# FA4 有限图窗口问题

以下记录保留修复前基线与复现。原两项问题已由 [TOOL_FIXES.md](../TOOL_FIXES.md) 修复；当前 reviewer 的有效→padding 样本也已通过（80 actions / 259 edges / 21 boundary skips）。最终三份主图与全部必要边界检查见 [REVIEW.md](REVIEW.md)。

复现文件：[tools_repro.py](tools_repro.py)。只调用 `pipeline.build_step`，不保存 YAML、不改工具、不运行 GPU：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_discussion/template_examples_20261117/fa4_prefill/tools_repro.py
```

工具基线 `7a8030f`，实际输出：

```text
flatten_boundary: ExpansionError DESIGN.Tile_pipeline.Step.MAIN.Role.USE.Waiting.Released[0]: cannot uniquely recover action-index coordinates: t
padding_window: Boundary DESIGN.Tile_pipeline.Step.MAIN.Role.USE.Waiting.Prepared[0]: n[t,tile_count(t)-1]: i=-1 outside its declared domain
```

第一例保留修改前的关键结构：`t=worktile`、`i=[0,tile_count(t)]`、`n=flatten[t,i]`；producer 分首项与后续循环声明同一动作族，等待为 `compute:k(n+2,0) ← load:k(n,0)`。样本 `tile_count(t)=4,valid(t)=1,worktiles=1`；移位越界应只跳过边界，混合声明中某项不能恢复 t 不应压过已明确的域外判断。原 FA4 对应 `MMA matmul:qk(n+2,0) ← SOFTMAX load:score(n)`。反向写 n/n-2 时，原稿的 LOAD→MMA 边也出现相同 t 恢复错误。

第二例保留修改前尾块动作 `compute:k(n[t,tile_count(t)-1],0)`，由 `if(valid(t))` 控制；样本 work0 长度1有效，work1 长度0无效。等待固定第二参数触发 window 全集枚举，错误求值无效工作的 last index。原 FA4 persistent 精确命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/packed_persistent.yaml --read-only --graph check \
  --set 'tile_count(0)=1' --set 'tile_count(1)=0' --set 'valid(0)=1' --set 'valid(1)=0' --set 'rescale(n)=0' --worktiles 2
```

曾尝试将 QK/PV 表为等价单 issue loop，该样本仍失败，错误定位为 false branch 中的 `n[t,j-1]`；现行正文保留源码的首项/稳态/末项组织。这不是源实现会执行该越界操作的证明。未用虚假正迭代数填充无效工作。

DOC 依据：`schema_docs/5.design.tile_pipeline.md` 的“依赖的索引窗口”要求目标因自己的 if 为假而不存在时不实例化等待；变量移位落域外项跳过；“连续序号”明确 n 绑定可恢复 t/i，且跨工作长度可变。“Pipe”也规定 false if 不产生动作。

已用的有限替代表达：源码本有的 ring acquire 显式拆在首次复用 `n>=2`（joint token为 `e>=4`）处，等待只对应已有旧读者；初次空闲槽没有旧读者，由搬运/写入动作取得其初次资格。未新增同步、零初始化或每工作 drain。split 的单工作 token 以 `e=2*p+d` 直接编号，K/V payload 仍独立；不再让嵌套 flatten 的 d 恢复影响另一动作族。三份主图均成功；persistent 有效工作后接无效 padding 的图仍待工具修正，源码路径已静态核对。

## Reviewer 最终表达

2026-11-17 在最终公共工具冻结后，reviewer 尝试把所有初次 acquire 统一显式写出，并把 joint MMA 写成首项/稳态/末项。反向 `acquire(n) ← pv(n-2)` 会遇到 `pv(n-1)` 的名义窗口包含 -1 而真实动作不存在；改为等价正向 `acquire(n+2) ← pv(n)` 后 root 与 persistent 均通过。没有改变源码同步。

joint 的 matmul 改为多个复合索引声明后，反求 d/i/n/t 超出有限展开能力；统一初始/后续 `pvslot(t,i+1)` 又遇到 e 反求。最终保持连续 `kvstage(e)` 动作族，用较短的 e%4 条件区分 stage，初次 S/Y 获取用 `initialscore / initialy`，复用等待仍绑定 `scoreslot / pvslot`。LOAD 与 SOFTMAX 也保留初次自由槽 acquire。工具未修改；这些表达调整不增加真实等待、不删载荷、不伪造初始化。主图、N=1、empty split 和 padding 均完成。
