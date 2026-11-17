# 有界局部域的 last 误判

稳定标识：`dynamic-worktile-dependent-local-last`。

复现：从仓库根目录运行 `/opt/venv/python3.13.12/bin/python3 -B schema_discussion/template_examples_20261117/qsa_prefill/last_local_domain_repro.py`。实际输出见 [last-local-domain.log](last-local-domain.log)：`last(i)` 报动态 worktile 域不可查询，等价 `tile_count(t)-1` 得到三个动作、无环。

最小结构是动态 `t: worktile`、有界 `i: [0,tile_count(t)]`，在已绑定 t 的工作内等待 `compute:x(t,last(i))`。i 的长度由当前已确认工作给定，查询没有获取动态 t 的总长。QSA 普通检查也对 `n[t,last(i)]` 报同样错误。

依据：[DOC5 索引](../../../schema_docs/5.design.tile_pipeline.md#索引) 仅禁止对动态 worktile 使用 count/last；[连续序号与索引窗口](../../../schema_docs/5.design.tile_pipeline.md#依赖的索引窗口) 允许当前工作窗口与局部域末项。`rules/pipeline.py:_dynamic` 及 `pipeline/expressions.py:Domains.has_worktiles` 把“依赖工作坐标”与“需要整个动态工作序列”混在一起。

当前 root 使用已定义的 `tile_count(t)-1`，语义、等待和源码方案未改变。建议公共工具区分这两类依赖，保留当前 t 绑定检查；本案例未修改工具。
