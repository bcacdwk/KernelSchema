# 巡检 84

采样与完成时间为 2026-11-20 15:47:33 北京时间（UTC+8）；`clock.committed=true`，下次到期为 16:17:33。

新增 KDA、BF16 FMHA、分页索引 Sparse 三项交付，累计 8/12。KDA 的 scan 摘要标为 stale，已先通过 `status --full` 核实权威状态为 ready/published、无在途/未结调用，checkpoint 为 d398e8618598d6d4bbaeffb419cb065530361c78f2fc5e091ae76376af38f647，未将过期摘要直接当作当前事实。

三项均核验全部调用结清、回执等于发布证明、run/checkpoint 一致，并逐一验证正式 YAML 与 dispatch 的 SHA。证据见 [KDA](../deliveries/kda_raw_prefill_state_bf16.json)、[BF16 FMHA](../deliveries/fmha_bf16.json)、[分页索引 Sparse](../deliveries/paged_indexed_latent_fp8.json)。

其余 3 项 P6、1 项 P5，共 4 个在途调用。FP8 FMHA 的 P6 回执引用末行 388，而当前稿只有 387 行；原 reviewer 正在自动补正，2/32 次。[定向 inspect](check-9-84-inspect.json) 确认进程存活，无控制屏障或未确认终止，本次不额外恢复或增额。
