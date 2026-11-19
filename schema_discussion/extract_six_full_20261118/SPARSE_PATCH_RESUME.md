# Sparse 维护 patch 公开恢复

2026-11-18 16:26:01 UTC 已确认恢复启动，随后停止恢复操作，不等待模型结束。

- Run：`8f0844f9-f279-4781-a19d-0216b7fc4ef5`；维护请求：`5677bbf8-286e-4169-b91e-59aa0f8e6445`。
- 恢复前为外层 `blocked/P4`、维护 `P1/patch/invalid_receipt`，拒绝原因 `missing bound implementation entry anchor`。已复核三个在途／未结清集合为空，所有 usage 已结清；原失败 attempt 的 native success、exit 0、termination confirmed、survivors=[]，旧 Runner 已不存在；active_hold=null、旧 launch finished、control 为旧 revision 4。P1 原计数 `4/32`。
- 本批第一次公开 whole-run resume（16:24:32 UTC，PID `1930244`）返回 `busy/committed=false`，进程退出且没有新增调用；第二次（16:25:17 UTC）提交成功，control revision 变为 `5`。未执行第三次重试。
- 新 active attempt：`e0779a2d-4364-446b-8f95-75cf04ad4667`；`purpose=maintenance`、`maintenance_step=patch`、`stage=P1`、`role=worker`，绑定原任务 `5677bbf8-286e-4169-b91e-59aa0f8e6445:P1`。
- 仍使用原 session `9728d37f-bac0-44c1-aff7-cf785a5ef028`，session 状态 `running`。P1 计数现在为 `5/32`，`call_additions=[]`；外层游标保持 `P4`，state 状态为 `running`。
- Runner PID `1930623` 与 attempt PID `1930749` 均存活；Runner 命令行运行该 run 原 snapshot 的公开入口。原 review 历史和 draft checkpoint `2c738fe37d10ec90906003f633718a39528d2913f4c5395f0944ea24e97187f8` 保留。

公开命令：

```bash
python3 -B /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume 8f0844f9-f279-4781-a19d-0216b7fc4ef5 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

启动日志：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runtime/supervisor-resume/8f0844f9-f279-4781-a19d-0216b7fc4ef5-20261118T162517Z.log`。两次命令及结果已追加到 `RECOVERIES.jsonl`。

未手写 state、回执或产物，未操作锁或其他进程，未增加额度、热更新冻结执行器、修改源码或启动新 run；未提交或推送。
