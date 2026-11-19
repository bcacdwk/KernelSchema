# MoE 公开 resume 短争用诊断与启动记录

2026-11-18 13:00–13:02 UTC，run `74bcac91-808a-41b2-aab4-85143935ab40`。

结论：此前三次公开 resume 的日志均为 `busy / committed=false`，没有提交启动，也没有新增模型调用。现场证据支持本批旧冻结 Runner 的共享短争用，未发现持续独占或新的重入问题。本次仅再执行一次原公开命令，已真正启动新 P3 worker。

恢复前复核：state revision 52，`invalid_receipt / P3 / worker`，失败 attempt `1bba285e-6616-4333-a4a0-da4f0b1419a2`，原因 `missing bound implementation entry anchor`。active、unsettled、unconfirmed 均空；P3 calls=1，max_calls=32，call_additions 为空。旧 Runner 与已记录的全部调用后代均无匹配的存活身份，MoE runner.lock 可非阻塞获取。失败调用 native success、exit=0、terminal outcome=completed、termination confirmed、survivors 为空；终止记录与结果中的 attempt canonical digest 一致，native/result/state/session 精确绑定 `373f4d3d-a750-4bbe-addf-c8a01c952229`，session ready。

持锁证据：100 次即时释放的非阻塞探测（约 3.46 秒）有 61 次 free、39 次 busy，连续 busy 最长仅 3 个采样点。另一次 100 次 fd 观察中，supervisor.lock 的打开 fd 在其他五个活跃 Runner PID `1744019`、`1744734`、`1757787`、`1776561`、`1776586` 之间切换；busy 与这些打开 fd 的关联同样轮换。fd 观察只是相关性证据，不单独证明某个 PID 此刻持锁；`/proc/locks` 和 fdinfo 未提供该工作区锁的可用持有者记录，因此不以其空结果断言锁自由。实际 flock 探测的 free/busy 交替，以及本次原命令直接成功，排除了现场持续占锁的表现。历史三次未同步采样，不能逐次追认唯一持锁 PID。

13:00:56.661445 UTC，通过 detached Popen 启动公开 whole-run 命令（`-B` 仅禁写字节码）：

```sh
/opt/venv/python3.13.12/bin/python3 -B /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume 74bcac91-808a-41b2-aab4-85143935ab40 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

启动后已见 state revision 57、`running / P3 / worker`，新 active attempt `82efa9ab-b36b-4b8c-92dc-5c26e5f189e2`，P3 calls=2、attempt count=9、call_additions 仍为空。Runner PID `1805019` 存活且 runner.lock held，实际入口仍为原 snapshot 的冻结脚本；supervisor launch `841f6815-e719-4948-ba9a-d9a8ff0d1ced` 已 claimed。调用 gate PID `1805074`、launch `7a6047cc-e9af-4e69-8600-666858858c1b` 已 released，invocation 已登记 `qwen3.8-max[1m] / max` 和原 resume_session_id。

日志：`.schema_extract/v2/runtime/supervisor-resume/74bcac91-808a-41b2-aab4-85143935ab40-20261118T130056Z.log`。请求与核验结果仅追加于本诊断文档同目录的 `RECOVERIES.jsonl`。确认真实调用启动后停止恢复操作，未等待该模型完成；未手工改源码、冻结内容、状态、产物或回执，未调整校验、额度或其他任务。
