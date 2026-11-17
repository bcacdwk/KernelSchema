# Extract 审阅与精简

基线：`be4fc48`。完成四轮 MAX、三轮 high、一轮 xhigh，每轮两位新代理。

| 类别 | 主要改动 |
| --- | --- |
| 阶段与恢复 | 写作次序只约束首次准入；维护保留当前草稿和待审计划；派发前响应最新暂停/取消；确证未执行时保留有效会话与尚未交付的恢复材料 |
| 检查与来源 | 补齐早期元数据文本检查，区分 warning/error；补全冻结指南及导航；standalone 使用与记录一致的 Template 版本 |
| 审计与运行 | 失败回执可按真实状态归档；已接受回执仍须完整校验证据；后台启动早退可重试；停止后的迟到提醒保留线程和消息身份 |
| 文字对齐 | 核对全部有效 changelog；修正作者入口的独立 Feedback 与旧字段引用，明确回执、修稿和 issue 职责；正式字段规则未改 |
| 精简 | 删除无用状态、参数、包装及重复核验；删除普通分页 reader、重复 handoff 文件和单独的唤醒探测；同步删减提示与重复测试 |

需要同步的使用变化：

- **新提取使用不存在的输出目录**，取消 `.gitkeep` 占位目录接管；同一已发布结果仍可幂等重放
- **`inspect` 只读**，排队任务用 `resume` 启动，后台新任务使用 `extract --background`
- 移除 `monitor start --probe-delay`；正式提醒继续使用同线程确认、扫描和有限重试
- 普通源码浏览使用原生读取工具；精确 `--evidence`、阶段自检和回执中的短交接继续保留

两位 xhigh 均签收，各自九项定向离线用例通过；四个 Template 案例的九份 YAML 只读检查通过、零变化，九张 PNG 的来源身份匹配。最终未发现需要用户决定的遗留；没有运行真实模型全流程、GPU 或 Wiki 批量转换。

完整核查见 [Template / Docs](R4_A_CHANGELOG_TEMPLATE.md)、[Extract / 作者入口](R4_B_CHANGELOG_EXTRACT.md)；最终验收见 [流程与产物](R8_A_FINAL_FLOW.md)、[执行与控制](R8_B_FINAL_RUNTIME.md)。
