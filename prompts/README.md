# Prompt 合同

`--config configs/experiment-single-label-v1.3.yaml` 为 `annotate` 和 `review-labels` 选用英文 Prompt 1.3.0。四张任务卡分别定义预判检测与最终判据，两个角色共享任务卡并填写各自的输出字段。模型响应仅含本阶段需要的字段，程序保存时补齐旧记录契约要求的固定值。使用单独批次 ID 和输出目录保存新版结果。

`--config configs/experiment-single-label.yaml` 启用英文 Prompt 1.2.0 和标签规范 2.0.0。四个模型维度使用 `rumorlab/prompts_v1_2.py` 的共享证据规范与任务卡，均为单选。现实语境加入 `PLAUSIBLE`，四个模型维度移除 `UNKNOWN`。下文的 1.1.2 基线由 `configs/experiment.yaml` 加载。

标签标注与审核、QA 生成与审核的 Prompt 版本均为 1.1.2。系统消息根据当前阶段注入 Annotator 或 Reviewer 的角色和职责。用户消息末尾包含按阶段和任务渲染的完整响应 Schema。模型指令、任务卡及模型撰写的解释和 QA 使用英文，原始陈述、引文和字符位置按来源保留。模板使用 StrictUndefined；变量缺失即报错。每条请求新建会话。

## 组成与响应

`system.txt` 放入系统消息。常规标注用户消息由 `annotate.j2` 与对应 `tasks/*.txt` 任务卡组成。任务卡各自独立，不拼接其他任务卡或预测结果。共享结构模板不改变任务级独立性。

| 阶段 | 模板 | 响应定义 |
|---|---|---|
| annotate | annotate.j2 加单一任务卡 | annotation_payload |
| verify_manipulation | verify_manipulation.j2 加 text_reasoning 或 image_perception 卡 | review_payload |
| review_labels | review_labels.j2 加单一任务卡 | review_payload |
| generate_qa | generate_qa.j2 加单一任务卡 | qa_payload 或带 skip_reason 的跳过响应 |
| review_qa | review_qa.j2 加单一任务卡 | review_payload |

text_reasoning 和 image_perception 的 OT/FT、OI/FI 由 manifest 产生 annotation 记录，不调用模型猜制作类型；它们的模型 Prompt 仅用于准备阶段的目标达成核验及可观察比较题。任何核验未通过都不能发布为成功受控篡改。

## 渲染变量

统一使用 `task_id`、`task_definition_json`、`task_instructions`、`target_claim_json`、`visible_input_json`、`response_schema_json`。它们均由白名单投影或合法配置生成。review_labels 另有 `candidate_annotation_json`；generate_qa 另有 `accepted_annotation_json`、`qa_policy_json`；review_qa 另有 `candidate_qa_json`、`solver_input_json`；verify_manipulation 另有 `operation_target_json` 和 `parent_input_json`。

图片以模型多模态消息的 image 内容项实际附加，JSON 中的 asset ID 只做映射，不能只传本地路径然后假定模型看到了图。figure/asset ID 与实际图片排序须可核验。

标签标注与审核的请求 Schema 将 `evidence_ids` 限制为本次可见证据列表中的正式 ID，列表为空时返回 `[]`。图片编号和文字细节可以写进 `rationale`。标注及审核建议的 `target_claim` 由输入值约束。

`response_schema_json` 必须是从完整 schema 解析 `$ref` 后、针对当前 task_id 收紧标签枚举的 payload schema。推理后再次进行本地 schema 与语义校验。模板禁止要求模型填写数据库状态、人工资格或质量等级。

## 维护

将完整渲染文本、标签定义、task card 和 schema 的哈希写入 run_manifest。修改模板即新 Prompt 版本；先用 pilot_dev 比较，再冻结。分组 Prompt、逐标签策略和 blind_first 为独立配置实验，不能改写本目录默认行为。
