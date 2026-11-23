# 数据与运行合同

本文件补充 EXPERIMENT_SPEC.md。schema 使用 JSON Schema Draft 2020-12；模型只返回对应 payload，程序负责记录 ID、时间、状态、来源、质量等级和输入版本。

## 记录与物理文件

推荐逐阶段保存 `samples.jsonl`、`evidence.jsonl`、`annotations.jsonl`、`reviews.jsonl`、`human_votes.jsonl`、`adjudications.jsonl`、`qa.jsonl`、`audit_selections.jsonl`。原始材料、语义 payload 与审计事件不可变。状态与质量等级属于可重建投影；JSONL 是冻结时点的快照，每个语义 ID 仅一行，旧快照不回写。大量数据可分 shard 并保留每片 SHA-256。

SQLite 建议包含 records、jobs、attempts、dependencies、audit_memberships 和 imports 六张表。records 的主键为 record_id，记录 type、content_hash 和 JSON 路径；jobs 对 stage、subject_id 和 input_hash 设唯一约束；attempts 保留请求摘要、finish_reason、重试、耗时和成本；dependencies 保存父记录 ID 与被依赖内容哈希；imports 按文件哈希和提交批次幂等；audit_memberships 保存冻结总体与纳入概率。

记录提交、工作项状态、追加事件和依赖写入须在同一事务中完成，避免恢复时只有 PASS 没有对应输出。候选通过审核后，annotation_id 和 payload 不变，更新状态投影；review 保持引用该语义 ID 与哈希。语义被人工修正才创建新 ID，以 supersedes_id 关联旧候选。日志文件写入使用临时文件后原子 rename；原始模型请求和返回值存内部运行目录，不混入公开数据。

## 哈希与版本

文本用 UTF-8，规范化 JSON 使用排序键、固定紧凑分隔符和不转义 Unicode；统一明确 Unicode 正规化策略但保留源文本。图片 SHA-256 基于原始字节。模型输入哈希基于实际送入的文本、图片与证据派生表示，并包含处理参数；不能只哈希文件名。

annotation、review 和 qa 的 content_hash 是规范化 payload JSON 的 SHA-256；input_hash 另覆盖实际允许输入及执行配置。sample 和 evidence 使用包含正文、媒体或证据内容的语义字段集合，排除工作状态、质量投影、时间戳和 content_hash 自身，并在实现中固定该集合及版本。任何语义内容变化都生成新记录；变更后通过依赖图失效审核与 QA。Prompt、模型或视觉处理变化产生新 job key，不回收旧结果充当新实验。发布索引固定所使用的记录版本，历史可重建。

## 模型请求

`ModelRequest` 至少含 role、stage、task_id、visible_inputs、rendered_prompt_hash、response_schema、model_config_hash 和 request_id。`ModelResponse` 至少含 final_content、parsed_payload、schema_valid、validation_errors、usage、latency_ms、finish_reason、provider_request_id 及 technical_error。

允许列表按 stage 定义。标注看原始允许输入和证据；candidate_aware Reviewer 额外看被审候选；QA 生成额外看已接受标签；solver 只看题目、选项和相应图文证据；独立人工标注看同样规定材料与指南。禁止把数据对象整体序列化后直接塞进 Prompt。

## 状态与引用校验

每条有效标注的 task_id 决定合法 labels。单标签恰有一个标签，多标签至少一个且去重；not_applicable 为空列表。UNKNOWN 不和实质标签共存。对 manifest 来源任务，模型预测不得覆盖来源属性。target_claim 为包含 claim_id、text、text_span 的对象；text_span 使用原始规范化文本的 Python Unicode 码点半开区间 [start,end)，不使用 UTF-8 字节或前端 UTF-16 单元偏移。

每个 evidence_id 必须存在并在当次输入白名单内。每条审核绑定被审 ID 和内容哈希；每条 QA 绑定上游 annotation ID、内容哈希、sample ID 和 solver_input 哈希。上下游外键、组与 split 均须一致。

gold 只能由完成资格检查的人工记录派生：独立首轮、规定人数、明确冲突处理和证据。qa 的 gold 还需题目本身的审核引用。人审展示过模型答案的普通修改可以保留，但不得无条件授予独立 gold。

## 人工导入导出

导出提供任务 ID、记录 ID、当前内容哈希、手册版本、中性媒体引用、可见材料和空白答案字段。CSV 的多标签用 JSON 数组字符串表示，避免逗号和中文分号的隐式拆分；JSONL 是无损导入格式。

首轮盲标文件不携带模型结果、Reviewer 意见、原始真假标签或其他标注员答案。文件头或伴随 manifest 标明审计批次、层和样本选择概率，但不把风险来源展示为答案提示。仲裁导出在独立标注完成后生成。

导入检查原始记录版本、匿名人员 ID、授权的轮次、合法标签、引用证据和重复提交。旧版本答案进入 conflict 队列，不覆盖新版本。空白答案是未完成，不转 UNKNOWN；解析失败是导入错误，不创造标注。原始人审票不删除，仲裁另存。

## 评测数据投影

`solver_input` 仅含 sample/QA 中性标识、问题、洗牌后展示选项、可见文本、媒体与允许证据。`answer_key` 在评分进程持有，包含稳定 option_id 答案集合、质量记录和上游标签。基准推理进程不读取 answer_key。

单选无效输出和超时记错并独立计数；多选先还原稳定选项 ID 后作集合比较。对扰动或去模态条件，保持题目 ID 及基础样本一致，记录故意省略的模态。缺失输入诊断不能被当作正常完整输入成绩。

## 数据量与统计单位

运行与质量报告同时列出原始来源样本数、独立事件/谱系组数、图像数、变体数、任务标注数和 QA 数。按标注项抽样的误差估计不自动代表按事件的总体误差；统计方法须与 population_unit 一致，置信区间以组处理依赖。

## 包内版本一致性

`schemas/records.schema.json` 对字段结构负责，labels.yaml 对类目定义负责，实验配置对运行参数负责，本文及主规范对跨记录语义负责。合同检查脚本仅覆盖可由小型夹具验证的部分；本体语义、人工培训和真实质量必须在 M2 至 M4 评估。
