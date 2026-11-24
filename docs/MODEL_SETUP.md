# 固定模型与部署接口

核实日期：2026-11-22。本文与 `configs/models.yaml` 共同定义模型接入约定；配置本身不是已经部署成功的服务，也不是已完成的实验结果。

## 两个固定角色

| 角色 | 精确模型 ID | 项目职责 |
| --- | --- | --- |
| Annotator | `Qwen/Qwen3.8-27B` | 独立任务 Prompt 下生成候选标签、可核对的简短依据，以及经标签审核后生成 QA |
| Reviewer | `google/gemma-4-31B-it` | 审核候选标签和 QA，输出结构化问题与路由建议 |

两个官方模型卡均列出公开权重、Apache 2.0 和图文输入能力。两者在本项目中的输出都是文本；它们不能直接生成或编辑 FI 图片像素。FI 的产生属于样本准备中的可追溯图像编辑操作或受控素材导入，不能通过让 Annotator 返回图片描述来假装已经完成图像编辑。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)；[Gemma 模型卡](https://huggingface.co/google/gemma-4-31B-it)。

Gemma 必须使用 `-it` 指令版本。`google/gemma-4-31B` 是不同用途的基座权重，不得静默替换。[Google 部署文档](https://gemma.ai/deployment/vllm/)。

## 客户端和版本锁定

项目以本地或自有服务器的 OpenAI-compatible Chat Completions 协议为首选统一接口，分别实现 Qwen 与 Gemma adapter。服务端可以使用 vLLM，调试后端可另行实现 Transformers；协议兼容并不表示调用闭源服务。正式批处理可以先完成 Annotator 阶段，再加载 Reviewer，模型不必同时常驻。

从 `ANNOTATOR_BASE_URL`、`REVIEWER_BASE_URL` 读取端点，从对应 `*_API_TOKEN` 环境变量读取凭证。凭证不得写入 YAML、日志或运行清单。

`revision`、`processor_revision`、`backend_version` 初始为 `null`。开发可以使用 mock 跑通解析、状态机、审核导出和评测；任何真实推理调用前必须补齐并生成 `model_lock.json`。正式运行必须 fail-fast 拒绝空版本、`main`、`latest` 等未锁定标识。模型权重、processor、tokenizer 和 chat template 应与同一已验证组合一致；服务端必须提供或由部署清单证明实际载入版本，不能仅凭 `/models` 返回一个名字就认定版本正确。

记录实际硬件、CUDA/驱动、后端构建版本、依赖锁文件摘要、模型 revision、模板摘要、精度和视觉处理配置。BF16 是初始实验建议；FP8/INT4 等量化属于独立 run，重新验证质量和吞吐，不得覆盖 BF16 结果。本文不承诺任何未经本地测量的显存需求。

部署时查阅 [Qwen 官方所链 vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) 与 [Google vLLM 文档](https://gemma.ai/deployment/vllm/)。Qwen recipe 的验证说明主要针对文本 serving，不能代替本项目的真实图片测试。

## Thinking 与最终输出

两者均显式设 `enable_thinking=true`，避免依赖服务默认值；vLLM 对应 reasoning parser 分别为 `qwen3` 和 `gemma4`。每个样本与每个任务均开启新会话，严禁复用其他维度预测或其他样本的会话历史。Qwen 的 `preserve_thinking=false` 仅是显式适配设置，不能代替新建会话。

模型卡给出的 thinking 初始采样设置为 Qwen `temperature=1.0, top_p=0.95, top_k=20`，Gemma `temperature=1.0, top_p=0.95, top_k=64`。配置中的输出预算、超时、重试次数、BF16 与 seed 是项目初始建议，须在开发集测量后冻结。seed 有助于审计，不能保证跨硬件和后端的逐位一致性。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)；[Gemma 模型卡](https://huggingface.co/google/gemma-4-31B-it)。

标签与 QA 只能从解析后的最终 `content` 中读取。单独的 reasoning/thinking 不是公开解释或事实证据，不进入公开数据集。需要的解释通过最终 JSON 内简短 `rationale` 与可核对 `evidence_refs` 字段提供。空最终答复、截断、未闭合推理输出和解析失败进入失败与有限重试流程，不能变成 `PASS`。

## 结构化输出与检查

首选服务端 `response_format` 的 JSON Schema 约束，同时始终进行本地 Schema 和语义校验。Prompt 必须完整说明字段语义与标签规则；约束解码只保证结构，不能保证标签正确、证据存在或 Reviewer 判断可靠。Gemma 的 vLLM 文档包含 thinking 与结构化输出联合示例；实际版本仍必须完成 smoke test。[vLLM 结构化输出](https://docs.vllm.ai/en/latest/features/structured_outputs/)；[Gemma vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)。

客户端不得静默丢弃不支持的配置参数或改用另一模型。任何经过明确配置的后端降级都要写入运行清单；格式约束不可用时，仍须通过最终 JSON 的严格解析和本地校验才能进入审核。

## Preflight 验收

1. 核对精确模型 ID、权重和 processor revision、实际后端版本，保存模型卡和许可证来源。
2. 验证文本、单图及数据集所需的最大图片数量。相同文本替换为明显不同图片，确认视觉输入确实生效。
3. 确认旋转方向、宽高比例、缩放与视觉 token 预算，记录处理后的图片信息和配置。
4. 验证 thinking 开关、parser、最终 JSON 分离，以及枚举、证据引用、跨字段关系检查。
5. 注入超时、空答复、截断和非法 JSON，确认有限重试、错误记录和人工路由行为。
6. 冻结模型与依赖清单并完成最小真实批次。任何模型、Prompt、后端、精度或视觉预算变化都创建新 run。

官方能力证明可以接入图文工作流；Reviewer 的实际检错能力仍由本项目的人工参考集衡量，不把模型之间的一致或不同厂商来源视为质量保证。

## A800 终端命令

在云服务器的 `MMFND` 下运行实验。`MMFND/models` 集中存放模型权重。本实验的两套权重以 Hugging Face 缓存快照的形式存放在这里，便于锁定精确 revision。两个 Bash 终端使用同一个 vLLM Python 环境。Qwen recipe 当前列出 vLLM 0.17.0 或更新版本，并要求与模型配置兼容的 Transformers。[Qwen vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

```bash
export MMFND=/data/home/202422011133/MMFND
export MODELS_DIR="$MMFND/models"
cd "$MMFND/multimodal_rumor_experiment"
nvidia-smi
python -c "import vllm, transformers, sqlite3; print(vllm.__version__, transformers.__version__, sqlite3.sqlite_version)"
python -m pip install -e .
mkdir -p runs/m1-qwen runs/m1-gemma
```

已经缓存的模型直接填写各自的 40 位提交哈希。需要首次下载时，运行下面的命令查询官方仓库当前提交，并按该提交下载到 `MMFND/models`。记下输出的两个哈希，部署和复跑使用相同值。[Hugging Face CLI 下载命令](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-download)

```bash
QWEN_REVISION=$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen3.8-27B").sha)')
GEMMA_REVISION=$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("google/gemma-4-31B-it").sha)')
printf 'Qwen  %s\nGemma %s\n' "$QWEN_REVISION" "$GEMMA_REVISION"
hf download Qwen/Qwen3.8-27B --revision "${QWEN_REVISION:?}" --cache-dir "$MODELS_DIR"
hf download google/gemma-4-31B-it --revision "${GEMMA_REVISION:?}" --cache-dir "$MODELS_DIR"
```

在两个终端设置同样的 `MODELS_DIR`、`QWEN_REVISION` 和 `GEMMA_REVISION`，由缓存路径定位快照。若终端已经执行上面的下载步骤，可直接执行路径导出。

```bash
export QWEN_SNAPSHOT="$MODELS_DIR/models--Qwen--Qwen3.8-27B/snapshots/${QWEN_REVISION:?}"
export GEMMA_SNAPSHOT="$MODELS_DIR/models--google--gemma-4-31B-it/snapshots/${GEMMA_REVISION:?}"
```

终端 A 先启动 Annotator。单卡参数用于初步测量。选用分配给你的 GPU。两个模型分开加载。

```bash
vllm serve "$QWEN_SNAPSHOT" --served-model-name Qwen/Qwen3.8-27B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 12288 --max-num-seqs 1 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":2}' --reasoning-parser qwen3 \
  2>&1 | tee runs/m1-qwen/server.log
```

服务就绪后在终端 B 执行探针并保存结果。报告的 `passed` 为 true 时，回到 A 按 Ctrl+C 结束当前服务，再启动 Reviewer。

```bash
export ANNOTATOR_BASE_URL=http://127.0.0.1:8000/v1
set -o pipefail
python -m rumorlab preflight --mode real --role annotator \
  --snapshot "$QWEN_SNAPSHOT" --model-lock runs/m1-qwen/model_lock.json \
  | tee runs/m1-qwen/smoke.json
```

终端 A 启动 Reviewer。官方 recipe 对 31B BF16 给出双卡实例，下面的单卡、短上下文参数是本项目待验证的起点。[Gemma vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html#multi-gpu-deployment)

```bash
vllm serve "$GEMMA_SNAPSHOT" --served-model-name google/gemma-4-31B-it \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 12288 --max-num-seqs 1 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":2}' --reasoning-parser gemma4 \
  2>&1 | tee runs/m1-gemma/server.log
```

终端 B 运行 Reviewer 探针，然后导入官方 val 数据。按服务器实际位置修正数据根目录。导入仅使用已有原始标注和解压图片。

```bash
export REVIEWER_BASE_URL=http://127.0.0.1:8000/v1
python -m rumorlab preflight --mode real --role reviewer \
  --snapshot "$GEMMA_SNAPSHOT" --model-lock runs/m1-gemma/model_lock.json \
  | tee runs/m1-gemma/smoke.json

python -m rumorlab ingest --adapter mmfakebench --run-id mmfakebench-val \
  --data-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench
```

探针检查纯文本、两次同文异图、双图交换顺序、最终 JSON 和独立 reasoning 字段，报告耗时和 token 用量。记录的 processor 配置来自本地快照，旋转方向和实际视觉 token 预算仍需实测。两个颜色探针报告只能作为 M1 接口结果，真实标注和人工质量评估属于后续步骤。

## M2 标注与审核试跑

在项目目录、已激活的 `rumorlab-m1` 环境中运行。两个阶段使用同一已导入的官方 val 批次、同一 Git 提交与同一 SQLite 状态库。以已导入的 12 条分层样本为例，先启动 Qwen，再执行：

```bash
ANNOTATOR_BASE_URL=http://127.0.0.1:8000/v1 \
  python -m rumorlab annotate --run-id mmfakebench-val-import-smoke-12 \
    --batch-id m2-pilot-01 --limit 12 \
    --annotator-lock runs/m1-qwen/model_lock.json \
    --reviewer-lock runs/m1-gemma/model_lock.json
```

标注对每个目标生成四个独立模型任务的候选，另从制作来源记录生成 OT/FT 与 OI/FI。成功任务在同一批次重跑时复用。报告中的 `failed` 为失败任务数，`SKIPPED` 表示按规则不适用。运行清单冻结模型锁摘要、初始数据、样本选择、划分与代码提交。

停止 Qwen，启动 Gemma 后执行：

```bash
REVIEWER_BASE_URL=http://127.0.0.1:8000/v1 \
  python -m rumorlab review-labels --run-id mmfakebench-val-import-smoke-12 \
    --batch-id m2-pilot-01 --reviewer-lock runs/m1-gemma/model_lock.json
```

Reviewer 只读取同一批次持久化且通过验证的模型候选。PASS 对原候选投影 silver，NEEDS_HUMAN 和 REJECT 分别投影待人工和排除。更改代码、锁或输入后使用新的 `--batch-id`。两阶段均只接受 MMFakeBench val。vLLM 解码使用结构和任务枚举约束，完整的条件式 Schema 保留在 Prompt 和本地校验中。

用 `python -m rumorlab export-snapshot --run-id mmfakebench-val-import-smoke-12` 导出源样本及该导入批次的当前标注与审核记录。
