# 多模态谣言检测实验开发包

本包是长期实验设计及开发合同。固定 Annotator 为 `Qwen/Qwen3.8-27B`，Reviewer 为 `google/gemma-4-31B-it`。Codex 应从合同与离线管线开始实现，再接入真实模型和数据。

## 阅读与开发入口

先读 `AGENTS.md`，随后完整阅读 `EXPERIMENT_SPEC.md`。后者是正式实验方案的完整 Markdown 版，Word 版由同一内容生成。

| 文件 | 用途 |
|---|---|
| EXPERIMENT_SPEC.md | 完整研究流程、任务、评测、开发阶段、长期维护和人工培训 |
| AGENTS.md | Codex 执行约束、阅读顺序、范围及交接规则 |
| configs/labels.yaml | 六任务、29 个类别条目、证据要求与边界 |
| configs/experiment.yaml | 数据、划分、人审、QA、审计、预算及发布配置 |
| configs/models.yaml | 固定模型、接口、采样初值及版本锁定 |
| schemas/records.schema.json | 记录和模型返回 payload 的 JSON Schema |
| prompts/ | 独立任务卡及标注、审核、QA 模板 |
| examples/ | 明确标识的合成测试夹具，不是研究数据 |
| docs/MODEL_SETUP.md | 模型官方接入依据和 preflight 要求 |
| docs/DATA_CONTRACT.md | 存储、哈希、依赖、人审导入和模型请求约定 |
| docs/ACCEPTANCE.md | 风险测试与阶段验收 |
| docs/PROGRESS.md | 开发进度入口 |
| docs/CHANGELOG.md | 版本与兼容性变化 |
| tools/validate_contract_examples.py | 仅检查设计包 schema 与示例的辅助脚本 |

## 可以直接交给 Codex 的启动任务

读取本目录 AGENTS.md 与 EXPERIMENT_SPEC.md，按 M0 实现 Python 包 rumorlab。先完成配置加载、记录 schema 与语义验证、generic_jsonl 导入、SQLite 任务状态、内容缓存、独立输入投影、mock 模型与核心风险测试。使用 examples 中的合成数据验证流程和失败路径。当前不要下载大模型或发起真实批量调用。把真实数据路径、模型 commit、端点与人工依赖记录到 docs/PROGRESS.md；这些依赖不阻止 M0 开发。M0 完成后给出可复核的命令和检查结果，再按文档推进 M1。

## 合同检查

在项目根目录使用具备 SQLite 和 Pillow 的 Python 环境复跑 M0。使用 CPython 3.9 复现已验证版本时，先安装 `requirements-m0-py39.txt` 中的依赖。

```bash
python -m pip install -r requirements-m0-py39.txt
python -m pip install --no-deps --no-build-isolation .
rumorlab preflight --mode mock
rumorlab validate-data
rumorlab ingest --fixture --run-id m0-local
rumorlab split --run-id m0-local
rumorlab mock-run --run-id m0-local --provenance
rumorlab mock-run --run-id m0-local --provenance
rumorlab export-snapshot --run-id m0-local > snapshot.jsonl
rumorlab validate-data --records snapshot.jsonl
python -m unittest discover -s tests -v
```

第二次 mock 运行应报告缓存命中。导出的合成快照应通过数据校验，并被正式发布检查拒绝。

开发环境准备好 `jsonschema` 和 `PyYAML` 后，在此目录执行：

```bash
python tools/validate_contract_examples.py --check-media
```

此命令只检查设计示例与合同。上面的 mock 复跑命令用于验证离线管线。

## 当前实现边界

M0 离线管线已完成。MMFakeBench 适配器可在持有原始标注 JSON 和已解压图片的机器上导入。两个真实模型尚待部署，目前无真实标注、人审或质量分数。配置中 `null` 字段表示部署或先导阶段需要填写的具体值。

示例记录和图片全部为软件开发用合成夹具，带 `is_mock=true`。发布门禁必须拒绝它们。格式正确不等于内容正确，人工质量验证按照正式方案另行执行。

## MMFakeBench 制作样本导入

制作样本使用官方 `val`（1,000 条，项目 split 为 `dev`）。官方 `test`（10,000 条）留作后续独立评测。在服务器的项目根目录执行，`--data-root` 指向已解压的 `MMFakeBench_val/` 的父目录。CLI 默认查找 `MMFakeBench_val.json` 和 `MMFakeBench_val/source/MMFakeBench_val.json`。

```bash
python -m rumorlab ingest --adapter mmfakebench --run-id mmfakebench-val \
  --data-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench
```

默认导入所选官方分区的全部唯一来源记录。需要小样本检查时，可以用 `--sample-size` 按 `gt_answers`、`fake_cls` 和实验种子做比例抽样。`True` 和 `Real` 归入同一个抽样层，记录保留原始标签值。整条原文作为初始目标陈述，OT/OI 表示文本和图片沿用来源。请使用原始 JSON，避免采用已改写文本或重命名 split 的 MAR 中间文件。[MMFakeBench 官方标注格式](https://github.com/liuxuannan/MMFakeBench#annotations)

导入输出包含实际样本数和各层计数。SQLite 的 `import_manifests` 表按 `run_id` 保存标注文件 SHA-256、种子、原始行号、来源与样本 ID，以及抽样过程中发现的隔离项。坏图在原层按排序顺位补足，配额不足会中止导入。相同输入和 `run-id` 重跑返回 `inserted=false`。抽样和导入只占用 CPU。来源和派生数据的再分发权限按下载协议标为 `restricted`。[MMFakeBench 数据使用协议](https://huggingface.co/datasets/liuxuannan/MMFakeBench/blob/main/README.md)

本模块可独立验证。

```bash
python -m unittest discover -s tests -p 'test_mmfakebench.py' -v
python -m unittest discover -s tests -p 'test_ingest.py' -v
```
