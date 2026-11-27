"""Load the project contract relative to the supplied experiment file."""

import json
from pathlib import Path

import jsonschema
import yaml

from . import prompts_v1_2, prompts_v1_3
from .validation import canonical_hash


def load_yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


class Contract:
    def __init__(self, path):
        path = Path(path).resolve()
        self.root = path.parent.parent
        overrides = load_yaml(path)
        base = overrides.pop("base_experiment", None)
        self.experiment = load_yaml(self.root / base) if base else {}
        for key, value in overrides.items():
            if key in {"annotation", "qa"} and base:
                self.experiment[key] = {**self.experiment[key], **value}
            else:
                self.experiment[key] = value
        self.labels = load_yaml(self.root / self.experiment["label_config"])
        self.models = load_yaml(self.root / self.experiment["model_config"])
        self.schema = json.loads((self.root / self.experiment["record_schema"]).read_text(encoding="utf-8"))
        self.label_prompt_version = self.experiment["annotation"]["prompt_version"]
        self.single_label = self.label_prompt_version in {prompts_v1_2.VERSION, prompts_v1_3.VERSION}
        if self.single_label:
            self._apply_single_label_contract()
        jsonschema.Draft202012Validator.check_schema(self.schema)
        self.gold_votes = self.experiment["human"]["gold_independent_votes"]
        if type(self.gold_votes) is not int or self.gold_votes < 2:
            raise ValueError("human.gold_independent_votes must be an integer >= 2")
        tasks = self.experiment["annotation"]["tasks"]
        if len(tasks) != len(set(tasks)) or set(tasks) != set(self.labels["tasks"]):
            raise ValueError("Annotation tasks differ from label registry")
        ids = {role: config["model_id"] for role, config in self.models["models"].items()}
        if ids != {"annotator": "Qwen/Qwen3.8-27B", "reviewer": "google/gemma-4-31B-it"}:
            raise ValueError("Model configuration differs from the fixed research roles")
        self.config_hash = canonical_hash([self.experiment, self.labels, self.models, self.schema])

    def _apply_single_label_contract(self):
        tasks = self.labels["tasks"]
        if set(self.experiment["annotation"]["model_tasks"]) != set(prompts_v1_2.TASKS):
            raise ValueError("Single-label task guides differ from configured model tasks")
        self.labels["label_schema_version"] = "2.0.0"
        rules = self.labels["global_rules"]
        del rules["unknown"]
        rules["applicability"]["rule"] = "有效输入必须选择一个业务标签。必要输入异常由执行流程记录。"
        rules["system_failures"]["rule"] = "读取、请求或解析失败须按执行流程处理，不生成标注。"
        rules["overlap"]["rule"] = "选择最符合当前任务的一项，写明关键取舍，无法可靠解决的实质分歧进入人审。"
        rules["evidence"]["rule"] = "区分可见线索、任务允许的世界知识及推断。正式证据 ID 必须来自本次输入。"
        for task_id, guide in prompts_v1_2.TASKS.items():
            task = tasks[task_id]
            local_names = {item["code"]: item["name_zh"] for item in task["labels"]}
            task.update(cardinality="single", decision_guide=guide["guide"], labels=[
                {"code": code, "name_zh": local_names.get(code, "语境合理"),
                 "name_en": name, "definition": definition}
                for code, (name, definition) in guide["labels"].items()
            ])

        definitions = self.schema["$defs"]
        annotation = definitions["annotation_payload"]
        # Keep the record envelope and required fields compatible with existing consumers.
        annotation["properties"].update(applicability={"const": "applicable"}, unknown_reason={"type": "null"})
        for name, field in (("annotation_payload", "labels"), ("qa_payload", "target_labels")):
            rules = definitions[name]["allOf"]
            retained = []
            for rule in rules:
                properties = rule["if"]["properties"]
                task_id = properties.get("task_id", {}).get("const")
                if task_id is None:
                    if name == "qa_payload" and field not in properties:
                        retained.append(rule)
                    continue
                task_properties = rule["then"]["properties"]
                task_properties[field].update(
                    items={"enum": [item["code"] for item in tasks[task_id]["labels"]]},
                    minItems=1, maxItems=1,
                )
                if name == "qa_payload" and task_id == "persuasion_strategy":
                    task_properties["question_type"] = {"const": "single_choice"}
                retained.append(rule)
            definitions[name]["allOf"] = retained

    def validator(self, definition):
        return jsonschema.Draft202012Validator(
            {"$schema": self.schema["$schema"], "$defs": self.schema["$defs"],
             "$ref": f"#/$defs/{definition}"},
            format_checker=jsonschema.FormatChecker(),
        )
