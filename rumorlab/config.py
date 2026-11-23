"""Load the project contract relative to the supplied experiment file."""

import json
from pathlib import Path

import jsonschema
import yaml

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
        self.experiment = load_yaml(path)
        self.labels = load_yaml(self.root / self.experiment["label_config"])
        self.models = load_yaml(self.root / self.experiment["model_config"])
        self.schema = json.loads((self.root / self.experiment["record_schema"]).read_text(encoding="utf-8"))
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

    def validator(self, definition):
        return jsonschema.Draft202012Validator(
            {"$schema": self.schema["$schema"], "$defs": self.schema["$defs"],
             "$ref": f"#/$defs/{definition}"},
            format_checker=jsonschema.FormatChecker(),
        )
