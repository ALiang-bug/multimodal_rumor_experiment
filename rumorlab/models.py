"""Offline model interface and explicit scripted responses."""

from dataclasses import dataclass
import json

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ModelResponse:
    final_content: str
    parsed_payload: object
    schema_valid: bool
    validation_errors: tuple
    usage: dict
    latency_ms: int
    finish_reason: str
    provider_request_id: str
    technical_error: object
    is_mock: bool = True


class MockModel:
    def __init__(self, responses):
        self.responses = responses

    def generate(self, request):
        if request.request_id not in self.responses:
            raise ValueError(f"Missing scripted mock response for {request.stage}/{request.task_id}")
        scripted = self.responses[request.request_id]
        if isinstance(scripted, TimeoutError):
            return ModelResponse("", None, False, (), {}, 0, "timeout", request.request_id, str(scripted))
        final = scripted if isinstance(scripted, str) else json.dumps(scripted, ensure_ascii=False)
        try:
            payload = json.loads(final)
        except json.JSONDecodeError as exc:
            return ModelResponse(final, None, False, (exc.msg,), {}, 0, "stop", request.request_id, None)
        errors = tuple(error.message for error in Draft202012Validator(request.response_schema).iter_errors(payload))
        return ModelResponse(final, payload, not errors, errors, {}, 0, "stop", request.request_id, None)
