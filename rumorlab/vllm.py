"""Chat Completions transport shared by the two fixed vLLM roles."""

from base64 import b64encode
import json
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from jsonschema import Draft202012Validator

from .models import ModelResponse


def _label_decoding_schema(schema, task_id):
    # Keep conditional cross-field rules in the prompt and strict local validation.
    result = {key: value for key, value in schema.items() if key not in {"allOf", "uniqueItems"}}
    if "pattern" in result and ("minLength" in result or "maxLength" in result):
        del result["pattern"]
    if "properties" in schema:
        properties = dict(schema["properties"])
        for rule in schema.get("allOf", ()):
            if rule.get("if", {}).get("properties", {}).get("task_id", {}).get("const") == task_id:
                properties.update(rule["then"]["properties"])
        result["properties"] = {key: _label_decoding_schema(value, task_id) for key, value in properties.items()}
    if "items" in schema:
        result["items"] = _label_decoding_schema(schema["items"], task_id)
    if "anyOf" in schema:
        result["anyOf"] = [_label_decoding_schema(value, task_id) for value in schema["anyOf"]]
    return result


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class VllmModel:
    def __init__(self, settings, base_url, token, timeout):
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment or url.path.rstrip("/") != "/v1"):
            raise ValueError("vLLM base URL must be an HTTP(S) URL ending in /v1")
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.opener = build_opener(NoRedirects)

    def _request(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(self.base_url + path, data=data, headers=headers)
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def verify_identity(self):
        try:
            models = self._request("/models")["data"]
            if not any(row["id"] == self.settings["model_id"] for row in models):
                raise ValueError("vLLM endpoint does not serve the configured model ID")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not verify vLLM model identity ({type(exc).__name__})") from exc

    def generate(self, request):
        response, _ = self.complete(request)
        return response

    def complete(self, request):
        messages = [{"role": role, "content": content} for role, content in request.messages]
        assets = request.visible_inputs["images"]
        if len(assets) != len(request.images):
            raise ValueError("Image metadata and attached image counts differ")
        if assets:
            images = []
            for asset, raw in zip(assets, request.images):
                encoded = b64encode(raw).decode("ascii")
                images.append({"type": "image_url", "image_url": {"url": f"data:{asset['mime_type']};base64,{encoded}"}})
            messages[-1]["content"] = [*images, {"type": "text", "text": messages[-1]["content"]}]
        generation = dict(self.settings["generation"])
        generation.pop("thinking")
        generation["max_tokens"] = generation.pop("max_output_tokens")
        decoding_schema = request.response_schema
        if request.stage in {"annotate", "review_labels"}:
            decoding_schema = _label_decoding_schema(decoding_schema, request.task_id)
        body = {
            **generation, "model": self.settings["model_id"], "messages": messages,
            "stream": False, "response_format": {"type": "json_schema", "json_schema": {
                "name": "rumorlab_response", "schema": decoding_schema,
            }},
        }
        started = monotonic()
        finish, provider_id, final, usage, reasoning_seen = "error", "", "", {}, False
        try:
            reply = self._request("/chat/completions", body)
            provider_id = reply.get("id", "")
            usage = reply.get("usage") or {}
            choice, = reply["choices"]
            finish = choice["finish_reason"]
            message = choice["message"]
            reasoning_seen = bool(message.get("reasoning") or message.get("reasoning_content"))
            final = message.get("content") or ""
            if choice.get("finish_reason") != "stop" or message.get("refusal") or not isinstance(final, str) or not final.strip():
                raise ValueError("incomplete, refused or empty final content")
            payload = json.loads(final)
            errors = tuple(error.message for error in Draft202012Validator(request.response_schema).iter_errors(payload))
            response = ModelResponse(final, payload, not errors, errors, usage, int((monotonic()-started)*1000),
                                     finish, provider_id, None, False)
            return response, reasoning_seen
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            technical = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            errors = ()
        except json.JSONDecodeError:
            technical, errors = None, ("Invalid final JSON or response envelope",)
        except (KeyError, TypeError, AttributeError, ValueError):
            technical, errors = "invalid or incomplete vLLM response", ()
        response = ModelResponse(final if isinstance(final, str) else "", None, False, errors, usage,
                                 int((monotonic()-started)*1000), finish, provider_id, technical, False)
        return response, reasoning_seen
