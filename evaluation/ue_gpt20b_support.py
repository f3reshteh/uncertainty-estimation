from pathlib import Path
from typing import Any, Dict, List, Optional
from requests import HTTPError
import json
import os
import re
import sys
import time

import numpy as np
import requests
from dotenv import load_dotenv


CONTROL_TOK_RE = re.compile(r"^<\|.*\|>$")
LOGPROB_FLOOR = -80.0
TOP_LOGPROBS = 5
MAX_CHAT_RETRIES = 8
CHAT_BACKOFF_BASE = 1.5
CHAT_BACKOFF_CAP = 120.0
MAX_TOKENS_DEFAULT = 4000
MAX_UPSTREAM_RETRIES = 10
BACKOFF_BASE = 2.0
BACKOFF_CAP = 180.0


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_legal_service_root() -> Path:
    candidates = [
        Path("intelliprocure-ai-legal/legal_service"),
        Path("../intelliprocure-ai-legal/legal_service"),
        Path("../../intelliprocure-ai-legal/legal_service"),
    ]
    root = _first_existing(candidates)
    if root is None:
        raise FileNotFoundError("Could not locate intelliprocure-ai-legal/legal_service.")
    return root


LEGAL_SERVICE_ROOT = _resolve_legal_service_root()


def _load_env_files() -> List[Path]:
    loaded = []
    candidates = [
        LEGAL_SERVICE_ROOT / ".env",
        Path(".env"),
        Path("uncertainty_estimation/.env"),
        Path("../uncertainty_estimation/.env"),
    ]
    seen = set()
    for env_path in candidates:
        key = str(env_path.resolve()) if env_path.exists() else str(env_path)
        if key in seen or not env_path.exists():
            continue
        load_dotenv(env_path, override=True)
        loaded.append(env_path)
        seen.add(key)
    return loaded


LOADED_ENV_FILES = _load_env_files()

LLM_BACKEND = os.getenv("LLM_BACKEND", "together").strip()
if LLM_BACKEND == "hf":
    LLM_BASE_URL = "https://router.huggingface.co/v1"
    LLM_API_KEY = os.environ["HF_TOKEN"]
    LLM_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-20b")
elif LLM_BACKEND == "openrouter":
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_API_KEY = os.environ["OPENROUTER_API_KEY"]
    LLM_MODEL = os.environ["OPENROUTER_MODEL"]
elif LLM_BACKEND == "together":
    LLM_BASE_URL = "https://api.together.xyz/v1"
    LLM_API_KEY = os.environ["TOGETHER_API_KEY"]
    LLM_MODEL = os.getenv("TOGETHER_MODEL", "openai/gpt-oss-20b")
    TOGETHER_LOGPROBS_K = int(os.getenv("TOGETHER_LOGPROBS_K", "5"))
else:
    raise ValueError(f"Unsupported LLM_BACKEND for this notebook helper: {LLM_BACKEND}")

DATA_SERVICE_API_KEY = os.getenv("DATA_SERVICE_API_KEY", "")
DATA_SERVICE_HOST_URL = os.getenv("DATA_SERVICE_HOST_URL", "http://localhost:8002")

uqlm_root = _first_existing([Path("uqlm-main"), Path("../uqlm-main")])
if uqlm_root is not None:
    uqlm_root_str = str(uqlm_root.resolve())
    if uqlm_root_str not in sys.path:
        sys.path.append(uqlm_root_str)

from uqlm.white_box.single_logprobs import SingleLogprobsScorer
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer
from uqlm.black_box.consistency import ConsistencyScorer
from uqlm.black_box.bert import BertScorer


def _safe_logprob(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return float(np.clip(x, LOGPROB_FLOOR, 0.0))


def extract_message_text(choice: Dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts).strip()
    for key in ("reasoning_content", "reasoning"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_api_response(text: Any) -> str:
    normalized = ("" if text is None else str(text))
    normalized = (
        normalized
        .replace("\u00A0", " ")
        .replace("\u2009", " ")
        .replace("\u202F", " ")
        .replace("\u200B", "")
        .replace("ï¼š", ":")
    )
    normalized = re.sub(r":\s{2,}", ": ", normalized)
    normalized = normalized.strip()
    normalized = re.sub(r"```+$", "", normalized).rstrip()
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized


def parse_boolean_result(api_response: Any) -> Optional[bool]:
    if api_response is None:
        return None

    trimmed = normalize_api_response(api_response)
    tail = trimmed[-400:]

    try:
        parsed = json.loads(trimmed)
        if isinstance(parsed, dict):
            for key in ("ergebnis", "Ergebnis", "ERGEBNIS", "result"):
                if key in parsed:
                    result = str(parsed[key]).lower()
                    if result in ("true", "false"):
                        return result == "true"
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    pattern_strict = re.compile(r'(?:ergebnis):\s*["\']?`?(true|false)`?["\']?(?:\s|`)*$', re.IGNORECASE)
    match = pattern_strict.search(tail) or pattern_strict.search(trimmed)
    if match:
        return match.group(1).lower() == "true"

    german_true = {"ja", "wahr"}
    german_false = {"nein", "falsch"}
    last_line = trimmed.split("\n")[-1].strip().lower()
    last_token_match = re.search(r"([\wÃ¤Ã¶Ã¼Ã„Ã–ÃœÃŸ]+)[\)\]\>\'\"\.,;:!\s`]*$", last_line)
    if last_token_match:
        token = last_token_match.group(1)
        if token in german_true:
            return True
        if token in german_false:
            return False

    match = re.search(r'["\']?`?(true|false)`?["\']?[\)\]\>\'\"\.,;:!\s?`]*$', tail, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"

    matches = list(re.finditer(r'(?:ergebnis):\s*["\']?`?(true|false)`?["\']?', trimmed, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).lower() == "true"

    last_line = trimmed.split("\n")[-1].strip()
    match = re.search(r"(true|false)[\)\]\>\'\"\.,;:!\s`]*$", last_line, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"

    return None


def canonical_result_line(text: Any) -> str:
    parsed = parse_boolean_result(text)
    if parsed is None:
        return ""
    return f"ERGEBNIS: {'true' if parsed else 'false'}"


def _is_transient_http_error(error: Exception) -> bool:
    if not isinstance(error, HTTPError):
        return False
    resp = getattr(error, "response", None)
    status = getattr(resp, "status_code", None)
    body = ""
    try:
        body = (resp.text or "").lower() if resp is not None else ""
    except Exception:
        pass
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    markers = ["resource exhausted", "rate limit", "quota", "too many requests", "temporarily unavailable"]
    return any(marker in body for marker in markers)


def _choice_has_logprobs(choice: Dict[str, Any]) -> bool:
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        return isinstance(logprobs.get("content"), list) and len(logprobs["content"]) > 0
    if isinstance(logprobs, list):
        return len(logprobs) > 0
    return False


def _require_logprobs_in_response(data: Dict[str, Any], context: str = "") -> None:
    choice = (data.get("choices") or [{}])[0]
    if not _choice_has_logprobs(choice):
        raise ValueError(f"Missing logprobs in successful response ({context}).")


def llm_chat(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    logprobs: bool = True,
    top_logprobs: int = TOP_LOGPROBS,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if LLM_BACKEND == "together":
        if logprobs:
            payload["logprobs"] = int(TOGETHER_LOGPROBS_K)
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=120)
        if response.status_code >= 400:
            raise HTTPError(f"{response.status_code} {response.reason}: {response.text[:1200]}", response=response)
        data = response.json()
        if logprobs:
            _require_logprobs_in_response(data, context="together")
        return data

    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(top_logprobs)
    if LLM_BACKEND == "openrouter":
        payload["provider"] = {"require_parameters": True}

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "HTTP-Referer": "https://intelliprocure.ch",
        "X-Title": "UQLM White-Box Eval",
    }
    response = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=120)
    if response.status_code >= 400:
        raise HTTPError(f"{response.status_code} {response.reason}: {response.text[:1200]}", response=response)
    data = response.json()
    if logprobs:
        _require_logprobs_in_response(data, context=LLM_BACKEND)
    return data


def llm_chat_with_backoff(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    logprobs: bool = True,
    top_logprobs: int = TOP_LOGPROBS,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(MAX_CHAT_RETRIES):
        try:
            return llm_chat(messages, temperature=temperature, max_tokens=max_tokens, logprobs=logprobs, top_logprobs=top_logprobs)
        except (HTTPError, ValueError) as error:
            last_err = error
            if isinstance(error, HTTPError) and not _is_transient_http_error(error):
                raise
            wait = min(CHAT_BACKOFF_CAP, CHAT_BACKOFF_BASE * (2 ** attempt)) + np.random.uniform(0, 0.6)
            print(f"llm_chat transient error; retry {attempt + 1}/{MAX_CHAT_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
    if last_err is None:
        raise RuntimeError("llm_chat_with_backoff failed without an exception.")
    raise last_err


def llm_chat_require_logprobs(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 50,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for temp in (temperature, 0.2):
        try:
            return llm_chat_with_backoff(messages, temperature=temp, max_tokens=max_tokens, logprobs=True, top_logprobs=TOP_LOGPROBS)
        except (HTTPError, ValueError) as error:
            last_err = error
    if last_err is None:
        raise RuntimeError("llm_chat_require_logprobs failed without an exception.")
    raise last_err


def to_uqlm_logprobs(logprobs_payload: Any) -> List[Dict[str, Any]]:
    if logprobs_payload is None:
        return []
    if isinstance(logprobs_payload, dict):
        tokens = logprobs_payload.get("content") or []
    elif isinstance(logprobs_payload, list):
        tokens = logprobs_payload
    else:
        return []

    cleaned = []
    final_only = []
    expect_channel_name = False
    in_final = False
    for token_info in tokens:
        token = (token_info.get("token") or "").strip()
        if not token:
            continue
        if token == "<|channel|>":
            expect_channel_name = True
            in_final = False
            continue
        if expect_channel_name:
            in_final = token.lower() == "final"
            expect_channel_name = False
            continue
        if CONTROL_TOK_RE.match(token):
            continue
        logprob = _safe_logprob(token_info.get("logprob"))
        if logprob is None:
            continue
        top_clean = []
        for alt in token_info.get("top_logprobs") or []:
            alt_token = (alt.get("token") or "").strip()
            alt_logprob = _safe_logprob(alt.get("logprob"))
            if not alt_token or alt_logprob is None or CONTROL_TOK_RE.match(alt_token):
                continue
            top_clean.append({**alt, "logprob": alt_logprob})
        row = {**token_info, "logprob": logprob, "top_logprobs": top_clean}
        cleaned.append(row)
        if in_final:
            final_only.append(row)
    return final_only if final_only else cleaned


def slice_logprobs_to_final_result(
    logprobs_payload: Any,
    expected_result_line: str = "",
) -> List[Dict[str, Any]]:
    tokens = to_uqlm_logprobs(logprobs_payload)
    if not tokens:
        return []

    token_texts = [str(tok.get("token") or "") for tok in tokens]
    expected_bool = parse_boolean_result(expected_result_line) if expected_result_line else None
    line_only_re = re.compile(
        r'^[\s"\'`]*ERGEBNIS\s*:\s*(true|false)[\s"\'`\)\]\>\.,;:!]*$',
        re.IGNORECASE,
    )

    for start in range(len(tokens) - 1, -1, -1):
        candidate = ""
        for end in range(start + 1, len(tokens) + 1):
            candidate += token_texts[end - 1]
            normalized = normalize_api_response(candidate)
            if not line_only_re.fullmatch(normalized):
                continue
            parsed = parse_boolean_result(normalized)
            if parsed is None:
                continue
            if expected_bool is not None and parsed != expected_bool:
                continue
            return tokens[start:end]

    return []


def extract_final_result_logprobs(logprobs_payload: Any, response_text: Any) -> List[Dict[str, Any]]:
    expected_result_line = canonical_result_line(response_text)
    if not expected_result_line:
        return []
    return slice_logprobs_to_final_result(logprobs_payload, expected_result_line=expected_result_line)


def build_messages(check_code: str, project_summary: Dict[str, Any]) -> List[Dict[str, str]]:
    checks_cfg = json.loads((LEGAL_SERVICE_ROOT / "test_definitions/checks.json").read_text(encoding="utf-8"))
    prompts_dir = LEGAL_SERVICE_ROOT / "test_definitions/prompts_checks"
    system_prompt = (LEGAL_SERVICE_ROOT / "test_definitions/system_prompt.txt").read_text(encoding="utf-8").strip()
    cfg = checks_cfg[check_code]
    required_fields = cfg["required_fields"]
    prompt_file = cfg["prompt_file"]

    formatted = ""
    for field in required_fields:
        value = project_summary.get(field, "")
        if value is None or value == "":
            value = "No answer"
        elif not isinstance(value, str):
            value = str(value)
        formatted += f"**{field}:**\n{value}\n\n"

    prompt = (prompts_dir / prompt_file).read_text(encoding="utf-8").replace("{PROJECT_DATA}", formatted)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


def fetch_project_summary(project_id: int, simap_version: str) -> Dict[str, Any]:
    url = f"{DATA_SERVICE_HOST_URL}/api/v1/projects/{project_id}?simap_version={simap_version}"
    headers = {"X-API-KEY": DATA_SERVICE_API_KEY} if DATA_SERVICE_API_KEY else {}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()["data"]["attributes"]["summary"]


def run_tests_for_project_with_backoff(project_id: int):
    last_err: Optional[Exception] = None
    for attempt in range(MAX_UPSTREAM_RETRIES):
        try:
            return run_tests_for_project(project_id)
        except HTTPError as error:
            last_err = error
            if not _is_transient_http_error(error):
                raise
            sleep_s = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt)) + np.random.uniform(0, 0.7)
            print(f"run_tests_for_project transient error; retry {attempt + 1}/{MAX_UPSTREAM_RETRIES} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    if last_err is None:
        raise RuntimeError("run_tests_for_project_with_backoff failed without an exception.")
    raise last_err
