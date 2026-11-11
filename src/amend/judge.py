import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-opus-5.5"
EFFORT = "max"
MAX_TOKENS = 64000
BATCH_CHUNK = 5000
TIMEOUT = 1200
FINAL = {"completed", "failed", "expired", "cancelled"}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "amend_fix",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"fixed": {"type": "string"}},
            "required": ["fixed"],
            "additionalProperties": False,
        },
    },
}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
NOT_FOUND_GRACE = 600


class APIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _text(body: dict | None) -> str | None:
    if not body or body.get("error"):
        return None
    choices = body.get("choices") or []
    if not choices or choices[0].get("finish_reason") != "stop":
        return None
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str):
        return None
    try:
        return json.loads(_FENCE.sub("", content.strip()))["fixed"].strip()
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _custom_id(index: int, prompt: str) -> str:
    return f"r{index}-{hashlib.sha1(prompt.encode()).hexdigest()[:16]}"


class Judge:
    def __init__(
        self,
        model: str = MODEL,
        effort: str | None = EFFORT,
        max_tokens: int = MAX_TOKENS,
        api_key: str | None = None,
        workers: int = 8,
        poll_seconds: int = 60,
        timeout: int = TIMEOUT,
        verbose: bool = False,
        resume_batches: list[str] | None = None,
        session: requests.Session | None = None,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("set OPENROUTER_API_KEY or pass api_key")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.workers = workers
        self.poll_seconds = poll_seconds
        self.timeout = timeout
        self.verbose = verbose
        self.resume_batches = list(resume_batches or [])
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _call(self, method: str, path: str, payload: dict | None = None, attempts: int = 5) -> dict:
        for attempt in range(attempts):
            status = None
            try:
                r = self.session.request(method, f"{API_URL}{path}", json=payload, timeout=(30, self.timeout))
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code < 400:
                    return r.json()
                status = r.status_code
                err = f"HTTP {status}: {r.text[:300]}"
                if status != 429 and status < 500:
                    raise APIError(err, status)
            if attempt == attempts - 1:
                raise APIError(err, status)
            time.sleep(min(2 ** attempt * 5, 120))
        raise APIError("unreachable")

    def _body(self, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "response_format": RESPONSE_FORMAT,
        }
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        return body

    def fix(self, prompt: str) -> str | None:
        try:
            body = self._call("POST", "/chat/completions", self._body(prompt))
        except APIError as e:
            self._log(f"request failed: {e}")
            return None
        text = _text(body)
        if text is None:
            choice = (body.get("choices") or [{}])[0]
            self._log(f"unusable response: finish_reason={choice.get('finish_reason')} error={body.get('error')}")
        return text

    def fix_many(self, prompts: list[str], mode: str = "sync") -> list[str | None]:
        unique = list(dict.fromkeys(prompts))
        if mode == "sync":
            results = self._sync(unique)
        elif mode == "batch":
            results = self._batch(unique)
        else:
            raise ValueError(f"unknown mode {mode!r}, expected 'sync' or 'batch'")
        lookup = dict(zip(unique, results))
        return [lookup[p] for p in prompts]

    def _sync(self, prompts: list[str]) -> list[str | None]:
        done = 0

        def run(p: str) -> str | None:
            nonlocal done
            out = self.fix(p)
            done += 1
            self._log(f"sync {done}/{len(prompts)}")
            return out

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(run, prompts))

    def _batch(self, prompts: list[str]) -> list[str | None]:
        if self.resume_batches:
            batch_ids = self.resume_batches
            self.resume_batches = []
            self._log(f"resuming {len(batch_ids)} batch(es): {' '.join(batch_ids)}")
        else:
            batch_ids = [self._submit(prompts, start) for start in range(0, len(prompts), BATCH_CHUNK)]
            self._log(f"to resume if interrupted, add: --resume {' '.join(batch_ids)}")
        results: list[str | None] = [None] * len(prompts)
        for batch_id in batch_ids:
            self._collect(self._wait(batch_id), prompts, results)
        return results

    def _submit(self, prompts: list[str], start: int) -> str:
        chunk = prompts[start : start + BATCH_CHUNK]
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": self.model,
            "completion_window": "24h",
            "requests": [{"custom_id": _custom_id(start + i, p), "body": self._body(p)} for i, p in enumerate(chunk)],
        }
        batch = self._call("POST", "/batches", payload)
        self._log(f"batch {batch['id']} submitted with {len(chunk)} requests")
        return batch["id"]

    def _wait(self, batch_id: str) -> dict:
        started = time.monotonic()
        while True:
            try:
                batch = self._call("GET", f"/batches/{batch_id}")
            except APIError as e:
                if e.status == 404 and time.monotonic() - started < NOT_FOUND_GRACE:
                    self._log(f"batch {batch_id} not visible yet, waiting")
                    time.sleep(min(self.poll_seconds, 15))
                    continue
                raise APIError(f"{e}; the batch may still be running, collect it later with --resume {batch_id}", e.status)
            status = batch.get("status")
            c = batch.get("request_counts") or {}
            if status in FINAL and (status != "completed" or batch.get("results") is not None):
                self._log(f"batch {batch_id} {status} completed={c.get('completed')} failed={c.get('failed')}")
                return batch
            self._log(f"batch {batch_id} {status} completed={c.get('completed')} failed={c.get('failed')} total={c.get('total')}")
            time.sleep(self.poll_seconds)

    def _collect(self, batch: dict, prompts: list[str], results: list[str | None]) -> None:
        if batch.get("status") != "completed":
            self._log(f"batch {batch.get('id')} ended {batch.get('status')}: {(batch.get('error') or {}).get('message')}")
            return
        mismatched = 0
        for item in batch.get("results") or []:
            cid = item.get("custom_id", "")
            index_part, _, _ = cid[1:].partition("-")
            idx = int(index_part) if index_part.isdigit() else -1
            if not 0 <= idx < len(prompts) or _custom_id(idx, prompts[idx]) != cid:
                mismatched += 1
                continue
            if item.get("error"):
                self._log(f"{cid} error: {item['error'].get('message')}")
                continue
            results[idx] = _text((item.get("response") or {}).get("body"))
        if mismatched:
            raise ValueError(
                f"batch {batch.get('id')}: {mismatched} results do not match the current inputs; "
                "resume with the same CSV, columns, --limit and prompt version that created the batch"
            )
