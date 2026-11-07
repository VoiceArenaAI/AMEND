import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
BATCH_CHUNK = 10000
SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"fixed": {"type": "string"}},
        "required": ["fixed"],
        "additionalProperties": False,
    },
}


def _text(message) -> str | None:
    if message.stop_reason != "end_turn":
        return None
    raw = "".join(b.text for b in message.content if b.type == "text")
    try:
        return json.loads(raw)["fixed"].strip()
    except (ValueError, KeyError, TypeError):
        return None


class Judge:
    def __init__(
        self,
        model: str = MODEL,
        effort: str = EFFORT,
        max_tokens: int = MAX_TOKENS,
        client: anthropic.Anthropic | None = None,
        workers: int = 8,
        poll_seconds: int = 30,
        verbose: bool = False,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()
        self.workers = workers
        self.poll_seconds = poll_seconds
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def fix(self, prompt: str) -> str | None:
        try:
            message = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort, "format": SCHEMA},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            self._log(f"request failed: {type(e).__name__}: {e}")
            return None
        text = _text(message)
        if text is None:
            self._log(f"unusable response: stop_reason={message.stop_reason}")
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
        results: list[str | None] = [None] * len(prompts)
        for start in range(0, len(prompts), BATCH_CHUNK):
            chunk = prompts[start : start + BATCH_CHUNK]
            requests = [
                Request(
                    custom_id=f"r{start + i}",
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        output_config={"effort": self.effort, "format": SCHEMA},
                        messages=[{"role": "user", "content": p}],
                    ),
                )
                for i, p in enumerate(chunk)
            ]
            batch = self.client.messages.batches.create(requests=requests)
            self._log(f"batch {batch.id} submitted with {len(requests)} requests")
            while batch.processing_status != "ended":
                time.sleep(self.poll_seconds)
                batch = self.client.messages.batches.retrieve(batch.id)
                c = batch.request_counts
                self._log(f"batch {batch.id} {batch.processing_status} processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
            for item in self.client.messages.batches.results(batch.id):
                idx = int(item.custom_id[1:])
                if item.result.type == "succeeded":
                    results[idx] = _text(item.result.message)
                else:
                    self._log(f"{item.custom_id} {item.result.type}")
        return results
