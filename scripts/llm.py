"""Chat client for the GEPA scripts: one shared budget, cost accounting and a full call log.

Calls go to OpenRouter (key from OPENROUTER_API_KEY; run `source ~/.bashrc` first so the newest key is
used), except Opus when opus_direct=True: then Opus goes to the Anthropic API (ANTHROPIC_API_KEY), with
server-side refusal fallbacks only if opus_fallbacks=True (off by default, so every reply comes from
the requested model and a refusal is just a refusal), because OpenRouter's Opus providers ("Claude Platform on AWS")
content-filters these attack transcripts. opus_provider pins OpenRouter's Opus calls to one provider. Each call asks
OpenRouter to report its cost (`usage.include`); if a response lacks it, cost is computed from the
token counts and PRICES. Before every call the client refuses to start if spent + margin > budget.
"""

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
OPUS = "anthropic/claude-opus-5.5"
OPUS48 = "anthropic/claude-opus-4.8"
DIRECT_ID = {OPUS: "claude-opus-5-5", OPUS48: "claude-opus-4-8"}  # Anthropic API model ids
QWEN = "qwen/qwen3-32b"
GEMMA = "google/gemma-4-31b-it"  # monitor_xf (cross-family instruction monitor, 2026-11-19)
PRICES = {  # $ per million tokens (OpenRouter, 2026-11-18): input, output, cache read, cache write
    OPUS: (4.0, 20.0, 0.20, 5.0),
    OPUS48: (5.0, 25.0, 0.50, 6.25),
    QWEN: (0.08, 0.28, 0.08, 0.08),
    GEMMA: (0.20, 0.40, 0.20, 0.20),  # OpenRouter list price 2026-11-19
}
MARGIN = {OPUS: 0.20, OPUS48: 0.25, QWEN: 0.01, GEMMA: 0.01}  # $ reserved per call when checking the budget


class BudgetExceeded(Exception):
    pass


class OpenRouter:
    def __init__(self, budget, log_path, dry_run=False, dry_reply=None, opus_direct=False, opus_provider=None, opus_fallbacks=False,
                 replay=None):
        self.budget, self.spent, self.dry_run, self.dry_reply = budget, 0.0, dry_run, dry_reply
        # replay: a previous run's calls.jsonl. A call with the same (tag, model, system, input) reuses the logged
        # reply at no cost; tags name trajectory/iteration/attempt, so a resumed run replays its own path exactly
        # as long as the scores (server cache) and the seeded parent choice are the same.
        self.replay = {}
        for path in ([replay] if isinstance(replay, (str, os.PathLike)) else replay or []):
            for line in open(path):
                c = json.loads(line)
                self.replay[self.replay_key(c["tag"], c["model"], c["system"], c["input"])] = c
        self.replayed = 0
        self.key = None if dry_run else os.environ["OPENROUTER_API_KEY"]
        self.opus_direct, self.opus_provider, self.opus_fallbacks = opus_direct, opus_provider, opus_fallbacks
        if opus_direct and not dry_run:
            import anthropic

            self.anthropic = anthropic.Anthropic()
        self.log_path = log_path
        self.lock = threading.Lock()

    @staticmethod
    def replay_key(tag, model, system, logged_input):
        return json.dumps([tag, model, system, logged_input])

    def cost(self, model, usage):
        if usage.get("cost") is not None:
            return float(usage["cost"])
        p_in, p_out, p_read, p_write = PRICES[model]
        details = usage.get("prompt_tokens_details") or {}
        read = details.get("cached_tokens", 0) or 0
        write = details.get("cache_write_tokens", 0) or 0
        fresh = max(0, usage.get("prompt_tokens", 0) - read - write)
        return (fresh * p_in + read * p_read + write * p_write + usage.get("completion_tokens", 0) * p_out) / 1e6

    def chat(self, kind, model, system, content, tag, max_tokens=16000, effort=None, retries=3, history=None):
        """One request. `content` is a string or a list of content parts (cache_control allowed).
        `history`: earlier turns of the same conversation ([{"role", "content"}, ...]) placed before it.
        Returns (text, finish_reason)."""
        history = history or []
        with self.lock:
            if self.spent + MARGIN.get(model, 0.05) > self.budget:
                raise BudgetExceeded(f"spent ${self.spent:.2f} of ${self.budget:.2f}")
        t0 = time.time()
        logged_input = content if isinstance(content, str) else [c.get("text", "") for c in content]
        hit = self.replay.get(self.replay_key(tag, model, system, logged_input))
        if hit is not None:
            with self.lock:
                self.replayed += 1
            return hit["output"], hit["finish_reason"]
        if self.dry_run:
            text, finish, usage, cost = self.dry_reply(kind, system, content), "stop", {}, 0.0
        elif model in DIRECT_ID and self.opus_direct:
            assert not history, "history is only supported via OpenRouter"
            text, finish, usage, cost = self.anthropic_chat(model, system, content, max_tokens, effort)
        else:
            body = {"model": model, "max_tokens": max_tokens, "usage": {"include": True},
                    "messages": [{"role": "system", "content": system}, *history, {"role": "user", "content": content}]}
            if effort:
                body["reasoning"] = {"effort": effort}
            if model in DIRECT_ID and self.opus_provider:
                body["provider"] = {"order": [self.opus_provider], "allow_fallbacks": False}
            for attempt in range(max(retries, 5)):
                try:
                    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                                 headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        out = json.load(r)
                    break
                except urllib.error.HTTPError as e:
                    msg = e.read().decode(errors="replace")[:300]
                    if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"OpenRouter HTTP {e.code} for {model}: {msg}") from None
                except (http.client.HTTPException, urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError) as e:
                    if attempt + 1 < max(retries, 5):  # dropped or truncated responses (e.g. IncompleteRead): retry
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"OpenRouter connection error for {model}: {type(e).__name__}: {e}") from None
            if "error" in out:
                raise RuntimeError(f"OpenRouter error for {model}: {out['error']}")
            choice = out["choices"][0]
            text, finish = choice["message"].get("content") or "", choice.get("finish_reason")
            usage = out.get("usage", {})
            cost = self.cost(model, usage)
        with self.lock:
            self.spent += cost
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"kind": kind, "model": model, "tag": tag, "time": round(time.time() - t0, 1),
                                    "finish_reason": finish, "cost": cost, "spent": self.spent, "usage": usage,
                                    "system": system, "input": content if isinstance(content, str) else [c.get("text", "") for c in content],
                                    "output": text}) + "\n")
        return text, finish


ANTHROPIC_PRICE = {  # direct API: input, cache write, cache read, output ($/M)
    OPUS: (4.0, 5.0, 0.20, 20.0),
    OPUS48: (5.0, 6.25, 0.50, 25.0),
}


def _anthropic_cost(model, usage):
    p_in, p_write, p_read, p_out = ANTHROPIC_PRICE[model]
    g = lambda k: getattr(usage, k, 0) or 0  # noqa: E731
    return (g("input_tokens") * p_in + g("cache_creation_input_tokens") * p_write
            + g("cache_read_input_tokens") * p_read + g("output_tokens") * p_out) / 1e6


def _anthropic_chat(self, model, system, content, max_tokens, effort):
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if self.opus_fallbacks else {}
    resp = self.anthropic.beta.messages.create(  # no fallback by default: every instruction must come from Opus 5.5
        model=DIRECT_ID[model], max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": parts}],
        output_config={"effort": effort or "medium"}, **kwargs,
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    fell_back = any(getattr(it, "type", "") == "fallback_message" for it in (getattr(resp.usage, "iterations", None) or []))
    cost = _anthropic_cost(model, resp.usage) * (1.25 if fell_back else 1.0)  # fallback models can cost more; stay conservative
    usage = resp.usage.to_dict() if hasattr(resp.usage, "to_dict") else {}
    return text, resp.stop_reason, usage, cost


OpenRouter.anthropic_chat = _anthropic_chat


def tag_text(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return (m.group(1) if m else text).strip()


def json_reply(text):
    """Parse the last JSON object in a reply (Qwen may think before answering)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for m in reversed(list(re.finditer(r"\{.*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    start = text.rfind("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pass
    return None
