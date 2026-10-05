"""The playground: a local web page where several models write the same story side by side.

`POST /write` streams server-sent events while the models take turns, one token each:
  start  {"seed", "prompt_tokens"}
  token  {"model", "text", "probability", "top": [[text, probability], ...], "ms"}
  done   {"model", "text", "reason"}   ("end": the model ended the story; "length": the limit)

`POST /judge` scores given answer options in one pass per model; each model answers or abstains.

One GPU, so one request at a time: a second request while one is running gets 409.
"""

import codecs
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from hello_tokens.generation.sampling import generate, generate_stream
from hello_tokens.judge.questions import Question
from hello_tokens.judge.scoring import score_options
from hello_tokens.model.gpt import GPT
from hello_tokens.tokenizer.tokenizer import Tokenizer

PAGE = Path(__file__).with_name("static") / "index.html"


@dataclass
class Entry:
    model: GPT
    label: str  # what the page shows, e.g. "classic"
    benchmark_tokens_per_s: float | None = None  # the saved benchmark row, for comparison
    judge: dict | None = None  # the judge run's settings: scoring method, temperature, threshold


class WriteRequest(BaseModel):
    prompt: str = Field("", max_length=4000)
    temperature: float = Field(0.8, ge=0, le=2)
    top_p: float = Field(0.95, ge=0.05, le=1)
    max_tokens: int = Field(200, ge=1, le=400)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    cache: bool = False  # use the KV cache (same text, less work per token)
    fused: bool = False  # use PyTorch's fused attention kernel (same text)
    graphs: bool = False  # replay each one-token step as a CUDA graph; implies the cache (same text)


class JudgeRequest(BaseModel):
    context: str = Field("", max_length=4000)
    options: list[str] = Field(min_length=2, max_length=8)
    fallback_tokens: int = Field(40, ge=0, le=200)  # when abstaining, write this much instead (0: don't)


class TextStream:
    """Turn token ids into text as they arrive.

    A token is a piece of bytes, and one character can be split across two tokens (é is two bytes
    in UTF-8). An incremental decoder holds back an unfinished character until the rest arrives.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.pieces = tokenizer.pieces
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        return self.decoder.decode(self.pieces[token_id])

    def finish(self) -> str:
        return self.decoder.decode(b"", final=True)  # an unfinished character becomes "�"


class Busy:
    """One story at a time. A story that hasn't produced anything for `stale_after` seconds counts as
    abandoned (e.g. the page closed before the stream started, so its cleanup never ran)."""

    def __init__(self, stale_after: float = 30.0):
        self.lock = threading.Lock()
        self.active = False
        self.last_seen = 0.0
        self.stale_after = stale_after

    def try_start(self) -> bool:
        with self.lock:
            if self.active and time.monotonic() - self.last_seen < self.stale_after:
                return False
            self.active, self.last_seen = True, time.monotonic()
            return True

    def alive(self) -> None:
        self.last_seen = time.monotonic()

    def finish(self) -> None:
        with self.lock:
            self.active = False


def event(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


def create_app(models: dict[str, Entry], tokenizer: Tokenizer, device: str) -> FastAPI:
    app = FastAPI(title="hello-tokens playground")
    app.state.busy = Busy()
    app.state.models = models
    end_id = tokenizer.end_of_text_id
    context = min(entry.model.config.context for entry in models.values())

    def token_text(token_id: int) -> str:
        if token_id == end_id:
            return "[end of story]"
        return tokenizer.pieces[token_id].decode("utf-8", errors="replace")

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/models")
    def list_models() -> list[dict]:
        return [{"name": name, "label": e.label, "parameters": e.model.parameter_count(),
                 "benchmark_tokens_per_s": e.benchmark_tokens_per_s, "context": e.model.config.context,
                 "device": device} for name, e in models.items()]

    @app.post("/write")
    def write(request: WriteRequest) -> StreamingResponse:
        ids = tokenizer.encode(request.prompt)
        if len(ids) > context:
            raise HTTPException(422, f"the prompt is {len(ids)} tokens; the models read at most {context}")
        ids = ids or [end_id]  # no prompt: start as if a new story begins
        if not app.state.busy.try_start():
            raise HTTPException(409, "a story is already being written; wait for it to finish")
        seed = request.seed if request.seed is not None else random.randrange(2**31)

        def events():
            try:
                yield event("start", {"seed": seed, "prompt_tokens": len(ids)})
                streams, texts, counts = {}, {}, {}
                for name, entry in models.items():
                    entry.model.use_fused_attention(request.fused)
                    # Each model gets its own random generator with the same seed, so one model's
                    # draws never change the other's story.
                    generator = torch.Generator(device=device).manual_seed(seed)
                    streams[name] = generate_stream(entry.model, ids, request.max_tokens, request.temperature,
                                                    None, request.top_p, end_id, generator, details=True,
                                                    cache=request.cache, graphs=request.graphs)
                    texts[name], counts[name] = TextStream(tokenizer), 0
                active = list(models)
                while active:
                    for name in list(active):  # take turns, one token each
                        step = next(streams[name], None)
                        if step is None:
                            reason = "length" if counts[name] == request.max_tokens else "end"
                            yield event("done", {"model": name, "text": texts[name].finish(), "reason": reason})
                            active.remove(name)
                            continue
                        counts[name] += 1
                        app.state.busy.alive()
                        yield event("token", {
                            "model": name,
                            "text": texts[name].feed(step.id),
                            "probability": step.probability,
                            "top": [[token_text(i), p] for i, p in step.top],
                            "ms": step.seconds * 1000,
                        })
            finally:
                app.state.busy.finish()  # also runs if the page is closed mid-story

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/judge")
    def judge(request: JudgeRequest) -> list[dict]:
        """Each model scores the options in one pass, then answers if its calibrated confidence reaches the
        threshold from its judge run, and otherwise abstains (and writes a continuation instead)."""
        if any(not option.strip() or len(option) > 500 for option in request.options):
            raise HTTPException(422, "options must be non-empty and at most 500 characters")
        if not app.state.busy.try_start():
            raise HTTPException(409, "the models are busy; try again in a moment")
        try:
            question = Question(request.context, request.options, 0)  # the answer is unknown here
            results = []
            for name, entry in models.items():
                settings = entry.judge or {}
                method = settings.get("method", "summed")
                temperature = settings.get("temperature", 1.0)
                threshold = settings.get("threshold", 0.9)
                started = time.perf_counter()
                try:
                    judgement = score_options(entry.model, tokenizer, question)
                except ValueError as error:
                    raise HTTPException(422, str(error))
                probabilities = judgement.probabilities(method, temperature)
                confidence, choice = probabilities.max(dim=0)
                scoring_ms = (time.perf_counter() - started) * 1000  # the judging itself, not the fallback
                answering = float(confidence) >= threshold
                fallback = None
                if not answering and request.fallback_tokens:
                    ids = (tokenizer.encode(request.context) or [end_id])[-context:]
                    fallback = tokenizer.decode(generate(entry.model, ids, request.fallback_tokens, temperature=0,
                                                         stop_id=end_id, cache=True))
                results.append({
                    "model": name,
                    "probabilities": probabilities.tolist(),
                    "choice": int(choice),
                    "confidence": float(confidence),
                    "decision": "answer" if answering else "abstain",
                    "threshold": threshold,
                    "calibrated": entry.judge is not None,
                    "ms": scoring_ms,
                    "fallback": fallback,
                })
            return results
        finally:
            app.state.busy.finish()

    return app
