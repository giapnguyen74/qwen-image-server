"""Qwen-Image-2.1 job server: JSON requests go into a FIFO queue, one GPU worker renders them, and
finished PNGs are downloaded by job id. The pipeline is loaded once at startup and stays loaded.

  uv run qwen_image_server.py                      # http://127.0.0.1:8000
  uv run qwen_image_server.py --host 0.0.0.0 --port 8000 --quant Q8_0
  uv run qwen_image_server.py --weights original          # official weights, 8-bit transformer
  uv run qwen_image_server.py --output-dir /data/qwen     # where finished PNGs are written
  uv run qwen_image_server.py --steps 50                  # default steps for requests that omit them

API
  POST   /jobs              submit; returns 202 {"id", "status", "position", ...}
  GET    /jobs              all jobs, newest first
  GET    /jobs/{id}         status: queued | running | done | failed | cancelled, with progress and
                            eta_s (estimated seconds until done, including jobs queued ahead)
  GET    /jobs/{id}/image   the PNG once status is "done"
  DELETE /jobs/{id}         cancel a queued or running job
  GET    /health            model settings and queue length

Request body (an agent using the qwen-image-t2i or qwen-image-edit skill in skills/ maps
its rewritten_prompt to "prompt" and its wh_ratio / ratio_follow across):
  {
    "prompt": "...",
    "images": ["<base64>", ...],  # optional, 1-10; present = edit (i2i), absent = text-to-image
    "wh_ratio": "2:3",            # optional W:H
    "ratio_follow": "<image1>",   # edit only: follow that input's aspect ratio (default <image1>)
    "steps": 40,                  # optional; server --steps default (40) if omitted
    "seed": 1234                  # optional; random if omitted
  }

Jobs live in memory and are lost on restart; images are kept in --output-dir
(default outputs/server/).
"""

import argparse
import base64
import binascii
import io
import queue
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from qwen_image_common import (add_model_args, build_pipeline, generate, resolve_model_args, save_png,
                               size_following, size_for_ratio)

MAX_IMAGES = 10
MAX_STEPS = 100
DEFAULT_STEPS = 40
DEFAULT_OUT_DIR = Path("outputs/server")
# ETA model: seconds per denoising step (updated from finished jobs) plus a fixed per-job overhead
# for prompt encoding and VAE decode. Measured on the RTX 3090 at 2K with GGUF Q8_0: ~6.5-7.4 s/step.
INITIAL_SEC_PER_STEP = 6.8
JOB_OVERHEAD_S = 8.0


class JobRequest(BaseModel):
    prompt: str
    images: list[str] = Field(default_factory=list)
    wh_ratio: str = ""
    ratio_follow: str = ""
    steps: int | None = Field(None, ge=1, le=MAX_STEPS)   # None: the server's --steps
    seed: int | None = Field(None, ge=0, lt=2**32)


@dataclass
class Job:
    id: str
    mode: str                      # "t2i" or "edit"
    prompt: str
    width: int
    height: int
    size_from: str
    steps: int
    seed: int
    n_images: int
    status: str = "queued"
    step: int = 0
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False
    images: list = field(default_factory=list, repr=False)   # PIL inputs, dropped after the run
    path: str | None = None


def decode_image(data, index):
    from PIL import Image

    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
        img.load()
    except (binascii.Error, ValueError, OSError) as e:
        raise HTTPException(422, f"images[{index}] is not a valid base64-encoded image: {e}")
    return img


def resolve_size(req, images):
    """Output size and a label for where it came from; follows the edit template's rules."""
    if req.wh_ratio and req.ratio_follow:
        raise HTTPException(422, "wh_ratio and ratio_follow are mutually exclusive")
    try:
        if req.wh_ratio:
            return (*size_for_ratio(req.wh_ratio), req.wh_ratio)
        if not images:
            if req.ratio_follow:
                raise HTTPException(422, "ratio_follow needs input images")
            return (*size_for_ratio("1:1"), "1:1")
        tag = req.ratio_follow or "<image1>"
        n = int(tag.strip("<>").removeprefix("image"))
    except SystemExit as e:  # size_for_ratio reports bad ratios this way for the CLI scripts
        raise HTTPException(422, str(e))
    except ValueError:
        raise HTTPException(422, f"ratio_follow must look like <image1>, got {req.ratio_follow!r}")
    if not 1 <= n <= len(images):
        raise HTTPException(422, f"ratio_follow {tag}: only {len(images)} input image(s)")
    return (*size_following(images[n - 1]), tag)


class JobQueue:
    def __init__(self, pipe, settings, out_dir=DEFAULT_OUT_DIR, default_steps=DEFAULT_STEPS):
        self.pipe = pipe
        self.default_steps = default_steps
        self.out_dir = Path(out_dir)
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.running: str | None = None
        self.sec_per_step = INITIAL_SEC_PER_STEP
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, job):
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.queue.put(job.id)

    def position(self, job):
        """0 = next to run; None when not queued."""
        if job.status != "queued":
            return None
        with self.lock:
            waiting = [j for j in self.order if self.jobs[j].status == "queued"]
        return waiting.index(job.id)

    def remaining_s(self, job, now):
        """Estimated seconds of GPU time this job still needs."""
        total = job.steps * self.sec_per_step + JOB_OVERHEAD_S
        if job.status == "running":
            return max(1.0, total - (now - job.started))
        return total

    def eta_s(self, job):
        """Estimated seconds until this job is done: queued jobs wait for the running job and
        everything queued ahead of them."""
        now = time.time()
        if job.status == "running":
            return round(self.remaining_s(job, now))
        if job.status != "queued":
            return None
        with self.lock:
            ahead = [self.jobs[j] for j in self.order if self.jobs[j].status in ("queued", "running")]
        eta = 0.0
        for j in ahead:
            eta += self.remaining_s(j, now)
            if j is job:
                break
        return round(eta)

    def view(self, job):
        d = {k: v for k, v in asdict(job).items() if k not in ("images", "cancel_requested", "path")}
        d["prompt"] = job.prompt if len(job.prompt) <= 200 else job.prompt[:200] + "..."
        d["position"] = self.position(job)
        d["progress"] = round(job.step / job.steps, 3)
        d["eta_s"] = self.eta_s(job)
        if job.status == "done":
            d["image_url"] = f"/jobs/{job.id}/image"
        return d

    def _worker(self):
        import torch

        while True:
            job = self.jobs[self.queue.get()]
            if job.status != "queued":        # cancelled while waiting
                continue
            job.status, job.started, self.running = "running", time.time(), job.id

            def on_step(pipe, i, t, kwargs):
                job.step = i + 1
                if job.cancel_requested:
                    pipe._interrupt = True    # the pipeline skips the remaining steps
                return kwargs

            try:
                image, _ = generate(self.pipe, job.prompt, job.width, job.height, job.steps, job.seed,
                                    images=job.images or None, cpu_offload=self.settings["cpu_offload"],
                                    keep_text_encoder=True, callback=on_step)
                if job.cancel_requested:
                    job.status = "cancelled"
                else:
                    path = self.out_dir / f"{job.id}.png"
                    save_png(image, path, {"prompt": job.prompt, "mode": job.mode, "seed": job.seed,
                                           "steps": job.steps, "size": f"{job.width}x{job.height}",
                                           "size_from": job.size_from, "n_images": job.n_images,
                                           "weights": self.settings["weights"], "quant": self.settings["quant"],
                                           "te_quant": self.settings["te_quant"]})
                    job.path, job.status = str(path), "done"
                    if job.steps >= 5:  # refine the ETA model from real runs
                        measured = (time.time() - job.started - JOB_OVERHEAD_S) / job.steps
                        self.sec_per_step = 0.7 * self.sec_per_step + 0.3 * measured
            except torch.OutOfMemoryError as e:
                job.status, job.error = "failed", f"CUDA out of memory: {e}".splitlines()[0]
            except Exception as e:  # keep the worker alive for the next job
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                job.finished, job.images, self.running = time.time(), [], None
                torch.cuda.empty_cache()


def create_app(jq):
    app = FastAPI(title="Qwen-Image-2.1 server")

    def get(job_id):
        job = jq.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job

    @app.post("/jobs", status_code=202)
    def submit(req: JobRequest):
        prompt = req.prompt.strip()
        if not prompt:
            raise HTTPException(422, "prompt is empty")
        if len(req.images) > MAX_IMAGES:
            raise HTTPException(422, f"{len(req.images)} images; the model supports at most {MAX_IMAGES}")
        images = [decode_image(d, i) for i, d in enumerate(req.images)]
        width, height, size_from = resolve_size(req, images)
        job = Job(id=uuid.uuid4().hex[:12], mode="edit" if images else "t2i", prompt=prompt, width=width,
                  height=height, size_from=size_from,
                  steps=req.steps if req.steps is not None else jq.default_steps,
                  seed=req.seed if req.seed is not None else secrets.randbelow(2**32),
                  n_images=len(images), images=images)
        jq.submit(job)
        return jq.view(job)

    @app.get("/jobs")
    def list_jobs():
        return [jq.view(jq.jobs[j]) for j in reversed(jq.order)]

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        return jq.view(get(job_id))

    @app.get("/jobs/{job_id}/image")
    def image(job_id: str):
        job = get(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        return FileResponse(job.path, media_type="image/png", filename=f"{job.id}.png")

    @app.delete("/jobs/{job_id}")
    def cancel(job_id: str):
        job = get(job_id)
        if job.status == "queued":
            job.status, job.finished = "cancelled", time.time()
        elif job.status == "running":
            job.cancel_requested = True   # the worker stops at the next step
        else:
            raise HTTPException(409, f"job is already {job.status}")
        return jq.view(job)

    @app.get("/health")
    def health():
        queued = sum(1 for j in jq.jobs.values() if j.status == "queued")
        return {"status": "ok", "queued": queued, "running": jq.running,
                "sec_per_step": round(jq.sec_per_step, 2),
                "default_steps": jq.default_steps, **jq.settings}

    return app


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines (no auth!)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"where finished PNGs are saved as <job id>.png (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"denoising steps for requests that don't set them (default: {DEFAULT_STEPS})")
    add_model_args(ap)
    args = resolve_model_args(ap.parse_args())
    if not 1 <= args.steps <= MAX_STEPS:
        ap.error(f"--steps must be 1-{MAX_STEPS}")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)  # fail now, not after the ~1 min model load
    print(f"output dir: {out_dir}")
    pipe, te_ok = build_pipeline(args.quant, args.te_quant, cpu_offload=args.cpu_offload, weights=args.weights)
    if not te_ok:
        raise SystemExit("text encoder weights did not all match; aborting")
    jq = JobQueue(pipe, {"weights": args.weights, "quant": args.quant, "te_quant": args.te_quant,
                         "cpu_offload": args.cpu_offload}, out_dir, args.steps)
    uvicorn.run(create_app(jq), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
