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
import json
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
                               size_following, size_for_ratio, working_set_gib)

MAX_IMAGES = 10
MAX_STEPS = 100
DEFAULT_STEPS = 40
DEFAULT_OUT_DIR = Path("outputs/server")
# ETA model: seconds per denoising step plus a per-job overhead for prompt encoding and VAE decode.
# Both are measured from finished jobs (step timestamps from the pipeline callback, so the overhead is
# what is left of the wall time) and persisted per GPU and weight configuration in TIMING_PATH, so a
# restarted server starts from what this card actually did. Until the first measurement the prior
# comes from the GPU name: 2K GGUF Q8_0 measured about 6.8 s/step on an RTX 3090 and 2.4 s/step on
# an RTX 5090. An unknown card gets the 3090 figure, which is the slower of the two.
SEC_PER_STEP_PRIORS = {"5090": 2.4, "4090": 3.5, "3090": 6.8}
DEFAULT_SEC_PER_STEP = 6.8
JOB_OVERHEAD_S = 8.0
MIN_SEC_PER_STEP = 0.05
# Weight of the newest measurement once one exists; the first measurement replaces the prior outright.
MEASUREMENT_WEIGHT = 0.5
# sec_per_step is kept for a 2048x2048 output; other sizes scale with their pixel count.
REFERENCE_PIXELS = 2048 * 2048
TIMING_PATH = Path(__file__).resolve().parent / "qwen_image_timing.json"
# Share of total VRAM a job may peak at. A 4-image edit measured 29.68 GiB of 31.36 (94.6%), so
# this leaves that working while still rejecting what cannot fit.
VRAM_USABLE_FRACTION = 0.97


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
    first_step_at: float | None = None    # end of denoising step 1 (encoding done)
    last_step_at: float | None = None     # end of the latest step
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


def check_fits(width, height, n_images, settings):
    """Reject a job that cannot fit before it is queued, rather than after it reaches the GPU.

    The resident weights never move, so what a job needs is its working set, and that is
    predictable from the output size and the number of input images.
    """
    total = settings.get("total_gib")
    resident = settings.get("resident_gib")
    if not total or not resident:        # no measurement (e.g. a test double): admit everything
        return
    need = working_set_gib(width, height, n_images)
    capacity = total * VRAM_USABLE_FRACTION - resident
    if need <= capacity:
        return
    hint = ""
    if n_images:
        fits = 0
        while working_set_gib(width, height, fits + 1) <= capacity:
            fits += 1
        hint = (f" At this size {fits} input image(s) fit; you sent {n_images}."
                if fits else " No edit fits at this size on this GPU.")
    # Not "use a smaller wh_ratio": every ratio targets the same ~2048x2048 pixel count, so
    # changing it does not change what the job costs.
    raise HTTPException(422, f"this job needs about {need:.1f} GiB of VRAM on top of the "
                             f"{resident:.1f} GiB of resident weights, and only {capacity:.1f} GiB "
                             f"of the {total:.1f} GiB card is available.{hint} Send fewer images, "
                             f"or restart the server with --cpu-offload or a smaller --quant")


def size_scale(job):
    return job.width * job.height / REFERENCE_PIXELS


def timing_key(settings):
    return "|".join(str(settings.get(k)) for k in ("gpu", "weights", "quant", "te_quant", "cpu_offload"))


def sec_per_step_prior(gpu):
    for tag, rate in SEC_PER_STEP_PRIORS.items():
        if tag in (gpu or ""):
            return rate
    return DEFAULT_SEC_PER_STEP


def read_timing(path, key):
    """(sec_per_step, overhead_s) recorded for this GPU and configuration, or None."""
    if path is None or not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text()).get(key)
        return (float(entry["sec_per_step"]), float(entry["overhead_s"]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        print(f"warning: ignoring timing file {path} ({type(e).__name__}: {e})")
        return None


def write_timing(path, key, sec_per_step, overhead_s):
    if path is None:
        return
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[key] = {"sec_per_step": round(sec_per_step, 3), "overhead_s": round(overhead_s, 2)}
    try:
        path.write_text(json.dumps(data, indent=1) + "\n")
    except OSError as e:
        print(f"warning: could not save timing to {path}: {e}")


class JobQueue:
    def __init__(self, pipe, settings, out_dir=DEFAULT_OUT_DIR, default_steps=DEFAULT_STEPS,
                 timing_path=TIMING_PATH):
        self.pipe = pipe
        self.default_steps = default_steps
        self.out_dir = Path(out_dir)
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.running: str | None = None
        self.timing_path, self.timing_key = timing_path, timing_key(settings)
        saved = read_timing(timing_path, self.timing_key)
        if saved:
            self.sec_per_step, self.overhead_s = saved
            self.eta_source = "measured"
        else:
            self.sec_per_step, self.overhead_s = sec_per_step_prior(settings.get("gpu")), JOB_OVERHEAD_S
            self.eta_source = "prior"
        print(f"ETA model: {self.sec_per_step:.2f} s/step + {self.overhead_s:.1f} s/job ({self.eta_source})")
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
        """Estimated seconds of GPU time this job still needs.

        A running job that has finished two or more steps is timed from its own steps, so the
        estimate is right within a few steps even when the model's rate is still a prior."""
        if job.status != "running":
            return job.steps * self.sec_per_step * size_scale(job) + self.overhead_s
        if job.step >= 2 and job.first_step_at and job.last_step_at:
            rate = (job.last_step_at - job.first_step_at) / (job.step - 1)
            # half the overhead is decoding, which is still to come
            left = (job.steps - job.step) * rate + self.overhead_s / 2 - (now - job.last_step_at)
        else:
            left = job.steps * self.sec_per_step * size_scale(job) + self.overhead_s - (now - job.started)
        return max(1.0, left)

    def record(self, job, finished):
        """Refine the ETA model from a completed job: the step rate from the step timestamps, the
        overhead from what is left of the wall time."""
        if job.steps < 2 or not job.first_step_at or not job.last_step_at:
            return
        rate = max(MIN_SEC_PER_STEP, (job.last_step_at - job.first_step_at) / (job.steps - 1))
        overhead = max(0.0, finished - job.started - rate * job.steps)
        rate /= size_scale(job)     # normalise to the 2K reference size
        if self.eta_source == "measured":
            w = MEASUREMENT_WEIGHT
            self.sec_per_step = (1 - w) * self.sec_per_step + w * rate
            self.overhead_s = (1 - w) * self.overhead_s + w * overhead
        else:
            self.sec_per_step, self.overhead_s, self.eta_source = rate, overhead, "measured"
        write_timing(self.timing_path, self.timing_key, self.sec_per_step, self.overhead_s)

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
        d = {k: v for k, v in asdict(job).items() if k not in ("images", "cancel_requested", "path", "first_step_at", "last_step_at")}
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
            # started first: remaining_s() reads it as soon as the status says "running".
            job.started = time.time()
            job.status, self.running = "running", job.id

            def on_step(pipe, i, t, kwargs):
                job.last_step_at = time.time()
                if i == 0:
                    job.first_step_at = job.last_step_at
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
                    self.record(job, time.time())
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
        check_fits(width, height, len(images), jq.settings)
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
        s = dict(jq.settings)
        for k in ("resident_gib", "total_gib"):
            if s.get(k) is not None:
                s[k] = round(s[k], 2)
        if s.get("total_gib"):
            # how many 2K input images an edit can carry right now, which is what an agent
            # actually needs to know before composing one
            capacity = s["total_gib"] * VRAM_USABLE_FRACTION - s["resident_gib"]
            n = 0
            while working_set_gib(2048, 2048, n + 1) <= capacity:
                n += 1
            s["max_edit_images_2k"] = n
        return {"status": "ok", "queued": queued, "running": jq.running,
                "sec_per_step": round(jq.sec_per_step, 2), "job_overhead_s": round(jq.overhead_s, 1),
                "eta_source": jq.eta_source, "default_steps": jq.default_steps, **s}

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

    # What the loaded pipeline actually costs, so submissions can be checked against the real card
    # rather than against the table's estimate. Offloaded weights are not resident, so skip it.
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    resident_gib = total_gib = None
    if not args.cpu_offload:
        torch.cuda.synchronize()
        resident_gib = torch.cuda.memory_allocated() / 2**30
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        headroom = total_gib * VRAM_USABLE_FRACTION - resident_gib
        print(f"resident weights {resident_gib:.2f} GiB of {total_gib:.2f} GiB; "
              f"{headroom:.2f} GiB left for a job's working set")

    jq = JobQueue(pipe, {"weights": args.weights, "quant": args.quant, "te_quant": args.te_quant,
                         "cpu_offload": args.cpu_offload, "gpu": gpu, "resident_gib": resident_gib,
                         "total_gib": total_gib}, out_dir, args.steps)
    uvicorn.run(create_app(jq), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
