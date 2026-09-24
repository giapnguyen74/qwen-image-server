#!/usr/bin/env python3
"""Client for the Qwen-Image-2.1 job server (qwen_image_server.py). Python standard library only.

Submit (prompt on stdin; prints one JSON line with the job id):
  qwen_image.py t2i  --ratio 2:3 < prompt.txt
  qwen_image.py t2i  --json < rewrite.json            # {"rewritten_prompt", "wh_ratio"} from the t2i template
  qwen_image.py edit -i a.png -i b.png --ratio-follow "<image1>" < prompt.txt
  qwen_image.py edit -i a.png --json < rewrite.json   # {"rewritten_prompt", "wh_ratio", "ratio_follow"}
  add --wait to block until done (subject to --timeout) and save the PNG

The server is asynchronous: submitting returns a job id at once, with "position" in the queue and
"eta_s", the estimated seconds until the image is ready (including jobs queued ahead).

Follow up:
  qwen_image.py status JOB_ID                                  # status, progress, eta_s
  qwen_image.py wait JOB_ID [--timeout 100] [--out file.png]   # poll until done, then save the PNG
  qwen_image.py download JOB_ID [--out file.png]               # save the PNG of a finished job now
  qwen_image.py cancel JOB_ID | list | health

Every command prints a single JSON object on stdout. Exit status: 0 = ok (including "still queued/
running" after a wait timeout; check "status"), 1 = error, failed or cancelled job.

Environment: QWEN_IMAGE_SERVER (default http://127.0.0.1:8000),
             QWEN_IMAGE_OUTPUT_DIR (default ./qwen_images) for saved PNGs.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = os.environ.get("QWEN_IMAGE_SERVER", "http://127.0.0.1:8000").rstrip("/")
OUTPUT_DIR = Path(os.environ.get("QWEN_IMAGE_OUTPUT_DIR", "qwen_images"))
SUMMARY_KEYS = ("id", "mode", "status", "position", "eta_s", "step", "steps", "progress", "width", "height",
                "size_from", "seed", "error")


class ClientError(Exception):
    pass


def request(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(SERVER + path, data=data, method=method,
                                 headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except ValueError:
            detail = e.reason
        raise ClientError(f"{method} {path} -> {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise ClientError(f"cannot reach the job server at {SERVER} ({e.reason}); is qwen_image_server.py running?")
    return payload if raw else json.loads(payload)


def summary(job):
    return {k: job[k] for k in SUMMARY_KEYS if job.get(k) is not None}


def read_prompt(args):
    """Prompt and size fields from stdin: plain text, or the rewriting template's JSON with --json."""
    text = sys.stdin.read().strip()
    if not text:
        raise ClientError("empty prompt on stdin")
    if not args.json:
        return text, {}
    try:
        obj = json.loads(text)
    except ValueError as e:
        raise ClientError(f"--json: stdin is not valid JSON ({e})")
    prompt = obj.get("rewritten_prompt") or obj.get("prompt")
    if not prompt:
        raise ClientError("--json: no rewritten_prompt in the JSON")
    return prompt, {k: obj[k] for k in ("wh_ratio", "ratio_follow") if obj.get(k)}


def submit(args):
    prompt, from_json = read_prompt(args)
    body = {"prompt": prompt, "seed": args.seed}
    if args.steps is not None:  # omitted: the server applies its --steps default
        body["steps"] = args.steps
    # explicit flags win over the template JSON
    body["wh_ratio"] = args.ratio or from_json.get("wh_ratio", "")
    if args.cmd == "edit":
        images = []
        for p in args.image:
            path = Path(p).expanduser()
            if not path.is_file():
                raise ClientError(f"input image not found: {p}")
            images.append(base64.b64encode(path.read_bytes()).decode())
        body["images"] = images
        body["ratio_follow"] = args.ratio_follow or ("" if args.ratio else from_json.get("ratio_follow", ""))
    job = request("POST", "/jobs", body)
    if args.wait:
        return wait(job["id"], args.timeout, args.out)
    return summary(job)


def wait(job_id, timeout, out):
    deadline = time.monotonic() + max(1, timeout)
    while True:
        job = request("GET", f"/jobs/{job_id}")
        if job["status"] in ("done", "failed", "cancelled") or time.monotonic() >= deadline:
            break
        time.sleep(3)
    result = summary(job)
    if job["status"] == "done":
        result["saved_to"] = download(job_id, out)
    return result


def download(job_id, out):
    """Save a finished job's PNG; the server answers 409 while it is not done."""
    path = Path(out).expanduser() if out else OUTPUT_DIR / f"{job_id}.png"
    png = request("GET", f"/jobs/{job_id}/image", raw=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return str(path.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_wait_opts(p):
        p.add_argument("--timeout", type=int, default=100, help="seconds to wait before returning (default 100)")
        p.add_argument("--out", default="", help="PNG path (default $QWEN_IMAGE_OUTPUT_DIR/<id>.png)")

    for name in ("t2i", "edit"):
        p = sub.add_parser(name, help=f"submit a {'text-to-image' if name == 't2i' else 'edit'} job; prompt on stdin")
        if name == "edit":
            p.add_argument("-i", "--image", action="append", required=True,
                           help="input image path; repeat for <image2>, <image3>, ...")
            p.add_argument("--ratio-follow", default="", help='"<imageN>": keep that input\'s aspect ratio')
        p.add_argument("--ratio", default="", help="output W:H" + (" (default 1:1)" if name == "t2i" else ""))
        p.add_argument("--json", action="store_true", help="stdin is the rewriting template's JSON output")
        p.add_argument("--steps", type=int, default=None, help="default: the server's --steps (40 unless changed)")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--wait", action="store_true", help="wait for the result (up to --timeout)")
        add_wait_opts(p)
    p = sub.add_parser("wait", help="wait for a job and save its PNG when done")
    p.add_argument("job_id")
    add_wait_opts(p)
    p = sub.add_parser("download", help="save the PNG of a finished job")
    p.add_argument("job_id")
    p.add_argument("--out", default="", help="PNG path (default $QWEN_IMAGE_OUTPUT_DIR/<id>.png)")
    for name in ("status", "cancel"):
        sub.add_parser(name).add_argument("job_id")
    sub.add_parser("list", help="all jobs, newest first")
    sub.add_parser("health", help="server status and model settings")
    args = ap.parse_args()

    try:
        if args.cmd in ("t2i", "edit"):
            result = submit(args)
        elif args.cmd == "wait":
            result = wait(args.job_id, args.timeout, args.out)
        elif args.cmd == "download":
            result = {"id": args.job_id, "saved_to": download(args.job_id, args.out)}
        elif args.cmd == "status":
            result = summary(request("GET", f"/jobs/{args.job_id}"))
        elif args.cmd == "cancel":
            result = summary(request("DELETE", f"/jobs/{args.job_id}"))
        elif args.cmd == "list":
            result = [summary(j) for j in request("GET", "/jobs")]
        else:
            result = request("GET", "/health")
    except ClientError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 1 if isinstance(result, dict) and result.get("status") in ("failed", "cancelled") else 0


if __name__ == "__main__":
    sys.exit(main())
