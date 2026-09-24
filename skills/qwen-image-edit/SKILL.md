---
name: qwen-image-edit
description: Edit or compose images with Qwen-Image-2.1 on the local GPU job server, using 1-10 input images. Use when the user supplies an image and asks to change it (swap clothes, recolor, replace background, add or remove objects, edit text), to put a subject into a new scene, to combine several images, or to derive new views of a subject (three-view turnaround, multi-panel grid, expressions). Rewrites the instruction into a precise edit directive first, then submits it with the bundled client script.
---

# Qwen-Image editing

Turns a vague edit instruction plus input images into a precise directive, runs it on the Qwen-Image-2.1 job
server (`qwen_image_server.py`), and saves the PNG.

The client is [scripts/qwen_image.py](scripts/qwen_image.py): Python standard library only, run with `python3`
from a shell. Below, `QI` stands for `python3 <this skill's folder>/scripts/qwen_image.py`. Every command prints
one JSON object. Server URL: `QWEN_IMAGE_SERVER` (default `http://127.0.0.1:8000`). PNGs are saved to
`QWEN_IMAGE_OUTPUT_DIR` (default `./qwen_images`) unless `--out` is given. Input images are local paths; the
script uploads them.

## Workflow

1. **Check the server:** `QI health`. If it returns an `error`, tell the user to start `qwen_image_server.py` on
   the GPU machine; do not retry in a loop.
2. **Look at every input image** before writing anything. The directive must be anchored on what is actually
   visible, including a complete reading of any text in the images. Decide the order of the images now: it is the
   order of the `-i` flags, and it defines `<image1>`, `<image2>`, …
3. **Rewrite the instruction yourself.** Read
   [references/edit_prompt_enhancer.md](references/edit_prompt_enhancer.md) and follow it as your own instructions
   for this step. Produce exactly the JSON it specifies:
   `{"rewritten_prompt": "...", "wh_ratio": "...", "ratio_follow": "..."}`.
   - One input image: no `<imageN>` tags, and refer to it as "the image". Two or more: tags are mandatory.
   - The two language decisions (prose language vs. rendered-text language) and the size rules in the template
     are strict; apply them exactly.
4. **Submit:** write the JSON to a file and pipe it in, passing the images in tag order. `wh_ratio` /
   `ratio_follow` are read from the JSON.

   ```bash
   QI edit -i first.png -i second.png --json < rewrite.json
   ```

   Add `--seed N` / `--steps N` only when the user asks for reproducibility or quality changes. The reply has the
   job `id`, queue `position`, and `eta_s`, the estimated seconds until it is ready.
5. **Wait:** `QI wait <id>` polls for up to 100 s (`--timeout` to change it; keep it below your shell tool's
   timeout). Run it again while `status` is `queued` or `running`. When `status` is `done`, `saved_to` has the PNG
   path. `QI status <id>` checks without waiting, and `QI download <id> --out file.png` fetches it again.
6. **Check and report.** Open the result next to the input: the requested change should be strong and
   unmistakable, and nothing else should have changed. Report the path and seed, and name any leakage (changed
   face, dropped accessory, reframing) or under-editing honestly.

## What works with this model (from testing)

- **Local edits hold well:** recoloring a uniform blouse or changing hair color kept the face, pose, outfit and whole
  street intact. The framing can shift slightly.
- **New views of a subject:** a three-view turnaround and a 2×2 expression grid kept identity and outfit consistent
  across panels. Use the template's grid rules for the ratio: three standing figures in a row is `1:1`, not `3:1`.
- **Preserve by naming, not describing:** "keep her face, pose and the street unchanged" works better than
  re-describing them. Detailed descriptions of kept content make the model regenerate it.
- **Memory:** edits keep the text encoder loaded. One input image at 2K peaks around 22 GiB of 24. Many large inputs
  may fail with a CUDA out-of-memory error; the job then reports `failed` with that message. Retry with fewer
  inputs or a smaller output ratio, or ask the user to restart the server with `--cpu-offload`.
- **Output size:** `ratio_follow` keeps the input's aspect ratio at a 2K-level size; `wh_ratio` takes any `W:H`.
- **Timing:** about 7 s per step (a 40-step edit takes about 4–5 minutes), plus any jobs queued ahead.
