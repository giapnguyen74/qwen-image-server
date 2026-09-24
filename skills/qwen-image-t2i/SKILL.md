---
name: qwen-image-t2i
description: Generate an image from a text request with Qwen-Image-2.1 on the local GPU job server. Use when the user asks to create, draw, render or generate a new picture, photo, poster, illustration, infographic or logo from a description (no input image). Expands the request into a full observational prompt and aspect ratio first, then submits it with the bundled client script.
---

# Qwen-Image text-to-image

Turns a short brief into a detailed prompt, renders it on the Qwen-Image-2.1 job server (`qwen_image_server.py`),
and saves the PNG.

The client is [scripts/qwen_image.py](scripts/qwen_image.py): Python standard library only, run with `python3`
from a shell. Below, `QI` stands for `python3 <this skill's folder>/scripts/qwen_image.py`. Every command prints
one JSON object. Server URL: `QWEN_IMAGE_SERVER` (default `http://127.0.0.1:8000`). PNGs are saved to
`QWEN_IMAGE_OUTPUT_DIR` (default `./qwen_images`) unless `--out` is given.

## Workflow

1. **Check the server:** `QI health`. If it returns an `error`, tell the user to start `qwen_image_server.py` on
   the GPU machine; do not retry in a loop.
2. **Rewrite the brief yourself.** Read [references/prompt_rewriter.md](references/prompt_rewriter.md) and follow
   it as your own instructions for this step, applied to the user's request. Produce exactly the JSON it
   specifies: `{"rewritten_prompt": "...", "wh_ratio": "..."}`.
   - Keep every piece of text the user wants shown, character for character, inside double quotes.
   - If the user gave a ratio or a size, it overrides the template's default choice.
3. **Submit:** write that JSON to a file and pipe it in, which avoids shell quoting problems:

   ```bash
   QI t2i --json < rewrite.json
   ```

   Add `--seed N` only when the user wants a reproducible or repeated result, and `--steps N` only when they ask
   for quality or speed (40 default; 50 is slightly crisper and 25% slower; below 25 is noticeably rough).
   The reply has the job `id`, its queue `position`, and `eta_s`, the estimated seconds until it is ready. Tell
   the user the estimate when it is more than a few minutes.
4. **Wait:** `QI wait <id>` polls for up to 100 s (`--timeout` to change it; keep it below your shell tool's
   timeout). While `status` is still `queued` or `running`, run it again. When `status` is `done`, the PNG has
   been saved and `saved_to` has its path. `QI status <id>` checks progress without waiting, and
   `QI download <id> --out file.png` fetches a finished image again.
5. **Check and report.** Open the saved PNG with your image-viewing tool and compare it with what was asked. Report
   the path, the seed, and any clear deviations (missing element, wrong text, wrong count) rather than claiming it
   matches.

To make variations, resubmit the same JSON with different seeds. To reproduce an image, reuse its prompt, ratio,
steps and seed. `QI cancel <id>` stops a job, and `QI list` shows all jobs.

## What works with this model (from testing)

- **Photorealism** comes from photographic wording: lens, aperture, natural light, skin texture, film grain. Words
  like "very beautiful", "porcelain skin", "flawless" or anime/donghua style words push faces toward a smooth
  beauty-filter look.
- **Text rendering is literal and strong:** anything in double quotes is drawn. Large headlines render cleanly;
  many small labels come out misspelled. Never put an instruction where a text value would go, because it will be
  drawn onto the image.
- **Everything in the prompt gets drawn:** requirement lists and section headings can appear as panels, so the
  prompt should only describe what is visible, which is what the rewriting template does.
- **Known artifact:** faint purple vertical specks sometimes appear; more steps do not remove them.
- **Timing:** about 6–7 s per step on the RTX 3090 (a 40-step 2K image takes about 4–5 minutes), plus any jobs
  queued ahead.

## Sizes

`wh_ratio` accepts any `W:H`. The model card's sizes are 1:1 = 2048×2048, 4:3 = 2400×1792, 3:2 = 2528×1696,
16:9 = 2752×1536 (and their portrait versions). Other ratios render at the same pixel count as 2048×2048.
