# qwen-image-server

HTTP job server for Qwen-Image-2.1 text-to-image and image editing in plain
[diffusers](https://github.com/huggingface/diffusers) (no ComfyUI), sized for a 24 GB RTX 3090. It loads the pipeline
once, queues JSON requests, renders them one at a time on the GPU, and serves the PNGs by job id.

Two weight sets:

| `--weights` | Transformer | Text encoder |
|---|---|---|
| `gguf` (default) | [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF) `qwen-image-2.1-UC-<quant>.gguf`, kept quantized | abenzerps `text_encoders/qwen3vl_8b_bf16.safetensors` (ComfyUI key layout, renamed on load) |
| `original` | official [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) `transformer/` (bf16, bitsandbytes 8/4-bit on load) | official `text_encoder/` |

Both use the official VAE, processor, scheduler and configs. The GGUF transformer is an uncensored fine-tune, so the
two sets do not give the same image for the same seed.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and the `hf` CLI (`uv tool install huggingface_hub`). The server never
downloads anything (`local_files_only=True`), so `setup.sh` runs `uv sync` and pulls the checkpoints into the Hugging
Face cache first:

```bash
./setup.sh gguf                    # Q8_0 GGUF + bf16 text encoder + official configs/VAE (~25 GB)
./setup.sh gguf --quant Q4_K_M     # another quant; repeat --quant for several, or --quant all (~40 GB of GGUFs)
./setup.sh original                # official transformer + text encoder + configs/VAE (~33 GB)
./setup.sh all                     # both
./setup.sh original --dry-run      # list what would be downloaded
./setup.sh gguf --no-sync          # skip uv sync
```

Files already cached are skipped, and `HF_HOME` / `HF_HUB_CACHE` / `HF_TOKEN` are respected. When the upstream repo
has a new commit, `hf download` moves the cache's `main` to it and relinks only the files you asked for. Rerun with
the quants you use (`--quant all`); unchanged files are relinked without downloading.

GGUF quants: `BF16` 14 GB, `Q8_0` 7.1 GB, `Q6_K` 5.5 GB, `Q5_K_M` 4.9 GB, `Q4_K_M` 4.3 GB, `Q4_0` 3.9 GB. The
abenzerps `qwen3vl_8b_int8_convrot.safetensors` text encoder uses ComfyUI's own int8 format and is not used.

Check that everything loads and render a 512×512, 4-step test to `outputs/dryrun.png`:

```bash
uv run qwen_image_dryrun.py                        # --no-generate to load only
uv run qwen_image_dryrun.py --weights original
```

## Running the server

```bash
uv run qwen_image_server.py                              # http://127.0.0.1:8000, docs at /docs
uv run qwen_image_server.py --quant Q4_K_M               # smaller GGUF
uv run qwen_image_server.py --weights original           # official weights, 8-bit transformer
uv run qwen_image_server.py --host 0.0.0.0 --port 8000   # reachable from other machines: there is no auth
uv run qwen_image_server.py --output-dir /data/qwen      # where finished PNGs go
uv run qwen_image_server.py --steps 50                   # default steps for requests that omit them
```

| Option | Default | |
|---|---|---|
| `--weights` | `gguf` | `gguf` or `original` |
| `--quant` | `Q8_0` / `8bit` | gguf: `BF16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q4_0`. original: `8bit`, `4bit`, `bf16` |
| `--te-quant` | `8bit` (`4bit` with `--cpu-offload`) | `8bit`, `4bit`, `none` (bf16) |
| `--steps` | `40` | Denoising steps for requests without `steps` (1–100) |
| `--output-dir` | `outputs/server` | Where finished PNGs are saved as `<id>.png`; created if missing |
| `--cpu-offload` | off | Keep components in system RAM and move each to the GPU only while it runs |

The server keeps the text encoder loaded between jobs. With GGUF Q8_0 at 2K, text-to-image peaks at 19.3 GiB and a
one-image edit at 21.9 GiB, both about 7 s/step. The `original` bf16 transformer is 13.3 GiB, too big to share the
GPU with the text encoder, so it needs `--cpu-offload`. The 8-bit option is about the size of Q8_0. bitsandbytes
keeps 8-bit weights on the GPU when offloading, so use `4bit` or `bf16` with `--cpu-offload`. The `original`
memory and speed numbers have not been measured on the 3090 yet.

## API

| Endpoint | |
|---|---|
| `POST /jobs` | Submit; returns `202` at once with `id`, `status: "queued"`, queue `position` and `eta_s` |
| `GET /jobs/{id}` | `queued` / `running` / `done` / `failed` / `cancelled`, with `eta_s`, `step`, `progress` and `image_url` when done |
| `GET /jobs/{id}/image` | The PNG (`409` until done) |
| `DELETE /jobs/{id}` | Cancel a queued job, or stop a running one at its next step |
| `GET /jobs`, `GET /health` | All jobs newest first; weights, quant settings, `default_steps`, queue length and measured `sec_per_step` |

```json
{
  "prompt": "...",
  "images": ["<base64 PNG/JPEG>"],
  "wh_ratio": "",
  "ratio_follow": "<image1>",
  "steps": 40,
  "seed": 1234
}
```

- `prompt` is required. `images` (1–10, base64 or `data:` URIs) makes it an edit, referred to as `<image1>`,
  `<image2>`, … in order; without images it is text-to-image.
- Size: `wh_ratio` (any `W:H`), or for edits `ratio_follow` (default `<image1>`), which keeps that input's aspect
  ratio. Setting both is rejected. Text-to-image defaults to `1:1`. Model card sizes are used for 1:1 (2048×2048),
  4:3 / 3:4 (2400×1792), 3:2 / 2:3 (2528×1696) and 16:9 / 9:16 (2752×1536); other ratios get the same pixel count as
  2048×2048, rounded to multiples of 32.
- `seed` is random when omitted and is returned in the job status. `steps` is 1–100; when omitted, the server's `--steps` (default 40) is used.

```bash
curl -s -XPOST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"prompt": "a lighthouse at dusk", "wh_ratio": "3:2"}'           # -> {"id": "e3969fd18827", ...}
curl -s localhost:8000/jobs/e3969fd18827                                  # poll
curl -s -o out.png localhost:8000/jobs/e3969fd18827/image                 # download when done
```

`eta_s` counts the running job and everything queued ahead. It uses a per-step time that starts at 6.8 s and is
updated from each finished job, plus about 8 s per job for encoding and decoding. Jobs are kept in memory and lost on
restart. Images are saved to `<output-dir>/<id>.png` with the prompt, seed, size, weights and quant settings in the
PNG text metadata.

## Agent skills

`skills/` holds `qwen-image-t2i`, `qwen-image-edit` and `qwen-image-character` in the standard `SKILL.md` format. Each
has its prompt template in `references/` and a standard-library client in `scripts/qwen_image.py` (submit, `status`,
`wait`, `download`, `cancel`). See [skills/README.md](skills/README.md).

## Workarounds in `qwen_image_common.py`

- **BF16 tensors in the GGUF.** diffusers' GGUF loader treats only F32/F16 as unquantized, so the norm layers get raw
  bytes and fail with a shape mismatch. `fix_unquantized_gguf_params()` decodes those weights after loading.
- **VAE decode at 2K.** Tiled decoding is always enabled.
- **8-bit models with offload.** bitsandbytes keeps int8 weights on the GPU when a model is moved to the CPU, so
  `--cpu-offload` defaults the text encoder to 4-bit and warns on 8-bit transformer or text encoder.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to reduce fragmentation.

## Files

| File | |
|---|---|
| `setup.sh` | `uv sync` + checkpoint download into the HF cache |
| `qwen_image_common.py` | Component loading for both weight sets, `build_pipeline()`, `generate()` |
| `qwen_image_server.py` | HTTP job server (queue, status, download) |
| `qwen_image_dryrun.py` | Load check and small test render |
| `skills/` | Agent skills and the job-server client |

## License

Qwen-Image-2.1 and derived weights are under the
[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), not a standard open-source
license. Read it before any commercial use.
