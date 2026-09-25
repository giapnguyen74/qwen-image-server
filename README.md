# qwen-image-server

HTTP job server for Qwen-Image-2.1 text-to-image and image editing in plain
[diffusers](https://github.com/huggingface/diffusers) (no ComfyUI), written for a 24 GB RTX 3090 and sized to
whatever card it finds: `setup.sh` probes the GPU and records a profile, and the server refuses jobs that will not
fit. It loads the pipeline once, queues JSON requests, renders them one at a time on the GPU, and serves the PNGs
by job id.

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
./setup.sh gguf --quant auto       # profile this GPU can run, sized for --max-images edits
./setup.sh gguf --quant Q4_K_M     # another quant; repeat --quant for several, or --quant all (~40 GB of GGUFs)
./setup.sh original                # official transformer + text encoder + configs/VAE (~33 GB)
./setup.sh all                     # both
./setup.sh original --dry-run      # list what would be downloaded
./setup.sh gguf --no-sync          # skip uv sync
./setup.sh gguf --no-verify        # record the profile without test-loading the checkpoints
./setup.sh gguf --quant auto --max-images 8   # size the auto profile for 8-input edits (default 4)
```

Files already cached are skipped, and `HF_HOME` / `HF_HUB_CACHE` / `HF_TOKEN` are respected. When the upstream repo
has a new commit, `hf download` moves the cache's `main` to it and relinks only the files you asked for. Rerun with
the quants you use (`--quant all`); unchanged files are relinked without downloading.

`--quant auto` reads the card's total VRAM and picks a whole profile - transformer quant, text encoder quant and
offload. It does not just fit the weights: it reserves room for an edit carrying `--max-images` input images
(default 4), using the measured working set below. A bigger transformer therefore costs input images, and asking
for more images buys a smaller one:

Rows are what `nvidia-smi` reports, which is a little under the card's advertised size (a 32 GB card reports
31 GiB):

| Reported | `--max-images 0` | `1` | `2` | `4` (default) | `8` |
|---|---|---|---|---|---|
| 12 GiB | Q4_0/4bit+offload | Q4_0/4bit+offload | Q4_0/4bit+offload | Q4_0/4bit+offload | Q4_0/4bit+offload |
| 16 GiB | Q8_0/4bit | Q5_K_M/4bit | Q4_0/4bit+offload | Q4_0/4bit+offload | Q4_0/4bit+offload |
| 24 GiB (3090) | Q8_0/8bit | Q8_0/8bit | Q6_K/8bit | Q5_K_M/4bit | Q4_0/4bit+offload |
| 31 GiB (5090) | BF16/8bit | BF16/8bit | Q8_0/8bit | Q8_0/8bit | Q4_0/4bit+offload |
| 47 GiB | BF16/8bit | BF16/8bit | BF16/8bit | BF16/8bit | Q8_0/8bit |

So a 24 GB 3090 keeps the built-in `Q8_0` only for text-to-image and single-image edits; reserving room for four
inputs costs it `Q5_K_M` with a 4-bit text encoder. Pick `--max-images` for the work you actually do: an edit with
more inputs than the profile reserved for is rejected at submit time, not silently run. Without `nvidia-smi` it
falls back to `Q8_0`.

### The profile

After downloading, `setup.sh` **test-loads the checkpoints** (`qwen_image_dryrun.py --no-generate`) and only
writes `qwen_image_profile.json` if they actually load, so a recorded profile is one this machine has run. The
server and the dry run read it, so **whatever setup.sh downloaded is what gets loaded**:

```bash
./setup.sh gguf --quant auto                  # 32 GB card, room for 4 inputs -> Q8_0
./setup.sh gguf --quant auto --max-images 1   # same card, single-image edits -> BF16
uv run qwen_image_server.py                   # loads whichever was recorded: no flags needed
```

```json
{"weights": "gguf", "quant": "Q8_0", "te_quant": "8bit", "cpu_offload": false,
 "gpu": "NVIDIA GeForce RTX 5090", "vram_gib": 31, "chosen_by": "auto", "written": "..."}
```

Precedence is **command line > profile > built-in defaults**, so any flag still overrides what was recorded.
The profile is honoured as written even on a different GPU than the one it names; `warn_if_tight()` reports when
that will not fit. If it names a checkpoint that is not in the cache the server stops and prints the exact
`./setup.sh` line to run, rather than silently loading something else. An unreadable profile is reported and
ignored. `--quant all`, and `./setup.sh all`, write no profile because the choice would be ambiguous; pass the
flags yourself. The file is per-machine and is gitignored.

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
| `--weights` | profile, else `gguf` | `gguf` or `original` |
| `--quant` | profile, else `Q8_0` / `8bit` | gguf: `BF16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q4_0`. original: `8bit`, `4bit`, `bf16` |
| `--te-quant` | profile, else `8bit` (`4bit` with `--cpu-offload`) | `8bit`, `4bit`, `none` (bf16) |
| `--steps` | `40` | Denoising steps for requests without `steps` (1–100) |
| `--output-dir` | `outputs/server` | Where finished PNGs are saved as `<id>.png`; created if missing |
| `--cpu-offload` | off | Keep components in system RAM and move each to the GPU only while it runs |

The server keeps the text encoder loaded between jobs. With GGUF Q8_0 at 2K the weights are 17.0 GiB resident,
text-to-image peaks at 19.3 GiB and a one-image edit at 21.9 GiB; see [VRAM admission](#vram-admission) for the
rest. Speed is per card: about 7 s/step on an RTX 3090, and 2.4 s/step measured on an RTX 5090 (a 40-step 2K image
in 97 s). The `original` bf16 transformer is 13.3 GiB, too big to share a 24 GB card with the text encoder, so
there it needs `--cpu-offload`. The 8-bit option is about the size of Q8_0. bitsandbytes keeps 8-bit weights on
the GPU when offloading, so use `4bit` or `bf16` with `--cpu-offload`. The `original` memory and speed numbers
have not been measured.

## API

| Endpoint | |
|---|---|
| `POST /jobs` | Submit; returns `202` at once with `id`, `status: "queued"`, queue `position` and `eta_s` |
| `GET /jobs/{id}` | `queued` / `running` / `done` / `failed` / `cancelled`, with `eta_s`, `step`, `progress` and `image_url` when done |
| `GET /jobs/{id}/image` | The PNG (`409` until done) |
| `DELETE /jobs/{id}` | Cancel a queued job, or stop a running one at its next step |
| `GET /jobs`, `GET /health` | All jobs newest first; weights, quant settings, `default_steps`, queue length, measured `sec_per_step`, and the VRAM budget (`resident_gib`, `total_gib`, `max_edit_images_2k`) |

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

- `prompt` is required. `images` (base64 or `data:` URIs) makes it an edit, referred to as `<image1>`,
  `<image2>`, … in order; without images it is text-to-image. Ten is the protocol maximum, but the real limit is
  VRAM: `GET /health` reports `max_edit_images_2k` for this server, and a job that cannot fit is rejected at
  submit time with `422` rather than failing on the GPU minutes later.
- Size: `wh_ratio` (any `W:H` up to 8:1 or 1:8), or for edits `ratio_follow` (default `<image1>`), which keeps
  that input's aspect ratio, clamped to the same limit. Setting both is rejected, as are ratios that are zero,
  negative, non-finite or more extreme than 8:1. Text-to-image defaults to `1:1`. Model card sizes are used for 1:1 (2048×2048),
  4:3 / 3:4 (2400×1792), 3:2 / 2:3 (2528×1696) and 16:9 / 9:16 (2752×1536); other ratios get the same pixel count as
  2048×2048, rounded to multiples of 32.
- `seed` is random when omitted and is returned in the job status. `steps` is 1–100; when omitted, the server's `--steps` (default 40) is used.

```bash
curl -s -XPOST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"prompt": "a lighthouse at dusk", "wh_ratio": "3:2"}'           # -> {"id": "e3969fd18827", ...}
curl -s localhost:8000/jobs/e3969fd18827                                  # poll
curl -s -o out.png localhost:8000/jobs/e3969fd18827/image                 # download when done
```

### VRAM admission

The resident weights never move, so what a job costs is its working set, and that is predictable. Measured at
2048×2048 with GGUF Q8_0 and an 8-bit text encoder (17.03 GiB resident on a 31.4 GiB card):

| Input images | Peak | Working set |
|---|---|---|
| 0 (text-to-image) | 19.34 GiB | 2.31 GiB |
| 1 | 21.93 GiB | 4.89 GiB |
| 2 | 24.51 GiB | 7.48 GiB |
| 4 | 29.68 GiB | 12.65 GiB |
| 8 | out of memory | — |

That is 2.31 GiB plus about 2.59 GiB per 2K input image, scaled by output area and independent of step count.
`working_set_gib()` models it, and `POST /jobs` refuses anything that will not fit, naming how many images do.
Every model card size is about the same 2048×2048 pixel count, so changing `wh_ratio` does not change the cost:
send fewer images, or use `--cpu-offload` or a smaller `--quant`.

`eta_s` counts the running job and everything queued ahead. It uses a per-step time plus a per-job overhead for
encoding and decoding, both measured from finished jobs and saved per GPU and weight configuration in
`qwen_image_timing.json` next to the server, so a restart keeps them. Before the first measurement the per-step
time is a prior from the GPU name (2.4 s on an RTX 5090, 6.8 s on an RTX 3090 or an unknown card) and `GET /health`
reports `eta_source: "prior"`; a running job is timed from its own steps after step 2, so its `eta_s` is right within
a few steps either way. Jobs are kept in memory and lost on restart. Images are saved to `<output-dir>/<id>.png` with the prompt, seed, size, weights and quant settings in the
PNG text metadata.

## Agent skills

`skills/` holds `qwen-image-t2i`, `qwen-image-edit` and `qwen-image-character` in the standard `SKILL.md` format. Each
has its prompt template in `references/` and a standard-library client in `scripts/qwen_image.py` (submit, `status`,
`wait`, `download`, `cancel`). See [skills/README.md](skills/README.md).

## Workarounds in `qwen_image_common.py`

- **BF16 tensors in the GGUF.** diffusers' GGUF loader treats only F32/F16 as unquantized, so the norm layers get raw
  bytes and fail with a shape mismatch. `fix_unquantized_gguf_params()` decodes those weights after loading.
- **VAE decode at 2K.** A full-frame decode costs about 6.26 GiB per megapixel of output (measured: 6.56 GiB at
  1024x1024, 14.76 at 1536x1536, out of memory at 2048x2048 with nothing but the VAE loaded), while a tiled decode
  costs 0.43-0.47 GiB at any size. Every size this server produces is about 2048x2048 pixels, so `generate()`
  always ends up tiling; it checks per job against the VRAM free at that moment rather than assuming.
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
| `qwen_image_profile.json` | Written by `setup.sh`, read by the server; per-machine, gitignored |
| `skills/` | Agent skills and the job-server client |

## License

Qwen-Image-2.1 and derived weights are under the
[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), not a standard open-source
license. Read it before any commercial use.
