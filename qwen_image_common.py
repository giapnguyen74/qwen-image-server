"""Shared loading for Qwen-Image-2.1 in plain diffusers, using already-cached checkpoints only.

Two weight sets (--weights):
  gguf (default)
    transformer  <- abenzerps/Qwen-Image-2.1-Uncensored-GGUF  qwen-image-2.1-UC-<quant>.gguf (kept quantized)
    text_encoder <- abenzerps/Qwen-Image-2.1-Uncensored-GGUF  text_encoders/qwen3vl_8b_bf16.safetensors
                    (ComfyUI key layout, renamed on load; optionally bitsandbytes 8/4-bit)
  original
    transformer  <- Qwen/Qwen-Image-2.1/transformer  (bf16, optionally bitsandbytes 8/4-bit)
    text_encoder <- Qwen/Qwen-Image-2.1/text_encoder (bf16, optionally bitsandbytes 8/4-bit)
  Both: vae, processor, scheduler, configs <- Qwen/Qwen-Image-2.1

Nothing is downloaded: every file is resolved with local_files_only=True. Run ./setup.sh first.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub import hf_hub_download, snapshot_download

GGUF_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
OFFICIAL_REPO = "Qwen/Qwen-Image-2.1"
TE_FILE = "text_encoders/qwen3vl_8b_bf16.safetensors"
WEIGHTS = ["gguf", "original"]
QUANTS = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"]      # --weights gguf
ORIGINAL_QUANTS = ["8bit", "4bit", "bf16"]                         # --weights original (bitsandbytes)
TE_QUANTS = ["8bit", "4bit", "none"]

# Peak VRAM of a full-frame (untiled) VAE decode, per megapixel of output. Measured on an RTX 5090
# with this bf16 VAE: 6.56 GiB at 1024x1024, 14.76 at 1536x1536, 25.32 at 2752x1536, and out of
# memory at 2048x2048 with nothing but the VAE loaded. Tiled decode costs 0.43-0.47 GiB at every
# one of those sizes. So the model card's 2K sizes never decode full-frame, on any card sold today;
# only smaller outputs can, which is what set_vae_tiling() checks for.
VAE_DECODE_GIB_PER_MPX = 6.26
VAE_DECODE_HEADROOM = 1.25         # margin for fragmentation and anything else sharing the card

# Weights that stay resident on the GPU, GiB. The GGUF quants are their size on disk; Q8_0 was
# measured at 7.07 GiB allocated, matching. The original entries are bitsandbytes' quantized sizes.
RESIDENT_GIB = {
    ("gguf", "BF16"): 14.0, ("gguf", "Q8_0"): 7.1, ("gguf", "Q6_K"): 5.5,
    ("gguf", "Q5_K_M"): 4.9, ("gguf", "Q4_K_M"): 4.3, ("gguf", "Q4_0"): 3.9,
    ("original", "bf16"): 13.3, ("original", "8bit"): 7.5, ("original", "4bit"): 4.5,
}
# 8bit measured at 9.34 GiB allocated (16.41 total with a 7.07 GiB transformer), not the 8.5 the
# file size suggests: bitsandbytes keeps fp16 outlier columns alongside the int8 weights.
TE_RESIDENT_GIB = {"8bit": 9.3, "4bit": 5.0, "none": 17.5}
VAE_RESIDENT_GIB = 0.62            # measured

# VRAM needed on top of the resident weights while a job runs, measured at 2048x2048 with GGUF
# Q8_0 + an 8-bit text encoder: 2.31 GiB for text-to-image, then about 2.59 GiB per 2K input image
# (1 -> 4.89, 2 -> 7.48, 4 -> 12.65 GiB; 8 ran out of memory on a 31.4 GiB card). Peak does not
# depend on step count. Scaled by output area, since these were taken at the 2048x2048 pixel count.
WORKING_SET_BASE_GIB = 2.31
WORKING_SET_PER_IMAGE_GIB = 2.59
MEASURED_MPX = 2048 * 2048 / 1e6


def working_set_gib(width, height, n_images):
    """Estimated VRAM a job needs beyond the resident weights."""
    scale = (width * height / 1e6) / MEASURED_MPX
    return (WORKING_SET_BASE_GIB + WORKING_SET_PER_IMAGE_GIB * n_images) * scale

# Written by setup.sh next to this file, read by the server and the dry run so that whatever
# setup.sh downloaded is what gets loaded. Command line flags still win over it.
PROFILE_PATH = Path(__file__).resolve().parent / "qwen_image_profile.json"
PROFILE_FIELDS = ("weights", "quant", "te_quant", "cpu_offload")


def gib(n_bytes):
    return f"{n_bytes / 2**30:.2f} GiB"


def vram_budget():
    """(free, total) GiB on the current CUDA device.

    Free rather than total: another process may already hold part of the card, so the amount
    actually available is what the defaults should be chosen against.
    """
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def untiled_decode_fits(width, height, free_gib):
    """Whether a full-frame VAE decode at this size fits in the VRAM free right now.

    Returns (fits, GiB needed); (False, None) when the decode cost has not been measured, so an
    unmeasured build keeps the conservative tiled path rather than guessing.
    """
    if VAE_DECODE_GIB_PER_MPX is None:
        return False, None
    need = VAE_DECODE_GIB_PER_MPX * (width * height / 1e6) * VAE_DECODE_HEADROOM
    return need < free_gib, need


def set_vae_tiling(pipe, width, height, cpu_offload=False):
    """Tile the VAE decode only when a full-frame decode would not fit.

    The 24 GB card this project was written for never had the room at 2K, so tiling used to be
    unconditional. On a larger card a full-frame decode is faster and leaves no tile seams.
    """
    if cpu_offload:   # offload exists to save VRAM; don't spend it back on the decode
        pipe.vae.enable_tiling()
        return
    free_gib, _ = vram_budget()
    fits, need = untiled_decode_fits(width, height, free_gib)
    if fits:
        pipe.vae.disable_tiling()
        print(f"  vae: full-frame decode ({need:.1f} GiB needed, {free_gib:.1f} GiB free)")
    elif need is None:
        pipe.vae.enable_tiling()
        print(f"  vae: tiled decode (full-frame cost not measured on this build)")
    else:
        pipe.vae.enable_tiling()
        print(f"  vae: tiled decode ({need:.1f} GiB needed for full-frame, {free_gib:.1f} GiB free)")


def vram(label):
    torch.cuda.synchronize()
    print(f"  [vram] {label}: allocated {gib(torch.cuda.memory_allocated())}, "
          f"peak {gib(torch.cuda.max_memory_allocated())}")


def report_loading_info(name, info):
    problems = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    if not problems:
        print(f"  {name}: all weights matched")
        return True
    for k, v in problems.items():
        v = sorted(v) if not isinstance(v, list) else v
        print(f"  {name}: {k} ({len(v)}): {v[:5]}{' ...' if len(v) > 5 else ''}")
    return False


def cached(fetch, setup_mode):
    """Run a local_files_only hub lookup; point at ./setup.sh when the files aren't cached."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return fetch()
    except (LocalEntryNotFoundError, FileNotFoundError, OSError) as e:
        raise SystemExit(f"checkpoint not in the Hugging Face cache ({type(e).__name__}: {e}).\n"
                         f"Download it first: ./setup.sh {setup_mode}")


def official_dir(weights="gguf"):
    patterns = ["*.json", "*.jinja", "*.txt", "vae/*"]
    if weights == "original":
        patterns += ["transformer/*", "text_encoder/*"]
    return cached(lambda: snapshot_download(OFFICIAL_REPO, allow_patterns=patterns, local_files_only=True), weights)


def bnb_config(quant, lib):
    """bitsandbytes config for "8bit" / "4bit" (nf4), None for unquantized. lib: diffusers or transformers."""
    if quant == "8bit":
        return lib.BitsAndBytesConfig(load_in_8bit=True)
    if quant == "4bit":
        return lib.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    return None


def resolve_quant(weights, quant, cpu_offload=False, te_quant=None):
    """Default and validate the transformer quant for the chosen weight set.

    Warnings are measured against the VRAM actually free on this card rather than against the
    24 GB the project was originally written for.
    """
    choices = QUANTS if weights == "gguf" else ORIGINAL_QUANTS
    if quant is None:
        quant = "Q8_0" if weights == "gguf" else "8bit"
    elif quant not in choices:
        raise SystemExit(f"--quant {quant} is not valid with --weights {weights}; choose from {', '.join(choices)}")
    if weights == "original" and quant == "8bit" and cpu_offload:
        print("warning: an 8-bit transformer does not offload (bitsandbytes keeps its int8 weights on the GPU); "
              "use 4bit or bf16 with --cpu-offload")
    if not cpu_offload:
        warn_if_tight(weights, quant, te_quant or "8bit")
    return quant


def warn_if_tight(weights, quant, te_quant):
    """Warn when the resident weights leave too little of this card for the denoising pass."""
    free_gib = gpu_free_gib()
    if free_gib is None:                     # no CUDA yet: nothing to measure against
        return
    resident = (RESIDENT_GIB.get((weights, quant), 0) + TE_RESIDENT_GIB.get(te_quant, 0)
                + VAE_RESIDENT_GIB)
    if resident > free_gib * 0.85:
        print(f"warning: {weights}/{quant} plus a {te_quant} text encoder is about {resident:.1f} GiB "
              f"resident, and only {free_gib:.1f} GiB is free on this GPU. Expect out-of-memory at 2K; "
              f"consider --cpu-offload, a smaller --quant, or --te-quant 4bit")


def gpu_free_gib():
    """Free VRAM in GiB, or None when there is no usable CUDA device."""
    try:
        if not torch.cuda.is_available():
            return None
        return vram_budget()[0]
    except (RuntimeError, AssertionError):
        return None


def load_original_transformer(quant, official_dir):
    import diffusers
    from diffusers import QwenImage21Transformer2DModel

    print(f"transformer: official Qwen/Qwen-Image-2.1/transformer, quant={quant}")
    t0 = time.time()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        official_dir,
        subfolder="transformer",
        quantization_config=bnb_config(quant, diffusers),  # bitsandbytes quantizes while loading onto the GPU
        torch_dtype=torch.bfloat16,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")
    return transformer


def load_transformer(quant, official_dir):
    from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel

    path = cached(lambda: hf_hub_download(GGUF_REPO, f"qwen-image-2.1-UC-{quant}.gguf", local_files_only=True),
                  f"gguf --quant {quant}")
    print(f"transformer: {Path(path).name} ({gib(os.path.getsize(path))} on disk)")
    t0 = time.time()
    transformer = QwenImage21Transformer2DModel.from_single_file(
        path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=official_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"  loaded in {time.time() - t0:.1f}s, {n_params / 1e9:.2f}B stored elements")
    fix_unquantized_gguf_params(transformer)
    return transformer


def fix_unquantized_gguf_params(model):
    """Decode BF16 GGUF tensors held by non-Linear modules (norms).

    diffusers' GGUF loader only treats F32/F16 as unquantized, so BF16 tensors stay wrapped as raw
    bytes (shape doubled). GGUFLinear dequantizes on the fly, but RMSNorm-style modules use
    `self.weight` directly and fail with a shape mismatch. Linear weights are left untouched.
    """
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor

    fixed = []
    for mod_name, module in model.named_modules():
        if isinstance(module, (GGUFLinear, torch.nn.Linear)):
            continue
        for p_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, GGUFParameter):
                plain = dequantize_gguf_tensor(param).to(torch.bfloat16)
                setattr(module, p_name, torch.nn.Parameter(plain, requires_grad=False))
                fixed.append(f"{mod_name}.{p_name}")
    print(f"  decoded {len(fixed)} BF16 GGUF params in non-Linear modules (e.g. {fixed[:2]})")


def load_text_encoder(te_quant, official_dir, device="cuda", weights="gguf"):
    import transformers
    from transformers import Qwen3VLForConditionalGeneration

    quant_cfg = bnb_config(te_quant, transformers)
    loading = dict(
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        # bitsandbytes quantizes while loading onto the GPU; unquantized weights can start on the CPU.
        device_map="cuda" if quant_cfg is not None else device,
        output_loading_info=True,
    )

    if weights == "original":
        print(f"text_encoder: official Qwen/Qwen-Image-2.1/text_encoder, quant={te_quant}")
        t0 = time.time()
        text_encoder, info = Qwen3VLForConditionalGeneration.from_pretrained(
            official_dir, subfolder="text_encoder", **loading)
        print(f"  loaded in {time.time() - t0:.1f}s")
        return text_encoder, report_loading_info("text_encoder", info)

    te_path = cached(lambda: hf_hub_download(GGUF_REPO, TE_FILE, local_files_only=True), "gguf")
    print(f"text_encoder: {TE_FILE} ({gib(os.path.getsize(te_path))} on disk), quant={te_quant}")

    # from_pretrained wants a model directory, so point a temp dir at the official config and
    # the ComfyUI weights file via symlinks (no copy of the 17.5 GB file).
    tmp = Path(tempfile.mkdtemp(prefix="qwen_te_"))
    try:
        for name in ("config.json", "generation_config.json"):
            (tmp / name).symlink_to(Path(official_dir) / "text_encoder" / name)
        (tmp / "model.safetensors").symlink_to(te_path)

        t0 = time.time()
        text_encoder, info = Qwen3VLForConditionalGeneration.from_pretrained(
            tmp,
            # ComfyUI stores the language model as model.layers.* / model.embed_tokens.* / model.norm.*;
            # transformers expects model.language_model.*. The vision tower (model.visual.*) and lm_head match.
            key_mapping={r"^model\.(?!visual\.|language_model\.)": "model.language_model."},
            **loading,
        )
        print(f"  loaded in {time.time() - t0:.1f}s")
    finally:
        # The weights are on the GPU by now and these are only symlinks, so the shim can go.
        shutil.rmtree(tmp, ignore_errors=True)
    ok = report_loading_info("text_encoder", info)
    return text_encoder, ok


def build_pipeline(quant="Q8_0", te_quant="8bit", cpu_offload=False, weights="gguf"):
    """Load all components. Returns (pipe, text_encoder_ok).

    weights="gguf":     abenzerps GGUF transformer (quant in QUANTS) + ComfyUI-layout bf16 text encoder.
    weights="original": official Qwen/Qwen-Image-2.1 transformer and text encoder (quant in ORIGINAL_QUANTS).

    cpu_offload=False: everything stays on the GPU.
    cpu_offload=True:  components wait in system RAM and diffusers' enable_model_cpu_offload() moves
                       each one to the GPU only while it runs (text_encoder -> transformer -> vae).
                       Lower VRAM, slower per image, and needs the weights to fit in system RAM.
    """
    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    assert torch.cuda.is_available(), "CUDA not available"
    free_gib, total_gib = vram_budget()
    print(f"GPU: {torch.cuda.get_device_name(0)}, {total_gib:.1f} GiB total, {free_gib:.1f} GiB free, "
          f"torch {torch.__version__}")
    odir = official_dir(weights)

    device = "cpu" if cpu_offload else "cuda"

    if weights == "original":
        transformer = load_original_transformer(quant, odir)
        if quant == "bf16":  # bitsandbytes models are placed on the GPU while loading and can't be .to()'d
            transformer = transformer.to(device)
    else:
        transformer = load_transformer(quant, odir).to(device)
    vram("transformer")

    text_encoder, te_ok = load_text_encoder(te_quant, odir, device, weights)
    vram("+ text_encoder")

    vae = AutoencoderKLQwenImage21.from_pretrained(odir, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    # Tiling stays on as the safe default; generate() reconsiders it per job, once the output size
    # and the VRAM actually free at that moment are both known.
    vae.enable_tiling()
    print("vae: official Qwen/Qwen-Image-2.1/vae")
    vram("+ vae")

    processor = Qwen3VLProcessor.from_pretrained(odir, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(odir, subfolder="scheduler")

    pipe = QwenImage21Pipeline(transformer=transformer, text_encoder=text_encoder, vae=vae,
                               processor=processor, scheduler=scheduler)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
        vram("model cpu offload enabled")
    return pipe, te_ok


# Supported output sizes from the model card (width, height).
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# Widest ratio accepted for sizes outside the table. The widest the skills' templates ask for is
# 9:2 (4.5), so this leaves room without allowing a size no GPU can render.
MAX_ASPECT = 8.0


def add_model_args(ap):
    """Model options shared by the server and the dry run. Resolve with resolve_model_args()."""
    ap.add_argument("--weights", default=None, choices=WEIGHTS,
                    help="gguf: abenzerps Uncensored GGUF transformer; original: official Qwen/Qwen-Image-2.1 "
                         "weights (default: the profile's, else gguf)")
    ap.add_argument("--quant", default=None,
                    help=f"transformer quant. gguf: {', '.join(QUANTS)} (default Q8_0); "
                         f"original: {', '.join(ORIGINAL_QUANTS)} (default 8bit, bitsandbytes)")
    ap.add_argument("--te-quant", default=None, choices=TE_QUANTS,
                    help="text encoder quantization (default: 8bit, or 4bit with --cpu-offload)")
    ap.add_argument("--cpu-offload", action="store_true", default=None,
                    help="optional fallback, off by default (everything runs on the GPU): keep components in "
                         "system RAM and move each to the GPU only while it runs (diffusers "
                         "enable_model_cpu_offload). Slower; for edits with several input images or bf16 weights")


def read_profile(path=PROFILE_PATH):
    """The profile setup.sh wrote, or None. A broken file is reported and ignored, never fatal."""
    import json

    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"warning: ignoring unreadable profile {path} ({type(e).__name__}: {e})")
        return None
    if not isinstance(profile, dict):
        print(f"warning: ignoring profile {path}: expected a JSON object")
        return None
    return profile


def describe_profile(profile):
    where = profile.get("chosen_by", "?")
    gpu = profile.get("gpu")
    on = f" for {gpu}" if gpu else ""
    return f"profile: {profile.get('weights')}/{profile.get('quant')}, te {profile.get('te_quant')} ({where}{on})"


def resolve_model_args(args, profile_path=PROFILE_PATH):
    """Settle the model options, in this order of precedence:

    1. what was passed on the command line
    2. the profile setup.sh wrote (honoured as recorded, even on a different GPU)
    3. the built-in defaults

    The profile is what makes `uv run qwen_image_server.py` with no flags load whatever
    setup.sh downloaded, instead of falling back to Q8_0 and asking for a file that is not there.
    """
    profile = read_profile(profile_path)
    if profile:
        from_profile = [f for f in PROFILE_FIELDS
                        if getattr(args, f, None) is None and profile.get(f) is not None]
        for field in from_profile:
            setattr(args, field, profile[field])
        if from_profile:
            print(f"{describe_profile(profile)}; using its {', '.join(from_profile)}")

    if args.weights is None:
        args.weights = "gguf"
    if args.cpu_offload is None:
        args.cpu_offload = False
    if args.te_quant is None:
        args.te_quant = "4bit" if args.cpu_offload else "8bit"
    args.quant = resolve_quant(args.weights, args.quant, args.cpu_offload, args.te_quant)
    if args.cpu_offload and args.te_quant == "8bit":
        # bitsandbytes leaves the int8 weights (weight.CB, ~7 GiB) on the GPU when an 8-bit model is
        # moved to the CPU, so offloading an 8-bit text encoder frees almost nothing.
        print("warning: --te-quant 8bit does not offload (bitsandbytes keeps ~7 GiB of int8 weights on the GPU); "
              "use 4bit or none with --cpu-offload")
    return args


def generate(pipe, prompt, width, height, steps, seed, images=None, cpu_offload=False,
             keep_text_encoder=False, callback=None):
    """Run one generation. Returns (PIL image, seconds).

    keep_text_encoder: leave the text encoder loaded (a long-running server needs it for the next
    job); otherwise text-to-image frees it after encoding to make room at 2K sizes.
    callback: optional callback_on_step_end(pipe, step, timestep, kwargs) -> kwargs.
    """
    import gc

    if cpu_offload or images or keep_text_encoder:
        # With offload, the hooks move the text encoder off the GPU before the transformer runs.
        # With condition images, the pipeline re-encodes prompt + images itself, so the text
        # encoder has to stay loaded.
        prompt_kwargs = {"prompt": prompt, "image": images or None}
    else:
        # Encode the prompt, then drop the text encoder: at 2K sizes the denoiser needs that VRAM.
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(prompt, device="cuda")
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        vram("after freeing text encoder")
        prompt_kwargs = {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask}

    n_img = f", {len(images)} reference image(s)" if images else ""
    print(f"generating {width}x{height}, {steps} steps, seed {seed}{n_img}")
    set_vae_tiling(pipe, width, height, cpu_offload)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    image = pipe(
        **prompt_kwargs,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        callback_on_step_end=callback,
    ).images[0]
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt / steps:.2f}s/step)")
    vram("generation")
    return image, dt


def save_png(image, out, metadata):
    """Save with generation settings in the PNG text chunks."""
    from PIL.PngImagePlugin import PngInfo

    meta = PngInfo()
    for k, v in metadata.items():
        meta.add_text(k, str(v))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, pnginfo=meta)
    print(f"  saved -> {out}")


def size_for_ratio(ratio):
    """(width, height) for a "w:h" ratio: the model card size if listed, else the same pixel
    count as 2048x2048 at that ratio, rounded to multiples of 32."""
    import math

    if ratio in ASPECT_RATIOS:
        return ASPECT_RATIOS[ratio]
    try:
        rw, rh = (float(x) for x in ratio.split(":"))
    except ValueError:
        raise SystemExit(f"invalid ratio {ratio!r}; expected e.g. 3:2")
    # Both sides have to be finite and positive, or the arithmetic below raises instead of
    # reporting a bad request: 0:1 and 1:0 divide by zero, -1:2 takes the root of a negative,
    # and nan/inf cannot be rounded to an int.
    if not (math.isfinite(rw) and math.isfinite(rh)) or rw <= 0 or rh <= 0:
        raise SystemExit(f"invalid ratio {ratio!r}: both sides must be finite and greater than zero")
    if not 1 / MAX_ASPECT <= rw / rh <= MAX_ASPECT:
        # Without this, 0.0001:1 asks for 32x131072: a latent far too large for any GPU.
        raise SystemExit(f"ratio {ratio!r} is more extreme than {MAX_ASPECT:g}:1; "
                         f"the widest size the model is documented for is 9:2")
    area = 2048 * 2048
    width = max(32, round(math.sqrt(area * rw / rh) / 32) * 32)
    height = max(32, round(area / width / 32) * 32)
    print(f"note: {ratio} is not one of the model card sizes; using {width}x{height}")
    return width, height


def size_following(image):
    """Keep the image's aspect ratio at the 2048x2048 pixel count, rounded to multiples of 32."""
    import math

    ratio = image.width / image.height
    # A pathologically thin input would otherwise produce a latent too large to render.
    ratio = min(MAX_ASPECT, max(1 / MAX_ASPECT, ratio))
    width = max(32, round(math.sqrt(2048 * 2048 * ratio) / 32) * 32)
    height = max(32, round(width / ratio / 32) * 32)
    return width, height
