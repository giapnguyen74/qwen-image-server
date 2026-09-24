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


def gib(n_bytes):
    return f"{n_bytes / 2**30:.2f} GiB"


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


def resolve_quant(weights, quant, cpu_offload=False):
    """Default and validate the transformer quant for the chosen weight set."""
    choices = QUANTS if weights == "gguf" else ORIGINAL_QUANTS
    if quant is None:
        return "Q8_0" if weights == "gguf" else "8bit"
    if quant not in choices:
        raise SystemExit(f"--quant {quant} is not valid with --weights {weights}; choose from {', '.join(choices)}")
    if weights == "original" and quant == "bf16" and not cpu_offload:
        print("warning: the bf16 original transformer is 13.3 GiB; with the text encoder kept loaded it will "
              "likely run out of VRAM on 24 GB. Consider --cpu-offload or --quant 8bit")
    if weights == "original" and quant == "8bit" and cpu_offload:
        print("warning: an 8-bit transformer does not offload (bitsandbytes keeps its int8 weights on the GPU); "
              "use 4bit or bf16 with --cpu-offload")
    return quant


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
    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
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
    vae.enable_tiling()  # the text encoder stays resident, so full-frame decode above ~768px runs out of VRAM
    print("vae: official Qwen/Qwen-Image-2.1/vae (tiled decode)")
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


def add_model_args(ap):
    """Model options shared by the server and the dry run. Resolve with resolve_model_args()."""
    ap.add_argument("--weights", default="gguf", choices=WEIGHTS,
                    help="gguf: abenzerps Uncensored GGUF transformer; original: official Qwen/Qwen-Image-2.1 weights")
    ap.add_argument("--quant", default=None,
                    help=f"transformer quant. gguf: {', '.join(QUANTS)} (default Q8_0); "
                         f"original: {', '.join(ORIGINAL_QUANTS)} (default 8bit, bitsandbytes)")
    ap.add_argument("--te-quant", default=None, choices=TE_QUANTS,
                    help="text encoder quantization (default: 8bit, or 4bit with --cpu-offload)")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="optional fallback, off by default (everything runs on the GPU): keep components in "
                         "system RAM and move each to the GPU only while it runs (diffusers "
                         "enable_model_cpu_offload). Slower; for edits with several input images or bf16 weights")


def resolve_model_args(args):
    """Fill in --quant / --te-quant defaults that depend on other options."""
    args.quant = resolve_quant(args.weights, args.quant, args.cpu_offload)
    if args.te_quant is None:
        args.te_quant = "4bit" if args.cpu_offload else "8bit"
    elif args.cpu_offload and args.te_quant == "8bit":
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
    area = 2048 * 2048
    width = round(math.sqrt(area * rw / rh) / 32) * 32
    height = round(area / width / 32) * 32
    print(f"note: {ratio} is not one of the model card sizes; using {width}x{height}")
    return width, height


def size_following(image):
    """Keep the image's aspect ratio at the 2048x2048 pixel count, rounded to multiples of 32."""
    import math

    ratio = image.width / image.height
    width = round(math.sqrt(2048 * 2048 * ratio) / 32) * 32
    height = round(width / ratio / 32) * 32
    return width, height
