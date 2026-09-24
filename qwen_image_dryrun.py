"""Dry run: load the Qwen-Image-2.1 pipeline (see qwen_image_common.py) and optionally render a small test image.

  uv run qwen_image_dryrun.py                  # load everything + tiny 512x512, 4-step generation
  uv run qwen_image_dryrun.py --no-generate    # load and check only
  uv run qwen_image_dryrun.py --quant Q4_K_M --te-quant 4bit --steps 40 --size 1024
  uv run qwen_image_dryrun.py --weights original      # check the official weights after ./setup.sh original
"""

import argparse
import time
from pathlib import Path

from qwen_image_common import add_model_args, build_pipeline, resolve_model_args, vram

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--no-generate", action="store_true", help="only load and check the components")
    ap.add_argument("--size", type=int, default=512, help="square output size, multiple of 32")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default='A neon shop sign that reads "DRY RUN", rainy night, reflections on wet pavement')
    ap.add_argument("--out", default="outputs/dryrun.png")
    args = resolve_model_args(ap.parse_args())

    torch.cuda.reset_peak_memory_stats()
    pipe, te_ok = build_pipeline(args.quant, args.te_quant, cpu_offload=args.cpu_offload, weights=args.weights)
    print("pipeline assembled")

    if not te_ok:
        print("text encoder weights did not all match; stopping before generation")
        raise SystemExit(1)
    if args.no_generate:
        return

    print(f"generating {args.size}x{args.size}, {args.steps} steps, seed {args.seed}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    image = pipe(
        prompt=args.prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    dt = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"  done in {dt:.1f}s ({dt / args.steps:.2f}s/step incl. encode+decode) -> {args.out}")
    vram("generation")


if __name__ == "__main__":
    main()
