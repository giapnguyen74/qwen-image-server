#!/usr/bin/env bash
# Install dependencies and download checkpoints into the Hugging Face cache ahead of time.
# The server never downloads anything (local_files_only=True), so run this first.
#
#   ./setup.sh gguf                    # abenzerps Uncensored GGUF (Q8_0) + bf16 text encoder + official configs/VAE
#   ./setup.sh gguf --quant Q4_K_M     # another GGUF quant; repeat --quant for several, or --quant all
#   ./setup.sh original                # official Qwen/Qwen-Image-2.1 transformer + text encoder + configs/VAE
#   ./setup.sh all                     # both
#   ./setup.sh gguf --dry-run          # show what would be downloaded
#   ./setup.sh original --no-sync      # skip `uv sync`
#
# Files already in the cache are skipped. HF_HOME / HF_HUB_CACHE / HF_TOKEN are respected.
set -euo pipefail

GGUF_REPO="abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
OFFICIAL_REPO="Qwen/Qwen-Image-2.1"
TE_FILE="text_encoders/qwen3vl_8b_bf16.safetensors"
ALL_QUANTS=(BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0)

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

mode="${1:-}"
[[ "$mode" =~ ^(gguf|original|all)$ ]] || usage 1
shift

quants=()
dry_run=()
sync=1
while (($#)); do
    case "$1" in
        --quant) [[ $# -ge 2 ]] || usage 1; quants+=("$2"); shift 2 ;;
        --dry-run) dry_run=(--dry-run); shift ;;
        --no-sync) sync=0; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done
((${#quants[@]})) || quants=(Q8_0)
if [[ " ${quants[*]} " == *" all "* ]]; then
    quants=("${ALL_QUANTS[@]}")
fi
for q in "${quants[@]}"; do
    [[ " ${ALL_QUANTS[*]} " == *" $q "* ]] || { echo "unknown GGUF quant $q; choose from ${ALL_QUANTS[*]} or all" >&2; exit 1; }
done

for tool in uv hf; do
    command -v "$tool" >/dev/null || {
        echo "$tool not found. Install uv from https://docs.astral.sh/uv/, then: uv tool install huggingface_hub" >&2
        exit 1
    }
done

cd "$(dirname "$0")"
if ((sync)) && ((${#dry_run[@]} == 0)); then
    echo "== uv sync"
    uv sync
fi

# hf download takes one --include per pattern; `--include "a/*" "b/*"` would treat b/* as a filename.
official_patterns=(--include "*.json" --include "*.jinja" --include "*.txt" --include "vae/*")

if [[ $mode == gguf || $mode == all ]]; then
    files=()
    for q in "${quants[@]}"; do files+=("qwen-image-2.1-UC-$q.gguf"); done
    echo "== $GGUF_REPO: ${files[*]} + $TE_FILE (~17 GB)"
    hf download "$GGUF_REPO" "${files[@]}" "$TE_FILE" "${dry_run[@]}"
fi

if [[ $mode == original || $mode == all ]]; then
    official_patterns+=(--include "transformer/*" --include "text_encoder/*")
    echo "== $OFFICIAL_REPO: transformer (~14 GB) + text encoder (~17.5 GB) + configs/VAE"
else
    echo "== $OFFICIAL_REPO: configs, processor, scheduler, VAE (~1.3 GB)"
fi
hf download "$OFFICIAL_REPO" "${official_patterns[@]}" "${dry_run[@]}"

((${#dry_run[@]})) && exit 0

echo
echo "Done. Start the server with:"
if [[ $mode == original ]]; then
    echo "  uv run qwen_image_server.py --weights original"
else
    if [[ " ${quants[*]} " == *" Q8_0 "* ]]; then
        echo "  uv run qwen_image_server.py"
    else
        echo "  uv run qwen_image_server.py --quant ${quants[0]}"
    fi
    [[ $mode == all ]] && echo "  uv run qwen_image_server.py --weights original"
fi
echo "Check the load first with: uv run qwen_image_dryrun.py [--weights original]"
