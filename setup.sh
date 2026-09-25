#!/usr/bin/env bash
# Install dependencies and download checkpoints into the Hugging Face cache ahead of time.
# The server never downloads anything (local_files_only=True), so run this first.
#
#   ./setup.sh gguf                    # abenzerps Uncensored GGUF (Q8_0) + bf16 text encoder + official configs/VAE
#   ./setup.sh gguf --quant auto       # pick the largest quant this GPU has room for (nvidia-smi)
#   ./setup.sh gguf --quant Q4_K_M     # another GGUF quant; repeat --quant for several, or --quant all
#   ./setup.sh original                # official Qwen/Qwen-Image-2.1 transformer + text encoder + configs/VAE
#   ./setup.sh all                     # both
#   ./setup.sh gguf --dry-run          # show what would be downloaded
#   ./setup.sh original --no-sync      # skip `uv sync`
#   ./setup.sh gguf --no-verify        # write the profile without test-loading the checkpoints
#   ./setup.sh gguf --quant auto --max-images 8   # size the profile for 8-input edits (default 4)
#
# Files already in the cache are skipped. HF_HOME / HF_HUB_CACHE / HF_TOKEN are respected.
set -euo pipefail

GGUF_REPO="abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
OFFICIAL_REPO="Qwen/Qwen-Image-2.1"
TE_FILE="text_encoders/qwen3vl_8b_bf16.safetensors"
ALL_QUANTS=(BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0)
PROFILE="$(dirname "$0")/qwen_image_profile.json"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# Pick a whole profile - transformer quant, text encoder quant, offload - from the GPU's total
# VRAM. The transformer sizes are the ones RESIDENT_GIB lists in qwen_image_common.py; a measured
# 8-bit text encoder adds 9.3 GiB and the VAE 0.6, and 2K denoising plus decode needs 2.5 more.
# Prints "QUANT TE_QUANT CPU_OFFLOAD" on stdout and a human note on stderr.
GPU_NAME=""
GPU_GIB=0

# Record the card even when it is not the one choosing, so the profile always says where it came
# from. The server honours the profile as written, so this is for the reader, not for a decision.
probe_gpu() {
    local mib
    mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    [[ $mib =~ ^[0-9]+$ ]] || return 1
    GPU_GIB=$((mib / 1024))
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    return 0
}

# Largest transformer that still leaves room for an edit with --max-images inputs. The resident
# sizes and the working-set model are the measured ones in qwen_image_common.py:
#   resident   = transformer + text encoder + 0.62 VAE
#   working set = 2.31 + 2.59 per 2K input image, at the 2048x2048 pixel count
# A bigger transformer therefore costs input images, which is the trade this makes explicit.
auto_profile() {
    if ! probe_gpu; then
        echo "no GPU detected via nvidia-smi; falling back to Q8_0" >&2
        echo "Q8_0 8bit false"
        return
    fi
    awk -v total="$GPU_GIB" -v want="$max_images" '
        BEGIN {
            nq = split("BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0", qs, " ")
            tsize["BF16"]=14.0; tsize["Q8_0"]=7.1;   tsize["Q6_K"]=5.5
            tsize["Q5_K_M"]=4.9; tsize["Q4_K_M"]=4.3; tsize["Q4_0"]=3.9
            nt = split("8bit 4bit", tes, " ")
            te_size["8bit"]=9.3; te_size["4bit"]=5.0
            usable = total * 0.97
            need = 2.31 + 2.59 * want
            for (t = 1; t <= nt; t++)
                for (i = 1; i <= nq; i++) {
                    resident = tsize[qs[i]] + te_size[tes[t]] + 0.62
                    if (resident + need <= usable) {
                        printf "%s %s false\n", qs[i], tes[t]
                        printf "auto: %d GiB GPU, room for %d input image(s) -> --quant %s --te-quant %s\n",
                               total, want, qs[i], tes[t] > "/dev/stderr"
                        exit
                    }
                }
            printf "Q4_0 4bit true\n"
            printf "auto: %d GiB GPU cannot hold %d input image(s) even at Q4_0/4bit -> --cpu-offload\n",
                   total, want > "/dev/stderr"
        }'
}

# The contract between this script and the server: whatever was downloaded is what gets loaded.
write_profile() {
    local weights=$1 quant=$2 te=$3 off=$4 by=$5
    cat > "$PROFILE" <<EOF
{
  "weights": "$weights",
  "quant": "$quant",
  "te_quant": "$te",
  "cpu_offload": $off,
  "gpu": "${GPU_NAME}",
  "vram_gib": ${GPU_GIB},
  "chosen_by": "$by",
  "written": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "profile -> $PROFILE ($weights/$quant, te $te, chosen_by $by)"
}

mode="${1:-}"
[[ "$mode" =~ ^(gguf|original|all)$ ]] || usage 1
shift

quants=()
dry_run=()
sync=1
verify=1
max_images=4      # how many 2K input images an edit should be able to carry; drives --quant auto
while (($#)); do
    case "$1" in
        --quant) [[ $# -ge 2 ]] || usage 1; quants+=("$2"); shift 2 ;;
        --dry-run) dry_run=(--dry-run); shift ;;
        --no-sync) sync=0; shift ;;
        --no-verify) verify=0; shift ;;
        --max-images) [[ $# -ge 2 ]] || usage 1; max_images="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done
chosen_by=user
te_quant=8bit
cpu_offload=false
[[ $max_images =~ ^[0-9]+$ ]] && ((max_images <= 10)) || {
    echo "--max-images must be 0-10 (the API's limit)" >&2; exit 1; }
probe_gpu || true      # for the record; auto_profile re-checks when it is the one deciding
if ((${#quants[@]} == 0)); then
    quants=(Q8_0)
    chosen_by=default
fi
if [[ " ${quants[*]} " == *" all "* ]]; then
    quants=("${ALL_QUANTS[@]}")
elif [[ " ${quants[*]} " == *" auto "* ]]; then
    read -r auto_q te_quant cpu_offload <<<"$(auto_profile)"
    quants=("$auto_q")
    chosen_by=auto
fi
for q in "${quants[@]}"; do
    [[ " ${ALL_QUANTS[*]} " == *" $q "* ]] || { echo "unknown GGUF quant $q; choose from ${ALL_QUANTS[*]}, auto or all" >&2; exit 1; }
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
    size=18   # the bf16 text encoder, ~17.5 GB
    for q in "${quants[@]}"; do
        files+=("qwen-image-2.1-UC-$q.gguf")
        case "$q" in
            BF16) size=$((size + 14)) ;; Q8_0) size=$((size + 7)) ;; Q6_K) size=$((size + 6)) ;;
            *) size=$((size + 5)) ;;   # Q5_K_M 4.9, Q4_K_M 4.3, Q4_0 3.9
        esac
    done
    echo "== $GGUF_REPO: ${files[*]} + $TE_FILE (~$size GB)"
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

# Only now that the files are really in the cache: the server treats the profile as authoritative
# and fails outright if it names a checkpoint that is not there.
profile_written=0
if [[ $mode == all ]]; then
    echo "profile not written: --weights is ambiguous after 'all'; pass --weights to the server"
elif ((${#quants[@]} > 1)); then
    echo "profile not written: several quants downloaded; pass --quant to the server"
else
    profile_written=1
    profile_weights=$mode
    profile_quant=${quants[0]}
    if [[ $mode == original ]]; then
        profile_quant=8bit          # setup.sh does not choose a bitsandbytes quant for the official weights
        chosen_by=default
    fi
    if ((verify)); then
        # Only persist a profile this machine has actually loaded: a download can succeed and the
        # weights still fail to load (too large for the VRAM, a bad file, a missing dependency).
        echo "== verifying the profile loads (qwen_image_dryrun.py --no-generate)"
        offload_flag=()
        [[ $cpu_offload == true ]] && offload_flag=(--cpu-offload)
        if uv run qwen_image_dryrun.py --no-generate --weights "$profile_weights" \
                --quant "$profile_quant" --te-quant "$te_quant" "${offload_flag[@]}"; then
            write_profile "$profile_weights" "$profile_quant" "$te_quant" "$cpu_offload" "$chosen_by"
        else
            echo "profile not written: $profile_weights/$profile_quant did not load here." >&2
            echo "Fix the load, or rerun with --no-verify to record it anyway." >&2
            exit 1
        fi
    else
        write_profile "$profile_weights" "$profile_quant" "$te_quant" "$cpu_offload" "$chosen_by"
    fi
fi

echo
echo "Done. Start the server with:"
if ((profile_written)); then
    # The profile carries the weights and quant, so no flags are needed.
    echo "  uv run qwen_image_server.py"
    echo "Check the load first with: uv run qwen_image_dryrun.py"
else
    if [[ $mode == original || $mode == all ]]; then
        echo "  uv run qwen_image_server.py --weights original"
    fi
    if [[ $mode != original ]]; then
        echo "  uv run qwen_image_server.py --quant ${quants[0]}"
    fi
    echo "Check the load first with: uv run qwen_image_dryrun.py [--weights original] [--quant Q]"
fi
