#!/usr/bin/env bash
set -euo pipefail

# Build a particles GIF timed to match a robot video duration.
# Defaults:
# - input directory: latest timestamped folder in src/amr_localization/images
# - output GIF: <input_dir>/particles_synced.gif
# - total duration: 174 seconds (2m54s)
# - initialization hold: 0.5 seconds

IMAGES_ROOT="src/amr_localization/images"
INPUT_DIR=""
OUTPUT_GIF=""
TOTAL_DURATION_SEC=55.7
INIT_HOLD_SEC=0.7

usage() {
  cat <<'EOF'
Usage: scripts/make_localization_gif.sh [options]

Options:
  --images-root <path>       Root folder containing timestamped image folders.
                             Default: src/amr_localization/images
  --input-dir <path>         Folder that contains PNG frames to include.
                             Default: latest folder inside --images-root
  --output <path>            Output GIF path.
                             Default: <input-dir>/particles_synced.gif
  --duration-sec <seconds>   Target total GIF duration in seconds.
                             Default: 174
  --init-hold-sec <seconds>  Hold time for initialization frame in seconds.
                             Default: 0.5
  -h, --help                 Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --images-root)
      IMAGES_ROOT="$2"
      shift 2
      ;;
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_GIF="$2"
      shift 2
      ;;
    --duration-sec)
      TOTAL_DURATION_SEC="$2"
      shift 2
      ;;
    --init-hold-sec)
      INIT_HOLD_SEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  if [[ ! -d "$IMAGES_ROOT" ]]; then
    echo "Images root not found: $IMAGES_ROOT" >&2
    exit 1
  fi

  latest_dir="$(find "$IMAGES_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -z "$latest_dir" ]]; then
    echo "No timestamped folders found in: $IMAGES_ROOT" >&2
    exit 1
  fi
  INPUT_DIR="$latest_dir"
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

if [[ -z "$OUTPUT_GIF" ]]; then
  OUTPUT_GIF="$INPUT_DIR/particles_synced.gif"
fi

mapfile -t png_files < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.png' | sort)

if [[ ${#png_files[@]} -eq 0 ]]; then
  echo "No PNG files found in: $INPUT_DIR" >&2
  exit 1
fi

init_file=""
for f in "${png_files[@]}"; do
  base_name="$(basename "$f")"
  lower_name="$(echo "$base_name" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower_name" == *initialization*.png ]]; then
    init_file="$f"
    break
  fi
done

# Convert seconds to centiseconds (GIF delay unit).
total_cs="$(awk -v s="$TOTAL_DURATION_SEC" 'BEGIN { printf "%d", (s * 100) + 0.5 }')"
init_cs="$(awk -v s="$INIT_HOLD_SEC" 'BEGIN { printf "%d", (s * 100) + 0.5 }')"

if [[ $total_cs -lt 1 ]]; then
  echo "Total duration must be > 0." >&2
  exit 1
fi

if [[ -n "$init_file" ]]; then
  if [[ ${#png_files[@]} -lt 2 ]]; then
    echo "Only one frame found; creating single-frame GIF with total duration hold." >&2
    convert -delay "$total_cs" "$init_file" -loop 0 "$OUTPUT_GIF"
    echo "Created GIF: $OUTPUT_GIF"
    exit 0
  fi

  # Remaining duration is spread across non-initialization frames.
  remaining_cs=$((total_cs - init_cs))
  non_init_count=$(( ${#png_files[@]} - 1 ))

  if [[ $remaining_cs -lt $non_init_count ]]; then
    echo "Duration too short for frame count after initialization hold." >&2
    echo "Increase --duration-sec or reduce --init-hold-sec." >&2
    exit 1
  fi

  base_delay=$((remaining_cs / non_init_count))
  remainder=$((remaining_cs % non_init_count))

  cmd=(convert)
  cmd+=( -delay "$init_cs" "$init_file" )

  idx=0
  for f in "${png_files[@]}"; do
    [[ "$f" == "$init_file" ]] && continue
    delay="$base_delay"
    if [[ $idx -lt $remainder ]]; then
      delay=$((delay + 1))
    fi
    cmd+=( -delay "$delay" "$f" )
    idx=$((idx + 1))
  done

  cmd+=( -loop 0 "$OUTPUT_GIF" )
  "${cmd[@]}"
else
  # No explicit initialization frame: spread total duration across all frames.
  frame_count=${#png_files[@]}
  if [[ $total_cs -lt $frame_count ]]; then
    echo "Duration too short for frame count." >&2
    exit 1
  fi

  base_delay=$((total_cs / frame_count))
  remainder=$((total_cs % frame_count))

  cmd=(convert)
  for i in "${!png_files[@]}"; do
    delay="$base_delay"
    if [[ $i -lt $remainder ]]; then
      delay=$((delay + 1))
    fi
    cmd+=( -delay "$delay" "${png_files[$i]}" )
  done

  cmd+=( -loop 0 "$OUTPUT_GIF" )
  "${cmd[@]}"
fi

echo "Created GIF: $OUTPUT_GIF"
echo "Input folder: $INPUT_DIR"
echo "Frames used: ${#png_files[@]}"
echo "Target duration: ${TOTAL_DURATION_SEC}s (initialization hold: ${INIT_HOLD_SEC}s)"
