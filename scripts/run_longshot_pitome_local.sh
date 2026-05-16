#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[VideoRLM] run_longshot_pitome_local.sh is now an alias for the official lazy PiToMe strategy." >&2
exec "${script_dir}/run_longshot_lazy_pitome_refinement_local.sh" "$@"
