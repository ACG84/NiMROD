#!/usr/bin/env bash
# Bootstrap the NiMROD toolchain from nothing.
#
# Psi4 is only distributed through conda-forge, so this fetches a standalone
# micromamba binary (no root, no pre-existing conda) and builds the environment
# from environment.yml.  Safe to re-run; existing environments are left alone.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAMBA_ROOT_PREFIX="${ROOT}/.mamba"
MICROMAMBA="${ROOT}/bin/micromamba"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

if [[ ! -x "${MICROMAMBA}" ]]; then
  log "Fetching micromamba"
  mkdir -p "${ROOT}/bin"
  curl -sSL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "${ROOT}" bin/micromamba
fi

if "${MICROMAMBA}" env list | grep -qE '^\s+nimrod\s'; then
  log "Environment 'nimrod' already exists; skipping creation"
else
  log "Creating the 'nimrod' environment (Psi4 from conda-forge, several minutes)"
  "${MICROMAMBA}" create -y -f "${ROOT}/environment.yml"
fi

log "Verifying"
"${MICROMAMBA}" run -n nimrod python - <<'PY'
import psi4, numpy, scipy
print(f"  psi4  {psi4.__version__}")
print(f"  numpy {numpy.__version__}")
print(f"  scipy {scipy.__version__}")
PY

cat <<EOF

Ready. Run calculations with:

  export MAMBA_ROOT_PREFIX="${ROOT}/.mamba"
  ${MICROMAMBA} run -n nimrod python -m nimrod.cli --help

Optional GPU offload through the Google Colab CLI: see gpu/README.md
EOF
