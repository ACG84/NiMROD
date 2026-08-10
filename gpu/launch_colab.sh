#!/usr/bin/env bash
#
# launch_colab.sh - run gpu4pyscf_job.py on a fresh Colab GPU VM.
#
# Wraps `colab run` (new + exec + stop in one command) and handles the four
# documented ways it goes wrong:
#
#   1. `colab run --timeout` defaults to THIRTY SECONDS.  Every real DFT job
#      exceeds that, so the default here is 2 hours.  This is the single
#      highest-value line in the file.
#   2. An unrecognised `--gpu` value silently falls back to A100 rather than
#      erroring, so a typo becomes an accelerator you did not ask for and
#      probably have no quota for.  Values are therefore validated locally
#      against the CLI's supported set before being passed on.
#   3. A 400 from `colab new`/`colab run` with an accelerator means no
#      quota/entitlement on this account, not a transient fault.  We recognise
#      it and step down the accelerator list instead of retrying.
#   4. An unauthenticated CLI prompts on stdin for an OAuth code.  With a TTY
#      attached that hangs forever.  Every invocation here gets </dev/null and
#      a timeout, and auth is pre-flighted once before any VM is allocated.
#
# The job's result never comes back as a file: `colab run` destroys the VM when
# the script exits, taking --out with it.  The script therefore also prints its
# JSON to stdout between sentinels, and that is what we harvest.
#
# Usage:
#   gpu/launch_colab.sh --xyz sensor.xyz --out data/gpu/sensor.json \
#       --mult 2 --functional pbe0 --basis def2-tzvp --n-states 12

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/gpu4pyscf_job.py"
readonly SCRIPT_DIR REPO_ROOT JOB_SCRIPT

readonly JSON_BEGIN='===NIMROD-GPU-JSON-BEGIN==='
readonly JSON_END='===NIMROD-GPU-JSON-END==='

# Accelerators the CLI actually recognises.  Anything else silently becomes
# A100 (gotcha 2), so we refuse it here instead.
readonly SUPPORTED_GPUS=(T4 L4 G4 H100 A100)

# Exit codes distinct from the job script's, so a caller can tell an
# infrastructure failure from a chemistry failure.
readonly EX_USAGE=2
readonly EX_NOAUTH=3
readonly EX_NOACCEL=4
readonly EX_NOJSON=5

# ---------------------------------------------------------------------------
# Defaults (override on the command line)
# ---------------------------------------------------------------------------

XYZ=""
OUT=""
CHARGE=0
MULT=1
FUNCTIONAL="pbe0"
BASIS="def2-tzvp"
N_STATES=0
TRIPLETS="none"
TDA=0
LABEL=""
# A100 first (the whole point of offloading), T4 as the widely-available
# fallback.  Append 'CPU' to accept a CPU runtime as a last resort.
GPU_ORDER=(A100 T4)
TIMEOUT=7200
KEEP=0
SESSION=""
DRY_RUN=0
VERBOSE=0
EXTRA_ARGS=()
#: Set to 1 once a `colab run` has been issued, i.e. once a VM may exist.
SESSION_ALLOCATED=0

# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

log() { printf '[launch] %s\n' "$*" >&2; }
# die MESSAGE [EXIT_CODE] -- note $1 only, so the code never leaks into the text.
die() { printf '[launch] ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

usage() {
    cat >&2 <<'USAGE'
Usage: launch_colab.sh --xyz FILE --out FILE [options] [-- extra job args]

Required:
  --xyz FILE            geometry to compute (.xyz, or a bare Cartesian block)
  --out FILE            where to write the returned JSON locally

Chemistry:
  --charge N            total charge                       (default 0)
  --mult N              spin multiplicity 2S+1             (default 1)
  --functional NAME     Psi4-style functional name         (default pbe0)
  --basis NAME          Psi4-style basis name              (default def2-tzvp)
  --n-states N          TDDFT roots; 0 skips TDDFT         (default 0)
  --triplets MODE       none|also|only, closed shell only  (default none)
  --tda                 Tamm-Dancoff instead of full TDDFT
  --label TEXT          label carried into the result

Infrastructure:
  --gpu "A100 T4"       accelerators to try in order; add CPU to allow a CPU
                        runtime as a last resort           (default "A100 T4")
  --timeout SEC         remote execution timeout           (default 7200)
                        NB the colab CLI's own default is 30 s
  --session NAME        session name (default: auto, always passed with -s)
  --keep                do not stop the VM afterwards (costs compute units)
  --dry-run             print the commands that would run, allocate nothing
  --verbose             echo the remote job's stderr as it arrives
  -h, --help            this message

Environment:
  NIMROD_COLAB_CMD      override the colab executable
  NIMROD_COLAB_AUTH     adc (default) or oauth2

Exit codes:
  0 ok | 2 usage | 3 unauthenticated | 4 no accelerator | 5 no JSON returned
  anything else is the remote job's own exit code (3 SCF, 4 TDDFT, 5 no pyscf)
USAGE
}

# Build the colab invocation as an array so quoting survives.
build_colab_cmd() {
    local auth="${NIMROD_COLAB_AUTH:-adc}"
    if [[ -n "${NIMROD_COLAB_CMD:-}" ]]; then
        # shellcheck disable=SC2206  # deliberate word-splitting of a user override
        COLAB=(${NIMROD_COLAB_CMD})
    elif [[ -x "${REPO_ROOT}/bin/micromamba" ]]; then
        COLAB=(env "MAMBA_ROOT_PREFIX=${REPO_ROOT}/.mamba"
               "${REPO_ROOT}/bin/micromamba" run -n colabcli colab)
    elif command -v colab >/dev/null 2>&1; then
        COLAB=(colab)
    else
        die "no colab CLI found: install it, or set NIMROD_COLAB_CMD" "${EX_USAGE}"
    fi
    # Global flags must precede the subcommand.  --config isolates this run's
    # session state so concurrent agents cannot clobber each other's sessions.
    COLAB+=("--auth=${auth}" "--config" "${STATE_FILE}")
}

is_supported_gpu() {
    local candidate="$1" gpu
    for gpu in "${SUPPORTED_GPUS[@]}"; do
        [[ "${candidate}" == "${gpu}" ]] && return 0
    done
    return 1
}

# Recognise the CLI's unauthenticated signatures.  Both auth strategies fail
# distinctly: oauth2 prints a consent URL and prompts for a code, adc raises
# DefaultCredentialsError.
looks_unauthenticated() {
    local text="$1"
    [[ "${text}" == *"DefaultCredentialsError"* ]] ||
    [[ "${text}" == *"Enter the authorization code"* ]] ||
    [[ "${text}" == *"default credentials were not found"* ]] ||
    [[ "${text}" == *"To authorize colab-cli"* ]]
}

auth_help() {
    cat >&2 <<'AUTHHELP'
[launch] The colab CLI is not authenticated. One-time setup (needs a browser
[launch] and the gcloud SDK, which this container does not have):
[launch]
[launch]   gcloud auth application-default login \
[launch]     --scopes=openid,\
[launch] https://www.googleapis.com/auth/cloud-platform,\
[launch] https://www.googleapis.com/auth/userinfo.email,\
[launch] https://www.googleapis.com/auth/colaboratory
[launch]
[launch] All four scopes are required: userinfo.email or the session backend
[launch] returns 401; colaboratory or the keep-alive RPC returns 403;
[launch] openid and cloud-platform are mandated by gcloud itself.
[launch] See gpu/README.md for the full explanation.
AUTHHELP
}

preflight_auth() {
    log "checking colab authentication ..."
    local output status
    set +e
    output="$(timeout 120 "${COLAB[@]}" sessions </dev/null 2>&1)"
    status=$?
    set -e
    if looks_unauthenticated "${output}"; then
        auth_help
        die "cannot proceed without authentication" "${EX_NOAUTH}"
    fi
    if (( status != 0 )); then
        log "colab sessions exited ${status}; continuing anyway. Output was:"
        printf '%s\n' "${output}" | sed 's/^/[launch]   /' >&2
        return
    fi
    log "authentication OK"
}

# shellcheck disable=SC2329  # invoked indirectly, via `trap cleanup EXIT INT TERM`
cleanup() {
    local status=$?
    # Only chase a session we actually asked the backend to allocate. Calling
    # `colab stop` before that would just trigger a pointless auth round-trip.
    if [[ "${SESSION_ALLOCATED}" -eq 1 && "${KEEP}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
        # `colab run` self-cleans, but an interrupt between allocation and
        # teardown leaks a billable VM.  Stopping an already-stopped session is
        # harmless, so this is unconditional.
        log "ensuring session '${SESSION}' is stopped"
        timeout 120 "${COLAB[@]}" stop -s "${SESSION}" </dev/null >/dev/null 2>&1 || true
    fi
    [[ -n "${WORK_DIR:-}" && -d "${WORK_DIR}" ]] && rm -rf "${WORK_DIR}"
    exit "${status}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --xyz)         XYZ="${2:?--xyz needs a value}"; shift 2 ;;
        --out)         OUT="${2:?--out needs a value}"; shift 2 ;;
        --charge)      CHARGE="${2:?--charge needs a value}"; shift 2 ;;
        --mult|--multiplicity) MULT="${2:?--mult needs a value}"; shift 2 ;;
        --functional)  FUNCTIONAL="${2:?--functional needs a value}"; shift 2 ;;
        --basis)       BASIS="${2:?--basis needs a value}"; shift 2 ;;
        --n-states|--nstates) N_STATES="${2:?--n-states needs a value}"; shift 2 ;;
        --triplets)    TRIPLETS="${2:?--triplets needs a value}"; shift 2 ;;
        --tda)         TDA=1; shift ;;
        --label)       LABEL="${2:?--label needs a value}"; shift 2 ;;
        --gpu)         read -r -a GPU_ORDER <<< "${2:?--gpu needs a value}"; shift 2 ;;
        --timeout)     TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
        --session)     SESSION="${2:?--session needs a value}"; shift 2 ;;
        --keep)        KEEP=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --verbose)     VERBOSE=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; EXTRA_ARGS=("$@"); break ;;
        *)             usage; die "unknown option: $1" "${EX_USAGE}" ;;
    esac
done

[[ -n "${XYZ}" ]] || { usage; die "--xyz is required" "${EX_USAGE}"; }
[[ -n "${OUT}" ]] || { usage; die "--out is required" "${EX_USAGE}"; }
[[ -f "${XYZ}" ]] || die "geometry file not found: ${XYZ}" "${EX_USAGE}"
[[ -f "${JOB_SCRIPT}" ]] || die "job script not found: ${JOB_SCRIPT}" "${EX_USAGE}"
[[ "${TIMEOUT}" =~ ^[0-9]+$ ]] || die "--timeout must be an integer, got '${TIMEOUT}'" "${EX_USAGE}"
[[ "${N_STATES}" =~ ^[0-9]+$ ]] || die "--n-states must be an integer, got '${N_STATES}'" "${EX_USAGE}"
(( ${#GPU_ORDER[@]} > 0 )) || die "--gpu list is empty" "${EX_USAGE}"

# Gotcha 2: validate accelerator names before the CLI silently rewrites them.
for candidate in "${GPU_ORDER[@]}"; do
    if [[ "${candidate}" != "CPU" ]] && ! is_supported_gpu "${candidate}"; then
        die "unsupported accelerator '${candidate}'. The colab CLI would
       silently fall back to A100 rather than reject it. Supported:
       ${SUPPORTED_GPUS[*]} (or CPU for a CPU runtime)." "${EX_USAGE}"
    fi
done

if (( TIMEOUT < 300 )); then
    log "WARNING: --timeout ${TIMEOUT}s is short for a DFT job; the CLI's own"
    log "         default of 30s would kill essentially any real calculation."
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nimrod-colab-XXXXXX")"
STATE_FILE="${WORK_DIR}/sessions.json"
readonly WORK_DIR STATE_FILE

if [[ -z "${SESSION}" ]]; then
    base="$(basename -- "${XYZ}")"
    base="${base%.*}"
    # Gotcha: an omitted -s becomes a random 6-hex name, which makes the
    # session impossible to refer to later. Always name it.
    SESSION="nimrod-${base//[^a-zA-Z0-9]/-}-$(date +%H%M%S)-$$"
fi

build_colab_cmd
trap cleanup EXIT INT TERM

mkdir -p "$(dirname -- "${OUT}")"

# Ship the geometry inside argv: `colab run` uploads no data files, and the VM
# is destroyed before we could download anything from it.
GEOM_B64="$(gzip -9 -c -- "${XYZ}" | base64 | tr -d '\n')"
readonly GEOM_B64

JOB_ARGS=(
    --xyz-b64 "${GEOM_B64}"
    --charge "${CHARGE}"
    --multiplicity "${MULT}"
    --functional "${FUNCTIONAL}"
    --basis "${BASIS}"
    --n-states "${N_STATES}"
    --triplets "${TRIPLETS}"
    --out "/content/$(basename -- "${OUT}")"
)
if (( TDA == 1 )); then
    JOB_ARGS+=(--tda)
fi
if [[ -n "${LABEL}" ]]; then
    JOB_ARGS+=(--label "${LABEL}")
fi
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    JOB_ARGS+=("${EXTRA_ARGS[@]}")
fi

log "system    : ${XYZ} (charge ${CHARGE}, multiplicity ${MULT})"
log "method    : ${FUNCTIONAL}/${BASIS}, ${N_STATES} TDDFT roots"
log "session   : ${SESSION}"
log "timeout   : ${TIMEOUT}s"
log "geometry  : ${#GEOM_B64} base64 chars on the command line"
log "trying    : ${GPU_ORDER[*]}"

if (( DRY_RUN == 1 )); then
    log "dry run: the following would be executed, in order, until one succeeds"
    for accel in "${GPU_ORDER[@]}"; do
        accel_flag=()
        if [[ "${accel}" != "CPU" ]]; then
            accel_flag=(--gpu "${accel}")
        fi
        printf '\n  # accelerator: %s\n  ' "${accel}"
        printf '%q ' "${COLAB[@]}" run "${accel_flag[@]}" \
            --timeout "${TIMEOUT}" -s "${SESSION}" "${JOB_SCRIPT}" "${JOB_ARGS[@]}"
        printf '\n'
    done
    printf '\n'
    log "then: extract JSON between the sentinels from stdout into ${OUT}"
    log "then: ${COLAB[*]} stop -s ${SESSION}"
    # The job script's own --dry-run validates the chemistry arguments without
    # a VM, so a launcher dry run can check them for free.
    log "validating job arguments locally via the job script's --dry-run:"
    python3 "${JOB_SCRIPT}" "${JOB_ARGS[@]}" --dry-run 2>&1 | sed 's/^/[launch]   /' >&2
    exit 0
fi

preflight_auth

# ---------------------------------------------------------------------------
# Run, stepping down the accelerator list
# ---------------------------------------------------------------------------

STDOUT_FILE="${WORK_DIR}/stdout.txt"
STDERR_FILE="${WORK_DIR}/stderr.txt"
job_status=-1
used_accel=""

for accel in "${GPU_ORDER[@]}"; do
    accel_flag=()
    if [[ "${accel}" != "CPU" ]]; then
        accel_flag=(--gpu "${accel}")
        log "requesting a ${accel} VM ..."
    else
        log "requesting a CPU VM (last resort; GPU4PySCF will not be used) ..."
    fi

    # Always capture stderr to a file rather than teeing through a process
    # substitution: the substitution is not waited on, so the file could still
    # be short when we read it a moment later. When following is wanted, a
    # background `tail -f` gives live output without that race.
    : >"${STDERR_FILE}"
    follow_pid=""
    if (( VERBOSE == 1 )); then
        tail -n +1 -f "${STDERR_FILE}" >&2 &
        follow_pid=$!
    fi

    SESSION_ALLOCATED=1
    set +e
    "${COLAB[@]}" run "${accel_flag[@]}" --timeout "${TIMEOUT}" \
        -s "${SESSION}" "${JOB_SCRIPT}" "${JOB_ARGS[@]}" \
        </dev/null >"${STDOUT_FILE}" 2>"${STDERR_FILE}"
    job_status=$?
    set -e

    if [[ -n "${follow_pid}" ]]; then
        kill "${follow_pid}" 2>/dev/null || true
        wait "${follow_pid}" 2>/dev/null || true
    fi

    combined="$(cat "${STDERR_FILE}" "${STDOUT_FILE}" 2>/dev/null || true)"

    if looks_unauthenticated "${combined}"; then
        auth_help
        die "authentication was lost mid-run" "${EX_NOAUTH}"
    fi

    # Did the remote script actually run?  If it emitted the sentinel, the
    # chemistry ran and its exit code is a chemistry verdict, not an
    # infrastructure one -- retrying on a different accelerator is pointless.
    if grep -qF "${JSON_BEGIN}" "${STDOUT_FILE}"; then
        used_accel="${accel}"
        log "job ran on ${accel} (exit ${job_status})"
        break
    fi

    if (( job_status == 0 )); then
        log "colab run reported success on ${accel} but emitted no result payload"
        used_accel="${accel}"
        break
    fi

    # Gotcha 3: a 400 is an entitlement problem, not a transient one.
    if [[ "${combined}" == *"400"* ]]; then
        log "${accel}: HTTP 400 - no quota/entitlement for this accelerator on this account"
    else
        log "${accel}: failed (exit ${job_status})"
        printf '%s\n' "${combined}" | tail -n 15 | sed 's/^/[launch]   /' >&2
    fi
    log "stepping down to the next accelerator"
done

if [[ -z "${used_accel}" ]]; then
    log "no accelerator in '${GPU_ORDER[*]}' could be allocated."
    log "Accelerator availability is Colab-tier-gated; most accounts get CPU only."
    log "Re-run with --gpu \"T4 CPU\" to accept a CPU runtime."
    exit "${EX_NOACCEL}"
fi

# ---------------------------------------------------------------------------
# Harvest the result
# ---------------------------------------------------------------------------

if ! awk -v b="${JSON_BEGIN}" -v e="${JSON_END}" '
        $0 == b { inside = 1; next }
        $0 == e { inside = 0; exit }
        inside  { print }
    ' "${STDOUT_FILE}" > "${OUT}.partial" || [[ ! -s "${OUT}.partial" ]]; then
    rm -f "${OUT}.partial"
    log "the remote job returned no parseable JSON. Remote output tail:"
    tail -n 30 "${STDERR_FILE}" | sed 's/^/[launch]   /' >&2
    exit "${EX_NOJSON}"
fi

if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "${OUT}.partial" 2>/dev/null; then
    log "harvested payload is not valid JSON; leaving it at ${OUT}.partial"
    exit "${EX_NOJSON}"
fi

mv -- "${OUT}.partial" "${OUT}"
log "wrote ${OUT} (accelerator: ${used_accel})"

if (( job_status != 0 )); then
    log "the remote job exited ${job_status}; ${OUT} records the failure"
fi

exit "${job_status}"
