# shellcheck shell=bash
#
# analysis/_cluster.sh — shared environment + submission helpers.
#
# Source this file from analysis jobs (it has no side effects beyond defining
# functions and exporting paths):
#
#     PHENONN_ROOT=/path/to/PhenoNN
#     source "$PHENONN_ROOT/analysis/_cluster.sh"
#     phenonn_cluster_init
#
# Environment:
#   PHENONN_ROOT      repository root (required for generated jobs)
#   PHENONN_VENV      activate script (default: $PHENONN_ROOT/.venv/bin/activate)
#   PHENONN_PYTHON    interpreter (default: python after activation)
#   DATA              dataset root (default: $PHENONN_ROOT/pixelset1 when
#                     present, otherwise $PHENONN_ROOT/../pixelset1)
#   PHENONN_SITE      hal | leonardo | local | auto (default: auto)
#   PHENONN_PARTITION / PHENONN_ACCOUNT / PHENONN_QOS / PHENONN_MEM
#   PHENONN_TIME / PHENONN_GRES
#   PHENONN_DEPENDENCY  extra sbatch dependency (e.g. afterok:12345)
#   PHENONN_MODULES   modules to load (default: site-specific; empty disables)
#   PHENONN_SKIP_MODULES=1
#   PHENONN_LOCAL=1   run jobs with bash instead of sbatch
#   PHENONN_DRY_RUN=1 print the sbatch commands instead of submitting

# Load modules without letting vendor activation hooks abort the caller.
# Conda-based modulefiles sometimes source scripts that reference unset
# variables; that is fatal under the wrappers' `set -u`, even with `|| true`.
phenonn_load_modules() {
    local modules="${PHENONN_MODULES:-}"
    [[ -n "$modules" ]] || return 0
    command -v module >/dev/null 2>&1 || return 0

    local restore_u=0 restore_e=0
    [[ $- == *u* ]] && restore_u=1
    [[ $- == *e* ]] && restore_e=1
    set +u
    set +e

    module purge >/dev/null 2>&1 || true
    local mod
    # shellcheck disable=SC2086  # word splitting is intentional
    for mod in $modules; do
        if ! module load "$mod" >/dev/null 2>&1; then
            echo "warning: module load $mod failed" >&2
        fi
    done

    if ((restore_e)); then set -e; fi
    if ((restore_u)); then set -u; fi
    return 0
}

phenonn_cluster_init() {
    if [[ -z "${PHENONN_ROOT:-}" ]]; then
        PHENONN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
    export PHENONN_ROOT
    cd "$PHENONN_ROOT" || return 1

    phenonn_site_defaults

    if [[ -z "${DATA:-}" ]]; then
        local candidate
        for candidate in "$PHENONN_ROOT/pixelset1" "$PHENONN_ROOT/../pixelset1"; do
            if [[ -d "$candidate" ]]; then
                DATA="$(cd "$candidate" && pwd)"
                break
            fi
        done
        : "${DATA:=$PHENONN_ROOT/../pixelset1}"
    fi
    export DATA

    if [[ "${PHENONN_SKIP_MODULES:-0}" != "1" \
        && "${PHENONN_LOCAL:-0}" != "1" \
        && "${PHENONN_DRY_RUN:-0}" != "1" ]]; then
        phenonn_load_modules
    fi

    local venv="${PHENONN_VENV:-$PHENONN_ROOT/.venv/bin/activate}"
    if [[ -f "$venv" ]]; then
        # shellcheck disable=SC1090
        source "$venv"
    elif [[ -n "${PHENONN_VENV:-}" ]]; then
        echo "warning: PHENONN_VENV does not exist: $venv" >&2
    fi

    export PHENONN_PYTHON="${PHENONN_PYTHON:-python}"
    export PHENONN_VENV="$venv"
}

phenonn_site_defaults() {
    case "${PHENONN_SITE:-auto}" in
        auto|"")
            : "${PHENONN_MODULES:=}"
            ;;
        hal)
            : "${PHENONN_MODULES:=}"
            : "${PHENONN_PARTITION:=batch}"
            ;;
        leonardo)
            : "${PHENONN_MODULES:=profile/deeplrn cineca-ai}"
            : "${PHENONN_PARTITION:=boost_usr_prod}"
            : "${PHENONN_QOS:=normal}"
            : "${PHENONN_ACCOUNT:=EUHPC_D36_053}"
            : "${PHENONN_MEM:=245G}"
            ;;
        local)
            : "${PHENONN_LOCAL:=1}"
            : "${PHENONN_MODULES:=}"
            ;;
        *)
            echo "warning: unknown PHENONN_SITE=${PHENONN_SITE}; using cluster defaults" >&2
            ;;
    esac
}

# Print one sbatch option per line; empty when nothing is configured.
phenonn_sbatch_opts() {
    [[ -n "${PHENONN_PARTITION:-}" ]] && printf '%s\n' "--partition=$PHENONN_PARTITION"
    [[ -n "${PHENONN_ACCOUNT:-}" ]] && printf '%s\n' "--account=$PHENONN_ACCOUNT"
    [[ -n "${PHENONN_QOS:-}" ]] && printf '%s\n' "--qos=$PHENONN_QOS"
    [[ -n "${PHENONN_MEM:-}" ]] && printf '%s\n' "--mem=$PHENONN_MEM"
    [[ -n "${PHENONN_TIME:-}" ]] && printf '%s\n' "--time=$PHENONN_TIME"
    [[ -n "${PHENONN_GRES:-}" ]] && printf '%s\n' "--gres=$PHENONN_GRES"
    [[ -n "${PHENONN_DEPENDENCY:-}" ]] \
        && printf '%s\n' "--dependency=$PHENONN_DEPENDENCY"
    return 0
}

# True when jobs must run locally (explicitly requested or no sbatch).
phenonn_is_local() {
    [[ "${PHENONN_LOCAL:-0}" == "1" ]] && return 0
    ! command -v sbatch >/dev/null 2>&1
}

# Submit (or run) one or more job scripts. PHENONN_DRY_RUN=1 never executes.
phenonn_submit() {
    phenonn_site_defaults

    if [[ "${PHENONN_DRY_RUN:-0}" == "1" ]]; then
        local -a opts=()
        mapfile -t opts < <(phenonn_sbatch_opts)
        local job
        for job in "$@"; do
            if phenonn_is_local; then
                printf '[dry-run] bash %q\n' "$job"
            else
                printf '[dry-run] sbatch'
                printf ' %q' "${opts[@]}" "$job"
                printf '\n'
            fi
        done
        return
    fi

    if phenonn_is_local; then
        local job
        for job in "$@"; do
            echo "[local] bash $job"
            bash "$job"
        done
        return
    fi

    local -a opts=()
    mapfile -t opts < <(phenonn_sbatch_opts)
    local job
    for job in "$@"; do
        sbatch "${opts[@]}" "$job"
    done
}

# Abort with a clear message when a dataset file/folder is missing.
phenonn_require_data() {
    local path
    for path in "$@"; do
        if [[ ! -e "$path" ]]; then
            echo "missing: $path" >&2
            echo "set DATA (or pass explicit data paths) and retry." >&2
            exit 1
        fi
    done
}
