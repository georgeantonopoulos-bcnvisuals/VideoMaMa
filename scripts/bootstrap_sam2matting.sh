#!/usr/bin/env bash
# Bootstrap the isolated SAM2Matting runtime.
#
# SAM2Matting ships modified top-level `sam2` and `sam3` packages and expects a
# different Torch build from our SAM 3 UI environment. Installing it alongside
# SAM 3 would shadow the official install, so it gets its own venv, its own
# source checkout pinned to an explicit upstream commit, and its own process.
#
# Layout (same spirit as bootstrap_tmp_venv.sh):
#   venv:        $VIDEOMAMA_VENV_ROOT/videomama-sam2matting-venv   (rebuildable)
#   source:      $VIDEOMAMA_VENV_ROOT/videomama-sam2matting-src    (pinned checkout)
#   checkpoints: $VIDEOMAMA_CHECKPOINTS/SAM2Matting/    (shared, persistent)
#   lock:        <repo>/tmp/runtime-locks/sam2matting-runtime.json
#
# Usage:
#   bash scripts/bootstrap_sam2matting.sh              # everything
#   bash scripts/bootstrap_sam2matting.sh env          # venv only
#   bash scripts/bootstrap_sam2matting.sh source       # pinned checkout only
#   bash scripts/bootstrap_sam2matting.sh checkpoints  # weights only
#
# Deliberate update of the pin:
#   SAM2MATTING_GIT_REF=<sha> REFRESH_SAM2MATTING_LOCK=1 bash scripts/bootstrap_sam2matting.sh source

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [[ -z "${SCRIPT_DIR}" ]]; then
  echo "Cannot resolve this script's directory; cd into the repo root and re-run." >&2
  exit 1
fi
cd "${SCRIPT_DIR}"

if [[ -n "${SUDO_USER:-}" ]]; then
  echo "Do not run this script with sudo; it only writes to /tmp, the repo, and the shared checkpoints." >&2
  exit 1
fi

REPO_ROOT="$(cd .. && pwd)"
ENV_FILE="${REPO_ROOT}/.videomama-env"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

DEFAULT_CHECKPOINTS="/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints"
CHECKPOINTS_DIR="${VIDEOMAMA_CHECKPOINTS:-${DEFAULT_CHECKPOINTS}}"
S2M_CHECKPOINTS="${CHECKPOINTS_DIR}/SAM2Matting"

VENV_ROOT="${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}"
S2M_VENV="${VIDEOMAMA_SAM2MATTING_VENV:-${VENV_ROOT}/videomama-sam2matting-venv}"
S2M_HOME="${VIDEOMAMA_SAM2MATTING_HOME:-${VENV_ROOT}/videomama-sam2matting-src}"

# Upstream has no releases or tags. This commit is the validated pin; keep it in
# sync with SAM2MATTING_PINNED_COMMIT in demo/matting_backends.py.
S2M_REPO_URL="${SAM2MATTING_REPO_URL:-https://github.com/FudanCVL/SAM2Matting.git}"
S2M_PINNED_COMMIT="73dd721d77b56749248aefe5e8824d7f61b9d13c"
S2M_GIT_REF="${SAM2MATTING_GIT_REF:-${S2M_PINNED_COMMIT}}"

S2M_HF_REPO="${SAM2MATTING_HF_REPO:-FudanCVL/SAM2Matting}"
S2M_HF_REVISION="${SAM2MATTING_HF_REVISION:-4315db9c60d27fde396b09765748a0ca6c97bed5}"

# Upstream pins torch 2.8.0 / torchvision 0.23.0. cu128 wheels cover the L4 (sm_89).
S2M_TORCH_INDEX_URL="${S2M_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
S2M_TORCH_VERSION="${S2M_TORCH_VERSION:-2.8.0}"
S2M_TORCHVISION_VERSION="${S2M_TORCHVISION_VERSION:-0.23.0}"

PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
PYENV_BIN="${PYENV_BIN:-${PYENV_ROOT}/bin/pyenv}"
PYTHON_S2M_BIN="${PYTHON_S2M_BIN:-}"

LOCK_DIR="${REPO_ROOT}/tmp/runtime-locks"
RUNTIME_LOCK="${LOCK_DIR}/sam2matting-runtime.json"

target="${1:-all}"

log() { printf '[sam2matting-bootstrap] %s\n' "$*"; }

if ! mkdir -p "${VENV_ROOT}" 2>/dev/null || [[ ! -w "${VENV_ROOT}" ]]; then
  echo "Venv root is not writable: ${VENV_ROOT}" >&2
  echo "Create it, or point VIDEOMAMA_VENV_ROOT somewhere with tens of GB free." >&2
  exit 1
fi

resolve_pyenv_python() {
  local version_prefix="$1" pyenv_cmd="" candidate
  candidate="$(command -v pyenv 2>/dev/null || true)"
  if [[ "${candidate}" == /* && -x "${candidate}" ]]; then
    pyenv_cmd="${candidate}"
  elif [[ -x "${PYENV_BIN}" ]]; then
    pyenv_cmd="${PYENV_BIN}"
  else
    return 1
  fi
  local version
  version="$("${pyenv_cmd}" versions --bare 2>/dev/null | awk -v prefix="${version_prefix}" '$0 ~ "^" prefix "(\\.|$)" { print; exit }')"
  [[ -n "${version}" ]] || return 1
  local py
  py="$(PYENV_VERSION="${version}" "${pyenv_cmd}" which python 2>/dev/null || true)"
  if [[ -z "${py}" || ! -x "${py}" ]]; then
    py="${PYENV_ROOT}/versions/${version}/bin/python"
  fi
  [[ -x "${py}" ]] || return 1
  printf '%s\n' "${py}"
}

resolve_python() {
  if [[ -n "${PYTHON_S2M_BIN}" ]]; then
    printf '%s\n' "${PYTHON_S2M_BIN}"
    return 0
  fi
  # Reuse the interpreter the SAM 3 UI venv was built with when we know it;
  # one fewer Python build to keep working on a production box.
  if [[ -n "${VIDEOMAMA_UI_PYTHON:-}" && -x "${VIDEOMAMA_UI_PYTHON}" ]]; then
    printf '%s\n' "${VIDEOMAMA_UI_PYTHON}"
    return 0
  fi
  local resolved
  if resolved="$(resolve_pyenv_python "3.12")"; then
    printf '%s\n' "${resolved}"
    return 0
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return 0
  fi
  printf '%s\n' "python3.12"
}

fetch_source() {
  log "Pinned SAM2Matting checkout at ${S2M_HOME} (ref ${S2M_GIT_REF})"
  if [[ -d "${S2M_HOME}/.git" ]]; then
    git -C "${S2M_HOME}" fetch --depth 1 origin "${S2M_GIT_REF}" 2>/dev/null \
      || git -C "${S2M_HOME}" fetch origin
  else
    rm -rf "${S2M_HOME}"
    git clone --filter=blob:none "${S2M_REPO_URL}" "${S2M_HOME}"
  fi
  git -C "${S2M_HOME}" checkout --detach "${S2M_GIT_REF}"
  local resolved
  resolved="$(git -C "${S2M_HOME}" rev-parse HEAD)"
  if [[ "${S2M_GIT_REF}" == "${S2M_PINNED_COMMIT}" && "${resolved}" != "${S2M_PINNED_COMMIT}" ]]; then
    echo "Checked-out commit ${resolved} does not match the pin ${S2M_PINNED_COMMIT}." >&2
    exit 1
  fi
  if [[ "${resolved}" != "${S2M_PINNED_COMMIT}" && "${REFRESH_SAM2MATTING_LOCK:-0}" != "1" ]]; then
    echo "Refusing to record commit ${resolved}, which differs from the pin in" >&2
    echo "demo/matting_backends.py (${S2M_PINNED_COMMIT})." >&2
    echo "Set REFRESH_SAM2MATTING_LOCK=1 and update SAM2MATTING_PINNED_COMMIT to move the pin." >&2
    exit 1
  fi
  log "SAM2Matting source at ${resolved}"
}

build_env() {
  local python_bin
  python_bin="$(resolve_python)"
  log "Building SAM2Matting venv at ${S2M_VENV} (Python: ${python_bin})"
  if ! command -v "${python_bin}" >/dev/null 2>&1 && [[ ! -x "${python_bin}" ]]; then
    echo "Missing interpreter: ${python_bin}. Override with PYTHON_S2M_BIN=/path/to/python" >&2
    exit 1
  fi
  if [[ -d "${S2M_VENV}" && ! -x "${S2M_VENV}/bin/python" ]]; then
    log "Removing incomplete venv: ${S2M_VENV}"
    rm -rf "${S2M_VENV}"
  fi
  if [[ ! -x "${S2M_VENV}/bin/python" ]]; then
    "${python_bin}" -m venv "${S2M_VENV}"
  fi
  "${S2M_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${S2M_VENV}/bin/python" -m pip install --index-url "${S2M_TORCH_INDEX_URL}" \
      "torch==${S2M_TORCH_VERSION}" "torchvision==${S2M_TORCHVISION_VERSION}"
  "${S2M_VENV}/bin/python" -m pip install -r "${REPO_ROOT}/scripts/requirements-sam2matting.txt"
  mkdir -p "${LOCK_DIR}"
  "${S2M_VENV}/bin/python" -m pip freeze > "${LOCK_DIR}/requirements-sam2matting.lock.txt"
  log "SAM2Matting venv ready."
}

fetch_checkpoints() {
  mkdir -p "${S2M_CHECKPOINTS}"
  log "Fetching SAM2Matting checkpoints into ${S2M_CHECKPOINTS} (revision ${S2M_HF_REVISION})"
  if [[ ! -x "${S2M_VENV}/bin/python" ]]; then
    echo "Build the venv first: bash scripts/bootstrap_sam2matting.sh env" >&2
    exit 1
  fi
  S2M_CHECKPOINTS="${S2M_CHECKPOINTS}" S2M_HF_REPO="${S2M_HF_REPO}" \
  S2M_HF_REVISION="${S2M_HF_REVISION}" "${S2M_VENV}/bin/python" - <<'PY'
import os
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(os.environ["S2M_CHECKPOINTS"])
repo = os.environ["S2M_HF_REPO"]
revision = os.environ["S2M_HF_REVISION"]
names = [
    "SAM2Matting-SAM2.1Base+.pt",
    "SAM2Matting-SAM2.1Tiny.pt",
    "SAM2Matting-SAM3.pt",
]
for name in names:
    destination = target / name
    if destination.is_file():
        print(f"[sam2matting-bootstrap] already present: {destination}")
        continue
    print(f"[sam2matting-bootstrap] downloading {name}")
    target.mkdir(parents=True, exist_ok=True)
    # local_dir downloads straight to the shared checkpoints directory. Going via
    # the default HF cache and copying would need twice the space, and these
    # checkpoints total ~3.9 GB on a root filesystem that is usually near full.
    with tempfile.TemporaryDirectory(dir=str(target)) as staging:
        downloaded = hf_hub_download(
            repo_id=repo,
            revision=revision,
            filename=f"checkpoints/{name}",
            local_dir=staging,
        )
        os.replace(downloaded, destination)
    print(f"[sam2matting-bootstrap] wrote {destination}")
PY
}

write_runtime_lock() {
  mkdir -p "${LOCK_DIR}"
  local commit="unknown" torch_version="unknown" python_version="unknown"
  if [[ -d "${S2M_HOME}/.git" ]]; then
    commit="$(git -C "${S2M_HOME}" rev-parse HEAD)"
  fi
  if [[ -x "${S2M_VENV}/bin/python" ]]; then
    torch_version="$("${S2M_VENV}/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
    python_version="$("${S2M_VENV}/bin/python" -c 'import platform; print(platform.python_version())' 2>/dev/null || echo unknown)"
  fi
  S2M_LOCK_PATH="${RUNTIME_LOCK}" S2M_COMMIT="${commit}" S2M_TORCH="${torch_version}" \
  S2M_PY="${python_version}" S2M_VENV="${S2M_VENV}" S2M_HOME="${S2M_HOME}" \
  S2M_CHECKPOINTS="${S2M_CHECKPOINTS}" S2M_HF_REPO="${S2M_HF_REPO}" \
  S2M_HF_REVISION="${S2M_HF_REVISION}" S2M_REPO_URL="${S2M_REPO_URL}" \
  python3 - <<'PY'
import hashlib, json, os, time
from pathlib import Path

checkpoints = Path(os.environ["S2M_CHECKPOINTS"])
digests = {}
for path in sorted(checkpoints.glob("*.pt")):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    digests[path.name] = digest.hexdigest()

payload = {
    "repo_url": os.environ["S2M_REPO_URL"],
    "commit": os.environ["S2M_COMMIT"],
    "venv": os.environ["S2M_VENV"],
    "home": os.environ["S2M_HOME"],
    "checkpoints_dir": str(checkpoints),
    "checkpoint_repo": os.environ["S2M_HF_REPO"],
    "checkpoint_revision": os.environ["S2M_HF_REVISION"],
    "checkpoint_sha256": digests,
    "torch_version": os.environ["S2M_TORCH"],
    "python_version": os.environ["S2M_PY"],
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
path = Path(os.environ["S2M_LOCK_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[sam2matting-bootstrap] wrote {path}")
PY
}

case "${target}" in
  all)
    fetch_source
    build_env
    fetch_checkpoints
    write_runtime_lock
    ;;
  source)
    fetch_source
    write_runtime_lock
    ;;
  env)
    build_env
    write_runtime_lock
    ;;
  checkpoints)
    fetch_checkpoints
    write_runtime_lock
    ;;
  lock)
    write_runtime_lock
    ;;
  *)
    echo "Unknown target: ${target} (expected: all | source | env | checkpoints | lock)" >&2
    exit 2
    ;;
esac

cat <<MSG

[sam2matting-bootstrap] Done.

  venv:        ${S2M_VENV}
  source:      ${S2M_HOME}
  checkpoints: ${S2M_CHECKPOINTS}
  lock:        ${RUNTIME_LOCK}

Launch the production UI as usual; pick "SAM2Matting SAM2.1 Base+" as the matting backend:
    bash "${REPO_ROOT}/scripts/run_production_frame_ui.sh"
MSG
