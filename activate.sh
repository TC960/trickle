# Set up and enter the project environment.  Usage:  source activate.sh
#
# Creates the venv with uv on first run, then activates it and exports the
# environment this machine needs. Safe to source repeatedly.
#
# Three machine-specific quirks are handled here, all of which cost real
# debugging time before they were understood:
#
#   1. On some machines ~/Desktop and ~/Documents are symlinks into a
#      cloud-synced folder, so anything written under them uploads. Model
#      weights (tens of GB) must live somewhere genuinely local, hence
#      AIRLLM_CACHE under ~/.cache. Check with `realpath ~/Desktop`.
#   2. Corporate TLS interception means Python does not trust the proxy CA that
#      curl accepts via the system keychain. We export the macOS trust store
#      once and point OpenSSL at it.
#   3. System sleep is set to 1 minute, which kills any long job. Run
#      `caffeinate -dims &` before training or large downloads.

# --- guard against being executed instead of sourced ------------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "This script must be sourced, not executed:"
    echo "    source activate.sh"
    exit 1
fi

_AIRLLM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# --- local cache (must NOT be under ~/Desktop, see note 1) ------------------
export AIRLLM_CACHE="${AIRLLM_CACHE:-$HOME/.cache/airllm-ternary}"
export HF_HOME="$AIRLLM_CACHE/hf-cache"
export AIRLLM_SHARDS="$AIRLLM_CACHE/shards-bitnet"
mkdir -p "$AIRLLM_CACHE" "$HF_HOME"

# --- TLS trust store (see note 2) -------------------------------------------
_CERT_BUNDLE="$AIRLLM_CACHE/certs/macos-trust.pem"
if [ ! -s "$_CERT_BUNDLE" ]; then
    echo "exporting macOS trust store for Python..."
    mkdir -p "$(dirname "$_CERT_BUNDLE")"
    security find-certificate -a -p \
        /System/Library/Keychains/SystemRootCertificates.keychain \
        > "$_CERT_BUNDLE" 2>/dev/null
    security find-certificate -a -p /Library/Keychains/System.keychain \
        >> "$_CERT_BUNDLE" 2>/dev/null
    security find-certificate -a -p ~/Library/Keychains/login.keychain-db \
        >> "$_CERT_BUNDLE" 2>/dev/null
    echo "  $(grep -c 'BEGIN CERTIFICATE' "$_CERT_BUNDLE") certificates"
fi
export SSL_CERT_FILE="$_CERT_BUNDLE"
export REQUESTS_CA_BUNDLE="$_CERT_BUNDLE"

# --- virtual environment ----------------------------------------------------
if [ ! -d "$_AIRLLM_ROOT/.venv" ]; then
    echo "creating .venv with uv..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "$_AIRLLM_ROOT/.venv" --python 3.12
        VIRTUAL_ENV="$_AIRLLM_ROOT/.venv" \
            uv pip install -r "$_AIRLLM_ROOT/requirements.txt"
    else
        echo "  uv not found, falling back to python -m venv (slower)"
        python3 -m venv "$_AIRLLM_ROOT/.venv"
        "$_AIRLLM_ROOT/.venv/bin/pip" install -q --upgrade pip
        "$_AIRLLM_ROOT/.venv/bin/pip" install -r "$_AIRLLM_ROOT/requirements.txt"
    fi
fi

# shellcheck disable=SC1091
source "$_AIRLLM_ROOT/.venv/bin/activate"

# Let `import airllm_ternary` work from anywhere.
export PYTHONPATH="$_AIRLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- report -----------------------------------------------------------------
printf '\n\033[1mairLLM + ternary\033[0m  environment ready\n'
printf '  python   %s\n' "$(python --version 2>&1 | cut -d' ' -f2) ($(which python))"
printf '  torch    %s\n' "$(python -c 'import torch; print(torch.__version__, "| mps:", torch.backends.mps.is_available())' 2>/dev/null || echo 'not installed')"
printf '  cache    %s\n' "$AIRLLM_CACHE"
if [ -f "$AIRLLM_SHARDS/manifest.json" ]; then
    printf '  shards   %s \033[32m(ready)\033[0m\n' "$AIRLLM_SHARDS"
else
    printf '  shards   \033[33mnot built\033[0m - run: python -m airllm_ternary.build_bitnet\n'
fi
cat <<'BANNER'

  python chat.py                    chat, 0.75 GB budget
  python chat.py --budget-gb 0.05   minimum footprint, heavy streaming
  pytest tests/ -q                  run the test suite
  caffeinate -dims &                stop the Mac sleeping during long jobs

BANNER

unset _AIRLLM_ROOT _CERT_BUNDLE
