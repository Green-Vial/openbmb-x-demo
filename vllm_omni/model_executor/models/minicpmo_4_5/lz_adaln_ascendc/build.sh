#!/usr/bin/env bash
# P28: build liblz_adaln_ops.so (AscendC kernel + torch extension host glue).
# Requires a CANN toolkit install; ASCEND_HOME_PATH is set up if missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for candidate in /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/ascend-toolkit/latest; do
        if [ -f "${candidate}/set_env.sh" ]; then
            # shellcheck disable=SC1091
            source "${candidate}/set_env.sh"
            break
        fi
    done
fi
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "lz_adaln build: ASCEND_HOME_PATH not found; install CANN toolkit" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"
cmake "${SCRIPT_DIR}" > cmake_config.log 2>&1 || { tail -n 40 cmake_config.log; exit 1; }
make -j4 > make.log 2>&1 || { tail -n 60 make.log; exit 1; }
echo "${SCRIPT_DIR}/build/liblz_adaln_ops.so"
