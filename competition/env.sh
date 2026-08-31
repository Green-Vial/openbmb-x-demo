#!/usr/bin/env bash
# 落云宗 · vLLM-Omni 子赛道 —— 提交脚本公共配置
# 所有竞赛脚本 source 本文件；评测环境差异通过环境变量覆盖，无需改脚本。

# ---- 服务端参数（与官方评测流程一致）----
export OMP_MODEL_PATH="${OMP_MODEL_PATH:-/models/MiniCPM-o-4_5}"      # 官方指定模型
export OMP_SERVED_NAME="${OMP_SERVED_NAME:-openbmb/MiniCPM-o-4_5}"    # bench 客户端 --model 需与之一致
export OMP_HOST="${OMP_HOST:-127.0.0.1}"
export OMP_PORT="${OMP_PORT:-8091}"
# 官方 deploy config：评测时由官方基准分支提供；本地演练时可指向仓库内 deploy/minicpmo_4_5.yaml
export OMP_DEPLOY_CONFIG="${OMP_DEPLOY_CONFIG:-}"

# ---- 客户端评测参数（官方口径）----
export OMP_SEED_TTS_PATH="${OMP_SEED_TTS_PATH:?请通过 OMP_SEED_TTS_PATH 指定 Seed-TTS 数据集路径}"
export OMP_TOKENIZER="${OMP_TOKENIZER:-$OMP_MODEL_PATH}"

# ---- 运行环境（离线评测容器惯例）----
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { log "ERROR: $*" >&2; exit 1; }

# 阻塞等待服务就绪（最多 15 分钟，覆盖三 stage 初始化 + 图捕获预热）
wait_service_ready() {
    local url="http://${OMP_HOST}:${OMP_PORT}/health"
    local deadline=$((SECONDS + 900))
    log "等待服务就绪: $url"
    while (( SECONDS < deadline )); do
        if curl -sf "$url" >/dev/null 2>&1; then log "服务已就绪"; return 0; fi
        sleep 10
    done
    fail "等待服务就绪超时（15 分钟）"
}

# 返回当前服务是否已在运行
service_running() { curl -sf "http://${OMP_HOST}:${OMP_PORT}/health" >/dev/null 2>&1; }
