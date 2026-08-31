#!/usr/bin/env bash
# 启动 MiniCPM-o 4.5 推理服务（与官方评测启动方式一致：vllm serve --omni）
# 用法: competition/start_service.sh
set -euo pipefail
cd "$(dirname "$0")/.." && source competition/env.sh

DEPLOY_ARGS=()
if [[ -n "$OMP_DEPLOY_CONFIG" ]]; then
    DEPLOY_ARGS+=(--deploy-config "$OMP_DEPLOY_CONFIG")
else
    log "未设置 OMP_DEPLOY_CONFIG，使用官方基准分支默认 deploy config（评测环境行为）"
fi

log "启动服务: model=$OMP_MODEL_PATH port=$OMP_PORT"
exec vllm serve "$OMP_MODEL_PATH" \
    --omni \
    --served-model-name "$OMP_SERVED_NAME" \
    --trust-remote-code \
    --host "$OMP_HOST" \
    --port "$OMP_PORT" \
    --stage-init-timeout 900 \
    "${DEPLOY_ARGS[@]}"
