#!/usr/bin/env bash
# 一键启动全模态交互 Demo：推理服务 + 官方 Web 前端
# 用法: competition/start_demo.sh
#   Web 界面: http://<本机IP>:7862
#   注意: 浏览器麦克风权限要求 HTTPS（localhost 除外）。
#   远程访问请配置 SSL 证书: OMP_DEMO_SSL_CERT / OMP_DEMO_SSL_KEY
set -euo pipefail
cd "$(dirname "$0")/.." && source competition/env.sh

DEMO_PORT="${OMP_DEMO_PORT:-7862}"

if service_running; then
    log "推理服务已在运行（http://${OMP_HOST}:${OMP_PORT}），跳过启动"
else
    log "先启动推理服务（后台进程，日志: /tmp/omp_service.log）"
    bash competition/start_service.sh >/tmp/omp_service.log 2>&1 &
    wait_service_ready
fi

DEMO_ARGS=(
    --minicpmo45-api-base "http://${OMP_HOST}:${OMP_PORT}/v1"
    --minicpmo45-model "$OMP_SERVED_NAME"
    --host 0.0.0.0
    --port "$DEMO_PORT"
)
if [[ -n "${OMP_DEMO_SSL_CERT:-}" && -n "${OMP_DEMO_SSL_KEY:-}" ]]; then
    DEMO_ARGS+=(--ssl-certfile "$OMP_DEMO_SSL_CERT" --ssl-keyfile "$OMP_DEMO_SSL_KEY")
    log "已启用 HTTPS（麦克风远程访问需要）"
fi

log "启动 Web Demo: http://0.0.0.0:${DEMO_PORT}"
exec python examples/online_serving/minicpmo/gradio_demo.py "${DEMO_ARGS[@]}"
