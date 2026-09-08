#!/usr/bin/env bash
# 一键启动 MiniCPM-o 4.5 普通服务（HTTP chat completions，含音频输出）
# 用法: bash start_serving.sh [端口号] [模型路径]
#   默认端口 8099，模型默认 openbmb/MiniCPM-o-4_5（可指向本地 checkpoint 目录）
set -uo pipefail
PORT="${1:-8099}"
MODEL="${2:-openbmb/MiniCPM-o-4_5}"

cd "$(dirname "$0")" || exit 1

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f StageEngineCore 2>/dev/null
pkill -9 -f spawn_main 2>/dev/null; pkill -9 -f forkserver 2>/dev/null; sleep 3
rm -f /dev/shm/chatcmpl* 2>/dev/null

unset ASCEND_RT_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

echo "[start_serving] serving $MODEL on port $PORT (standard chat pipeline)"
echo "  OpenAI API:  http://localhost:$PORT/v1/chat/completions"
echo "  Health:      http://localhost:$PORT/health"

exec vllm serve "$MODEL" --omni --served-model-name openbmb/MiniCPM-o-4_5 \
  --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --stage-init-timeout 900 --allowed-local-media-path /workspace \
  --host 0.0.0.0 --port "$PORT"
