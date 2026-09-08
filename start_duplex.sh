#!/usr/bin/env bash
# 一键启动 MiniCPM-o 4.5 全双工（Realtime WebSocket）服务
# 用法: bash start_duplex.sh [端口号] [模型路径]
#   默认端口 8099。启动后可用 examples/online_serving/minicpmo/realtime_duplex_demo.py
#   或浏览器客户端（examples/online_serving/minicpmo/realtime_web.py）连接
#   ws://<host>:<port>/v1/realtime?duplex=1
set -uo pipefail
PORT="${1:-8099}"
MODEL="${2:-openbmb/MiniCPM-o-4_5}"

cd "$(dirname "$0")" || exit 1

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f StageEngineCore 2>/dev/null
pkill -9 -f spawn_main 2>/dev/null; pkill -9 -f forkserver 2>/dev/null; sleep 3
rm -f /dev/shm/chatcmpl* 2>/dev/null

unset ASCEND_RT_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

echo "[start_duplex] serving $MODEL on port $PORT (native full-duplex runtime)"
echo "  Realtime WS: ws://localhost:$PORT/v1/realtime?duplex=1"
echo "  Health:      http://localhost:$PORT/health"
echo "  Demo client: python examples/online_serving/minicpmo/realtime_duplex_demo.py \\"
echo "      --url ws://localhost:$PORT/v1/realtime?duplex=1 --model openbmb/MiniCPM-o-4_5 \\"
echo "      --input-wav <16k-mono-pcm16.wav> --ref-audio <ref.wav> --output-dir /tmp/out"

exec vllm serve "$MODEL" --omni --served-model-name openbmb/MiniCPM-o-4_5 \
  --trust-remote-code --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
  --stage-init-timeout 900 --allowed-local-media-path /workspace \
  --host 0.0.0.0 --port "$PORT"
