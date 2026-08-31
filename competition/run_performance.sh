#!/usr/bin/env bash
# Seed-TTS（中文）性能评测 —— 官方口径：单并发、预热 2 轮
# 用法: competition/run_performance.sh [subset|full]
#   subset（默认）: 32 条快速口径，用于迭代验证
#   full         : 全量 2020 条，用于最终提交数据
set -euo pipefail
cd "$(dirname "$0")/.." && source competition/env.sh

MODE="${1:-subset}"
NUM_PROMPTS=2020
[[ "$MODE" == "subset" ]] && NUM_PROMPTS=32

service_running || fail "服务未运行，请先执行 competition/start_service.sh"

log "性能评测: 模式=$MODE 条数=$NUM_PROMPTS（单并发 / --num-warmups 2 / zh）"
SEED_TTS_SIM_EVAL=1 vllm bench serve --omni \
    --port "$OMP_PORT" \
    --max-concurrency 1 --num-warmups 2 \
    --dataset-name seed-tts --dataset-path "$OMP_SEED_TTS_PATH" \
    --seed-tts-locale zh --num-prompts "$NUM_PROMPTS" --no-oversample --disable-shuffle \
    --model "$OMP_SERVED_NAME" --trust-remote-code \
    --tokenizer "$OMP_TOKENIZER" \
    --endpoint /v1/chat/completions --backend openai-chat-omni \
    --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
    --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}'
