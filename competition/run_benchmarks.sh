#!/usr/bin/env bash
# 三项精度 Benchmark（Daily-Omni / Video-MME / Seed-TTS），参数与官方精度评测一致
# 用法: competition/run_benchmarks.sh <daily-omni|video-mme|seed-tts> [数据集路径等参数]
set -euo pipefail
cd "$(dirname "$0")/.." && source competition/env.sh

TARGET="${1:?用法: run_benchmarks.sh <daily-omni|video-mme|seed-tts> [路径参数]}"
shift || true

service_running || fail "服务未运行，请先执行 competition/start_service.sh"

bench_common=(
    --port "$OMP_PORT"
    --trust-remote-code
    --model "$OMP_SERVED_NAME"
    --tokenizer "$OMP_TOKENIZER"
    --endpoint /v1/chat/completions
    --backend openai-chat-omni
    --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf
)

case "$TARGET" in
    daily-omni)
        VIDEO_DIR="${1:?缺少视频目录}"; QA_JSON="${2:?缺少 QA json}"
        vllm bench serve --omni "${bench_common[@]}" \
            --dataset-name daily-omni --num-prompts 2000 --no-oversample \
            --temperature 0 --output-len 512 \
            --daily-omni-input-mode all --daily-omni-pack-mode minicpm-interleave \
            --daily-omni-video-dir "$VIDEO_DIR" --daily-omni-qa-json "$QA_JSON" \
            --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}'
        ;;
    video-mme)
        VROOT="${1:?缺少 Video-MME 本地根目录}"
        vllm bench serve --omni "${bench_common[@]}" \
            --dataset-name video-mme --dataset-path "$VROOT" --num-prompts 2700 \
            --no-oversample --disable-shuffle \
            --temperature 0 --output-len 128 \
            --videomme-pack-mode minicpm-frames --videomme-max-frames 96 --videomme-duration all \
            --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}'
        ;;
    seed-tts)
        # SEED_TTS_SIM_EVAL=1 开启 ASV SIM 评测（WavLM 说话人相似度）
        vllm bench serve --omni "${bench_common[@]}" \
            --dataset-name seed-tts --dataset-path "$OMP_SEED_TTS_PATH" \
            --seed-tts-locale zh --num-prompts 2020 --no-oversample --disable-shuffle \
            --seed-tts-wer-eval --seed-tts-wer-save-items --temperature 0 \
            --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}'
        ;;
    *) fail "未知目标: $TARGET（可选 daily-omni | video-mme | seed-tts）" ;;
esac
