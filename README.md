# MiniCPM-o 4.5 × vLLM-Omni 推理优化（昇腾 910B）

基于 vLLM-Omni 的 MiniCPM-o 4.5 全模态推理优化实践：在**昇腾 910B 单卡、原始 bf16 权重（非量化）**上，对三级流水线（Thinker 8B 理解 → Talker 语音生成 → Code2Wav 波形合成）做了 28 个 commit 的框架层 / 模型层 / 算子层优化，支持标准 chat 服务与**原生全双工语音服务**，精度全部保持。

## 性能报告（单并发 × Seed-TTS 32 条，vs 官方 910B 基线）

| 指标 | 官方 910B 基线 | 本仓库最终版本 | 变化 |
|---|---|---|---|
| RTF（音频实时率，越低越好） | 0.83 | **0.2629** | **-68.3%** |
| TTFT（首 token 延迟） | 442 ms | **103.3 ms** | **-76.6%** |
| TTFP（首音频包延迟） | 1732 ms | **599.66 ms** | **-65.4%** |

- 评测协议：Seed-TTS EN 前 32 条、`request-rate inf`、3 次预热、no-oversample、disable-shuffle；`vllm bench serve --omni`（`--backend openai-chat-omni`），与官方 `tests/dfx/perf` 口径一致。
- 精度：多模态问答准确率、语音 WER、说话人相似度、Daily-Omni/Video-MME 全部保持不低于官方基线（W8A8 量化实验另行验证后未纳入本仓库默认链路）。

## 背景

MiniCPM-o 4.5 在 vLLM-Omni 上按三个独立引擎进程跑一条流水线：8B 的 Thinker 理解多模态输入并生成文本 / TTS 计划，20 层小 Talker 把语义逐 token 翻译成音频 codec token，Code2Wav（流匹配 DiT + HiFT 声码器）把 codec token 合成波形。

逐 token 解码的每一步都要付一整轮引擎往返（调度 → 跨进程 → 前向 → 采样 → 回传 → 输出处理），对 20 层的小模型而言**固定 host 开销远超设备计算本身**——这是 RTF 的第一瓶颈；Code2Wav 的流匹配整循环和声码器由几百个逐算子下发的微型 kernel 组成，是第二瓶颈； thinker 会越过分界符多走"没人消费"的空步、跨进程隐状态传输按 Python 对象逐元素序列化，是第三、第四瓶颈。

本仓库沿"profile 证据 → 预期指标 → A/B 验证"的方法论组织了五条优化主线：**摊薄**（多步合并解码）、**图化**（CFM/声码器/采样链 NPUGraph）、**造轮子**（AscendC 融合算子）、**少算**（投机解码 + 早停）、**搬运**（跨进程单拷贝传输）。每一项都带 env 回滚开关，任何一项失败都不会破坏正确性（fail-closed）。

## 快速开始

环境要求：昇腾 910B 单卡、CANN + torch_npu、vllm-ascend、`pip install stepaudio2-minicpmo`（talker 依赖）。

```bash
# 1) 标准服务（OpenAI 兼容 chat completions，支持文本+音频输出）
bash start_serving.sh [端口，默认 8099] [模型路径，默认 openbmb/MiniCPM-o-4_5]

# 2) 全双工服务（原生 Realtime WebSocket 运行时）
bash start_duplex.sh [端口，默认 8099]
```

服务启动需要数分钟（三阶段引擎初始化 + 图捕获），就绪后 `/health` 返回 200。

### 标准服务使用

```bash
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "介绍一下你自己"}],
    "modalities": ["text", "audio"],
    "stream": true
  }'
```

音频以 base64 WAV 帧的形式随流式增量返回（`delta.content` 中 `UklGR...` 开头的片段）。完整基准复现：

```bash
vllm bench serve --omni --host localhost --port 8099 \
  --model openbmb/MiniCPM-o-4_5 --endpoint /v1/chat/completions --backend openai-chat-omni \
  --request-rate inf --num-prompts 32 --max-concurrency 1 --no-oversample --disable-shuffle \
  --num-warmups 3 --trust-remote-code --dataset-name seed-tts \
  --dataset-path <seedtts_testset> --seed-tts-locale en \
  --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}' \
  --percentile-metrics ttft,e2el,audio_ttfp,audio_rtf
```

### 全双工服务使用

全双工运行时通过 `vllm_omni/deploy/minicpmo_4_5_duplex.yaml` 启用（固定 Thinker → Talker → Code2Wav 拓扑、双会话上限、软打断策略），客户端经 WebSocket `/v1/realtime?duplex=1` 连接，实时流式送入 16 kHz 单声道 PCM16 音频并实时取回语音回复（支持插话 / barge-in）。

CLI demo（流式送入一个 WAV，收回文本 + 音频产物）：

```bash
bash start_duplex.sh 8099 &
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
    --url ws://localhost:8099/v1/realtime?duplex=1 \
    --model openbmb/MiniCPM-o-4_5 \
    --input-wav /path/to/input_16k_mono_pcm16.wav \
    --ref-audio /path/to/ref_audio.wav \
    --output-dir /tmp/minicpmo_realtime_duplex_demo --require-audio
```

浏览器客户端（页面 + 同源 WebSocket 代理）：

```bash
python -m examples.online_serving.minicpmo.realtime_web \
    --port 7862 --ws-backend ws://127.0.0.1:8099 --ref-audio /path/to/ref_audio.wav
# 打开 http://<host>:7862/
```

架构与生命周期不变量见 [`vllm_omni/experimental/fullduplex/DESIGN.md`](vllm_omni/experimental/fullduplex/DESIGN.md)；上游原始说明见 [`docs/README_upstream.md`](docs/README_upstream.md)。

## 优化 Commit 全览

基线为 vLLM-Omni 官方 `minicpm-challenge` 分支（`11dcde9`）。以下 28 个 commit 按主题分组，全部独立可回滚（env 开关或 `git revert`）。

**调度与解码（摊薄引擎往返）**

| Commit | 优化 |
|---|---|
| `a63bcba4` | **P24 runner-local K 窗口解码**：调度器把单步批发为 K-token 步（预分配 KV + 占位符记账），runner 单次 `execute_model` 内连跑 K 步，摊薄引擎循环固定开销；fail-closed 三层门控，25 项单测 |
| `d2f21710` | P24 默认窗口 K 8 → 10 |
| `e83f4c48` | **P24 修复**：修三层拒绝门（架构名不匹配 / prefer_model_sampler / mm_inputs / duplex hook / 过期批视图），窗口真实生效；codec 流逐位等价验证；c1x32 实测 E2EL -17.5%、RTF -16.8% |
| `7f2fa613` | C2.1 Code2Wav CFM 步数 10 → 3 |
| `312b3cef` | B3 跨 stage 交接张量保活 |
| `d50f6da2` | T2/T3/T4 + Code2Wav 预热 |
| `275eacae` | T3 修复 `_find_tts_span` 的 bos 索引偏移 |
| `ed3d2606` | B4 连接器轮询参数接入 recv/save 循环 |

**图捕获（消除 kernel 下发风暴）**

| Commit | 优化 |
|---|---|
| `b2aee855` | **P5 FULL_AND_PIECEWISE 图模式**：decode 走整图回放，消除逐层 attention 气泡 |
| `ca6d91a4` | **P15 CFM 解码整循环捕获 NPUGraph** |
| `7f2fa613`→`e5ae0491` | P13 Code2Wav setup 提前，prompt 预热与 talker 解码重叠 |
| `1aead621` | **P16 HiFT 声码器整链捕获 NPUGraph**（f0 预测、SineGen、STFT、decode 链） |
| `e90c4729` | P16b 只捕获首 chunk + 稳态两个宽度（防尾部宽度捕获风暴，修复 RTF 回退） |
| `e111d741` | **P27 确定性采样链捕获 NPUGraph**（head 线性 + Gumbel 尾部） |
| `17de9aba` | P27 默认关闭（图回放固定成本超过收益），env 可选开启 |

**算子开发（AscendC 自定义融合）**

| Commit | 优化 |
|---|---|
| `ce657185` | **P28 DiT adaLN-Zero 调制链融合为单个 AscendC kernel**（每 CFM chunk 84 处调用点，每处 4-6 个微型 kernel → 1 个）；tiling 按值缓存保证图捕获兼容，fail-closed 降级 |
| `32979939` | **P28b q_norm+k_norm 双 LayerNorm 融合**（batched-heads 单 kernel 单屏障） |

**少算（投机 + 早停）**

| Commit | 优化 |
|---|---|
| `3f31a83a` | **P17 代码级默认**：AR 阶段强制 FULL_AND_PIECEWISE 图 + 静态 kernel；thinker ngram 投机 K=12；thinker 早停（TTS 边界 token 注入 stop_token_ids） |
| `7f16edac` | P17b 修复 CPU ngram 草稿与异步调度的兼容 |
| `499e4878` | P18 外部复盘正确性修复（handoff hidden 偏移 / chat 音频输出 key / 音频 token 预算对齐原生启发式） |

**搬运与摊销（跨进程与缓存）**

| Commit | 优化 |
|---|---|
| `c4ef8da3` | **P19-P23 五连**：二进制 handoff 传输（~7e5 Python 对象 → 单拷贝 bytes，25-45ms/请求 → 1 次拷贝）、CFM 解析切线跳步、投机验证桶对齐（K+1=16 零 padding）、预热宽度覆盖、Gumbel-max codec 采样（RNG 出采样 kernel 链，add+argmax 两个 kernel） |
| `a63bcba4`→`24672e25` | **P25 初始状态模板缓存**（同参考音频的 prompt conformer / speaker 投影 / prompt-mel CFM 结果模板化，40-90ms/新请求） |
| `fd9ee897` | P25 修复空模板库 UnboundLocalError + speaker 投影缓存 |
| `8d8879dc` | **P26 CPU 绑核**：orchestrator + 三 stage 引擎核物理隔离（cgroup 安全、防御式），压制 RTF 方差 |
| `152d44c0` | 竞赛提交辅助脚本 |

## 仓库结构

```
vllm_omni/                     优化后的 vLLM-Omni 源码
  core/sched/                  调度器（P24 窗口规划 / 记账 / 对账）
  platforms/npu/worker/        NPU 运行器（K 步窗口执行 / 计时插桩）
  model_executor/models/minicpmo_4_5/   模型实现（图捕获 / AscendC 算子 / 采样链）
  experimental/fullduplex/     原生全双工运行时
  deploy/                      部署配置（标准 / 全双工 yaml）
docs/                          上游文档与设计说明
examples/online_serving/minicpmo/   全双工 demo / 浏览器客户端
start_serving.sh               一键启动标准服务
start_duplex.sh                一键启动全双工服务
```

## 精度声明

所有优化均为以下三类之一：无损（输出逐位一致，如多步窗口、图捕获、传输重构）、分布等价（Gumbel 采样，WER/SIM 门禁通过）、有界数值变更（CFM 切线跳步，WER/SIM 门禁通过）。最终精度四项指标全部保持不低于官方基线。
