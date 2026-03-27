# Qwen3.5-35B-A3B on GCP L4 (24GB VRAM)

Deploy Qwen3.5-35B-A3B Uncensored (HauhauCS Aggressive Q4_K_M GGUF, 20GB + 858MB mmproj) on a single NVIDIA L4 GPU with llama.cpp server, Open WebUI, and nginx reverse proxy. Multimodal capable via mmproj vision projector.

## Architecture

```
Internet → nginx (:80/:443)
              ├── /v1/*    → llama.cpp server (:8080)  [OpenAI-compatible API]
              ├── /health  → llama.cpp server (:8080)
              └── /*       → Open WebUI (:3000)         [Chat UI]
```

## Quick Start

### 1. Create GCP Instance

```bash
gcloud compute instances create qwen35-serving-l4 \
  --project=$GCP_PROJECT \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --image=pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20260129 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --tags=llama-server
```

- **g2-standard-8**: 1x NVIDIA L4 (24GB VRAM), 8 vCPUs, 32GB RAM
- **Spot instance**: ~60-70% cheaper than on-demand (~$0.26/hr vs $0.86/hr)
- Deep Learning VM image: CUDA 12.8, NVIDIA driver 570 pre-installed
- Spot instances may be preempted; use `--instance-termination-action=STOP` to preserve disk

### 2. Firewall Rules

```bash
gcloud compute firewall-rules create allow-llama-server \
  --project=$GCP_PROJECT \
  --allow=tcp:8080 --target-tags=llama-server

gcloud compute firewall-rules create allow-llama-http \
  --project=$GCP_PROJECT \
  --allow=tcp:80 --target-tags=llama-server

gcloud compute firewall-rules create allow-llama-https \
  --project=$GCP_PROJECT \
  --allow=tcp:443 --target-tags=llama-server
```

### 3. Download Model

```bash
gcloud compute ssh qwen35-serving-l4 --project=$GCP_PROJECT --zone=us-central1-a

mkdir -p ~/models
# HauhauCS Aggressive Q4_K_M GGUF (~20GB) + mmproj (~858MB)
huggingface-cli download HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF \
  Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  mmproj-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  --local-dir ~/models
```

### 4. Deploy Services

```bash
# Copy config files
scp -r nginx/ docker-compose.yml <instance>:~/

# SSH into instance and run
docker compose up -d
```

Or manually:

```bash
# llama.cpp server
docker run -d --name llama-server --gpus all \
  -v ~/models:/models \
  -v ~/templates:/templates \
  -p 8080:8080 \
  --restart unless-stopped \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --mmproj /models/mmproj-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 98304 \
  --parallel 1 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --jinja \
  --threads 8 \
  --chat-template-file /templates/chat_template.jinja \
  --checkpoint-every-n-tokens 256 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --chat-template-kwargs '{"enable_thinking": false}'

# Open WebUI
docker run -d --name open-webui \
  -p 3000:8080 \
  -v open-webui-data:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  --add-host=host.docker.internal:host-gateway \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

### 5. nginx Reverse Proxy

```bash
sudo apt install -y nginx
sudo cp nginx/qwen-api /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/qwen-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

For HTTPS with Cloudflare Origin cert:
```bash
sudo mkdir -p /etc/nginx/ssl
# Place your Cloudflare Origin cert + key:
sudo cp cf-origin-cert.pem /etc/nginx/ssl/
sudo cp cf-origin-key.pem /etc/nginx/ssl/
```

### 6. DNS (Cloudflare)

Create an A record pointing your domain to the instance's external IP:
```
Type: A
Name: qwen-api (or your subdomain)
Content: <INSTANCE_EXTERNAL_IP>
Proxy: OFF (DNS only, gray cloud)
```

## Parameters

### llama.cpp Server

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--ctx-size` | 98304 | ~96K context. Current config on L4 24GB with Q4_K_M + mmproj + KV cache quantization. |
| `--mmproj` | mmproj-...-f16.gguf | Multimodal vision projector (858MB). Enables image understanding. |
| `--parallel` | 1 | Single request slot (trade-off for longer context). |
| `--cache-type-k` | q4_0 | Quantize KV cache keys to 4-bit (saves ~75% KV VRAM vs fp16). |
| `--cache-type-v` | q4_0 | Quantize KV cache values to 4-bit. |
| `--n-gpu-layers` | 999 | Offload all layers to GPU |
| `--flash-attn` | on | Required for memory efficiency |
| `--jinja` | - | Enables Jinja2 chat templates (required for Qwen3.5) |
| `--threads` | 8 | CPU threads for prompt processing |
| `--chat-template-file` | `/templates/chat_template.jinja` | Fixed Jinja template that preserves empty think blocks in non-thinking mode (see issue #7 below) |
| `--checkpoint-every-n-tokens` | 256 | Create more frequent checkpoints during prefill to reduce reprocessing distance for hybrid/recurrent models (see issue #1 below) |
| `--chat-template-kwargs` | `{"enable_thinking": false}` | Disable thinking mode by default (saves tokens) |

### VRAM Budget

The L4 has 22,383 MiB (~22 GB) usable VRAM. Model weights (Q4_K_M, 20GB) + mmproj (858MB) take ~20.8 GB, leaving ~1.2 GB for KV cache + compute buffers.

| Config | KV Cache | VRAM Status | Notes |
|--------|----------|-------------|-------|
| ctx=98K, parallel=1, KV q4_0 | ~540 MB | OK | **Current config**. With mmproj loaded. |
| ctx=131K, parallel=1, KV q4_0 | ~720 MB | OK | May be tight with mmproj. |
| ctx=180K, parallel=1, KV q4_0 | ~990 MB | Borderline | Would need to drop mmproj. |
| ctx=32K, parallel=1, KV fp16 | 640 MB | OK | Safe fallback. |
| ctx=32K, parallel=4, KV fp16 | ~2560 MB | OOM | Not feasible with mmproj. |

### KV Cache Quantization Trade-offs

Using `--cache-type-k q4_0 --cache-type-v q4_0` reduces KV cache VRAM by ~75% vs fp16, enabling much longer context. The trade-off:

- **Short conversations (<32K)**: virtually no quality difference
- **Long context (32K-128K)**: minor precision loss in long-range attention
- **Very long context (128K+)**: may lose fine-grained details from early tokens; best for summarization, less ideal for needle-in-haystack retrieval
- Alternative: `--cache-type-k q8_0 --cache-type-v q8_0` for better quality at ~50% savings (max ~131K on L4)

### Performance

Measured on NVIDIA L4 (24GB VRAM), llama.cpp `server-cuda`, KV cache q4_0, thinking disabled, single request slot.

#### Qwen3.5-35B-A3B (MoE, Q4_K_M, 20GB) vs Qwen3.5-27B Dense (Q6_K, 21GB)

| Metric | 35B-A3B (MoE) | 27B Dense | Ratio |
|--------|---------------|-----------|-------|
| **Max Context (no mmproj)** | **256K** (22.4 GB VRAM) | **68K** (22.5 GB VRAM) | **3.8x** |
| Prefill (5K tokens) | 1,877 tok/s | 557 tok/s | **3.4x** |
| Decode | 67 tok/s | 10 tok/s | **6.7x** |
| TTFT (short prompt) | ~67 ms | ~235 ms | **3.5x** |

**Why the difference:**
- **35B-A3B (hybrid MoE)**: Only 10 of 40 layers use full attention (need KV cache), the other 30 use GDN (linear, no KV cache) → can push to model's native max 256K
- **27B Dense**: All 28 layers need KV cache → maxes out at 68K on L4 24GB with KV q4_0 quantization
- Both use `--cache-type-k q4_0 --cache-type-v q4_0` (4-bit KV cache quantization)

The MoE variant activates only 3B parameters per token (of 35B total), while the 27B dense model runs all 27B parameters. Despite similar GGUF file sizes (~20-21GB), the MoE architecture delivers 3-7x better throughput and 3.8x longer context on L4.

#### Raw Timings (35B-A3B, ctx 98K)

| Metric | Value |
|--------|-------|
| Prefill | ~1980 tok/s (2316 tok in 1.17s) |
| Decode | 69.3 tok/s |

### Reasoning Quality Comparison

Tested with a multi-step math riddle (sheep problem requiring tracking state across 3 operations). Both models generated 2000 tokens.

| Config | Time | Thinking | Result |
|--------|------|----------|--------|
| 35B-A3B OFF | 30.3s | - | ✅ Steps 1-2 correct, ❌ verbose, no clear final answer |
| 35B-A3B ON | 30.3s | 6636 chars | ⚠️ Answer truncated (thinking consumed max_tokens) |
| 27B Dense OFF | 200.4s | - | ✅ Steps 1-2 correct, ❌ verbose, no clear final answer |
| 27B Dense ON | 200.4s | 6499 chars | ⚠️ Answer truncated (thinking consumed max_tokens) |

**Findings:**
- **No quality difference**: Both models produce nearly identical reasoning and get stuck on the same edge case (odd-number division)
- **Speed advantage**: 35B-A3B is **6.6x faster** (30.3s vs 200.4s) for identical output quality
- **Thinking mode**: Generates internal reasoning but causes answer truncation; thinking OFF is more reliable for complete answers

## Known Issues / Pitfalls

### 1. KV Cache for Hybrid/Recurrent Models

**Background**: Qwen3.5 uses a hybrid architecture with 30 Gated DeltaNet (linear attention, recurrent) layers + 10 full attention layers. Early llama.cpp versions had broken cache reuse for this architecture - every request triggered full prompt re-processing ("forcing full prompt re-processing due to lack of cache data").

**Root Cause**: The checkpoint validation logic used `n_swa = max(1, llama_model_n_swa(model))`. Since Qwen3.5 has no SWA (`sliding_window` is null), `n_swa` defaulted to 1. The GDN recurrent state pushed `pos_min` to the end of the sequence, so `pos_min > n_swa` was always true, discarding the cache.

**Status**: **FIXED** in upstream llama.cpp via two PRs:
- [#16382](https://github.com/ggml-org/llama.cpp/pull/16382) (2025-10-03): "context checkpointing for hybrid and recurrent models"
- [#19045](https://github.com/ggml-org/llama.cpp/pull/19045) (2026-01-25): "fix prompt cache for recurrent models"

**Docker image `ghcr.io/ggml-org/llama.cpp:server-cuda` (2026-03-13+ builds) includes both fixes.** No manual patching needed.

**Verified Performance** (80-turn multi-turn conversation, 32K context, L4 GPU):
- Prefill: 320-400ms stable (only processes new tokens, cache reused)
- `cache_n` grows correctly from 0 to 32K tokens
- Decode: 7.0-8.3s for 400 tokens (~39 tok/s)
- Total wall time: 641s for 80 turns

**Ablation Study** (Snake game iterative coding, 10 turns to 32K context):

| Config | Prefill Range | Decode tok/s | Cache Reprocess |
|--------|--------------|--------------|-----------------|
| patched template + checkpoint-256 | 249-3072ms | 56→49 | 0 times |
| patched template, no checkpoint | 238-3041ms | 56→49 | 0 times |
| default template, no checkpoint | 238-3621ms | 56→48 | 1 time (Turn 2 only) |

**Conclusion**: The patched Jinja template (`--chat-template-file`) is the only meaningful fix. `--checkpoint-every-n-tokens 256` has no measurable impact. The default template causes one cache miss on Turn 2 (when the template strips the empty `</think>` tag from Turn 1), then self-corrects.

### 2. `response_format: json_object` Doesn't Work

**Symptom**: Setting `response_format: {"type": "json_object"}` still produces markdown-fenced output (````json ... ````).

**Root Cause**: Grammar enforcement doesn't work properly with Qwen3.5's chat template. Related: [#20345](https://github.com/ggml-org/llama.cpp/issues/20345).

**Workaround**: Strip markdown fences and `<think>` tags in your client code:
```python
def clean_response(content: str) -> str:
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content[:-3].strip()
    return content
```

### 3. JSON Output Truncation

**Symptom**: Long JSON responses get cut off mid-string, causing parse failures.

**Root Cause**: `max_tokens` limit reached before the JSON array is complete. Qwen3.5's tokenizer uses more tokens for non-Latin scripts (Hindi, Korean, etc.).

**Workaround**: Implement truncated JSON recovery:
```python
def parse_json_list(content: str) -> list | None:
    content = clean_response(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Recover truncated JSON array
        last = content.rfind("},")
        if last == -1:
            last = content.rfind("}")
        if last > 0:
            try:
                return json.loads(content[:last + 1] + "]")
            except json.JSONDecodeError:
                pass
        return None
```

### 4. Thinking Mode

- Default: **disabled** via `--chat-template-kwargs '{"enable_thinking": false}'`
- To enable per-request: set `enable_thinking: true` in the request body
- When thinking is enabled, grammar/structured output is completely bypassed ([#20345](https://github.com/ggml-org/llama.cpp/issues/20345))
- On Windows, use: `--chat-template-kwargs "{\"enable_thinking\":false}"`

### 6. Structured JSON Output

- `response_format: {"type": "json_object"}` does **NOT** work -- still wraps output in markdown fences
- `response_format: {"type": "json_schema", "json_schema": {...}}` **WORKS** -- grammar constrained decoding enforces valid JSON
- Fixed in PR [#20223](https://github.com/ggml-org/llama.cpp/pull/20223) (commit 62b8143, merged 3/8), included in b8323+
- ~23% slower than unconstrained output, but eliminates parse failures entirely
- Use `json_schema` mode instead of `json_object` for reliable structured output

### 7. Jinja Template Think Block Bug (Multi-turn)

**Symptom**: In multi-turn conversations with `enable_thinking: false`, the model reprocesses the full prompt on every turn even when the conversation prefix hasn't changed.

**Root Cause**: The default Jinja template injects `<think>\n\n</think>\n\n` before generation when thinking is disabled. On the next turn, the template strips the `</think>` tag from the previous assistant message's history. From llama.cpp's perspective, the prompt changed, triggering full reprocessing.

**Fix**: Use the patched template in `templates/chat_template.jinja` (from [Reddit r/LocalLLaMA](https://reddit.com/r/LocalLLaMA/comments/1rt0g8y)). The fix checks whether the think block has actual content: if yes, strip it (save context); if empty, keep it (avoid triggering reprocess).

```bash
--chat-template-file /templates/chat_template.jinja
```

**Note**: This only affects multi-turn conversations (thinking disabled). Single-request batch inference is not affected, but it's still recommended to use the patched template.

### 8. Docker Image Version

- Using `ghcr.io/ggml-org/llama.cpp:server-cuda` (latest)
- Tested version: 2026-03-13 build (includes hybrid cache fixes #16382 + #19045)
- **Must use builds from 2026-01-25 or later** to get working KV cache for Qwen3.5
- Older images will trigger full prompt re-processing on every turn

### 9. Open WebUI Connection

- Open WebUI connects to llama.cpp via `OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1`
- `--add-host=host.docker.internal:host-gateway` is required for Docker-to-host networking
- Open WebUI is on port 3000 (mapped from container's 8080)

## Model Details

| Property | Value |
|----------|-------|
| Model | Qwen3.5-35B-A3B-Uncensored (HauhauCS Aggressive) |
| Quantization | Q4_K_M |
| File size | 20 GB model + 858 MB mmproj |
| Multimodal | Yes (via mmproj vision projector, f16) |
| Architecture | Hybrid: 30x Gated DeltaNet + 10x Gated Attention (MoE, 256 experts, 8 active) |
| Layer pattern | 10 x (3 x GDN-MoE + 1 x GA-MoE) |
| Context | 262K native, 98K current on L4 (with KV q4_0 + mmproj) |
| Active params | 3B (of 35B total) |
| Vocab | 248,320 |

## File Structure

```
├── README.md
├── docker-compose.yml
├── nginx/
│   └── qwen-api              # nginx site config
├── templates/
│   └── chat_template.jinja   # Patched Jinja template (fixes think block reprocessing)
└── scripts/
    └── setup.sh               # One-shot setup script
```
