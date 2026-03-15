# Qwen3.5-35B-A3B on GCP L4 (24GB VRAM)

Deploy Qwen3.5-35B-A3B (Unsloth Q4_K_XL GGUF, 21GB) on a single NVIDIA L4 GPU with llama.cpp server, Open WebUI, and nginx reverse proxy.

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
  --machine-type=g2-standard-16 \
  --image=pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20260129 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --tags=llama-server
```

- **g2-standard-16**: 1x NVIDIA L4 (24GB VRAM), 16 vCPUs, 64GB RAM
- Deep Learning VM image: CUDA 12.8, NVIDIA driver 570 pre-installed
- Cost: ~$0.86/hr (~$620/month)

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
# Unsloth Q4_K_XL GGUF (~21GB)
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
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
  -p 8080:8080 \
  --restart unless-stopped \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  --parallel 4 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --jinja \
  --threads 8 \
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
| `--ctx-size` | 32768 | Max 32K on L4 24GB with Q4_K_XL. Higher OOMs. |
| `--parallel` | 4 | 4 concurrent request slots. More slots = more VRAM per slot. |
| `--n-gpu-layers` | 999 | Offload all layers to GPU |
| `--flash-attn` | on | Required for memory efficiency |
| `--jinja` | - | Enables Jinja2 chat templates (required for Qwen3.5) |
| `--threads` | 8 | CPU threads for prompt processing |
| `--chat-template-kwargs` | `{"enable_thinking": false}` | Disable thinking mode by default (saves tokens) |

### VRAM Budget

| Config | VRAM Usage | Notes |
|--------|-----------|-------|
| ctx=32768, parallel=1 | ~22.3 GB | Original config, near limit |
| ctx=4096, parallel=4 | ~21.8 GB | Optimized for batch inference |
| ctx=32768, parallel=4 | OOM | Don't try |

### Context Length vs VRAM

The L4 has 23,034 MiB (22.5 GB) total VRAM. The Q4_K_XL model weights take ~20.5 GB, leaving ~2 GB for KV cache.

- **Max safe ctx-size**: 32768 with parallel=1
- **For batch inference**: reduce ctx-size to 4096-8192, increase parallel to 4
- Qwen3.5 supports up to 262K context, but L4 can only fit 32K

### Performance

| Metric | Value |
|--------|-------|
| Tokens/sec (single request) | ~39 tok/s |
| Tokens/sec (4 parallel) | ~20 tok/s per request, ~80 tok/s aggregate |
| Time per request (500 tokens out) | ~13-17s |
| Prompt processing | ~280 tok/s (prefill) |

## Known Issues / Pitfalls

### 1. KV Cache Never Reuses (Prompt Caching Broken)

**Symptom**: Server log shows "forcing full prompt re-processing due to lack of cache data" on every request, even with identical system prompts.

**Root Cause**: Qwen3.5 uses a hybrid architecture with 30 Gated DeltaNet (linear attention, recurrent) layers + 10 full attention layers. llama.cpp's checkpoint validation uses `n_swa = max(1, llama_model_n_swa(model))`. Since Qwen3.5 has no SWA (`sliding_window` is null in config), `n_swa` defaults to 1. The GDN recurrent state pushes `pos_min` to the end of the sequence (e.g., 1022), so `pos_min > n_swa` (1022 > 1) is always true, causing the cache to be discarded every time.

**Impact**: ~4x slower than it should be. Every request does full prompt re-prefill regardless of shared prefix.

**Status**: llama.cpp issue [#20225](https://github.com/ggml-org/llama.cpp/issues/20225). Community patch exists but not merged upstream. Affects ALL Qwen3.5 variants (0.6B to 397B).

**Workaround**: None available. Accept the ~16s per request latency for now. Using `cache_prompt: true` and `id_slot` in requests has no effect due to this bug.

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

### 5. Docker Image Version

- Using `ghcr.io/ggml-org/llama.cpp:server-cuda` (latest)
- Tested version: b8323 (commit 57819b8d4)
- PR [#19877](https://github.com/ggml-org/llama.cpp/pull/19877) (merged at b8153) fixed some cache issues but NOT the GDN recurrent state problem

### 6. Open WebUI Connection

- Open WebUI connects to llama.cpp via `OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1`
- `--add-host=host.docker.internal:host-gateway` is required for Docker-to-host networking
- Open WebUI is on port 3000 (mapped from container's 8080)

## Model Details

| Property | Value |
|----------|-------|
| Model | Qwen3.5-35B-A3B |
| Quantization | Unsloth UD-Q4_K_XL |
| File size | 21 GB |
| Architecture | Hybrid: 30× Gated DeltaNet + 10× Gated Attention (MoE, 256 experts, 8 active) |
| Layer pattern | 10 × (3 × GDN-MoE + 1 × GA-MoE) |
| Context | 262K native, 32K max on L4 |
| Active params | 3B (of 35B total) |
| Vocab | 248,320 |

## File Structure

```
├── README.md
├── docker-compose.yml
├── nginx/
│   └── qwen-api            # nginx site config
└── scripts/
    └── setup.sh             # One-shot setup script
```
