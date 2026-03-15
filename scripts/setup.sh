#!/bin/bash
# One-shot setup for Qwen3.5-35B-A3B on GCP L4
# Run on the GCP instance after creation
set -e

echo "=== Installing dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq nginx docker-compose-plugin

echo "=== Downloading model ==="
mkdir -p ~/models
if [ ! -f ~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf ]; then
    pip install -q huggingface-hub
    huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
        Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
        --local-dir ~/models
else
    echo "Model already downloaded"
fi

echo "=== Starting services ==="
cd "$(dirname "$0")/.."
docker compose up -d

echo "=== Configuring nginx ==="
DOMAIN="${DOMAIN:-_}"
sed "s/\${DOMAIN}/$DOMAIN/g" nginx/qwen-api | sudo tee /etc/nginx/sites-available/qwen-api > /dev/null
sudo ln -sf /etc/nginx/sites-available/qwen-api /etc/nginx/sites-enabled/

# Remove default site to avoid conflicts
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t && sudo systemctl reload nginx

echo "=== Waiting for llama-server to be healthy ==="
for i in $(seq 1 60); do
    if curl -s http://localhost:8080/health | grep -q ok; then
        echo "llama-server is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

echo ""
echo "=== Deployment complete ==="
echo "API: http://$(curl -s ifconfig.me):8080/v1/chat/completions"
echo "WebUI: http://$(curl -s ifconfig.me):3000"
echo ""
echo "Test:"
echo "  curl http://localhost:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":50}'"
