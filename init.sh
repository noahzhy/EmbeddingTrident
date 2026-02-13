#!/bin/bash
set -e

echo "=============================="
echo " Step 1. Install Docker"
echo "=============================="

sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl start docker
sudo systemctl enable docker

echo "=============================="
echo " Step 2. Write docker-compose.yml"
echo "=============================="

cat > docker-compose.yml << 'EOF'
# version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: ccr.ccs.tencentyun.com/3rd-proxy/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: ccr.ccs.tencentyun.com/3rd-proxy/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: ccr.ccs.tencentyun.com/3rd-proxy/milvus:v2.4.23-gpu
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              device_ids: ["0"]
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
EOF

echo "=============================="
echo " Step 3. Install NVIDIA Container Toolkit"
echo "=============================="

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=============================="
echo " Step 3.1. Check GPU availability"
echo "=============================="

if command -v nvidia-smi &> /dev/null
then
    echo ">>> Running nvidia-smi to check GPU:"
    nvidia-smi
else
    echo "⚠️  nvidia-smi not found. Please install NVIDIA driver manually."
    echo "   https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html"
fi

echo "=============================="
echo " Step 4. Install pymilvus (Python client)"
echo "=============================="

pip install --upgrade pymilvus

echo "=============================="
echo " Step 5. Pull Milvus related images from Tencent registry"
echo "=============================="

sudo docker login ccr.ccs.tencentyun.com

sudo docker pull ccr.ccs.tencentyun.com/3rd-proxy/etcd:v3.5.5
sudo docker pull ccr.ccs.tencentyun.com/3rd-proxy/minio:RELEASE.2023-03-20T20-16-18Z
sudo docker pull ccr.ccs.tencentyun.com/3rd-proxy/milvus:v2.4.23-gpu

echo "=============================="
echo " Step 6. Start Milvus server"
echo "=============================="

sudo docker compose up -d

echo "=============================="
echo "✅ Installation & Startup Finished!"
echo "=============================="
echo "👉 You can check Milvus logs with: sudo docker compose logs -f"
