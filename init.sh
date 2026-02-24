# 终端1：cd web && npm run api
# 终端2：cd web && npm run dev#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$ROOT_DIR/web"

DO_BUILD=false
if [[ "${1:-}" == "--build" ]]; then
  DO_BUILD=true
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[init] 错误: 未找到 npm"
  exit 1
fi

cleanup() {
  echo ""
  echo "[init] 正在停止 API 服务..."
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$WEB_DIR"

echo "[init] 安装依赖(如已安装会很快)"
npm install

if [[ "$DO_BUILD" == "true" ]]; then
  echo "[init] 构建前端"
  npm run build
fi

echo "[init] 启动 API 服务: npm run api"
npm run api &
API_PID=$!

sleep 1

echo "[init] 启动前端开发服务: npm run dev"
npm run dev