#!/bin/bash
cd /home/user2/wqq/masa

echo "🚀 终止所有旧 gunicorn/uvicorn 相关进程..."
pids=$(ps aux | grep 'masaenv' | grep -E 'gunicorn|uvicorn' | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
  echo "找到旧进程，逐个杀死:"
  for pid in $pids; do
    echo "  Killing PID $pid"
    kill -9 $pid
  done
else
  echo "没有找到旧进程"
fi

echo "🔧 初始化 Redis..."
python z_init_redis.py

echo "📡 读取 WORKERS 数量..."
WORKERS=$(python -c "from z_cuda_redis_config import WORKERS; print(WORKERS)")
echo "工作进程数量: $WORKERS"

BASE_PORT=$(python -c "from z_cuda_redis_config import BASE_PORT; print(BASE_PORT)")

echo "🔥 启动 $WORKERS 个 gunicorn 实例..."
for ((i=0; i<WORKERS; i++))
do
  PORT=$((BASE_PORT + i))
  nohup gunicorn z_masaserver:app \
    -k uvicorn.workers.UvicornWorker \
    -w 1 \
    -b 0.0.0.0:$PORT \
    --timeout 300 \
    --log-level info \
    > z_masaserver_$PORT.log 2>&1 &
  echo "启动实例监听端口 $PORT"
done

echo "✅ 全部实例启动完毕"
