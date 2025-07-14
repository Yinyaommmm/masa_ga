import redis
from z_cuda_redis_config import CUDA_DEVICES, MODEL_PER_DEVICE, REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_KEY

def init_worker_queue():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.delete(REDIS_KEY)  # 清空旧的 worker_id 队列

    total_workers = len(CUDA_DEVICES) * MODEL_PER_DEVICE
    worker_ids = list(range(total_workers))

    # 将 worker_id 依次压入 Redis 列表
    for wid in worker_ids:
        r.rpush(REDIS_KEY, wid)

    print(f"✅ Redis 初始化完成，共有 {total_workers} 个 worker_id 被写入到 {REDIS_KEY}")

if __name__ == "__main__":
    init_worker_queue()
