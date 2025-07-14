# 使用的 GPU 设备列表
CUDA_DEVICES = [4,5,6,7]

# 每张 GPU 上部署的模型数量
MODEL_PER_DEVICE = 2

# 启动的工作进程数量
WORKERS = len(CUDA_DEVICES) * MODEL_PER_DEVICE

# 第一个服务进程使用的起始端口号 *注意要和Nginx匹配*
BASE_PORT = 10021

# Redis 配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_KEY = "uvicorn_workers_id"