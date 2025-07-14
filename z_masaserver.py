import asyncio
import json
import io
from pathlib import Path
import time
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from typing import List
from demo.masaqq_module import MASAQQ
from logger_config import setup_logger
from server_type import serialize_instances_to_dicts, serialize_categories, SavePngRequest
from z_cuda_redis_config import CUDA_DEVICES, MODEL_PER_DEVICE, REDIS_KEY, REDIS_HOST, REDIS_PORT
import redis
from datetime import datetime
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# === 获取WorkerID ===
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
worker_id = int(r.lpop(REDIS_KEY))

# === 初始化 logger ===
server_start_time = time.time()
logger = setup_logger("./z_serverlog",worker_id=worker_id) 
logger.info("==============New Server Start!================")
# === FastAPI app ===
app = FastAPI()
logger.info("🚀 FastAPI MASAQQ Server started. Loading MASA Model Pool...")
# === 模型加载 ===
device_index = CUDA_DEVICES[worker_id % len(CUDA_DEVICES) ]
model = MASAQQ(device=f"cuda:{device_index}")
logger.info(f"✅UviID:{worker_id} Cuda Idx: {device_index}. 模型初始化完毕.  耗时: {time.time() - server_start_time:.2f}秒")


@app.post("/inference")
async def inference(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        video_np = np.load(io.BytesIO(contents))
        logger.info(f"[/inference] 🟢 Start inference at {now_str()}")
        start_time = time.time()
        _, _, _, pred_instances_list, categories = model.inference_byVideoNumpy(video_np)
        logger.info(f"[/inference] 🔴 End inference at {now_str()}, elapse = {time.time()-start_time :.2f}")

        logger.info(f"[/inference] ✅ Inference complete. Instances: {len(pred_instances_list)}")

        response_data = {
            "pred_instances_list": serialize_instances_to_dicts(pred_instances_list),
            "categories": serialize_categories(categories)
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception("❌ Inference failed")
        return JSONResponse(status_code=500, content={"error": str(e)})




@app.post("/savepng")
async def save_png_endpoint(
    file: UploadFile = File(...),
    body: str = Form(...)
):
    # body 是字符串，需要转字典
    body_dict = json.loads(body)
    savepng_req = SavePngRequest.model_validate(body_dict)
    try:

        logger.info(f"[worker {worker_id}] [/savepng] Received file: {file.filename}")
        contents = await file.read()
        video_np = np.load(io.BytesIO(contents))

        output_path = Path(savepng_req.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[worker {worker_id}] [/savepng] Saving to {output_path}")
        result = MASAQQ.save_event_png(
            video_data=video_np,
            pred_instances_list=savepng_req.pred_instances_list,
            output_path=output_path,
            image_file_prefix=savepng_req.image_file_prefix,
            enlarge_scale=savepng_req.enlarge_scale,
            select_percentile=savepng_req.select_percentile
        )

        logger.info(f"✅ [savepng] Saved {len(result)} PNGs")
        return JSONResponse(content={"success": True, "records": result})

    except Exception as e:
        logger.exception("❌ [savepng] Failed")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/hello")
def hello():
    logger.info("👋 Hello endpoint called")
    return {"message": "Hello, MASAQQ!"}