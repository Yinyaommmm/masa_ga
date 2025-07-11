from pathlib import Path
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from flask import json
import numpy as np
import io
from typing import List
from demo.masaqq_module import MASAQQ, CoarseCategory
from logger_config import setup_logger
from pydantic import BaseModel
from typing import List, Dict

# === 初始化 logger ===
logger = setup_logger("./z_serverlog") 
logger.info("==============New Server Start!================")
logger.info("🚀 Logger started")
# === FastAPI app ===
app = FastAPI()
logger.info("🚀 FastAPI MASAQQ Server started")

# === 模型加载 ===
logger.info("✅ Loading MASA Model...")
model = MASAQQ(device="cuda:1")
logger.info("✅ MASAQQ model initialized on cuda:1")

def serialize_instances_to_dicts(pred_instances_list):
    result = []
    for instance in pred_instances_list:
        if instance is None:
            continue
        bboxes = instance.bboxes
        scores = instance.scores if hasattr(instance, "scores") else None
        labels = instance.labels if hasattr(instance, "labels") else None
        ids = instance.instances_id

        # 把每个InstanceData的所有字段先转成纯Python类型
        new_instance = {
            "bboxes": bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes),
            "scores": scores.tolist() if scores is not None and hasattr(scores, "tolist") else (list(scores) if scores is not None else []),
            "labels": labels.tolist() if labels is not None and hasattr(labels, "tolist") else (list(labels) if labels is not None else []),
            "instances_id": ids.tolist() if hasattr(ids, "tolist") else list(ids),
        }
        result.append(new_instance)
    return result

def serialize_categories(categories: List[CoarseCategory]) -> List[str]:
    return [cat.real for cat in categories]

class SavePngRequest(BaseModel):
    pred_instances_list: List[Dict]
    output_path: str
    image_file_prefix: str
    enlarge_scale: float
    select_percentile: float

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    try:
        logger.info(f"[/inference] Received file: {file.filename}")
        contents = await file.read()
        video_np = np.load(io.BytesIO(contents))

        logger.info(f"[/inference] Running inference...")
        _, _, _, pred_instances_list, categories = model.inference_byVideoNumpy(video_np)

        logger.info(f"[/inference]✅ Inference complete. Instances: {len(pred_instances_list)}")

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

        logger.info(f"[savepng] Received file: {file.filename}")
        contents = await file.read()
        video_np = np.load(io.BytesIO(contents))

        output_path = Path(savepng_req.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[savepng] Saving to {output_path}")
        result = model.save_event_png(
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