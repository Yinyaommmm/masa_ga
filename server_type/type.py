
from typing import Dict, List

from pydantic import BaseModel
from demo.masaqq_module import CoarseCategory


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
