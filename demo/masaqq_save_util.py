import os
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import List, Tuple, Dict


def enlarge_bbox_norm(norm_bbox: List[float], scale: float) -> List[float]:
    """
    放大归一化后的bbox（仍然保持归一化格式），并限制在 [0,1] 范围内
    """
    x1, y1, x2, y2 = norm_bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * scale
    new_h = h * scale

    new_x1 = max(cx - new_w / 2, 0.0)
    new_y1 = max(cy - new_h / 2, 0.0)
    new_x2 = min(cx + new_w / 2, 1.0)
    new_y2 = min(cy + new_h / 2, 1.0)

    return [new_x1, new_y1, new_x2, new_y2]


def denormalize_bbox(norm_bbox: List[float], image_shape: Tuple[int, int]) -> List[int]:
    """将归一化 bbox 转换为像素级坐标"""
    H, W = image_shape
    x1, y1, x2, y2 = norm_bbox
    return [
        int(x1 * W),
        int(y1 * H),
        int(x2 * W),
        int(y2 * H),
    ]


# def group_instances_by_id(pred_instances_list: List[dict]) -> Dict[int, List[Tuple[int, List[float], float, int]]]:
#     """
#     本函数用于整理pred_instances_list中的实例数据,  将所有 instance 的 bbox 按 instance_id 分组（归一化坐标 + 分数 + 标签）
#     返回的Dict格式: {instance_id , List[(frame_idx, norm_bbox, score, label)]}
#     """
#     instance_dict = defaultdict(list)
#     for frame_idx, inst_data in enumerate(pred_instances_list):
#         if inst_data is None:
#             continue
#         ids = inst_data.get("instances_id", None)
#         bboxes = inst_data.get("bboxes", None)
#         scores = inst_data.get("scores", None)
#         labels = inst_data.get("labels", None)
#         if ids is None or bboxes is None:
#             continue

#         for i in range(len(ids)):
#             inst_id = int(ids[i].item())
#             norm_bbox = bboxes[i].tolist()
#             score = float(scores[i].item()) if scores is not None else 1.0
#             label = int(labels[i].item()) if labels is not None else -1
#             instance_dict[inst_id].append((frame_idx, norm_bbox, score, label))

#     return instance_dict

def group_instances_by_id(pred_instances_list: List[dict]) -> Dict[int, List[Tuple[int, List[float], float, int]]]:
    """
    整理 pred_instances_list 中的实例数据，将所有 instance 按 instance_id 分组。
    返回格式: {instance_id: [(frame_idx, norm_bbox, score, label), ...]}
    其中，norm_bbox是List[float]，score是float，label是int。
    """
    instance_dict = defaultdict(list)

    for frame_idx, inst_data in enumerate(pred_instances_list):
        if inst_data is None:
            continue

        ids = inst_data.get("instances_id")
        bboxes = inst_data.get("bboxes")
        scores = inst_data.get("scores")
        labels = inst_data.get("labels")

        if ids is None or bboxes is None:
            continue

        for i in range(len(ids)):
            inst_id = int(ids[i])
            norm_bbox = list(bboxes[i])  # 保证是list类型
            score = float(scores[i]) if scores is not None else 1.0
            label = int(labels[i]) if labels is not None else -1

            instance_dict[inst_id].append((frame_idx, norm_bbox, score, label))

    return instance_dict


def select_bbox_by_percentile(bbox_list: List[Tuple[int, List[float], float, int]], 
                              percentile: float = 1.0) -> Dict:
    """
    从给定 bbox 列表中选取给定百分位的 bbox（按面积排序）

    返回:
        {
            "frame_idx": int,
            "bbox_norm": List[float],
            "score": float,
            "label": int
        }
    """
    area_list = []
    for frame_idx, norm_bbox, score, label in bbox_list:
        x1, y1, x2, y2 = norm_bbox
        area = (x2 - x1) * (y2 - y1)
        area_list.append((area, frame_idx, norm_bbox, score, label))

    if not area_list:
        return None

    area_list.sort(key=lambda x: x[0])
    idx = int(len(area_list) * percentile) - 1
    idx = max(0, min(idx, len(area_list) - 1))
    _, frame_idx, norm_bbox, score, label = area_list[idx]

    return {
        "frame_idx": frame_idx,
        "bbox_norm": norm_bbox,
        "score": score,
        "label": label
    }
