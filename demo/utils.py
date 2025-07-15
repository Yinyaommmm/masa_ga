import time
import torch
import numpy as np
from collections import defaultdict

from mmdet.models.task_modules.assigners import BboxOverlaps2D
from mmengine.structures import InstanceData
import functools
import torch

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查是否为类方法
        if len(args) > 0 and hasattr(args[0], '__class__'):
            self_obj = args[0]
            if hasattr(self_obj, 'debug') and not getattr(self_obj, 'debug'):
                # 如果有 self.debug 并且是 False，就直接跳过计时
                return func(*args, **kwargs)

        # 正常计时逻辑
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__qualname__} took {end - start:.4f} seconds")
        return result

    return wrapper

def average_score_filter(instances_list):
    # Extract instance IDs and their scores
    instance_id_to_frames = defaultdict(list)
    instance_id_to_scores = defaultdict(list)
    for frame_idx, instances in enumerate(instances_list):
        for i, instance_id in enumerate(instances[0].pred_track_instances.instances_id):
            instance_id_to_frames[instance_id.item()].append(frame_idx)
            instance_id_to_scores[instance_id.item()].append(instances[0].pred_track_instances.scores[i].cpu().numpy())

    # Compute average scores for each segment of each instance ID
    for instance_id, frames in instance_id_to_frames.items():
        scores = np.array(instance_id_to_scores[instance_id])

        # Identify segments
        segments = []
        segment = [frames[0]]
        for idx in range(1, len(frames)):
            if frames[idx] == frames[idx - 1] + 1:
                segment.append(frames[idx])
            else:
                segments.append(segment)
                segment = [frames[idx]]
        segments.append(segment)

        # Compute average score for each segment
        avg_scores = np.copy(scores)
        for segment in segments:
            segment_scores = scores[frames.index(segment[0]):frames.index(segment[-1]) + 1]
            avg_score = np.mean(segment_scores)
            avg_scores[frames.index(segment[0]):frames.index(segment[-1]) + 1] = avg_score

        # Update instances_list with average scores
        for frame_idx, avg_score in zip(frames, avg_scores):
            instances_list[frame_idx][0].pred_track_instances.scores[
                instances_list[frame_idx][0].pred_track_instances.instances_id == instance_id] = torch.tensor(avg_score, dtype=instances_list[frame_idx][0].pred_track_instances.scores.dtype)

    return instances_list


def moving_average_filter(instances_list, window_size=5):
    # Helper function to compute the moving average
    def smooth_bbox(bboxes, window_size):
        smoothed_bboxes = np.copy(bboxes)
        half_window = window_size // 2
        for i in range(4):
            padded_bboxes = np.pad(bboxes[:, i], (half_window, half_window), mode='edge')
            smoothed_bboxes[:, i] = np.convolve(padded_bboxes, np.ones(window_size) / window_size, mode='valid')
        return smoothed_bboxes

    # Extract bounding boxes and instance IDs
    instance_id_to_frames = defaultdict(list)
    instance_id_to_bboxes = defaultdict(list)
    for frame_idx, instances in enumerate(instances_list):
        for i, instance_id in enumerate(instances[0].pred_track_instances.instances_id):
            instance_id_to_frames[instance_id.item()].append(frame_idx)
            instance_id_to_bboxes[instance_id.item()].append(instances[0].pred_track_instances.bboxes[i].cpu().numpy())

    # Apply moving average filter to each segment
    for instance_id, frames in instance_id_to_frames.items():
        bboxes = np.array(instance_id_to_bboxes[instance_id])

        # Identify segments
        segments = []
        segment = [frames[0]]
        for idx in range(1, len(frames)):
            if frames[idx] == frames[idx - 1] + 1:
                segment.append(frames[idx])
            else:
                segments.append(segment)
                segment = [frames[idx]]
        segments.append(segment)

        # Smooth bounding boxes for each segment
        smoothed_bboxes = np.copy(bboxes)
        for segment in segments:
            if len(segment) >= window_size:
                segment_bboxes = bboxes[frames.index(segment[0]):frames.index(segment[-1]) + 1]
                smoothed_segment_bboxes = smooth_bbox(segment_bboxes, window_size)
                smoothed_bboxes[frames.index(segment[0]):frames.index(segment[-1]) + 1] = smoothed_segment_bboxes

        # Update instances_list with smoothed bounding boxes
        for frame_idx, smoothed_bbox in zip(frames, smoothed_bboxes):
            instances_list[frame_idx][0].pred_track_instances.bboxes[
                instances_list[frame_idx][0].pred_track_instances.instances_id == instance_id] = torch.tensor(smoothed_bbox, dtype=instances_list[frame_idx][0].pred_track_instances.bboxes.dtype).to(instances_list[frame_idx][0].pred_track_instances.bboxes.device)

    return instances_list


def identify_and_remove_giant_bounding_boxes(instances_list, image_size, size_threshold, confidence_threshold,
                                             coverage_threshold, object_num_thr=4, max_objects_in_box=6):
    # Initialize BboxOverlaps2D with 'iof' mode
    bbox_overlaps_calculator = BboxOverlaps2D()

    # Initialize data structures
    invalid_instance_ids = set()

    image_width, image_height = image_size
    two_thirds_image_area = (2 / 3) * (image_width * image_height)

    # Step 1: Identify giant bounding boxes and record their instance_ids
    for frame_idx, instances in enumerate(instances_list):
        bounding_boxes = instances[0].pred_track_instances.bboxes
        confidence_scores = instances[0].pred_track_instances.scores
        instance_ids = instances[0].pred_track_instances.instances_id

        N = bounding_boxes.size(0)

        for i in range(N):
            current_box = bounding_boxes[i]
            box_size = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])

            if box_size < size_threshold:
                continue

            other_boxes = torch.cat([bounding_boxes[:i], bounding_boxes[i + 1:]])
            other_confidences = torch.cat([confidence_scores[:i], confidence_scores[i + 1:]])
            iofs = bbox_overlaps_calculator(other_boxes, current_box.unsqueeze(0), mode='iof', is_aligned=False)

            if iofs.numel() == 0:
                continue

            high_conf_mask = other_confidences > confidence_threshold

            if high_conf_mask.numel() == 0 or torch.sum(high_conf_mask) == 0:
                continue

            high_conf_masked_iofs = iofs[high_conf_mask]

            covered_high_conf_boxes_count = torch.sum(high_conf_masked_iofs > coverage_threshold)

            if covered_high_conf_boxes_count >= object_num_thr and torch.all(
                    confidence_scores[i] < other_confidences[high_conf_mask]):
                invalid_instance_ids.add(instance_ids[i].item())
                continue

            if box_size > two_thirds_image_area:
                invalid_instance_ids.add(instance_ids[i].item())
                continue

            # New condition: if the bounding box contains more than 6 objects
            if covered_high_conf_boxes_count > max_objects_in_box:
                invalid_instance_ids.add(instance_ids[i].item())
                continue

    # Remove invalid tracks
    for frame_idx, instances in enumerate(instances_list):
        valid_mask = torch.tensor(
            [instance_id.item() not in invalid_instance_ids for instance_id in
             instances[0].pred_track_instances.instances_id])
        if len(valid_mask) == 0:
            continue
        new_instance_data = InstanceData()
        new_instance_data.bboxes = instances[0].pred_track_instances.bboxes[valid_mask]
        new_instance_data.scores = instances[0].pred_track_instances.scores[valid_mask]
        new_instance_data.instances_id = instances[0].pred_track_instances.instances_id[valid_mask]
        new_instance_data.labels = instances[0].pred_track_instances.labels[valid_mask]
        if 'masks' in instances[0].pred_track_instances:
            new_instance_data.masks = instances[0].pred_track_instances.masks[valid_mask]
        instances[0].pred_track_instances = new_instance_data

    return instances_list



# wqq add 
# wqq add 
def remove_static_tracks(instances_list, image_size, base_static_threshold=20, base_height=360):
    """
    根据轨迹移动距离移除静态目标。

    Args:
        instances_list: 帧列表，每帧含有 pred_track_instances。
        image_size: (height, width)，用于自动缩放静止判断阈值。
        base_static_threshold: 在 base_height 分辨率下判断为“静止”的阈值（像素）。
        base_height: 用于缩放 static_threshold 的基准高度（默认360）。
    """


    def compute_center(bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def trajectory_movement(centers):
        total = 0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            total += (dx**2 + dy**2)**0.5
        return total

    def create_empty_instance_data(template):
        """创建一个空的 InstanceData，结构与 template 相同。"""
        empty = InstanceData()
        empty.bboxes = template.bboxes.new_zeros((0, 4))
        empty.scores = template.scores.new_zeros((0,))
        empty.instances_id = template.instances_id.new_zeros((0,), dtype=torch.long)
        empty.labels = template.labels.new_zeros((0,))
        if hasattr(template, 'masks') and isinstance(template.masks, np.ndarray):
            empty.masks = np.zeros((0, *template.masks.shape[1:]), dtype=template.masks.dtype)
        return empty

    # 自动缩放 threshold
    W, H = image_size
    scale_factor = H / base_height
    scaled_threshold = base_static_threshold * scale_factor
    # print(f"[remove_static_tracks] Adjusted static_threshold: {scaled_threshold:.2f} (scale: {scale_factor:.2f})")

    id_to_centers = defaultdict(list)

    for frame_idx, inst in enumerate(instances_list):
        tracks = inst[0].pred_track_instances  # 每帧
        for i, tid in enumerate(tracks.instances_id):
            tid = tid.item()
            bbox = tracks.bboxes[i].tolist()
            center = compute_center(bbox)
            id_to_centers[tid].append(center)

    dynamic_ids = set()
    for tid, centers in id_to_centers.items():
        if len(centers) <= 1:
            # 如果该轨迹只出现过 1 帧，默认保留
            dynamic_ids.add(tid)
        elif trajectory_movement(centers) >= scaled_threshold:
            dynamic_ids.add(tid)

    for inst in instances_list:
        tracks = inst[0].pred_track_instances
        keep_mask = torch.tensor([int(tid) in dynamic_ids for tid in tracks.instances_id],
                                 device=tracks.bboxes.device)
        
        if keep_mask.sum() == 0:
            inst[0].pred_track_instances = create_empty_instance_data(tracks)
            continue

        new_instance_data = InstanceData()
        new_instance_data.bboxes = tracks.bboxes[keep_mask]
        new_instance_data.scores = tracks.scores[keep_mask]
        new_instance_data.instances_id = tracks.instances_id[keep_mask]
        new_instance_data.labels = tracks.labels[keep_mask]
        if hasattr(tracks, 'masks') and isinstance(tracks.masks, np.ndarray):
            new_instance_data.masks = tracks.masks[keep_mask.cpu().numpy()]
        inst[0].pred_track_instances = new_instance_data

    return instances_list

# wqq add
def create_empty_instance_data(reference_tracks):

    new_instance_data = InstanceData()
    device = reference_tracks.bboxes.device if hasattr(reference_tracks, 'bboxes') else None
    dtype = reference_tracks.bboxes.dtype if hasattr(reference_tracks, 'bboxes') else torch.float32
    # 初始化空张量，shape为(0, 4)等
    new_instance_data.bboxes = torch.empty((0, 4), dtype=dtype, device=device)
    new_instance_data.scores = torch.empty((0,), dtype=dtype, device=device)
    new_instance_data.instances_id = torch.empty((0,), dtype=torch.long, device=device)
    new_instance_data.labels = torch.empty((0,), dtype=torch.long, device=device)
    if hasattr(reference_tracks, 'masks'):
        import numpy as np
        new_instance_data.masks = np.empty((0,), dtype=object)  # 如果用其他结构请调整
    return new_instance_data

# wqq add
def remove_short_tracks(instances_list, min_duration=3):

    # 收集每个 track_id 出现在哪些帧
    id_to_frames = defaultdict(list)
    for frame_idx, inst in enumerate(instances_list):
        tracks = inst[0].pred_track_instances
        
        # 跳过空帧
        if not hasattr(tracks, 'instances_id'):
            continue

        for tid in tracks.instances_id:
            id_to_frames[int(tid.item())].append(frame_idx)

    # 找到需要保留的轨迹（帧数 >= min_duration）
    valid_ids = {tid for tid, frames in id_to_frames.items() if len(frames) >= min_duration}

    # 构造保留掩码
    for inst in instances_list:
        tracks = inst[0].pred_track_instances
        # 跳过空帧
        if not hasattr(tracks, 'instances_id'):
            continue
        keep_mask = torch.tensor(
            [int(tid) in valid_ids for tid in tracks.instances_id],
            device=tracks.bboxes.device
        )
        if keep_mask.sum() == 0:
            inst[0].pred_track_instances = create_empty_instance_data(tracks)
            continue
        new_instance_data = InstanceData()
        new_instance_data.bboxes = tracks.bboxes[keep_mask]
        new_instance_data.scores = tracks.scores[keep_mask]
        new_instance_data.instances_id = tracks.instances_id[keep_mask]
        new_instance_data.labels = tracks.labels[keep_mask]
        if hasattr(tracks, 'masks') and isinstance(tracks.masks, np.ndarray):
            new_instance_data.masks = tracks.masks[keep_mask.cpu().numpy()]
        inst[0].pred_track_instances = new_instance_data

    return instances_list

# wqq add 
def filter_low_threshold_instances(instances_list, confidence_threshold=0.2):
    """
    Filter out instances with confidence scores below a specified threshold.

    Args:
        instances_list (list): List of InstanceData objects for each frame.
        confidence_threshold (float): The minimum confidence score to keep an instance.

    Returns:
        list: Filtered instances_list with low-confidence instances removed.
    """
    for inst in instances_list:
        tracks = inst[0].pred_track_instances
        if not hasattr(tracks, 'scores'):
            continue  # 跳过没有分数的帧

        keep_mask = tracks.scores >= confidence_threshold
        if keep_mask.sum() == 0:
            inst[0].pred_track_instances = create_empty_instance_data(tracks)
            continue

        new_instance_data = InstanceData()
        new_instance_data.bboxes = tracks.bboxes[keep_mask]
        new_instance_data.scores = tracks.scores[keep_mask]
        new_instance_data.instances_id = tracks.instances_id[keep_mask]
        new_instance_data.labels = tracks.labels[keep_mask]
        if hasattr(tracks, 'masks') and isinstance(tracks.masks, np.ndarray):
            new_instance_data.masks = tracks.masks[keep_mask.cpu().numpy()]
        inst[0].pred_track_instances = new_instance_data

    return instances_list

#wqq add
def stabilize_labels(instances_list):

    # 第一步：收集每个 instance_id 的所有标签及出现次数
    id_to_label_counts = defaultdict(lambda: defaultdict(int))  # 格式: {track_id: {label: 出现次数}}
    for frame_idx, inst in enumerate(instances_list):
        tracks = inst[0].pred_track_instances
        
        # 跳过无标签或无轨迹ID的帧
        if not (hasattr(tracks, 'instances_id') and hasattr(tracks, 'labels')):
            continue
        if len(tracks.instances_id) != len(tracks.labels):
            continue  # 避免ID与标签数量不匹配的异常情况
        
        # 遍历当前帧的所有轨迹，统计标签出现次数
        for tid, label in zip(tracks.instances_id, tracks.labels):
            tid_int = int(tid.item())  # 转为整数ID
            label_int = int(label.item())  # 转为整数标签
            id_to_label_counts[tid_int][label_int] += 1

    # 第二步：为每个 instance_id 确定“主导标签”（出现次数最多的标签）
    id_to_dominant_label = {}
    # 新增：记录每个轨迹的原始标签分布（用于打印）
    id_label_distribution = {}
    for tid, label_counts in id_to_label_counts.items():
        # 按出现次数排序，次数相同则保留数值最小的标签
        sorted_labels = sorted(label_counts.items(), key=lambda x: (x[1], -x[0]), reverse=True)
        dominant_label = sorted_labels[0][0]
        id_to_dominant_label[tid] = dominant_label
        # 保存标签分布信息（用于打印）
        id_label_distribution[tid] = sorted_labels

    # 打印每个轨迹的标签统计（可选，帮助了解整体情况）

    for tid, dist in id_label_distribution.items():
        total = sum(c for _, c in dist)
        dist_str = ", ".join([f"标签{l}: {c}次" for l, c in dist])
        # print(f"轨迹ID {tid}（共{total}帧）：{dist_str} → 主导标签：{id_to_dominant_label[tid]}")

    # 第三步：遍历所有帧，将同一 instance_id 的标签统一替换为主导标签，并打印替换记录
    replace_total = 0  # 统计总替换次数
    for frame_idx, inst in enumerate(instances_list):
        tracks = inst[0].pred_track_instances
        
        # 跳过无标签或无轨迹ID的帧
        if not (hasattr(tracks, 'instances_id') and hasattr(tracks, 'labels')):
            continue
        if len(tracks.instances_id) != len(tracks.labels):
            continue
        
        # 生成新标签（替换为主导标签）
        new_labels = []
        for tid, original_label in zip(tracks.instances_id, tracks.labels):
            tid_int = int(tid.item())
            original_label_int = int(original_label.item())
            
            # 检查是否需要替换
            if tid_int in id_to_dominant_label:
                dominant_label = id_to_dominant_label[tid_int]
                new_labels.append(dominant_label)
                # 若标签不同，则打印替换信息
                if dominant_label != original_label_int:
                    replace_total += 1
            else:
                # 无主导标签时保留原标签
                new_labels.append(original_label_int)
        
        # 将新标签转为与原标签同设备、同类型的张量
        new_labels_tensor = torch.tensor(
            new_labels,
            dtype=tracks.labels.dtype,
            device=tracks.labels.device
        )
        # 更新轨迹的标签
        tracks.labels = new_labels_tensor


    return instances_list
def filter_and_update_tracks(instances_list, image_size, size_threshold=10000, coverage_threshold=0.75,
                             confidence_threshold=0.2, smoothing_window_size=5,base_static_threshold=20,min_duration =3): 

    # Step 1: Identify and remove giant bounding boxes
    instances_list = identify_and_remove_giant_bounding_boxes(instances_list, image_size, size_threshold, confidence_threshold, coverage_threshold)

     # Step 2: Smooth interpolated bounding boxes
    instances_list = moving_average_filter(instances_list, window_size=smoothing_window_size)

    # Step 3: compute the track average score
    instances_list = average_score_filter(instances_list)

    # Step 4: Remove nearly-static tracks
    instances_list = remove_static_tracks(instances_list, image_size, base_static_threshold=base_static_threshold, base_height=360)

    # # Step 5: Remove short-lifetime tracks
    # instances_list = remove_short_tracks(instances_list, min_duration=min_duration)

    # Step 6: Remove low-conf tracks
    instances_list= filter_low_threshold_instances(instances_list, confidence_threshold=confidence_threshold)

    # Step 7: Stabilize labels
    instances_list = stabilize_labels(instances_list)
    
    return instances_list
