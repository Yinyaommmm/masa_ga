import os
from typing import List,Dict
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import torch
import imageio.v2 as imageio
from torch.multiprocessing import Pool
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops import batched_nms

from ..masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from .utils import filter_and_update_tracks,timer
from .convert import convert_webp_to_mp4
current_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
import copy 
from pathlib import Path
from .masaqq_save_util import enlarge_bbox_norm, denormalize_bbox,group_instances_by_id,select_bbox_by_percentile
from PIL import Image

from enum import IntEnum
import warnings
warnings.filterwarnings('ignore')

class MASAQQ:
    def __init__(self,
                 device='cuda:0',
                 score_thr=0.3,
                 save_dir=None, # gif保存路径
                 line_width=5, # box框粗细
                 fp16=False,  # fp16推理
                 show_fps=False, # 输出的gif是否带show_fps
                 no_post=False # 是否后处理
                 ): 
        

        self.texts = "person . moving car . two-wheelers . three-wheelers . cat . dog . other "  # 开放式分类用.当间隔符，例如"car . person . cat . dog"
        params = {
            "device": device,
            "score_thr": score_thr,
            "save_dir": save_dir if save_dir is not None else "None",
            "texts": self.texts,
            "line_width": line_width,
            "fp16": fp16,
            "show_fps": show_fps,
            "no_post": no_post
        }
        print("MASAQQ初始化参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        self.args = locals()
        self.device = device
        self.show_fps = show_fps
        self.score_thr = score_thr
        self.no_post = no_post
        self.fp16 = fp16

        # 设置绝对路径
        masa_config = os.path.join(current_dir, "../configs/masa-gdino/masa_gdino_swinb_inference.py")
        masa_config = os.path.abspath(masa_config)
        masa_checkpoint = os.path.join(current_dir, "../saved_models/masa_models/gdino_masa.pth")
        masa_checkpoint = os.path.abspath(masa_checkpoint)
        self.det_model = None
        self.unified=True
        self.det_config=None
        self.det_checkpoint=None
        self.detector_type='mmdet'
        self.masa_model = init_masa(masa_config, masa_checkpoint, device=self.device)

        if not self.unified:
            self.det_model = init_detector(self.det_config, self.det_checkpoint, palette='random', device=self.device)
            self.det_model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
            self.test_pipeline = Compose(self.det_model.cfg.test_dataloader.dataset.pipeline)

        self.masa_test_pipeline = build_test_pipeline(self.masa_model.cfg, with_text=(self.texts is not None))
        self.masa_model.cfg.visualizer['texts'] = self.texts if self.texts is not None else (
            self.det_model.dataset_meta['classes'] if not self.unified else [])
        self.masa_model.cfg.visualizer['save_dir'] = save_dir
        self.masa_model.cfg.visualizer['line_width'] = line_width
        self.visualizer = VISUALIZERS.build(self.masa_model.cfg.visualizer)

        # # -- 初始化数据库连接
        # self.db = DBConnector('mysql+mysqlconnector://user:password@localhost/your_database')

    @timer
    def inference_byVideoNumpy(self, video_data: np.ndarray):
        """
        推理函数，接受一个 numpy array 格式的视频数据，而不是视频文件路径。
        video_data 的形状应为 (frames, height, width, channels)。

        Args:
            video_data (np.ndarray): 视频数据，形状为 (frames, height, width, channels)，
                                    例如 (30, 360, 640, 3)，表示 30 帧，分辨率 360x640，RGB 通道。

        Returns:
            frames, instances_list, fps_list, pred_instances_list, categories
        """
        assert self.unified == True  # 目前只支持统一模型

        # 通过 video_data 获取视频的帧数与分辨率
        num_frames, height, width, channels = video_data.shape

        instances_list, frames, fps_list = [], [], []
        print('Video frames', num_frames)
        # 通过遍历 numpy 数组的方式来访问每一帧
        for frame_idx in range(num_frames):
            frame = video_data[frame_idx]  # 取出当前帧
            track_result = inference_masa(self.masa_model, frame,
                                        frame_id=frame_idx,
                                        video_len=num_frames,
                                        test_pipeline=self.masa_test_pipeline,
                                        text_prompt=self.texts,
                                        fp16=self.fp16,
                                        detector_type=self.detector_type,
                                        show_fps=self.show_fps)

            if self.show_fps:
                track_result, fps = track_result

            if 'masks' in track_result[0].pred_track_instances and len(track_result[0].pred_track_instances.masks) > 0:
                track_result[0].pred_track_instances.masks = torch.stack(
                    track_result[0].pred_track_instances.masks, dim=0).cpu().numpy()

            track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)

            instances_list.append(track_result.to('cpu'))
            frames.append(frame)
            if self.show_fps:
                fps_list.append(fps)

        
        if not self.no_post:
            instances_list = filter_and_update_tracks(instances_list, (width, height),
                                                    base_static_threshold=20, min_duration=5, confidence_threshold=self.score_thr)
            
        pred_instances_list = [copy.deepcopy(item[0].pred_track_instances) for item in instances_list]          # 归一化边界框坐标
        for pred_instances in pred_instances_list:
            bboxes = pred_instances.bboxes
            # 将 bboxes 归一化到 [0, 1] 范围
            normalized_bboxes = bboxes.clone()
            normalized_bboxes[:, 0] = bboxes[:, 0] / width  # x_min
            normalized_bboxes[:, 1] = bboxes[:, 1] / height  # y_min
            normalized_bboxes[:, 2] = bboxes[:, 2] / width  # x_max
            normalized_bboxes[:, 3] = bboxes[:, 3] / height  # y_max
            # 将归一化后的 bboxes 替换回原始数据
            pred_instances.bboxes = normalized_bboxes
           
        # 统一收集所有帧中的标签ID
        all_labels = set()
        for instances in pred_instances_list:
            if hasattr(instances, 'labels') and len(instances.labels) > 0:
                frame_labels = instances.labels.cpu().numpy().tolist()
                all_labels.update(frame_labels)
        
        # 将收集的标签ID映射为粗分类枚举类型
        categories = [map_label(label) for label in sorted(all_labels)]
        # 过滤掉可能的None值（如果有未映射的标签ID）
        categories = [cat for cat in categories if cat is not None] 

        return frames, instances_list, fps_list, pred_instances_list,categories  # video_fps should be set based on video data if needed

    @timer
    def destroy(self):
        del self.masa_model
        if self.det_model:
            del self.det_model
        torch.cuda.empty_cache()
        gc.collect()
        
    @timer
    def save_gif(self, frames, instances_list, fps_list, output_path, fps_value):
        num_cores = max(1, min(os.cpu_count() - 1, 8))

        if not self.show_fps:
            fps_list = [None] * len(frames)

        args_list = [
            (frame, track_result.to('cpu'), idx, fps, self.visualizer, self.score_thr)
            for idx, (frame, fps, track_result) in enumerate(zip(frames, fps_list, instances_list))
        ]

        with Pool(processes=num_cores) as pool:
            frames = pool.map(_visualize_frame_static, args_list)

        imageio.mimsave(output_path, frames, duration=1 / fps_value, loop=0)

    # @timer 太短了根本不用
    def save_event_png(self,
                    video_data: np.ndarray,
                    pred_instances_list: List[dict],
                    output_path: Path,
                    image_file_prefix: str = "11580_360P_0",
                    enlarge_scale: float = 1.1,
                    select_percentile: float = 1.0) -> List[Dict]:
        """
        保存每个 instance_id 的事件图像, 并返回结构化信息, 用于交给后端存储。
        output_path: 保存文件夹名
        image_file_prefix: 保存的图片前缀
        Returns:
            List[Dict]：每个 instance 的记录，如下结构：
            {
                "instance_id": int,
                "frame_idx": int,
                "score": float,
                "label": int,
                "bbox_norm": List[float],      # 原始归一化坐标
                "bbox_enlarged": List[float],  # 放大后的归一化坐标
                "bbox_pixel": List[int],       # 像素坐标
                "saved_path": str
            }
        """
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        H, W = video_data.shape[1:3]
        instance_groups = group_instances_by_id(pred_instances_list)

        result_records = []

        for inst_id, bbox_list in instance_groups.items():
            selected = select_bbox_by_percentile(bbox_list, select_percentile)
            if selected is None:
                continue

            frame_idx = selected["frame_idx"]
            bbox_norm = selected["bbox_norm"]
            score = selected["score"]
            label = selected["label"]

            # 放大并还原为像素坐标
            bbox_enlarged = enlarge_bbox_norm(bbox_norm, enlarge_scale)
            bbox_pixel = denormalize_bbox(bbox_enlarged, (H, W))
            x1, y1, x2, y2 = bbox_pixel

            # 裁剪图像
            frame = video_data[frame_idx]
            cropped = frame[y1:y2, x1:x2]
            cropped_rgb = cropped[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(cropped_rgb)
            save_path = output_path / f"{image_file_prefix}_instance_{inst_id}.png"
            img.save(save_path)

            # 保存记录（便于后续写入数据库）
            result_records.append({
                "instance_id": inst_id,
                "frame_idx": frame_idx,
                "score": score,
                "label": label,
                "bbox_norm": bbox_norm,
                "bbox_enlarged": bbox_enlarged,
                "bbox_pixel": bbox_pixel,
                "saved_path": str(save_path)
            })

        return result_records

def _visualize_frame_static(args):
    frame, track_result, frame_idx, fps, visualizer, score_thr = args

    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=score_thr,
        fps=fps
    )
    out_frame = visualizer.get_image()
    gc.collect()
    return out_frame

def load_video_as_numpy(video_path: str):
    video_cap = cv2.VideoCapture(video_path)
    
    # 获取视频的 FPS
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        frames.append(frame)
    
    video_cap.release()
    
    # 返回帧数据和 FPS
    return np.array(frames), fps # 返回形状为 (frames, height, width, channels) 的 numpy array

class CoarseCategory(IntEnum):
    """粗分类枚举"""
    PERSON = 0       # 人
    VEHICLE = 1      # 车
    NON_VEHICLE = 2  # 非机动车
    OTHER = 3        # 其他

def map_label(label: int) -> CoarseCategory:
    """
    将原始分类标签映射到粗分类枚举
    
    Args:
        label: 原始分类标签(0-6)
    
    Returns:
        CoarseCategory: 粗分类枚举值
    """
    mapping = {
        0: CoarseCategory.PERSON,
        1: CoarseCategory.VEHICLE,
        2: CoarseCategory.NON_VEHICLE,
        3: CoarseCategory.NON_VEHICLE,
        4: CoarseCategory.OTHER,
        5: CoarseCategory.OTHER,
        6: CoarseCategory.OTHER,
    }
    return mapping.get(label, None)