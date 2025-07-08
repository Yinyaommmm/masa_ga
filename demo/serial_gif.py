import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import numpy as np # 不能删，会引起线程库的冲突
import gc
import resource
import argparse
import time
import torch
from torch.multiprocessing import Pool, set_start_method
import imageio.v2 as imageio  # 用于 gif 输出

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms

from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from utils import filter_and_update_tracks

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)
from convert import convert_webp_to_mp4
# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

set_file_descriptor_limit(65535)

def visualize_frame(args, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=args.score_thr,
        fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame

def parse_args():

    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument( '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--save_dir', type=str, help='Output for video frames')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--line_width', type=int, default=5, help='Line width')
    parser.add_argument('--unified', action='store_true', help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--show_fps', action='store_true', help='Visualize the fps')
    parser.add_argument('--sam_mask', action='store_true', help='Use SAM to generate mask for segmentation tracking')
    parser.add_argument('--sam_path',  type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def main():
    start_time = time.time()
    print(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    args = parse_args()
    print("Args:  ", args)
    assert os.path.isdir(args.video), 'Now --video must be a folder containing .webp files.'
    if not os.path.exists(args.out):
        print(f'[INFO] Output folder "{args.out}" does not exist. Creating it...')
        os.makedirs(args.out)

    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        det_model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)

    texts = args.texts
    masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=(texts is not None))
    masa_model.cfg.visualizer['texts'] = texts if texts is not None else (
        det_model.dataset_meta['classes'] if not args.unified else [])
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)

    video_dir = args.video
    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)
    webp_files = sorted([f for f in os.listdir(video_dir)])
    total = len(webp_files)

    for idx, webp_file in enumerate(webp_files, 1):
        start = time.time()
        print(f"[{idx}/{total}] Processing {webp_file}...")
        input_path = os.path.join(video_dir, webp_file)
        output_name = webp_file.replace('.webp', '.gif')
        output_path = os.path.join(output_dir, output_name)

        video_input_path = convert_webp_to_mp4(input_path)
        video_reader = mmcv.VideoReader(video_input_path)

        frame_idx = 0
        instances_list = []
        frames = []
        fps_list = []
        last_time = time.time()

        for frame in track_iter_progress((video_reader, len(video_reader))):
            if args.unified:
                track_result = inference_masa(masa_model, frame,
                                              frame_id=frame_idx,
                                              video_len=len(video_reader),
                                              test_pipeline=masa_test_pipeline,
                                              text_prompt=texts,
                                              fp16=args.fp16,
                                              detector_type=args.detector_type,
                                              show_fps=args.show_fps)
                if args.show_fps:
                    track_result, fps = track_result
            else:
                if args.detector_type == 'mmdet':
                    # 走的这儿进行推理
                    result = inference_detector(det_model, frame,
                                                text_prompt=texts,
                                                test_pipeline=test_pipeline,
                                                fp16=args.fp16)
                det_bboxes, keep_idx = batched_nms(
                    boxes=result.pred_instances.bboxes,
                    scores=result.pred_instances.scores,
                    idxs=result.pred_instances.labels,
                    class_agnostic=True,
                    nms_cfg=dict(type='nms', iou_threshold=0.5,
                                 class_agnostic=True, split_thr=100000))
                det_bboxes = torch.cat([det_bboxes,
                                        result.pred_instances.scores[keep_idx].unsqueeze(1)], dim=1)
                det_labels = result.pred_instances.labels[keep_idx]

                track_result = inference_masa(masa_model, frame,
                                              frame_id=frame_idx,
                                              video_len=len(video_reader),
                                              test_pipeline=masa_test_pipeline,
                                              det_bboxes=det_bboxes,
                                              det_labels=det_labels,
                                              fp16=args.fp16,
                                              show_fps=args.show_fps)
                if args.show_fps:
                    track_result, fps = track_result

            frame_idx += 1
            if 'masks' in track_result[0].pred_track_instances and len(track_result[0].pred_track_instances.masks) > 0:
                track_result[0].pred_track_instances.masks = torch.stack(
                    track_result[0].pred_track_instances.masks, dim=0).cpu().numpy()
            track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
            instances_list.append(track_result.to('cpu'))
            frames.append(frame)
            if args.show_fps:
                fps_list.append(fps)

        if not args.no_post:
            instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]),
                                                      static_threshold=20,min_duration=5)

        elpase = time.time() - start
        print('Start to visualize the results...')
        num_cores = max(1, min(os.cpu_count() - 1, 16))
        print('Using {} cores for visualization'.format(num_cores))
        with Pool(processes=num_cores) as pool:
            if not args.show_fps:
                fps_list = [None] * len(frames) # 统一长度
            frames = pool.starmap(
                visualize_frame,
                [(args, visualizer, frame, track_result.to('cpu'), idx3, fps)
                for idx3, (frame, fps, track_result) in enumerate(zip(frames, fps_list, instances_list))])

        imageio.mimsave(output_path, [frame for frame in frames], duration=1/video_reader.fps)
        print(f'Deal all frames elapse {elpase}')
        print(f"[DONE] {output_name} saved. Time: {(time.time() - last_time):.2f}s")
        last_time = time.time()

    print('All done!')
    end_time = time.time()
    print(f"[END] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"[TOTAL TIME] {(end_time - start_time):.2f} seconds")

if __name__ == '__main__':
    main()
