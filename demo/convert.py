import os
import imageio
import cv2
import tempfile

def convert_webp_to_mp4(webp_path):
    # 使用 imageio 读取所有帧
    reader = imageio.get_reader(webp_path)
    frames = [frame for frame in reader]
    height, width, _ = frames[0].shape

    # 创建临时 MP4 文件
    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_video_fd)  # 关闭文件描述符，不影响后续写入

    # 初始化视频写入器
    writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame in frames:
        # OpenCV 要求 BGR 格式
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()
    return temp_video_path  # 返回临时生成的 mp4 路径

