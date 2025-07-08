import os
from PIL import Image
import tqdm

folder_left = '/home/wqq/masa/demo_outputs/dynamic/82F_1_nopost'
folder_right = '/home/wqq/masa/demo_outputs/dynamic/82F_2_post'
output_folder = '/home/wqq/masa/demo_outputs/dynamic/82F1and2_gifs'

os.makedirs(output_folder, exist_ok=True)

left_files = {f for f in os.listdir(folder_left) if f.lower().endswith('.gif')}
right_files = {f for f in os.listdir(folder_right) if f.lower().endswith('.gif')}
common_files = left_files.intersection(right_files)

files_len = len(common_files)
print(f"Found {files_len} common GIF files to merge.")
idx = 0
for filename in common_files:
    idx+=1
    print(f"Progress: {idx} / {files_len}")
    left_path = os.path.join(folder_left, filename)
    right_path = os.path.join(folder_right, filename)

    left_gif = Image.open(left_path)
    right_gif = Image.open(right_path)

    frames_left = []
    frames_right = []

    # 读取左边GIF的所有帧
    try:
        while True:
            frames_left.append(left_gif.copy())
            left_gif.seek(left_gif.tell() + 1)
    except EOFError:
        pass

    # 读取右边GIF的所有帧
    try:
        while True:
            frames_right.append(right_gif.copy())
            right_gif.seek(right_gif.tell() + 1)
    except EOFError:
        pass

    # 帧数可能不一致，取最小帧数保证对应
    frame_count = min(len(frames_left), len(frames_right))

    merged_frames = []
    for i in range(frame_count):
        left_frame = frames_left[i].convert("RGBA")
        right_frame = frames_right[i].convert("RGBA")

        # 统一高度，按比例缩放右边图片
        h_left = left_frame.height
        h_right = right_frame.height
        if h_left != h_right:
            # 缩放右图高度匹配左图
            new_width = int(right_frame.width * (h_left / h_right))
            right_frame = right_frame.resize((new_width, h_left), Image.ANTIALIAS)

        # 新图宽度 = 左图宽 + 右图宽
        new_width = left_frame.width + right_frame.width
        new_img = Image.new("RGBA", (new_width, h_left))

        # 粘贴左右两边
        new_img.paste(left_frame, (0, 0))
        new_img.paste(right_frame, (left_frame.width, 0))

        merged_frames.append(new_img)

    # 保存合成后的gif，循环次数设为0(无限循环)
    out_path = os.path.join(output_folder, filename)
    merged_frames[0].save(out_path, save_all=True, append_images=merged_frames[1:], duration=left_gif.info.get('duration', 100), loop=0, disposal=2)

    print(f"Merged {filename} saved to {out_path}")

print("All done!")
