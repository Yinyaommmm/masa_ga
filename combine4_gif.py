import os
from PIL import Image

def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(gif.convert("RGBA").copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

def resize_with_aspect(img, target_height):
    # 按目标高度等比缩放图片
    w, h = img.size
    new_w = int(w * (target_height / h))
    return img.resize((new_w, target_height), Image.Resampling.LANCZOS)

def merge_4_frames(f1, f2, f3, f4):
    # 统一4张图高度，取最大高度作为统一高度
    target_height = max(f1.height, f2.height, f3.height, f4.height)
    f1 = resize_with_aspect(f1, target_height)
    f2 = resize_with_aspect(f2, target_height)
    f3 = resize_with_aspect(f3, target_height)
    f4 = resize_with_aspect(f4, target_height)

    # 计算左右两列宽度
    left_width = max(f1.width, f3.width)
    right_width = max(f2.width, f4.width)

    # 计算上下两行高度 (都是 target_height)
    total_width = left_width + right_width
    total_height = target_height * 2

    # 新建空图
    new_img = Image.new("RGBA", (total_width, total_height), (0,0,0,0))

    # 粘贴四张图（左上，右上，左下，右下）
    new_img.paste(f1, (0, 0))
    new_img.paste(f2, (left_width, 0))
    new_img.paste(f3, (0, target_height))
    new_img.paste(f4, (left_width, target_height))

    return new_img

def merge_4folder_gifs(folder1, folder2, folder3, folder4, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files1 = {f for f in os.listdir(folder1) if f.lower().endswith('.gif')}
    files2 = {f for f in os.listdir(folder2) if f.lower().endswith('.gif')}
    files3 = {f for f in os.listdir(folder3) if f.lower().endswith('.gif')}
    files4 = {f for f in os.listdir(folder4) if f.lower().endswith('.gif')}

    common_files = files1 & files2 & files3 & files4
    print(f"Found {len(common_files)} common GIF files to merge.")

    for idx, filename in enumerate(sorted(common_files), 1):
        print(f"Processing {idx}/{len(common_files)}: {filename}")

        frames1 = load_gif_frames(os.path.join(folder1, filename))
        frames2 = load_gif_frames(os.path.join(folder2, filename))
        frames3 = load_gif_frames(os.path.join(folder3, filename))
        frames4 = load_gif_frames(os.path.join(folder4, filename))

        frame_count = min(len(frames1), len(frames2), len(frames3), len(frames4))

        merged_frames = []
        for i in range(frame_count):
            merged_frame = merge_4_frames(frames1[i], frames2[i], frames3[i], frames4[i])
            merged_frames.append(merged_frame)

        # duration 使用第一个gif的 duration，默认100ms
        first_gif_path = os.path.join(folder1, filename)
        with Image.open(first_gif_path) as img:
            duration = img.info.get('duration', 100)

        out_path = os.path.join(output_folder, filename)
        merged_frames[0].save(
            out_path,
            save_all=True,
            append_images=merged_frames[1:],
            duration=duration,
            loop=0,
            disposal=2
        )
        print(f"Saved merged gif: {out_path}")

    print("All GIFs merged successfully.")

if __name__ == "__main__":
    folder_1 = '/home/wqq/masa/demo_outputs/dynamic/9D_1_nopost'
    folder_2 = '/home/wqq/masa/demo_outputs/dynamic/9D_3_post_threshd10'
    folder_3 = '/home/wqq/masa/demo_outputs/dynamic/9D_4_post_threshd20'
    folder_4 = '/home/wqq/masa/demo_outputs/dynamic/9D_6_post_trhd20_min5'
    output_folder = '/home/wqq/masa/demo_outputs/dynamic/merged1346_4grid_gifs'

    merge_4folder_gifs(folder_1, folder_2, folder_3, folder_4, output_folder)
