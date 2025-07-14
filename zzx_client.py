import json
import random
import time
import requests
import io
import numpy as np
from demo.masaqq_module import load_video_as_numpy, convert_webp_to_mp4
import concurrent.futures
def get_video_numpy(input_path):
    # 1. 转成mp4临时路径
    video_path = convert_webp_to_mp4(input_path)

    # 2. 读取mp4成numpy数组和fps（fps可以不用上传）
    video_data, video_fps = load_video_as_numpy(video_path)
    return video_data

def get_video_buffer(input_path):
    video_data = get_video_numpy(input_path)
    buffer = io.BytesIO()
    np.save(buffer, video_data)
    buffer.seek(0)
    return buffer

def test_inference():
    buffer = get_video_buffer("/home/user2/video_dev/output/NVR_ch28_main_20250327070013_20250327080013_1080P/fullImage/gif/27810_0.webp")

    # 4. 准备请求
    url = "http://127.0.0.1:10020/inference"  # 修改成你的服务地址和端口
    files = {
        "file": ("video.npy", buffer, "application/octet-stream")
    }

    # 5. 发送POST请求
    start_time = time.time()
    response = requests.post(url, files=files,headers={"Connection": "close"})
    elpased_time = time.time() - start_time
    # 6. 打印结果
    print(f"inference请求耗时: {elpased_time:.2f}秒, 状态码:, {response.status_code}")
    try:
        res = response.json()
        # print("返回JSON:", res)
        return res
    except Exception as e:
        print("解析JSON失败:", e)
        print("响应文本:", response.text)

def test_save_png(res):
    buffer = get_video_buffer("/home/user2/video_dev/output/NVR_ch28_main_20250327070013_20250327080013_1080P/fullImage/gif/27810_0.webp")
    if not res or "pred_instances_list" not in res:
        print("Inference failed or no instances found.")
        return

    url = "http://127.0.0.1:10020/savepng"
    pred_instances_list = res["pred_instances_list"]
    output_path = "/home/user2/wqq/masa/server_trash"
    random_suffix = random.randint(0, 9999)
    image_file_prefix = f"test_image_{random_suffix:04d}"
    enlarge_scale = 1.5
    select_percentile = 0.8


    # JSON 序列化字段
    body_data = {
        "pred_instances_list": pred_instances_list,
        "output_path": output_path,
        "image_file_prefix": image_file_prefix,
        "enlarge_scale": enlarge_scale,
        "select_percentile": select_percentile
    }

    files = {
        "file": ("video.npy", buffer, "application/octet-stream")
    }

    headers = {"accept": "application/json"}

    # 关键改动：把结构化 body 放到 data 里（作为表单字段）
    start_time = time.time()
    response = requests.post(
        url,
        data={"body": json.dumps(body_data)},
        files=files,
        headers=headers
    )
    elapsed_time = time.time() - start_time
    try:
        print(f"savepng请求耗时: {elapsed_time:.2f}秒, 状态码: {response.status_code}")
        # print("返回JSON:", json.dumps(response.json(), indent=2, ensure_ascii=False))

    except Exception as e:
        print("解析JSON失败:", e)
        print("响应文本:", response.text)

def inference_and_save(index: int):
    print(f"[{index}] 开始 test_inference")
    res = test_inference()
    print(f"[{index}] 开始 test_save_png")
    test_save_png(res)
    print(f"[{index}] ✅ 完成一组任务")
    return index
def inference_only(index: int):
    print(f"[{index}] 开始 test_inference")
    res = test_inference()
    print(f"[{index}] ✅ 完成Infer Only的测试")
    return index

if __name__ == "__main__":
    para_num = 128
    print(f"开始并行测试 {para_num} 组 test_inference + test_save_png...")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(para_num,16)) as executor:
        futures = [executor.submit(inference_only, i) for i in range(para_num)]
        for future in concurrent.futures.as_completed(futures):
            print(f"✅ 线程 {future.result()} 完成")
    
    print(f"✅ 所有任务完成, elapsed time: {time.time() - start_time:.2f}秒 ")
