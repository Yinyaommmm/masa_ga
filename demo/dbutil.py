# 读取 YAML 配置文件
def load_config(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件 {file_path} 不存在")
        return None
    except yaml.YAMLError as e:
        print(f"解析 YAML 失败: {e}")
        return None