import json
import os

def to_abs(base_path, path):
    """将相对路径转换为绝对路径，并验证文件是否存在"""
    if not path or os.path.isabs(path):
        return path
    abs_path = os.path.join(base_path, path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"图像文件不存在: {abs_path}")
    return abs_path

def process_jsonl(input_path, output_path, base_dir):
    processed_lines = 0
    current_id = 0  # 用于生成递增的 id 字段

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)

            # 添加 id 字段
            data['id'] = current_id
            current_id += 1

            # 将 images 列表转换为 image 字段（取第一个图像路径）
            if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
                data['image'] = to_abs(base_dir, data['images'][0])
                del data['images']
            else:
                data['image'] = None  # 如果没有 images 字段或为空，设置为 None

            # 将 messages 转换为 conversations
            if 'messages' in data:
                conversations = []
                for msg in data['messages']:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    from_role = 'human' if role == 'user' else 'gpt'
                    conversations.append({'from': from_role, 'value': content})
                data['conversations'] = conversations
                del data['messages']

            # 写入处理后的数据
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_lines += 1

    print(f"处理完成：总处理行数 = {processed_lines}，输出路径：{output_path}")

# 配置路径
INPUT_PATH = "/root/dataset/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl"
OUTPUT_PATH = "/root/dataset/VLM-formula-recognition-dataset_intern_camp/train/train_mini_msmm.jsonl"
BASE_DIR = "/root/dataset/VLM-formula-recognition-dataset_intern_camp/train/"

# 执行处理
process_jsonl(INPUT_PATH, OUTPUT_PATH, BASE_DIR)
