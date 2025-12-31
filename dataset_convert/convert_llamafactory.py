import json
import os

def to_abs(base_path, path):
    if not path or os.path.isabs(path):
        return path
    abs_path = os.path.join(base_path, path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"图像文件不存在: {abs_path}")
    return abs_path

def process_jsonl(input_path, output_path, base_dir):
    processed_lines = 0
    modified_lines = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)
            changed = False

            # 处理顶层 images 列表
            if 'images' in data and isinstance(data['images'], list):
                data['images'] = [to_abs(base_dir, img) for img in data['images']]
                changed = True

            # 处理 messages 中的 content
            if 'messages' in data:
                for msg in data['messages']:
                    # 转换 role -> from
                    if 'role' in msg:
                        role = msg.pop('role')
                        msg['from'] = 'human' if role == 'user' else 'gpt'
                    
                    # 转换 content -> value，并确保保留 <image> 标签
                    if 'content' in msg:
                        content = msg.pop('content')
                        # 如果 content 是字符串且包含 <image>，保留标签
                        if isinstance(content, str) and '<image>' in content:
                            msg['value'] = content  # 直接保留原始字符串
                        else:
                            msg['value'] = content  # 其他情况直接赋值
                        changed = True

            # 写入数据
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_lines += 1
            if changed:
                modified_lines += 1

    print(f"处理完成：总行数={processed_lines}，修改行数={modified_lines}，输出路径：{output_path}")

# 配置路径
INPUT_PATH = "/root/dataset/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl"
OUTPUT_PATH = "/root/dataset/VLM-formula-recognition-dataset_intern_camp/train/train_mini_lf.jsonl"
BASE_DIR = "/root/dataset/VLM-formula-recognition-dataset_intern_camp/train/"

# 执行处理
process_jsonl(INPUT_PATH, OUTPUT_PATH, BASE_DIR)