import os
os.environ["USE_FLASH_ATTENTION"] = "0"

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = "/tmp/code/workdir/vl_1031/20251104125845/hf-450"

# 加载处理器和模型
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,  # 使用 torch_dtype 而不是 dtype
    trust_remote_code=True,
    attn_implementation="eager"  # 禁用 flash attention
)

# 加载图像
image = Image.open("/tmp/code/dataset/VLM-formula-recognition-dataset_intern_camp/train/mini_train/sample19996.png")
prompt = "describle this img"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

# 处理输入
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 确保输入数据类型与模型一致
inputs = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

# 生成
with torch.inference_mode():
    generate_ids = model.generate(**inputs, max_new_tokens=200)

decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("current model: ", model_name)
print(decoded_output)
