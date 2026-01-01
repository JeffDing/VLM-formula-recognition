from xtuner.v1.model import InternVL3P5Dense1BConfig
from xtuner.v1.train import Trainer, TrainerConfig
from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig, DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
import sys
# model config - 启用梯度检查点
model_cfg = InternVL3P5Dense1BConfig(
    use_gradient_checkpointing=True, freeze_vision=True, freeze_projector=False, freeze_language=False
)
# dataset and dataloader config
sample_max_length = 8000
pack_max_length = 8000

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="formula_recognition",
            anno_path="/root/data/dataset/VLM-formula-recognition-dataset-intern-camp/train/train_mini_xt.jsonl",
            media_root="/root/data/dataset/VLM-formula-recognition-dataset-intern-camp/train/",
            sample_ratio=1.0,
            class_name="VLMJsonlDataset",
        ),
        # 使用 InternVL3.5 模板，确保 prompt 与视觉 token 对齐
        "tokenize_fn": InternS1VLTokenizeFnConfig(
            model_cfg=model_cfg,
            max_length=sample_max_length,
            template_name="internvl-3.5",
        ),
    }
]
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    num_workers=16,
    pack_level="soft",
    collator="intern_s1_vl_sft_collator",
)

# 优化学习率配置 - 提高学习率以加快收敛
optim_cfg = AdamWConfig(
    lr=3e-5,           # 提高学习率，从1e-6到3e-5
    weight_decay=0.01, # 添加权重衰减防止过拟合
    betas=(0.9, 0.95), # 优化Adam参数
    foreach=False
)
lr_cfg = LRConfig(
    lr_type="cosine",
    warmup_ratio=0.1,  # 增加warmup比例，让模型更稳定地开始训练
    min_lr_ratio=0.1,   # 添加最小学习率比例
    lora_rank=128,
    lora_alpha=1024
)

load_from = "/root/data/model/InternVL3_5-1B-HF"
tokenizer = "/root/data/model/InternVL3_5-1B-HF"

# trainer config
trainer = TrainerConfig(
    load_from=load_from,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    global_batch_size=8,
    gradient_accumulation_steps=4,
    total_epoch=5,
    work_dir="/root/xtuner_workdir/output_ckpt/",
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    hf_interval=200,
    hf_max_keep=2,
)
trainer = Trainer.from_config(trainer)
# 检查模型是否正确加载了预训练权重
print(f"Model device: {next(trainer._engine.model.        parameters()).device}")
print(f"Model dtype: {next(trainer._engine.model.parameters()).dtype}")
# sys.exit(0)
trainer.fit()
