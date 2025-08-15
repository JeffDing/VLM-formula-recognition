# metax-swift-internlm-finetune-paper_classificaiton

## 数据集信息

VLM-formula-recognition-dataset_intern_camp：

[https://www.modelscope.cn/datasets/JimmyMa99/VLM-formula-recognition-dataset_intern_camp](https://www.modelscope.cn/datasets/JimmyMa99/VLM-formula-recognition-dataset_intern_camp)

## GPU环境配置
```bash
conda create -n swift python=3.12 -y
conda activate swift
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# flash-attention从github代码仓下载，根据python、torch、cuda版本下载对应的whl安装文件
https://github.com/Dao-AILab/flash-attention/
pip install flash_attnXXX.whl --no-build-isolation  #前面下载下来的文件名

# 安装ms-swift
pip install ms-swift==3.5.0 -U

# 如果想源码安装的执行以下步骤，直接pip的话直接跳过
cd /tmp/code
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout v3.5.0
pip install -e .

# 安装wandb
pip install wandb
# 安装deepspeed
pip install deepspeed

pip install transformers==4.48.3
pip install numpy==1.26.4

cd /tmp/code
```

## 沐曦环境准备
### 沐曦开发者社区
[https://developer.metax-tech.com/](https://developer.metax-tech.com/)


### 克隆代码仓
```bash
git clone https://openi.pcl.ac.cn/fresh-little-lemon/metax-swift-internlm-finetune-paper_classificaiton
cd metax-swift-internlm-finetune-paper_classificaiton
```
### 安装环境
```bash
conda create -n swift python=3.10 -y
conda activate swift
pip install -r requirements.txt -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com --no-build-isolation
```
### 源码安装
```bash
tar -Jxvf mxc500-deepspeed-py310-2.32.0.5-linux-x86_64.tar.xz
cd mxc500-deepspeed-2.32.0.5/wheel/
pip install deepspeed-0.15.1+4225e38d-py3-none-any.whl

# 安装ms-swift
pip install ms-swift==3.5.0 -U

# 如果想源码安装的执行以下步骤，直接pip的话直接跳过
cd /tmp/code
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout v3.5.0
pip install -e .

pip install wandb
pip install transformers==4.48.3
pip install numpy==1.26.4

cd /tmp/code
```

## 沐曦安装Xtuner
```bash
# 安装环境
pip install -r requirements_xtuner.txt -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com --no-build-isolation

# 安装bitsandbytes
tar -Jxvf mxc500-bitsandbytes-py310-2.32.0.5-linux-x86_64.tar.xz
cd mxc500-bitsandbytes-2.32.0.5/wheel
pip install bitsandbytes-0.45.2+maca2.32.0.5-cp310-cp310-linux_x86_64.whl

# 先安装deepspeed,然后下载xtuner安装
tar -Jxvf mxc500-xtuner-py310-2.32.0.7-linux-x86_64.tar.xz
cd mxc500-xtuner-2.32.0.7/xtuner/

pip install -e .[all]

pip install timm
pip install numpy==1.26.4

# 如果提示LibGL不操作的话安装下面两个库
apt install libgl1-mesa-glx libgl1-mesa-dri libglib2.0-0
```

## NPU环境配置
###  创建新的conda虚拟环境(可选)
```bash
conda create -n swift-npu python=3.10 -y
conda activate swift-npu
```
### 安装torch-npu
```bash
pip install torch==2.3.1 torch-npu==2.3.1 torchaudio==2.3.1 torchvision decorator
pip install ms-swift==3.5.0 -U

# 如果想源码安装的执行以下步骤，直接pip的话直接跳过
cd /tmp/code
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout v3.5.0
pip install -e .

pip install wandb -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
### 如果你想要使用deepspeed (控制显存占用,训练速度会有一定下降)
```bash
pip install deepspeed -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install transformers==4.48.3
pip install numpy==1.26.4
```

## NPU安装Xtuner
```bash
# 安装torch
pip install torch==2.3.1 torch-npu==2.3.1 torchaudio==2.3.1 torchvision

# clone代码仓
git clone https://github.moeyy.xyz/https://github.com/InternLM/lmdeploy
git clone https://github.moeyy.xyz/https://github.com/InternLM/xtuner.git

# 安装LMDeploy
cd lmdeploy
pip install -r requirements_ascend.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
LMDEPLOY_TARGET_DEVICE=ascend pip3 install -v --no-build-isolation -e .

#安装Transformers4.48.0
pip install transformers==4.48.3

#安装deepspeed及mpi4py
pip install deepspeed==0.16.2
conda install mpi4py

#安装XTuner:
cd ../xtuner/
# 删除requirements/runtime.txt中的第一行bitsandbytes==0.45.0
# 删除requirements.txt文件中的-r requirements/deepspeed.txt 这一行
pip install -e '.[all]'
```


## 脚本参数解析

### 预训练
```python
#!/bin/bash

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR  # 确保日志目录存在，如果不存在则创建

# 获取当前时间戳，用于生成唯一的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/internlm3-8b_lora_sft_${TIMESTAMP}.log"  # 设置日志文件路径

# 断点续训Weights only load failed解决
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 设置CUDA环境变量
export NPROC_PER_NODE=1  # 设置每个节点使用的进程数为1
export OMP_NUM_THREADS=1  # 限制OpenMP线程数为1，避免过多线程竞争
export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU编号为0
export MASTER_PORT=$((10000 + RANDOM % 50000))

# 使用nohup命令在后台运行训练任务，即使终端关闭也能继续运行
nohup swift sft \
    --model {model_path} \  # 指定基础模型路径
    --train_type lora \  # 使用LoRA训练方法
    --dataset {data_path} \  # 指定训练数据集
    --torch_dtype bfloat16 \  # 使用bfloat16精度以节省显存
    --num_train_epochs 1 \  # 设置训练轮数为2
    --per_device_train_batch_size 2 \  # 每个设备的训练批次大小为4
    --learning_rate 5e-5 \  # 学习率设置为5e-5
    --warmup_ratio 0.1 \  # 预热阶段占总训练步数的10%
    --split_dataset_ratio 0 \  # 不拆分数据集
    --lora_rank 8 \  # LoRA的秩设置为8
    --lora_alpha 32 \  # LoRA的alpha参数设置为32
    --use_chat_template false \  # 不使用聊天模板
    --target_modules all-linear \  # 对所有线性层应用LoRA
    --gradient_accumulation_steps 2 \  # 梯度累积步数为2，用于增大有效批次大小
    --save_steps 2000 \  # 每2000步保存一次模型
    --save_total_limit 5 \  # 最多保存5个检查点
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \  # 梯度检查点设置，禁用重入
    --logging_steps 5 \  # 每5步记录一次日志
    --max_length 2048 \  # 最大序列长度设为2048
    --output_dir ./swift_output/InternLM3-8B-Lora \  # 输出目录
    --dataloader_num_workers 256 \  # 数据加载器使用256个工作线程
    --model_author JeffDing \  # 模型作者信息
    --model_name InternLM3-8B-Lora \  # 模型名称
    --resume_from_checkpoint checkpoint_dir # 断点续训参数，传入checkpoint路径。默认为None。断点续训请保持其他参数不变，额外增加 --resume_from_checkpoint checkpoint_dir
    #注意:resume_from_checkpoint会读取模型权重，优化器权重，随机种子，并从上次训练的steps继续开始训练。你可以指定--resume_only_mode1 只读取模型权重。

# 打印进程ID和日志文件位置，便于用户跟踪
echo "Training started with PID $!"  # 显示后台进程的PID
echo "Log file: $LOG_FILE"  # 显示日志文件位置

# 提示用户如何实时查看日志
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"
```

### 微调
```python
#!/bin/bash
# 指定使用bash解释器执行脚本

# 创建日志目录
LOG_DIR="logs"
# 定义日志存储目录变量
mkdir -p $LOG_DIR
# 创建日志目录，-p参数确保即使目录已存在也不会报错

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 获取当前时间并格式化为年月日_时分秒格式
LOG_FILE="$LOG_DIR/internlm3-8b_lora_sft_${TIMESTAMP}.log"
# 组合日志文件路径，使用时间戳确保文件名唯一

# 断点续训Weights only load failed解决
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 设置CUDA设备
export NPROC_PER_NODE=1
# 设置每个节点的进程数为1
export OMP_NUM_THREADS=1
# 设置OpenMP线程数为1，限制并行线程数
export CUDA_VISIBLE_DEVICES=0
# 指定使用的GPU设备为0号设备

nohup swift sft \
# 使用nohup命令在后台运行swift sft命令，即使终端关闭也能继续运行
    --model /root/code/camp5_course/swift_output/InternLM3-8B-Lora/v1-20250416-140542/checkpoint-74-merged \
    # 指定基础模型路径，使用之前训练的checkpoint-74的合并模型
    --train_type lora \
    # 设置训练类型为LoRA（低秩适应）
    --dataset '/root/code/camp5_course/data/swift_formatted_sft_train_data.jsonl' \
    # 指定训练数据集的路径
    --torch_dtype bfloat16 \
    # 设置模型参数的数据类型为bfloat16，减少内存占用
    --num_train_epochs 1 \
    # 设置训练轮数为5轮
    --per_device_train_batch_size 22 \
    # 设置每个设备的训练批次大小为8
    --learning_rate 1e-4 \
    # 设置学习率为0.0001
    --lr_scheduler_type cosine \
    # lr_scheduler类型，默认 consine, consine：余弦退火 constant：常量lr
    --warmup_ratio 0.1 \
    # 设置预热比例为0.1，即10%的训练步骤用于学习率从小到大的预热
    --split_dataset_ratio 0 \
    # 设置数据集分割比例为0，不进行训练/验证分割
    --report_to wandb \
    # 设置训练报告发送到Weights & Biases平台
    --lora_rank 8 \
    # 设置LoRA的秩为8，控制可训练参数的数量
    --lora_alpha 32 \
    # 设置LoRA的alpha为32，影响LoRA更新的缩放程度
    --target_modules all-linear \
    # 设置LoRA目标模块为所有线性层
    --gradient_accumulation_steps 2 \
    # 设置梯度累积步数为2，相当于扩大了批次大小
    --save_steps 2000 \
    # 每2000步保存一次检查点
    --save_total_limit 5 \
    # 最多保存5个检查点，超过会删除旧的
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    # 设置梯度检查点的参数，关闭重入功能以提高稳定性
    --logging_steps 5 \
    # 每5步记录一次日志
    --max_length 2048 \
    # 设置最大序列长度为2048
    --output_dir ./swift_output/InternLM3-8B-Lora \
    # 设置输出目录
    --dataloader_num_workers 256 \
    # 设置数据加载器的工作进程数为256，加速数据加载
    --model_author JeffDing \
    # 设置模型作者信息
    --model_name InternLM3-8B-Lora \
    # 设置模型名称
    --resume_from_checkpoint checkpoint_dir # 断点续训参数，传入checkpoint路径。默认为None。断点续训请保持其他参数不变，额外增加 --resume_from_checkpoint checkpoint_dir
    #注意:resume_from_checkpoint会读取模型权重，优化器权重，随机种子，并从上次训练的steps继续开始训练。你可以指定--resume_only_mode1 只读取模型权重。

# 打印进程ID和日志文件位置
echo "Training started with PID $!"
# 显示训练进程的PID（$!代表最近一个后台进程的PID）
echo "Log file: $LOG_FILE"
# 显示日志文件的路径

# 显示查看日志的命令
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"
# 提示用户如何实时查看日志文件内容
```
