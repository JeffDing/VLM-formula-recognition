from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

# 配置基本信息
YOUR_ACCESS_TOKEN = "ms-"  # 填写自己的 api token ，获取方式点击：https://modelscope.cn/my/myaccesstoken
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# 取名字
owner_name = ""  # ModelScope 的用户名
model_name = "formula-recognition-internvl_3_5-1b-metax-20251117"  # 为模型库取个响亮优雅又好听的名字，需根据自己情况修改
model_id = f"{owner_name}/{model_name}"

# 创建模型仓库
api.create_model(
    model_id,
    visibility=ModelVisibility.PUBLIC,
    license=Licenses.APACHE_V2,
    chinese_name=f"{owner_name}的书生大模型实战营-公式微调分类比赛",
)

# 上传模型到仓库
api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path="/root/data/VLM-formula-recognition-dataset/swift_output/SFT-InternVL3_5-1B/v3-20251117-155608/checkpoint-1125-merged",  # 微调后模型的文件夹名称
    commit_message="upload model folder to repo",  # 写上传信息
)

