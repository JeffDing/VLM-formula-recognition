from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

YOUR_ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN' #这边填你的modelscope token
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'user_name'   #这边填你的modelscope用户名
model_name = 'model_name'   #这边填模型名，自己起
model_id = f"{owner_name}/{model_name}"

api.create_model(
    model_id,
    visibility=ModelVisibility.PUBLIC,
    license=Licenses.APACHE_V2,
)


api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path='FILE_PATH',   #这边填sft后的模型路径地址
    commit_message='upload model folder to repo',
)