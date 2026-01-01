from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
#from modelscope.hub.api.HubApi import delete_model, list_models


YOUR_ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'JeffDing'

listmodels=api.list_models(
    owner_or_group="YOUR_NAME OR NAMESPACE",
    page_size=100
)

# 遍历 Models 列表，提取每个模型的 model_id
model_ids = []
for model in listmodels.get("Models", []):
    backend_support = model.get("BackendSupport", {})
    model_id = backend_support.get("model_id")
    if model_id:
        model_ids.append(model_id)

# 输出提取到的 model_id
print("提取到的 model_id 列表：")
for idx, model_id in enumerate(model_ids, 1):
    print(f"{idx}. {model_id}")

print("输出过滤后的：")

# 遍历模型并过滤
for model in listmodels["Models"]:
    delete_model_id = model["BackendSupport"]["model_id"]
    if "prefix" in delete_model_id:
        print(f"Deleting: {delete_model_id }")
        api.delete_model(model_id=delete_model_id )