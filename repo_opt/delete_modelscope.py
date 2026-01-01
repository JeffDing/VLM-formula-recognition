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

for model in listmodels["Models"]:
    model_id = model["BackendSupport"]["model_id"]
    if "prefix" in model_id:
        print(f"Deleting: {model_id }")
        api.delete_model(model_id=model_id )