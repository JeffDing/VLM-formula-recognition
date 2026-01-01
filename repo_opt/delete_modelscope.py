from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


YOUR_ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'owner_or_group'
prefix='prefix'

listmodels = api.list_models(
    owner_or_group=owner_name,
    page_size=100
)

for model in listmodels["Models"]:
    model_id = model["BackendSupport"]["model_id"]
    if prefix in model_id:
        repo_id = model_id
        print(f"Deleting: {repo_id}")
        api.delete_model(model_id=repo_id)