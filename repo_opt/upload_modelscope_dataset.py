from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'username'
dataset_name = 'dataset_name'

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='file_path',
    commit_message='upload dataset folder to repo',
    repo_type = 'dataset'
)