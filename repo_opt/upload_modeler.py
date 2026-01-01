from openmind_hub import create_repo
from openmind_hub import upload_folder

my_repo="REPO" #模型空间名字,自己取名
my_token="YOUR_TOKEN" #获取方式：https://modelers.cn/my/tokens

create_repo(
    repo_id=my_repo,
    token=my_token,
    private=True,
    license="unknown",
)

upload_folder(
    token=my_token,
    folder_path="FILE_PATH",#模型位置，
    repo_id=my_repo,
)