from openmind_hub import list_models, delete_repo

# 修改为自己的token和组织ID
Token_write = "YOUR_TOKEN"
prefix = 'PREFIX'

# 遍历 MyTest 下所有模型
for m in list_models(author="YOUR_NAME", token=Token_write):
    # 检查模型名称是否包含 "openmind"（不区分大小写）
    if prefix in m.name.lower():  # 过滤条件：名称中包含 openmind
        repo_id = f"{m.owner}/{m.name}"  # 构造仓库ID
        print("Deleting:", repo_id)