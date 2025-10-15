import json, os, sys

in_path  = "train_mini.jsonl"       # 你的现用 jsonl
out_path = "train_mini_abs.jsonl"   # 输出到新文件jsonl
base = "./"  # train_swift.jsonl所在的文件夹

def to_abs(p):
    if not p: return p
    # 仅当是相对路径时拼接；已是绝对路径则保留
    return p if os.path.isabs(p) else os.path.join(base, p)

n=0;m=0
with open(in_path,'r',encoding='utf-8') as fin, open(out_path,'w',encoding='utf-8') as fout:
    for line in fin:
        if not line.strip(): continue
        obj = json.loads(line)
        changed = False
        # 兼容两种常见结构：images: [..] 或 messages[].content[].image
        if 'images' in obj and isinstance(obj['images'], list):
            obj['images'] = [to_abs(x) for x in obj['images']]
            changed = True
        if 'messages' in obj:
            for msg in obj['messages']:
                c = msg.get('content')
                if isinstance(c, list):
                    for part in c:
                        if part.get('type') == 'image' and 'image' in part:
                            part['image'] = to_abs(part['image'])
                            changed = True
        fout.write(json.dumps(obj,ensure_ascii=False)+'\n')
        n+=1
        m+=changed
print(f"processed lines={n}, patched={m}, saved -> {out_path}")
