# Post-training (fine-tuning) a base Qwen2.5-0.5B on 3000 Chinese Q&A pairs.
# Ran on one RTX 4090 (AutoDL), 400 steps, a few minutes. Loss 4.67 -> ~1.5.
# Needs: pip install torch transformers "modelscope[datasets]" addict
# Models and data come from ModelScope (works from China).
import random
import torch
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 第 1 格：拿模型 + 一个"问它"的函数 =====
base_path = snapshot_download("Qwen/Qwen2.5-0.5B")            # 只预训练过的，下到本地
inst_path = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")   # 后训练过的，只借字表和包记号的写法

tok = AutoTokenizer.from_pretrained(inst_path)
model = AutoModelForCausalLM.from_pretrained(base_path).to("cuda")   # 旋钮用只预训练过的
end_id = tok.convert_tokens_to_ids("<|im_end|>")

def ask(q, n=80):
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # 包上记号
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    start = ids.shape[1]                       # 问题包完一共几个 token，后面的才是它写的
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            logits = model(ids).logits
            p = torch.softmax(logits[0, -1].float(), dim=-1)
            nxt = torch.multinomial(p, 1)
            ids = torch.cat([ids, nxt[None]], dim=1)
            if nxt.item() == end_id:
                break
    return tok.decode(ids[0, start:])          # 只显示它写的部分

print("===== 训练前 =====")
torch.manual_seed(0)
for q in ["春天开什么花？", "怎么煮米饭？", "1 加 1 等于几？"]:
    print(q, "→", ask(q))
    print("=====")

# ===== 第 2 格：拿问答数据（约 4.9 万条，挑 3000 条短的）=====
ds = MsDataset.load("AI-ModelScope/alpaca-gpt4-data-zh", split="train")

data = []
for row in ds:
    if row["input"]:                 # 有的题还附了一段材料，先不要，只留"一个问题 + 一个回答"的
        continue
    if len(row["output"]) > 300:     # 回答太长的不要，训得快
        continue
    msgs = [{"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)   # 问题和回答一起包上记号
    data.append(tok(text).input_ids[:256])                  # 变成号，最多留 256 个
    if len(data) == 3000:
        break

print(len(data))
print(tok.decode(data[0]))           # 看第 0 条包完长什么样

# ===== 第 3 格：训 =====
opt = torch.optim.Adam(model.parameters(), lr=1e-5)   # 比迷你 GPT 小 300 倍：旋钮已经练好，轻轻挪
B = 8
model.train()

for step in range(400):
    batch = random.sample(data, B)                     # 随机抽 8 条
    L = max(len(x) for x in batch)                     # 最长那条几个号
    inp = torch.full((B, L), end_id)                   # 短的后面拿 <|im_end|> 补齐
    lab = torch.full((B, L), -100)                     # -100 = 这一格不打分
    for i, x in enumerate(batch):
        inp[i, :len(x)] = torch.tensor(x)
        lab[i, :len(x)] = torch.tensor(x)              # 补齐的那几格保持 -100
    inp, lab = inp.cuda(), lab.cuda()

    loss = model(inp, labels=lab).loss                 # 每个位置猜下一个号，算 −log
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(step, round(loss.item(), 3))

print("===== 训练后 =====")
torch.manual_seed(0)
for q in ["春天开什么花？", "怎么煮米饭？", "1 加 1 等于几？"]:
    print(q, "→", ask(q))
    print("=====")
