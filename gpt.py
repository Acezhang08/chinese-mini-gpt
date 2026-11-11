# A character-level Chinese mini GPT, trained on Tang poems.
# Needs: pip install torch opencc-python-reimplemented
# Test loss 5.26 (T4 GPU, 20000 steps).
import json, urllib.request, time
import torch
from opencc import OpenCC
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== 块 0：数据 =====
cc = OpenCC('t2s')                      # 繁体 → 简体
base = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.{}.json'
poems = []                              # 一首诗一个字符串
for i in range(58):                     # 58 个文件：0、1000、2000 …… 57000
    data = json.loads(urllib.request.urlopen(base.format(i * 1000)).read())
    for p in data:
        poems.append(cc.convert(''.join(p['paragraphs'])))
first5 = poems[:5003]                   # 前 5 个文件
cut = int(len(first5) * 0.9)
test_text  = ''.join(first5[cut:])                          # 测试：前 5 个文件的后 10%（一直不换）
train_text = ''.join(first5[:cut]) + ''.join(poems[5003:])  # 训练：其余全部
print('训练', len(train_text), '字；测试', len(test_text), '字')

# ===== 块 1：字表、字 → 号 =====
chars = sorted(set(train_text + test_text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}          # 字 → 号
itos = {i: c for i, c in enumerate(chars)}          # 号 → 字
train_ids = torch.tensor([stoi[c] for c in train_text], device=dev)
test_ids  = torch.tensor([stoi[c] for c in test_text],  device=dev)
print('字表', V, '个字')

# ===== 块 2：设置 + 旋钮 =====
T, B = 32, 32                  # 一行看 32 个字；一次拿 32 行
D, H, L, F = 32, 4, 4, 128     # 每字几个数、几个头、几层、前馈中间几个数
torch.manual_seed(0)
def knob(*shape, scale=None):
    w = torch.randn(*shape, device=dev) * (scale if scale is not None else shape[0] ** -0.5)
    return w.requires_grad_()
C = knob(V, D, scale=1.0)      # 字表：每个字一串数
P = knob(T, D, scale=1.0)      # 位置表：每个位置一串数
layers = []
for _ in range(L):             # 每层一套自己的旋钮
    layers.append(dict(
        g1=torch.ones(D, device=dev, requires_grad=True), b1n=torch.zeros(D, device=dev, requires_grad=True),
        Wq=knob(D, D), Wk=knob(D, D), Wv=knob(D, D), Wo=knob(D, D),
        g2=torch.ones(D, device=dev, requires_grad=True), b2n=torch.zeros(D, device=dev, requires_grad=True),
        W1=knob(D, F), b1=torch.zeros(F, device=dev, requires_grad=True),
        W2=knob(F, D), b2=torch.zeros(D, device=dev, requires_grad=True)))
gf = torch.ones(D, device=dev, requires_grad=True); bf = torch.zeros(D, device=dev, requires_grad=True)
Wout = knob(D, V, scale=0.01)  # 起点很小：一开始每个字的分都差不多
params = [C, P, gf, bf, Wout] + [w for l in layers for w in l.values()]
print('旋钮一共', sum(p.numel() for p in params), '个')

# ===== 块 3：往前走 =====
mask = torch.tril(torch.ones(T, T, device=dev))    # 下三角是 1：只能看自己和左边
def layernorm(x, g, b):
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - m) / (s + 1e-5) * g + b
def attention(x, l):
    n, t, _ = x.shape
    q = (x @ l['Wq']).view(n, t, H, D // H).transpose(1, 2)   # n×4×t×8：每个头 t 个位置 × 8 个数
    k = (x @ l['Wk']).view(n, t, H, D // H).transpose(1, 2)
    v = (x @ l['Wv']).view(n, t, H, D // H).transpose(1, 2)
    s = q @ k.transpose(-2, -1) / (D // H) ** 0.5              # 打分表
    s = s.masked_fill(mask[:t, :t] == 0, float('-inf'))        # 右上三角填 −∞
    a = torch.softmax(s, dim=-1)
    out = (a @ v).transpose(1, 2).reshape(n, t, D)             # 4 个头接起来
    return out @ l['Wo']
def feedforward(x, l):
    return torch.relu(x @ l['W1'] + l['b1']) @ l['W2'] + l['b2']
def forward(X):
    t = X.shape[1]
    x = C[X] + P[torch.arange(t, device=dev)]                  # 字 + 位置
    for l in layers:
        x = x + attention(layernorm(x, l['g1'], l['b1n']), l)  # 残差
        x = x + feedforward(layernorm(x, l['g2'], l['b2n']), l)
    return layernorm(x, gf, bf) @ Wout                         # 先归一化，再 × Wout

# ===== 块 4：拿一批、测试打分 =====
def get_batch(ids):
    i = torch.randint(0, len(ids) - T - 1, (B,), device=dev)
    X = torch.stack([ids[j:j+T] for j in i]); Y = torch.stack([ids[j+1:j+T+1] for j in i])
    return X, Y
@torch.no_grad()
def test_loss():
    n = (len(test_ids) - 1) // T
    X = test_ids[:n*T].view(n, T); Y = test_ids[1:n*T+1].view(n, T)
    tot = 0
    for s in range(0, n, 512):
        lo = forward(X[s:s+512])
        tot += torch.nn.functional.cross_entropy(lo.reshape(-1, V), Y[s:s+512].reshape(-1), reduction='sum').item()
    return tot / (n * T)

# ===== 块 5：训练（Adam）=====
STEPS = 20000
LR = 0.003
opt = torch.optim.Adam(params, lr=LR)          # Adam：每个旋钮自己记着最近几圈的坡
t0 = time.time()
for step in range(STEPS):
    X, Y = get_batch(train_ids)
    logits = forward(X)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), Y.reshape(-1))
    opt.zero_grad()                              # 清零
    loss.backward()                              # 往回
    for grp in opt.param_groups:
        grp['lr'] = LR if step < STEPS * 3 // 4 else LR / 10   # 最后四分之一变成十分之一
    opt.step()                                   # 照 Adam 的公式挪
    if step % 2000 == 0:
        print(step, '训练', round(loss.item(), 3), '用了', int(time.time() - t0), '秒')
print('测试损失', round(test_loss(), 3))

# ===== 块 6：写诗 =====
@torch.no_grad()
def write(start, n=48):
    ids = [stoi[c] for c in start]                  # 开头的字 → 号
    for _ in range(n):
        X = torch.tensor([ids[-T:]], device=dev)    # 只拿最后 T 个号，排成 1 行
        logits = forward(X)                         # 1 × t × V 个分
        p = torch.softmax(logits[0, -1], dim=-1)    # 只要最后一个位置的 V 个概率
        nxt = torch.multinomial(p, 1).item()        # 按概率抽一个
        ids.append(nxt)                             # 接到后面
    return ''.join(itos[i] for i in ids)            # 号 → 字

torch.manual_seed(1)
for start in ['春', '月', '山']:
    print(write(start))
torch.save(params, 'gpt_t32.pt')
