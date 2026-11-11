# "1 plus 1 equals 1": building a Chinese GPT from scratch

In this stage I started from "how does text become numbers" and worked all the way up to a small Chinese GPT that writes five-character Tang poems. At the end I post-trained a real Qwen model. Below is what I figured out along the way. Every number here is one I ran myself or ran into.

## Why "猫" is 3

Under the hood, a computer stores text as numbers from 0 to 255, called bytes. An English letter is 1 byte (a is 97). "猫" (cat) is 3 bytes: 231, 140, 171.

My first thought: the pinyin for 猫 is "mao", which is 3 letters. So I assumed Chinese first turns into pinyin letters, then into numbers. It doesn't. "我" is "wo" in pinyin, only 2 letters. "张" is "zhang", 5 letters. Both are still 3 bytes. Every common Chinese character is 3 bytes. It has nothing to do with how the character sounds. It is just the character's number in one worldwide code table, split into 3 pieces.

Real models (GPT, Qwen, DeepSeek) use BPE. Start from the 256 bytes. Count which two pieces sit next to each other most often, merge them into one piece, and repeat, round after round. So how many pieces a piece of text becomes depends on how common it is, not how hard it is. With Qwen's real table: "我爱你" (I love you) is 1 piece. "反向传播" (backpropagation) becomes 反 / 向 / 传播. 🦖 is 1 piece. But "龘", a very rare character, is cut into two half-characters. In my own BPE, the first merge produced half of "我", which shows up as �.

To keep my mini GPT simple, I still give each character its own number (9,175 characters).

## A setting that worked in English broke in Chinese

The dumbest language model only looks at the previous character. Count how often each pair of characters appears side by side, then turn the counts into probabilities. A pair that never appeared gets probability 0, and −log 0 is infinity. One such pair ruins the whole loss. So you smooth: add a little to every cell first.

I copied makemore's "add 1 to every cell", and it stopped working in Chinese. makemore has 27 characters. I had 5,157. The row for "我" had 337 real counts, but the 1s I added came to 5,157, more than the real counts. They washed out what the model had learned. Test loss was 7.28, not far from random guessing (8.55). Adding 0.01 instead brought it down to 6.33. You can't copy how much to add. You pick it by the test loss.

Later I replaced the count table with learned vectors: each character gets a list of numbers. I only asked it to guess the next character. I never told it which characters are alike. It put "春" (spring) and "秋" (autumn) next to each other on its own (similarity 0.86). Three and four, moon and night, too. They tend to be followed by similar characters. To guess well for both, the cheapest way is to give them similar numbers.

## Looking further back

Looking at only one character is too dumb. Joining the vectors of the last two characters and feeding them to a network gave a test loss of 6.29. But joining has two problems. How many characters it sees is fixed, and every extra character needs another block of knobs. And the same character in a different position is handled by a different block of knobs, so what it learned isn't shared.

Attention does it differently. Each character makes three things: a query (what I'm looking for), a key (what I am), and a value (what I hand over when picked). The character deciding takes its own query and dot-products it with the key of every character before it to get scores. Softmax turns the scores into shares that add up to 1. Then it adds up the values of those characters by their shares. However many characters came before, the output is always the same length. Characters to the right must be hidden: each position's answer is the character right after it, and seeing it would be cheating.

Attention over 8 characters, with one head, got a test loss of 6.17, 0.12 better than joining two characters. And it can't even tell order. Keys and values depend only on which character it is, and addition doesn't care about order. To it, "孤烟直" and "烟孤直" look the same.

To add order, give each position its own list of numbers (a position table) and add it to the character's vector: test loss 6.18 → 6.12, worth only 0.06. I also ran into a surprise: test loss got better, but training loss went up (5.64 → 5.78). I only ran it once, and I still can't say why.

Then I added the rest of the parts. Four heads: one head's shares add up to 1, so it can't take two places in full at once. A feedforward network in every layer. Residuals: add the original x back, which also gives the gradient a straight road back with slope 1. Layer norm: residuals keep adding, the numbers keep growing, and if they aren't pulled back, softmax gets too sharp and the gradient can't flow back. Four layers stacked: 6.12 → 5.96. But that changed several things at once, so I couldn't tell which one helped.

## Change one thing at a time

From here on I ran everything in my own Colab, and I decided what to change and what to compare. Test loss along the way:

| What changed | Test loss |
|---|---|
| Moved the previous model into Colab | 6.17 |
| Only changed the starting size of Wout (× 0.01) | 5.93 |
| Switched to the big dataset (3.64M characters, 13× more) | 6.09 |
| Rows per step 32 → 128 | 5.98 |
| Switched the optimizer to Adam | 5.59 |
| Look at 8 characters → 32 characters | 5.26 |

**One number worth 0.25.** The code I rewrote gave 6.17. It didn't match the earlier 5.96. The clue was at step 0: loss 9.015, higher than random guessing (ln 5157 ≈ 8.55). Worse than random means the right answer started with less than 1/5157 of the probability. The model was confidently wrong from the start. The cause was the last table, Wout, starting too large. The scores weren't all near 0, so the model randomly favored some characters. Changing that one number: 6.17 → 5.93. That's 4 times what adding positions was worth.

**More data, worse model.** I bet it would get better. It got 6.09. The test set is all early-Tang poems, and only 8% of the big dataset is from that batch. Each character was seen only 1.4 times on average, down from 18. My judgment then was "probably more than one cause." I made each step take 4× the rows, so each character was seen 5.6 times. It came back to 5.98, about two thirds of the way.

**Adam's "just divide".** When every knob shares one learning rate, the knobs with big slopes overshoot and the ones with small slopes barely move. The fix I found myself: move = learning rate × slope ÷ size of the slope. Every knob moves the same amount; only the direction is left. Real Adam also averages the last few slopes. If the direction is steady, it moves. If the slope keeps flipping, it barely moves. I bet it would help a little. It helped 0.40, the biggest step of the whole stage. By step 2,000 it matched what the old optimizer reached after the full 20,000.

**It couldn't see the last line.** At 5.59 the poems mixed five- and seven-character lines. A seven-character line plus punctuation is 8 characters, so looking at 8 characters, it couldn't see how long the previous line was. I changed it to look at 32 characters. I bet 5.3. It got 5.26, and the five-character lines lined up neatly: 「独酌风凉朝，归时近晚砧。水移双甲角，光逼瀑兰凉。」 The characters formed lines. The lines still didn't add up to meaning.

I never changed the test set. If I had, I couldn't tell whether the model got better or the test got easier.

## From continuing text to answering

My GPT only continues text. Real Qwen comes in two versions too. Ask the pretrained-only one "春天开什么花？" (What flowers bloom in spring?), and it continues with another question, then turns it into a web article.

The post-trained version, given the same sentence, still didn't answer. It turned it into a multiple-choice question. The reason: in post-training, every Q&A pair is wrapped in special tokens. After `<|im_start|>user` comes the question; after `<|im_start|>assistant` comes the answer. Without the tokens, it can't tell it's its turn to answer.

I post-trained the pretrained-only Qwen myself: 3,000 Chinese Q&A pairs, wrapped in the tokens. Training was exactly the same as pretraining: guess the next token, −log, Adam. Only the text changed. Starting loss was 4.67, far below random guessing (11.9), because it already knew Chinese. 400 steps, a few minutes, loss down to 1.5. All three test questions came back in the shape of an answer.

But it said "1 plus 1 equals 1". Nobody wrote it a rule like "when you see assistant, answer". In all 3,000 examples, an answer came after assistant. Guessing the next token got scored 3,000 times, and that's what it was scored into. The shape took a few minutes to learn. Whether what it knows is right comes from pretraining and model size. Post-training can't make up for it.

## Running it myself

Halfway through this stage I saw a problem. Claude was always the one explaining and running things on its side. I wasn't doing it myself. I was learning fast, but fast isn't a compliment, and what I learned wasn't that strong. So from the mini GPT on, the code ran in my own Colab, and I decided what to change and what to compare. The 6.17 that didn't match, and more data making it worse, both showed up on my own screen, and I tracked them down myself.

What matters is doing it yourself. Learning can't be done behind closed doors, just listening without doing. You have to try it and go through it yourself. When you hit a problem, don't back off. Get over it. I'd rather be stuck than not know.

Next: reproducing papers.

## Files

| File | What it is |
|---|---|
| `gpt.py` | The mini GPT: data, vocab, knobs, forward pass, batches + test loss, Adam training, writing poems (blocks 0–6) |
| `finetune.py` | Post-training the pretrained-only Qwen2.5-0.5B on 3,000 Chinese Q&A pairs |

```
pip install torch opencc-python-reimplemented
python gpt.py
```

Notes, so the numbers are honest:
- 5.26 came from running the original notebook cells on a Colab T4 GPU (20,000 steps). `gpt.py` is the same code put into one file, with no settings changed. I checked it for 400 steps on CPU: 9,175 characters, 638,592 knobs, starting loss 9.125 ≈ ln 9175. I did not re-run the full 20,000 steps, and a CPU gives different random numbers than a GPU, so expect something near 5.26, not exactly.
- `finetune.py` needs a GPU. I ran it on one RTX 4090 (AutoDL). It downloads the model and data from ModelScope.
- Every number was run once. I haven't measured how much they move between runs.

*Most of the code was written by Claude; at every step I worked it out by hand first, then read the code. Earlier projects: [backprop-from-scratch](https://github.com/Acezhang08/backprop-from-scratch), [mnist-handwritten-digits](https://github.com/Acezhang08/mnist-handwritten-digits).*

---

## 中文

### 「1 加 1 等于 1」：从零造一个中文 GPT

这一段我从「文字怎么变成数」开始，一路造到一个能写五言诗的中文迷你 GPT，最后拿真的 Qwen 做了一次后训练。下面是我在这一路上弄明白的东西，每个数都是自己跑出来、撞见过的。

#### 「猫」为什么是 3 个

电脑存文字，底下是一串 0～255 的数，叫字节。一个英文字母是 1 个字节（a 是 97），「猫」是 3 个（231、140、171）。

我第一眼看到就想：猫的拼音 mao 不正好 3 个字母吗？所以以为中文是先变成拼音字母，再变成数。其实不是：「我」的拼音 wo 只有 2 个字母，「张」的 zhang 有 5 个，可它们也都是 3 个字节。常用的中文字全是 3 个，跟读音没关系，只是这个字在全世界统一的编码表里的编号，拆成了 3 段。

真模型（GPT、Qwen、DeepSeek）用的是 BPE：从这 256 个字节起步，数一数哪两块最常挨在一起，就合成一块，一轮一轮合下去。所以合成几块，看的是常不常见，不看难不难。拿真 Qwen 的表一试：「我爱你」是 1 块；「反向传播」被切成 反 / 向 / 传播；🦖 是 1 块；「龘」却被切成两个半字。我自己写的 BPE，第一轮合出来的是「我」的半个字，显示出来是 �。

我的迷你 GPT 为了简单，还是一个字一个号（字表 9175 个字）。

#### 照抄别人的设置，换成中文就失效

最笨的语言模型只看前一个字：数一数语料里每一对字挨着出现几次，再变成概率。没出现过的一对，概率是 0，−log 0 是无穷大，一对就能把整个损失弄坏。所以要平滑：每格先加一点。

我照抄 makemore 的「每格加 1」，在中文上就不灵了。makemore 的字表只有 27 个字符，我的有 5157 个字。「我」那一行真数出来 337 次，可是加的 1 一共有 5157 个，比真的还多，把学到的冲没了。测试损失 7.28，离瞎猜的 8.55 不远。改成每格加 0.01，降到 6.33。加多少不能抄，要看测试损失挑。

后来把数表换成训练出来的向量，每个字一串数。我只让它猜下一个字，没告诉它谁跟谁像，它自己把「春」和「秋」放到了一起（相似度 0.86），三和四、月和夜也是。原因是它们后面常跟差不多的字，想都猜准，最省事的办法就是让它们的数长得像。

#### 看前面的字

只看前一个字太笨。把前两个字的向量接起来交给网络，测试 6.29。可是接起来有两个毛病：看几个字是写死的，多看一个就要多一截旋钮；同一个字换个位置，就归另一截旋钮管，学过的不共用。

注意力换了个做法：每个字变出三样东西，问（我在找什么）、招牌（我是什么）、内容（被看中后交出去的）。拿主意的字用自己的问，跟前面每个字的招牌做点积打分，softmax 变成一行加起来是 1 的比例，再按比例把各个字的内容加起来。不管前面几个字，出来都是同样长的一串数。右边的字要遮住，因为每个位置的答案就是它右边那个字，不遮就等于偷看答案。

8 个字的注意力，只有一个头，测试 6.17，比接起来看两个字好 0.12。它其实连顺序都分不出：招牌和内容只看是哪个字，加法又不管先后，「孤烟直」和「烟孤直」在它眼里一样。

补上顺序，给每个位置一串数（位置表），加到字的向量上：测试 6.18 → 6.12，只值 0.06。我还碰到了一个意外：测试变好了，训练损失却变高了（5.64 → 5.78）。只跑了一次，到现在也说不清为什么。

再把零件装全：分成 4 个头（一个头一行比例加起来是 1，想同时整份看两个地方做不到）、每层后面一个前馈网络、残差（把原来的 x 加回去，往回交坡时多一条坡是 1 的直路）、层归一化（残差越加数越大，要拉回来，不然 softmax 太尖、坡传不回），叠 4 层：6.12 → 5.96。可这一下改了好几样，分不出是哪一样起的作用。

#### 一次只改一样

接下来我在自己的 Colab 里跑，改哪一样、跟谁比我来定。一路的测试损失：

| 改了什么 | 测试 |
|---|---|
| 把上一步的模型搬进 Colab | 6.17 |
| 只改 Wout 一开始的大小（× 0.01） | 5.93 |
| 换成大数据（364 万字，多 13 倍） | 6.09 |
| 每圈拿的行数 32 → 128 | 5.98 |
| 挪法换成 Adam | 5.59 |
| 一次看 8 个字 → 32 个字 | 5.26 |

**一个数值 0.25。** 我重写的代码跑出 6.17，对不上之前的 5.96。线索在第 0 圈：损失 9.015，比瞎猜的 ln 5157 ≈ 8.55 还高。比瞎猜还差，说明正确答案一开始拿到的概率比 1/5157 还小，模型一开始就在很自信地猜错。查下去是最后那张表 Wout 起点取大了，分不全挨着 0，随机偏心了某些字。只改这一个数，6.17 → 5.93，比加位置值得多 4 倍。

**数据多了，反而变差。** 我押的是变好，结果 6.09。测试卷全是初唐的诗，大数据里只有 8% 是这一批；每个字平均只见了 1.4 遍，原来是 18 遍。我当时的判断是「多半不止一个原因」。把每圈的行数加到 4 倍，每个字见 5.6 遍，回到 5.98，补回了大约三分之二。

**Adam 的「除一下」。** 所有旋钮共用一个学习率，坡大的跨过头，坡小的挪不动。我自己摸到的办法是：挪多少 = 学习率 × 坡 ÷ 坡的大小，每个旋钮都挪一样大，只剩方向。真的 Adam 再看最近几圈坡的平均，方向稳就挪，来回晃就几乎不动。我押它只会好一点，结果好 0.40，是一路上最大的一步，第 2000 圈就到了旧挪法跑满 20000 圈的水平。

**看不到上一句。** 5.59 的诗五言七言混着写。七言一句加标点就 8 个字，只看 8 个字，它看不到上一句是几个字。改成看 32 个字，我押 5.3，实际 5.26，五言整整齐齐：「独酌风凉朝，归时近晚砧。水移双甲角，光逼瀑兰凉。」字连成了句子，句子还连不成意思。

测试卷从头到尾没换过。换了，就分不清是模型变好了，还是卷子变容易了。

#### 从接话到回答

我的 GPT 只会往下接。真的 Qwen 也分两个版本：只预训练过的，问它「春天开什么花？」，它接了另一个问题，再写成一篇网页文章。

后训练过的版本，喂同一句话，照样不回答，还出成了选择题。原因是后训练的每条问答都用几个特殊记号包着：`<|im_start|>user` 后面是问题，`<|im_start|>assistant` 后面是回答。不包上记号，它看不出该它答了。

我自己拿只预训练过的 Qwen 做了一次后训练：3000 条中文问答，包好记号，训练的做法跟预训练一模一样，还是猜下一个字、−log、Adam，只换了喂的文字。起点损失 4.67，远低于瞎猜的 11.9，因为它本来就会中文。400 圈、几分钟，损失降到 1.5，三个问题全都答成了回答的样子。

可它答「1 加 1 等于 1」。没有人给它写过「见到 assistant 就回答」这条规矩。3000 条数据里，assistant 后面每一条都是回答，猜下一个字被打分打了 3000 次，就被打成了这个样子。样子几分钟就学会了；知道的东西对不对，是预训练和模型大小定的，后训练补不上。

#### 自己去跑

这一段走到一半，我发现了一个问题：一直是 Claude 讲、Claude 在它那边跑，我没有自己体验去做。学得快，可快不是褒义词，学出来的不是那么强大。所以从训练迷你 GPT 开始，代码在我自己的 Colab 里跑，改哪一样、跟谁比我来定。上面 6.17 对不上、数据多了反而变差，都是在我自己屏幕上撞见、自己查出来的。

重要的是自己亲身去做。学习不能闭门造车、光听不做，要自己亲身去体验、去经历。碰见问题也不要退缩，要把这个槛跨过去。宁愿卡死，也不能不会。

下一步是复现论文。

#### 文件

- `gpt.py`：迷你 GPT（块 0～6：数据、字表、旋钮、往前走、拿一批 + 测试、Adam 训练、写诗）
- `finetune.py`：拿只预训练过的 Qwen2.5-0.5B，用 3000 条中文问答做后训练

说明：5.26 是在 Colab T4 上用原来那几格跑出来的（20000 圈）。`gpt.py` 是同一份代码拼成一个文件，设置没改，在 CPU 上核对过 400 圈（字表 9175、旋钮 638,592、起点 9.125），没有重新跑满 20000 圈；CPU 和显卡的随机数不同，结果会在 5.26 附近，不会一模一样。`finetune.py` 要显卡，我在一张 RTX 4090（AutoDL）上跑的。每个数都只跑了一次，没量过随机晃动多大。

*代码大部分由 Claude 写，每一步我先手算、再读代码。*
