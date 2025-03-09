
# 用PyTorch构建情感分析模型



🎭 **情感分析炼丹指南：用PyTorch熬制"情绪探测器"**
——从数据预处理到模型训练の魔法全流程

---

### Ⅰ. **任务背景：AI读心术速成班**
**业务需求**：
- 输入句子 → 输出情绪极性（正面/负面）
- 示例：
  "这手机牛逼！" → 😄 (正面)
  "客服态度像僵尸" → 😠 (负面)

**技术选型**：
```
方案 = 词向量(emoji化) + LSTM(情绪捕手) + 全连接(情感裁判)
```

**灵魂吐槽**：
"情感分析就像给AI装了个测谎仪，专治网上阴阳怪气！"

---

### Ⅱ. **数据预处理：制作情绪魔法药剂**
#### 1. 加载IMDB电影评论数据集（自带正负标签）
```python
from torchtext.datasets import IMDB

# 查看数据样例
train_data = IMDB(split='train')
print(next(iter(train_data)))  # 输出: ("This film is terrible...", 0)
```

#### 2. 文本分词与词表构建（使用spacy加速）
```python
import spacy
nlp = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [token.text for token in nlp(text)]

from torchtext.vocab import build_vocab_from_iterator

# 构建词表（自动过滤低频词）
vocab = build_vocab_from_iterator(
    map(tokenizer, [text for text, label in train_data]),
    min_freq=5,
    specials=['<unk>', '<pad>']
)
vocab.set_default_index(vocab['<unk>'])

print("词表大小:", len(vocab))  # 输出: 约25000
```

**魔法注释**：
- `<unk>`: 未知词占位符（遇到生僻词时的万能胶）
- `<pad>`: 填充符（像方便面里的脱水蔬菜，充数用的）

---

### Ⅲ. **数据管道：打造文本流水线**
#### 1. 文本向量化函数（文字→数字）
```python
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1 if x == 'pos' else 0

# 测试效果
print(text_pipeline("I love PyTorch!"))  # 输出: [23, 56, 345]
print(label_pipeline('pos'))             # 输出: 1
```

#### 2. 封装DataLoader（自动分批+填充）
```python
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    text_list, label_list = [], []
    for (_text, _label) in batch:
        text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(text)
        label_list.append(label_pipeline(_label))
  
    # 填充到相同长度（像整理不同高度的书架）
    padded_text = pad_sequence(text_list, padding_value=vocab['<pad>'])
    return padded_text.T, torch.tensor(label_list)

# 创建DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
```

**技术吐槽**：
"pad_sequence就像给不同身高的学生发增高鞋垫！"

---

### Ⅳ. **模型构建：组装情绪分析机甲**
```python
import torch.nn as nn

class EmotionDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)  # 双向LSTM需要*2
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x形状: [batch_size, seq_len]
        embedded = self.embedding(x)          # → [batch, seq, embed]
        lstm_out, _ = self.lstm(embedded)     # → [batch, seq, hidden*2]
        # 取最后时刻的输出（情绪累积结果）
        last_output = lstm_out[:, -1, :]      # → [batch, hidden*2]
        return self.fc(self.dropout(last_output))

# 初始化模型
model = EmotionDetector(len(vocab), 128, 256)
print(model)
```

**机甲部件解析**：
1. **Embedding层**：把单词变成128维向量（文字→数学坐标）
2. **Bi-LSTM**：双向扫描句子，捕捉前后语境（像同时用左右脑思考）
3. **全连接层**：把LSTM的输出压缩成1个概率值

---

### Ⅴ. **训练准备：配置炼丹炉参数**
```python
import torch.optim as optim

# 损失函数（二分类任务用BCEWithLogitsLoss更高效）
criterion = nn.BCEWithLogitsLoss()
# 优化器（带学习率衰减的Adam）
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 把模型扔到GPU（如果有的话）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**炼丹口诀**：
- BCELoss自带Sigmoid → 省去手动计算
- StepLR → 每5轮学习率打9折

---

### Ⅵ. **训练循环：启动情绪熔炉**
```python
for epoch in range(10):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.float().to(device)
      
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
      
        total_loss += loss.item()
  
    scheduler.step()
    print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# 示例输出：
# Epoch: 1, Loss: 0.5321
# Epoch: 2, Loss: 0.3216
# ...
# Epoch: 10, Loss: 0.1124
```

**避坑指南**：
- `clip_grad_norm_`：防止梯度爆炸（像给水管加压力阀）
- `squeeze()`：去掉多余的维度（把[[0.5], [0.7]] → [0.5, 0.7]）

---

### Ⅶ. **模型验证：AI情感大师考试**
```python
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts).squeeze()
        predicted = (torch.sigmoid(outputs) > 0.5).int()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'准确率: {100*correct/total:.2f}%')
# 输出: 准确率: 87.32% （经过调参可达更高）
```

**性能优化技巧**：
- 尝试Transformer模型（如BERT）
- 使用预训练词向量（GloVe）
- 增加注意力机制

---

🎯 **课后三问**：
1. 为什么要用双向LSTM？
   → 正向看句子的前半段，反向看后半段，像两人接力读论文

2. 如何处理长文本？
   → 截断到固定长度 或 使用Transformer（可并行处理长序列）

3. 遇到脏话干扰怎么办？
   → 数据清洗 或 在词向量中给脏话特殊标记

```python
# 彩蛋：实时情绪测试
def predict(text):
    with torch.no_grad():
        tokenized = torch.tensor(text_pipeline(text)).unsqueeze(0).to(device)
        output = model(tokenized)
        prob = torch.sigmoid(output).item()
        return "😄" if prob > 0.5 else "😠"

print(predict("This movie blew my mind!"))  # 输出: 😄
print(predict("Waste of time and money"))   # 输出: 😠
```

🔔 **玄学总结**：
"情感分析模型就是个数字时代的读心神探：
👉 词向量是它的心理学词典
👉 LSTM是它的微表情分析仪
👉 全连接层是它的最终直觉判断！"