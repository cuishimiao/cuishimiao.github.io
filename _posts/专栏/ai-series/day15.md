---
title: "你的文章标题"
date: 2024-03-20  # ← 必须包含日期字段
series: "AI学习之路"  # ← 系列标识符需完全一致
permalink: /专栏/ai-series/:title/  # ← 自定义URL结构
---
# Day 15:自定义Dataset类（处理非标准数据）



🍔 **自定义Dataset类：把乱炖数据变成AI能吃的大餐**
——当你不是厨神但被迫处理“垃圾食品”数据集时的生存指南

---

### Ⅰ. **数据现状：比衣柜还乱的文件堆**
**经典烂摊子场景**：
- 图片存在20个不同后缀的文件夹里（`.jpg`, `.jpeg`, `.png_临时`）
- 标签是个Excel，但第三列被同事改成了颜文字 (`好图→ (✧ω✧)`)
- 有些图片被家里的猫吃了（实际是误删但不敢承认）

**PyTorch官方Dataset**：
"亲，我们只接受标准套餐哦~"
**你**：
"不！我要把这些黑暗料理变成米其林！"

---

### Ⅱ. **Dataset类本质：AI界的食堂大妈**
#### 核心任务：
```python
class 你的Dataset(Dataset):
    def __init__(self):
        # 准备工作：洗菜、磨刀、深呼吸
    def __len__(self):
        # 告诉厨房还有多少食材
    def __getitem__(self, idx):
        # 现炒第idx道菜（别把锅烧穿了）
```
**灵魂比喻**：
- `__init__`：打开冰箱发现过期牛奶时的表情管理
- `__len__`：数清楚还有多少包泡面可以吃
- `__getitem__`：用泡面+老干妈做米其林三星料理

---

### Ⅲ. **实战：处理散装猫狗照片**
#### 场景设定：
- 图片路径： `数据集/喵/张三_2020.jpg` , `数据集/汪/李四_手机拍糊了.jpg`
- 要求：返回图片张量 + 标签(0猫/1狗)

#### 代码整活：
```python
from torch.utils.data import Dataset
import os
from PIL import Image

class 猫狗Dataset(Dataset):
    def __init__(self, 根目录, 变身器=None):
        self.所有图片 = []
        for 类别 in ["喵", "汪"]:
            文件夹 = os.path.join(根目录, 类别)
            for 文件 in os.listdir(文件夹):
                if 文件.endswith(("jpg", "jpeg", "png")):
                    self.所有图片.append( (os.path.join(文件夹, 文件), 类别) )
        self.变身器 = 变身器  # 比如Resize+ToTensor

    def __len__(self):
        return len(self.所有图片)

    def __getitem__(self, idx):
        路径, 类别 = self.所有图片[idx]
        图片 = Image.open(路径).convert("RGB")
      
        # 处理那些被猫啃过的图片
        try:
            if self.变身器:
                图片 = self.变身器(图片)
        except Exception as e:
            print(f"文件{路径}可能被猫吃了，错误：{e}")
            return None  # 或者随便返回一张空白图

        标签 = 0 if 类别 == "喵" else 1
        return 图片, 标签
```

**重点吐槽**：
1. `os.path.join`：防止在Mac/Linux/Windows之间反复横跳路径错误
2. `try-except`：防御你家猫对数据集的物理攻击
3. `变身器`：比如加上`transforms.Resize((256,256))` 把歪图掰正

---

### Ⅳ. **Debug小剧场：程序员 vs 数据集**
**Bug 1**：`FileNotFoundError`
- **你说**："代码明明没问题！"
- **数据集**："但你把路径写成`数据集/猫`了，我文件夹叫`数据集/喵`啊！"

**Bug 2**：`OSError: image file is truncated`
- **你说**："这图片怎么打不开？"
- **真相**：同事用手机传图时断网了，文件只有一半

**解决方案**：
```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 强行读取残血图片
```

---

### Ⅴ. **高级技巧：给Dataset加Buff**
#### 1. **缓存大法**：
```python
def __getitem__(self, idx):
    if idx in self.缓存:
        return self.缓存[idx]  # 避免重复读硬盘
    # ...正常处理...
    self.缓存[idx] = (图片, 标签)
```
**适合场景**：
- 数据读取比蜗牛还慢（比如要解压）
- 你的SSD已经在冒烟

#### 2. **动态增强**：
```python
# 在__getitem__里加入：
if random.random() > 0.5:
    图片 = 图片.transpose(Image.FLIP_LEFT_RIGHT)  # 随机镜像
```
**哲学意义**：
"让AI以为你的数据集有20倍大，其实都是PS的！"

#### 3. **懒加载优化**：
```python
def __init__(self):
    self.图片路径列表 = [...]  # 先只存路径，等用的时候再加载
```
**省内存效果**：
从"吃内存怪兽"变成"内存管理大师"

---

### Ⅵ. **Dataset的究极形态：万能瑞士军刀版**
**适用场景**：
- 数据分散在CSV、数据库、云存储
- 需要实时下载图片（比如爬虫现场）

```python
class 宇宙最强Dataset(Dataset):
    def __getitem__(self, idx):
        # 步骤1：从数据库读URL
        # 步骤2：爬虫下载图片（如果本地没有）
        # 步骤3：检测图中是否有NSFW内容
        # 步骤4：生成标签并返回
        # 如果中间出错：返回随机噪声+错误日志
        # 同时祈祷老板不要看日志
```

**程序员警告**：
"这代码跑起来，不是封IP就是被封号！"

---

### Ⅶ. **总结：自定义Dataset三步走**
1. **收集食材**：
   `__init__`里把分散的数据路径整理成列表
2. **告诉厨房有多少菜**：
   `__len__`返回正确的数量（别把空文件算进去！）
3. **现炒每道菜**：
   `__getitem__`里处理每个数据，记得防御式编程

**终极心法**：
"把Dataset类当成AI的数据火锅——什么乱七八糟的都能往里扔，但别忘了开火（DataLoader）才能煮熟！"

```python
# 测试你的Dataset
if __name__ == "__main__":
    dataset = 猫狗Dataset("数据集/")
    print(dataset[0])  # 应该输出(张量, 0)
    print(len(dataset))  # 输出总数量
    # 如果报错：
    # 1. 检查路径 2.检查文件权限 3.检查你的猫
```

🎉 **恭喜！** 现在你可以对凌乱数据说："拿来吧你！"