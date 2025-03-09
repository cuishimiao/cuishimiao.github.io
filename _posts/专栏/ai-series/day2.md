---
# 正确示例 ✅
layout: post
title: "修仙Day1：神经网络入门"
date: 2024-03-20 14:30:00 +0800
series: "AI学习之路"  # 必须与配置一致
categories: AI修仙
---


# Day 2: 概率统计：贝叶斯定理、高斯分布（Python代码验证）


✨**概率世界の奇妙物语**✨ —— 用剧本杀和天气预报玩转概率（附🔍破案代码）

---

### Ⅰ. **贝叶斯定理：剧本杀破案指南**
**剧情设定**：豪华游轮发生钻石失窃案，已知：
- 船上20%乘客戴金丝眼镜（先验概率）
- 目击者称凶手戴金丝眼镜（似然度）
- 目击者在强光下可能看错（识别准确率90%）

**问题**：船长抓住一个戴金丝眼镜的乘客，他是凶手的概率多大？

```python
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    # 计算P(B) = P(B|A)P(A) + P(B|¬A)P(¬A)
    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
    # 贝叶斯公式
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b

# 输入参数
p_a = 0.2           # 戴眼镜乘客比例
p_b_given_a = 0.9   # 凶手戴眼镜且被正确识别的概率
p_b_given_not_a = 0.1 # 非凶手戴眼镜却被误判的概率

# 计算后验概率
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)
print(f"这个眼镜仔是真凶的概率：{result:.2%}")

# 输出：这个眼镜仔是真凶的概率：69.23%
```

**🕵️♂️破案启示**：
- 即使目击者90%可靠，真凶概率也只有69%
- 关键原因是戴眼镜的普通乘客太多（先验概率影响）
- 就像垃圾邮件过滤：即使包含"优惠"这个词，也要考虑这个词本身的常见程度

---

### Ⅱ. **高斯分布：天气预报の魔法**
**场景设定**：预测明天温度，气象台说服从高斯分布（μ=25℃, σ=3℃）

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成温度数据
mu = 25    # 平均温度
sigma = 3  # 标准差
temperatures = np.linspace(15, 35, 100)

# 计算概率密度
probabilities = norm.pdf(temperatures, mu, sigma)

# 可视化
plt.figure(figsize=(10,5))
plt.plot(temperatures, probabilities, 'r-', lw=3)
plt.fill_between(temperatures, probabilities, alpha=0.3)
plt.title('🌡️ 温度概率分布曲线', fontsize=14)
plt.xlabel('温度(℃)')
plt.ylabel('概率密度')
plt.grid(True)
plt.show()
```

**📊图形解读**：
- 钟形曲线最高点在25℃ → 最可能温度
- 68%概率落在22℃~28℃之间（μ±σ）
- 95%概率落在19℃~31℃之间（μ±2σ）

**🎲实战应用**：用鸢尾花数据集验证高斯分布
```python
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
petal_length = iris.data[:, 2]  # 花瓣长度

# 拟合高斯分布
mu_real = np.mean(petal_length)
sigma_real = np.std(petal_length)
print(f"真实分布参数：μ={mu_real:.2f}, σ={sigma_real:.2f}")

# 输出：真实分布参数：μ=3.76, σ=1.76
```

**🌺生物学发现**：不同品种鸢尾花的花瓣长度遵循不同高斯分布 → 这就是朴素贝叶斯分类的数学基础！

---

### 🎮**概率实验室**
**挑战任务**：用贝叶斯定理破解「三门问题」
```python
# 三门问题（Monty Hall Problem）模拟
import random

def monty_hall(switch_door, num_trials=10000):
    wins = 0
    for _ in range(num_trials):
        prize_door = random.randint(0, 2)
        first_choice = random.randint(0, 2)
        # 主持人打开一扇没有奖品的门
        open_door = next(i for i in range(3) 
                       if i != first_choice and i != prize_door)
        if switch_door:
            # 换到剩下的那扇门
            second_choice = next(i for i in range(3) 
                               if i != first_choice and i != open_door)
        else:
            second_choice = first_choice
        wins += (second_choice == prize_door)
    return wins / num_trials

print(f"不换门胜率：{monty_hall(False):.2%}")
print(f"换门后胜率：{monty_hall(True):.2%}")

# 输出示例：
# 不换门胜率：33.22%
# 换门后胜率：66.86%
```

**🤯反直觉结论**：换门策略将胜率提高一倍！贝叶斯定理告诉我们，新信息会改变概率判断。

---

💡**行业应用彩蛋**：
- 高斯分布：人脸识别中的特征分布建模
- 贝叶斯定理：医疗诊断系统（如COVID检测结果可靠性评估）
- 推荐系统：利用用户行为数据更新推荐概率