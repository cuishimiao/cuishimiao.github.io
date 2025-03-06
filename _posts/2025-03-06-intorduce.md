---
title: 如何用github搭建博客
tags: 
    - 开源
    - DevOps
    - 前端工程
---


### **一、选择技术栈**
#### **推荐组合：GitHub Pages + 静态网站生成器**
- **GitHub Pages**：免费托管静态网页，支持自定义域名，自动部署。
- **静态生成器**：
  - **Jekyll**（GitHub 原生支持，适合新手，主题丰富）
  - **Hugo**（速度快，适合技术文档）
  - **Hexo**（Node.js 驱动，插件生态强）

**建议选择 Jekyll**：无需构建环境，GitHub 直接渲染，适合快速搭建。


### **二、搭建步骤**
#### **1. 创建 GitHub 仓库**
   - 仓库名格式：`<你的GitHub用户名>.github.io`（例如 `zhangsan.github.io`）
   - 初始化时勾选 `Add a README file`。

#### **2. 选择并配置主题**
   - **推荐技术博客主题**：
     - [TeXt 主题](https://github.com/kitian616/jekyll-TeXt-theme)（支持文档、项目展示）
     - [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes)（高度可定制）
   - **简历专用主题**：
     - [Online CV](https://github.com/sharu725/online-cv)（响应式简历模板）
     - [Resume](https://github.com/jglovier/resume-template)

   **操作**：
   ```bash
   # 以 Jekyll 为例，克隆主题到本地
   git clone https://github.com/kitian616/jekyll-TeXt-theme.git
   cd jekyll-TeXt-theme
   ```

#### **3. 编写内容**
   - **技术博客**：在 `_posts` 目录下用 Markdown 写技术文章，文件名格式 `YYYY-MM-DD-title.md`。
   - **简历页面**：
     - 创建独立页面 `resume.md`，使用 HTML+CSS 或 Markdown 编写。
     - 示例结构：
       ```markdown
     
       layout: page
       title: 我的简历
    
       ## 个人信息
       - 姓名：张三
       - 邮箱：zhangsan@email.com
       - GitHub：[github.com/zhangsan](https://github.com/zhangsan)

       ## 技术栈
       - 语言：C++/Python/JavaScript
       - 框架：React/Node.js
       ```

#### **4. 配置导航和展示**
   - 修改 `_config.yml`，添加简历入口：
     ```yaml
     navigation:
       - title: 博客
         url: /
       - title: 简历
         url: /resume
     ```
   - 在首页添加项目展示区块（示例）：
     ```html
     <!-- 在 index.md 中 -->
     ## 我的项目
     - [项目1：分布式系统](https://github.com/zhangsan/project1)
     - [项目2：机器学习模型](https://github.com/zhangsan/project2)
     ```

#### **5. 部署到 GitHub**
   - 推送代码到仓库：
     ```bash
     git add .
     git commit -m "init blog"
     git push origin main
     ```
   - 访问 `https://<你的用户名>.github.io` 查看效果。



### **三、增强功能**
#### **1. 绑定自定义域名**
   - 在域名商处添加 `CNAME` 记录指向 `xxx.github.io`。
   - 在仓库根目录创建 `CNAME` 文件，写入域名（如 `blog.zhangsan.com`）。

#### **2. SEO 优化**
   - 在 `_config.yml` 中添加关键词和描述：
     ```yaml
     description: "张三的技术博客 | C++/Python开发工程师"
     keywords: "编程, 求职, 分布式系统"
     ```
   - 提交到 Google Search Console。

#### **3. 集成评论和统计**
   - **评论系统**：使用 [Disqus](https://disqus.com) 或 [Gitalk](https://github.com/gitalk/gitalk)（基于 GitHub Issue）。
   - **访问统计**：接入 [Google Analytics](https://analytics.google.com) 或 [Umami](https://umami.is)。



### **四、维护与求职应用**
#### **1. 持续更新**
   - 每周更新技术文章（如源码解析、项目总结）。
   - 在博客中展示 GitHub 项目链接和 Demo。

#### **2. 简历优化**
   - 在独立简历页面附加 **PDF 下载链接**。
   - 在博客导航栏高亮显示“简历”入口。


### **五、示例模板结构**
```
├── _config.yml          # 全局配置
├── _posts/              # 技术文章
│   └── 2024-05-20-cpp-concurrency.md
├── resume.md            # 简历页面
├── images/              # 图片资源
├── CNAME                # 自定义域名
└── _includes/           # 网页组件（导航栏、页脚）
```



### **六、高级技巧**
- **自动化部署**：通过 GitHub Actions 实现 CI/CD。
  ```yaml
  # .github/workflows/deploy.yml
  name: Deploy Blog
  on:
    push:
      branches: [main]
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/jekyll-build-pages@v1
  ```
- **暗黑模式**：在主题 CSS 中添加 `prefers-color-scheme` 适配。



通过以上步骤，你可以快速搭建一个兼具技术深度和求职竞争力的个人博客，所有内容通过 GitHub 免费托管，适合长期维护。