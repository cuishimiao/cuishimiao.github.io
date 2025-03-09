---
layout: page                  # 必须使用page布局
title: "AI修仙系列导航"
permalink: /专栏/ai-series/    # 注意结尾斜杠
nav_order: 2                   # 导航排序（可选）
---

{% raw %}<section class="series-index">
  <h1>{{ page.title }}</h1>
  
  <!-- 按时间倒序排列 -->
  {% assign sorted_posts = site.posts 
    | where: "series", "AI修仙专栏" 
    | sort: "date" | reverse 
  %}
  
  <!-- 带摘要的列表 -->
  <div class="post-list">
    {% for post in sorted_posts %}
      <article class="post-item">
        <header>
          <h2>
            <a href="{{ post.url | relative_url }}">
              {{ post.title }}
              <time>({{ post.date | date: "%Y.%m.%d" }})</time>
            </a>
          </h2>
          <div class="post-meta">
            {% if post.tags %}
              <span class="tags">🏷 {{ post.tags | join: " · " }}</span>
            {% endif %}
          </div>
        </header>
        
        {{ post.excerpt | markdownify }}
        <a href="{{ post.url | relative_url }}" class="read-more">继续阅读...</a>
      </article>
    {% endfor %}
  </div>
</section>{% endraw %}
