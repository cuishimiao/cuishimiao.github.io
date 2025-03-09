---  
layout: page  
title: "🔥AI从入门到入土：我的21天修仙笔记"  
permalink: /专栏/ai-series/  # 自定义访问路径  
---  

{% raw %}{% assign sorted_posts = site.posts | where: "series", "AI学习之路" | sort: "date" %}
{% for post in sorted_posts %}
  <div class="series-item">
    <h3><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h3>
    <p>{{ post.excerpt | strip_html | truncate: 150 }}</p>
    <span class="date">{{ post.date | date: "%Y.%m.%d" }}</span>
  </div>
{% endfor %}{% endraw %}

