---  
layout: page  
title: "🔥AI从入门到入土：我的21天修仙笔记"  
permalink: /专栏/ai-series/  # 自定义访问路径  
---  

{% raw %}{% for post in site.posts %}  
  {% if post.series == "AI学习之路" %}  
    <div class="series-item">  
      <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>  
      <p>{{ post.excerpt }}</p>  
      <span class="date">{{ post.date | date: "%Y.%m.%d" }}</span>  
    </div>  
  {% endif %}  
{% endfor %}{% endraw %}  
