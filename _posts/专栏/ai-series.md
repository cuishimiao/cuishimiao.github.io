{% raw %}<!-- /专栏/ai-series.md -->
---
layout: page
title: "AI修仙专栏"
permalink: /专栏/ai-series/
---

<ul class="post-list">
  {% assign series_posts = site.posts 
    | where: "series", "AI学习之路" 
    | sort: "date" 
  %}
  
  {% for post in series_posts %}
    <li>
      <a href="{{ post.url | relative_url }}" class="post-link">
        {{ post.title }} 
        <time>{{ post.date | date: "%Y.%m.%d" }}</time>
      </a>
    </li>
  {% endfor %}
</ul>{% endraw %}
