---
layout: page
title: "AI修仙之路"
permalink: /专栏/ai-series/
---

{% raw %}{% assign series_posts = site.posts 
  | where: "series", "AI学习之路" 
  | sort: "date" 
%}

<ul class="post-list">
  {% for post in series_posts %}
    <li>
      <h2>
        <a href="{{ post.url | relative_url }}">
          {{ post.title }} 
          <time>{{ post.date | date: "%Y.%m.%d" }}</time>
        </a>
      </h2>
      {{ post.excerpt | strip_html }}
    </li>
  {% endfor %}
</ul>{% endraw %}
