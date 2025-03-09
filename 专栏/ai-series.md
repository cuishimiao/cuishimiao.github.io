{% raw %}
---
layout: page
title: "🔥AI从入门到入土：我的21天修仙笔记"
permalink: /专栏/ai-series/
---

<div class="series-container">
  {% assign series_posts = site.posts 
    | where_exp: "post", "post.path contains '/专栏/ai-series/'" 
    | sort: "date" 
  %}

  {% for post in series_posts %}
    <article class="post-card">
      <header>
        <h2>
          <a href="{{ post.url | relative_url }}" 
             class="post-link"
             title="{{ post.title | escape }}">
            {{ post.title | escape }}
          </a>
        </h2>
        <time datetime="{{ post.date | date_to_xmlschema }}">
          {{ post.date | date: "%Y.%m.%d" }}
        </time>
      </header>

      <div class="excerpt">
        {{ post.excerpt | default: post.content 
           | strip_html 
           | truncate: 150, "..." }}
      </div>

      <div class="post-meta">
        <span class="reading-time">
          ⏱ {{ post.content | reading_time }}
        </span>
        <span class="word-count">
          📝 {{ post.content | number_of_words }}字
        </span>
      </div>
    </article>
  {% endfor %}
</div>

<style>
/* 专业级CSS样式 */
.series-container {
  max-width: 800px;
  margin: 2rem auto;
  padding: 0 20px;
}

.post-card {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 3px 6px rgba(0,0,0,0.1);
  padding: 2rem;
  margin-bottom: 2rem;
  transition: transform 0.2s;
}

.post-card:hover {
  transform: translateY(-3px);
}

.post-link {
  color: #2c3e50;
  text-decoration: none;
  border-bottom: 2px solid #3498db;
}

time {
  color: #7f8c8d;
  font-size: 0.9em;
}

.excerpt {
  color: #34495e;
  line-height: 1.6;
  margin: 1em 0;
}

.post-meta {
  display: flex;
  gap: 15px;
  font-size: 0.85em;
  color: #95a5a6;
}

.reading-time::before {
  content: "⏱ ";
}
</style>{% endraw %}
