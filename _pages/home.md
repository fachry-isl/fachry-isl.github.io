---
layout: archive
permalink: /
title: " "
author_profile: true
header:
    overlay_color: "#000"
    overlay_filter: "0.1"
    overlay_image: /assets/images/headerGlencoe.jpg
---
<h2>Recent Post<h2>
<!-- {% assign postsByYear = site.posts | group_by_exp:"post", "post.date | date: '%Y'"  %}
{% for year in postsByYear %}
  <h2 id="{{ year.name | slugify }}" class="archive__subtitle">{{ year.name }}</h2>
  {% for post in year.items %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %} -->

{% assign postsByYear = site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in postsByYear %}
  <h2 id="{{ year.name | slugify }}" class="archive__subtitle">{{ year.name }}</h2>
  {% for post in year.items %}
    {% include archive-single.html %}
    <div class="tags">
      {% for tag in post.tags %}
        <a class="page__taxonomy-item" rel="tag" style="font-size: 16px;">{{ tag }}</a>
      {% endfor %}
    </div>
  {% endfor %}
{% endfor %}


<link rel="icon" href="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/assets/images/evergreen.png?raw=true" type="image/x-icon" />