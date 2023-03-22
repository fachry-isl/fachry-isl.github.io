---
title: "Spotify Data Analysis"
classes: wide
layout: single
categories:
  - Portfolio
tags:
  - Portfolio
  - Data Analysis
  - Data Visualization
  - API
  - Tableau
excerpt: "Using online music platform such as Spotify generate data that we can analyze for learning purposes."
author_profile: true
# toc: true
# header:
#     image: /assets/images/fachry2.png
#     caption: "Photo credit: [Andrew Elliott](https://www.reddit.com/r/dataisbeautiful/comments/5l39mu/my_daughters_sleeping_patterns_for_the_first_4/)"
---

## Analyzing Personal Spotify Data
I like listening to music, almost every day I hear music especially when I'm studying or in my free time. Using online music platform such as Spotify generate data that we can analyze, for learning purposes. This project taught me about different kinds of R packages from processing and analyzing the data to requesting data through API, followed by advanced visualization using Tableau

### 1. Request the Data
Access your Spotify account dashboard at https://www.spotify.com/. In the privacy settings, you’ll find the option to request your data. This requires some patience. Spotify says it takes up to thirty days, but it’s usually much faster. In my case, I waited three days. Eventually, you will get an email with your Spotify data in a .zip file. Extract the MyData folder and copy it into your working folder.

### 2. Processing The Data for Visualization
After requesting the data, we can preprocess the data for visualization, in my case I do some join and API calls to get more data because the data isn't complete, there are more data and features that we can apply to our project. This depends on what we need there are so many different API endpoints that we could call. Here's my example code using R as a programming language [Link](https://github.com/fachry-isl/personal-spotify-data-visualization/blob/main/main.Rmd)

### 3. Visualizing the Data
Finally, the fun part, in this part I visualize my data using Tableau dashboard and customize the dashboard using Figma to make it beautiful. This is the final result. 
[Link](https://public.tableau.com/app/profile/fachry.ikhsal/viz/YearinRewind-MySpotifyActivity/SpotifyDashboard)
[![test](/assets/images/spotify_tableau_dashboard.jpg)](https://public.tableau.com/app/profile/fachry.ikhsal/viz/YearinRewind-MySpotifyActivity/SpotifyDashboard)

<div class='tableauPlaceholder' id='viz1679316703512' style='position: relative'>
   <noscript><a href='#'><img alt='Spotify Dashboard ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ye&#47;YearinRewind-MySpotifyActivity&#47;SpotifyDashboard&#47;1_rss.png' style='border: none' /></a></noscript>
   <object class='tableauViz'  style='display:none;'>
      <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
      <param name='embed_code_version' value='3' />
      <param name='site_root' value='' />
      <param name='name' value='YearinRewind-MySpotifyActivity&#47;SpotifyDashboard' />
      <param name='tabs' value='no' />
      <param name='toolbar' value='yes' />
      <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ye&#47;YearinRewind-MySpotifyActivity&#47;SpotifyDashboard&#47;1.png' />
      <param name='animate_transition' value='yes' />
      <param name='display_static_image' value='yes' />
      <param name='display_spinner' value='yes' />
      <param name='display_overlay' value='yes' />
      <param name='display_count' value='yes' />
      <param name='language' value='en-US' />
   </object>
</div>
<script type='text/javascript'>                    var divElement = document.getElementById('viz1679316703512');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='800px';vizElement.style.height='1600px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>