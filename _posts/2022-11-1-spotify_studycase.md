---
title: "Spotify Study Case"
categories:
  - Portfolio
tags:
  - Portfolio
  - Data Analysis
  - Data Visualization
  - API
  - Tableau
toc: true
---

## Analyzing Personal Spotify Data
I like listening to music, almost every day I hear music especially when I'm studying or in my free time. Using online music platform such as Spotify generate data that we can analyze, for learning purposes. This project taught me about different kinds of R packages from processing and analyzing the data to requesting data through API, followed by advanced visualization using Tableau

### 1. Request the Data
Access your Spotify account dashboard at https://www.spotify.com/. In the privacy settings, you’ll find the option to request your data. This requires some patience. Spotify says it takes up to thirty days, but it’s usually much faster. In my case, I waited three days. Eventually, you will get an email with your Spotify data in a .zip file. Extract the MyData folder and copy it into your working folder.

### 2. Processing The Data for Visualization
After requesting the data, we can preprocess the data for visualization, in my case I do some join and API calls to get more data because the data isn't complete, there are more data and features that we can apply to our project. This depends on what we need there are so many different API endpoints that we could call. Here's my example code using R as a programming language [Link](https://github.com/fachry-isl/personal-spotify-data-visualization/blob/main/main.Rmd)

### 3. Visualizing the Data
Finally, the fun part, in this part I visualize my data using Tableau dashboard and customize the dashboard using Figma to make it beautiful. This is the final result. [Link](https://public.tableau.com/app/profile/fachry.ikhsal/viz/YearinRewind-MySpotifyActivity/SpotifyDashboard)
[![test](/assets/images/spotify_tableau_dashboard.jpg)](https://public.tableau.com/app/profile/fachry.ikhsal/viz/YearinRewind-MySpotifyActivity/SpotifyDashboard)

