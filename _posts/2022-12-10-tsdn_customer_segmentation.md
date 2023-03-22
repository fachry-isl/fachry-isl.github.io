---
title: "TSDN Customer Segmentation"
classes: wide
layout: single
categories:
  - Portfolio
tags:
  - Portfolio
  - Machine Learning
  - Unsupervised Learning
  - Customer Segmentation
  - Competition
excerpt: "To commemorate Youth Pledge Day 2022, I joined National Data Science Competition 2022 and make it into final"
author_profile: true
---

## Project Report

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSfJl2xsCwgoBSlx4yKH1C231qOUuYmsXLBWeofqf1QGiyUdk9QJFX9rtXY0gp0sA/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

## Code

```python
import numpy as np
import pandas as pd
import scipy

#These are the visualization libraries. Matplotlib is standard and is what most people use.
#Seaborn works on top of matplotlib, as we mentioned in the course.
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
#For standardizing features. We'll use the StandardScaler module.
from sklearn.preprocessing import StandardScaler
#Hierarchical clustering with the Sci Py library. We'll use the dendrogram and linkage modules.
from scipy.cluster.hierarchy import dendrogram, linkage
#Sk learn is one of the most widely used libraries for machine learning. We'll use the k means and pca modules.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# We need to save the models, which we'll use in the next section. We'll use pickle for that.
import pickle
```

## Import Data


```python
# Load the data, contained in the segmentation data csv file.
df_segmentation = pd.read_csv('data/Mall_Customers.csv', index_col = 0)
```

## Explore Data


```python
df_segmentation.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_segmentation.shape
```




    (200, 4)




```python
# Encode gender to numeric
df_segmentation['Gender'] = df_segmentation['Gender'].map({'Male': 0, 'Female': 1})
```


```python
# Rename the columns
df_segmentation.rename({'Annual Income (k$)': 'Income',
                       'Spending Score (1-100)': 'Spending Score'}, axis=1, inplace=True,                       )
```


```python
df_segmentation.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.560000</td>
      <td>38.850000</td>
      <td>60.560000</td>
      <td>50.200000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.497633</td>
      <td>13.969007</td>
      <td>26.264721</td>
      <td>25.823522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>28.750000</td>
      <td>41.500000</td>
      <td>34.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>61.500000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>49.000000</td>
      <td>78.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>70.000000</td>
      <td>137.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Correlation Estimate


```python
# Compute Pearson correlation coefficient for the features in our data set.
# The correlation method in pandas, it has the Pearson correlation set as default.
df_segmentation.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gender</th>
      <td>1.000000</td>
      <td>-0.060867</td>
      <td>-0.056410</td>
      <td>0.058109</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.060867</td>
      <td>1.000000</td>
      <td>-0.012398</td>
      <td>-0.327227</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>-0.056410</td>
      <td>-0.012398</td>
      <td>1.000000</td>
      <td>0.009903</td>
    </tr>
    <tr>
      <th>Spending Score</th>
      <td>0.058109</td>
      <td>-0.327227</td>
      <td>0.009903</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We'll plot the correlations using a Heat Map. Heat Maps are a great way to visualize correlations using color coding.
# We use RdBu as a color scheme, but you can use viridis, Blues, YlGnBu or many others.
# We set the range from -1 to 1, as it is the range of the Pearson Correlation. 
# Otherwise the function infers the boundaries from the input.
plt.figure(figsize = (12, 9))
s = sns.heatmap(df_segmentation.corr(),
               annot = True, 
               cmap = 'RdBu',
               vmin = -1, 
               vmax = 1)
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](/assets/images/tsdn2022_files/output_11_0.png)
    


## Visualize Raw Data


```python
# We'll plot the data. We create a 12 by 9 inches figure.
# We have 2000 data points, which we'll scatter acrros Age and Income, located on positions 2 and 4 in our data set. 
plt.figure(figsize = (12, 9))
plt.scatter(df_segmentation.iloc[:,1], df_segmentation.iloc[:,2])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Visualization of raw data')
```




    Text(0.5, 1.0, 'Visualization of raw data')




    
![png](/assets/images/tsdn2022_files/output_13_1.png)
    


## Standarization


```python
# Standardizing data, so that all features have equal weight. This is important for modelling.
# Otherwise, in our case Income would be considered much more important than Education for Instance. 
# We do not know if this is the case, so we would not like to introduce it to our model. 
# This is what is also refered to as bias.
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)
```

## Hierarchical Clustering


```python
# Perform Hierarchical Clustering. The results are returned as a linkage matrix. 
hier_clust = linkage(segmentation_std, method = 'ward')
```


```python
# We plot the results from the Hierarchical Clustering using a Dendrogram. 
# We truncate the dendrogram for better readability. The level p shows only the last p merged clusters
# We also omit showing the labels for each point.
plt.figure(figsize = (12,9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust,
           truncate_mode = 'level', 
           p = 5, 
           show_leaf_counts = False, 
           no_labels = True)
plt.show()
```


    
![png](/assets/images/tsdn2022_files/output_18_0.png)
    


## K-means Clustering


```python
# Perform K-means clustering. We consider 1 to 10 clusters, so our for loop runs 10 iterations.
# In addition we run the algortihm at many different starting points - k means plus plus. 
# And we set a random state for reproducibility.
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)
```

    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    


```python
# Plot the Within Cluster Sum of Squares for the different number of clusters.
# From this plot we choose the number of clusters. 
# We look for a kink in the graphic, after which the descent of wcss isn't as pronounced.
plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering')
plt.show()
```


    
![png](/assets/images/tsdn2022_files/output_21_0.png)
    



```python
# We run K-means with a fixed number of clusters. In our case 4.
kmeans = KMeans(n_clusters =4, init = 'k-means++', random_state = 42)
```


```python
# We divide our data into the four clusters.
kmeans.fit(segmentation_std)
```

    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=4, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=4, random_state=42)</pre></div></div></div></div></div>



## Results


```python
# We create a new data frame with the original features and add a new column with the assigned clusters for each point.
df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment K-means'] = kmeans.labels_
```


```python
df_segm_kmeans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>Segment K-means</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate mean values for the clusters
df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()
df_segm_analysis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
    </tr>
    <tr>
      <th>Segment K-means</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>49.437500</td>
      <td>62.416667</td>
      <td>29.208333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>28.438596</td>
      <td>59.666667</td>
      <td>67.684211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>48.109091</td>
      <td>58.818182</td>
      <td>34.781818</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>28.250000</td>
      <td>62.000000</td>
      <td>71.675000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compute the size and proportions of the four clusters
df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment K-means','Gender']].groupby(['Segment K-means']).count()
df_segm_analysis['Prop Obs'] = df_segm_analysis['N Obs'] / df_segm_analysis['N Obs'].sum()
```


```python
df_segm_analysis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>N Obs</th>
      <th>Prop Obs</th>
    </tr>
    <tr>
      <th>Segment K-means</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>49.437500</td>
      <td>62.416667</td>
      <td>29.208333</td>
      <td>48</td>
      <td>0.240</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>28.438596</td>
      <td>59.666667</td>
      <td>67.684211</td>
      <td>57</td>
      <td>0.285</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>48.109091</td>
      <td>58.818182</td>
      <td>34.781818</td>
      <td>55</td>
      <td>0.275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>28.250000</td>
      <td>62.000000</td>
      <td>71.675000</td>
      <td>40</td>
      <td>0.200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add the segment labels to our table
df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'wise customer', 
                                                                  1:'standard customer',
                                                                  2:'rare customer', 
                                                                  3:'active customer'})
```


```python
# We plot the results from the K-means algorithm. 
# Each point in our data set is plotted with the color of the clusters it has been assigned to.
x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm'])
plt.title('Segmentation K-means')
plt.show()
```


    
![png](/assets/images/tsdn2022_files/output_31_0.png)
    



```python
# We plot the results from the K-means algorithm. 
# Each point in our data set is plotted with the color of the clusters it has been assigned to.
x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (5, 4))
sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm', 'y'], legend=False)
plt.title('Segmentation K-means')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\2911316098.py:6: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm', 'y'], legend=False)
    


    
![png](/assets/images/tsdn2022_files/output_32_1.png)
    



```python
# We plot the results from the K-means algorithm. 
# Each point in our data set is plotted with the color of the clusters it has been assigned to.
x_axis = df_segm_kmeans['Spending Score']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm', 'y'])
plt.title('Segmentation K-means')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\1279089453.py:6: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm', 'y'])
    


    
![png](/assets/images/tsdn2022_files/output_33_1.png)
    



```python
# We plot the results from the K-means algorithm. 
# Each point in our data set is plotted with the color of the clusters it has been assigned to.
x_axis = df_segm_kmeans['Gender']
y_axis = df_segm_kmeans['Spending Score']
plt.figure(figsize = (10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm', 'y'])
plt.title('Segmentation K-means')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\2421465087.py:6: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm', 'y'])
    


    
![png](/assets/images/tsdn2022_files/output_34_1.png)
    


### PCA


```python
# Employ PCA to find a subset of components, which explain the variance in the data.
pca = PCA()
```


```python
# Fit PCA with our standardized data.
pca.fit(segmentation_std)
```




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA()</pre></div></div></div></div></div>




```python
# The attribute shows how much variance is explained by each of the seven individual components.
pca.explained_variance_ratio_
```




    array([0.33690046, 0.26230645, 0.23260639, 0.16818671])




```python
pca.explained_variance_ratio_.cumsum()
```




    array([0.33690046, 0.5992069 , 0.83181329, 1.        ])




```python
# Plot the cumulative variance explained by total number of components.
# On this graph we choose the subset of components we want to keep. 
# Generally, we want to keep around 80 % of the explained variance.
plt.figure(figsize = (12,9))
plt.plot(range(1,5), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
```




    Text(0, 0.5, 'Cumulative Explained Variance')




    
![png](/assets/images/tsdn2022_files/output_40_1.png)
    



```python
# We choose three components. 3 or 4 seems the right choice according to the previous graph.
pca = PCA(n_components = 3)
```


```python
#Fit the model the our data with the selected number of components. In our case three.
pca.fit(segmentation_std)
```




<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(n_components=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=3)</pre></div></div></div></div></div>



### PCA Results


```python
# Here we discucss the results from the PCA.
# The components attribute shows the loadings of each component on each of the seven original features.
# The loadings are the correlations between the components and the original features. 
pca.components_
```




    array([[-0.23430156,  0.68790025, -0.00608217, -0.68691996],
           [-0.62688553, -0.10368955,  0.7652519 ,  0.10321115],
           [ 0.74300906,  0.12238438,  0.64366712, -0.13657317]])




```python
df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = df_segmentation.columns.values,
                           index = ['Component 1', 'Component 2', 'Component 3'])
df_pca_comp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Component 1</th>
      <td>-0.234302</td>
      <td>0.687900</td>
      <td>-0.006082</td>
      <td>-0.686920</td>
    </tr>
    <tr>
      <th>Component 2</th>
      <td>-0.626886</td>
      <td>-0.103690</td>
      <td>0.765252</td>
      <td>0.103211</td>
    </tr>
    <tr>
      <th>Component 3</th>
      <td>0.743009</td>
      <td>0.122384</td>
      <td>0.643667</td>
      <td>-0.136573</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Heat Map for Principal Components against original features. Again we use the RdBu color scheme and set borders to -1 and 1.
sns.heatmap(df_pca_comp,
            vmin = -1, 
            vmax = 1,
            cmap = 'RdBu',
            annot = True)
plt.yticks([0, 1, 2], 
           ['Component 1', 'Component 2', 'Component 3'],
           rotation = 45,
           fontsize = 9)
```




    ([<matplotlib.axis.YTick at 0x1fd12327df0>,
      <matplotlib.axis.YTick at 0x1fd0fac4fa0>,
      <matplotlib.axis.YTick at 0x1fd12265e20>],
     [Text(0, 0, 'Component 1'),
      Text(0, 1, 'Component 2'),
      Text(0, 2, 'Component 3')])




    
![png](/assets/images/tsdn2022_files/output_46_1.png)
    



```python
pca.transform(segmentation_std)
```




    array([[-4.06382715e-01, -5.20713635e-01, -2.07252663e+00],
           [-1.42767287e+00, -3.67310199e-01, -2.27764369e+00],
           [ 5.07605706e-02, -1.89406774e+00, -3.67375232e-01],
           [-1.69451310e+00, -1.63190805e+00, -7.17466691e-01],
           [-3.13108383e-01, -1.81048272e+00, -4.26459924e-01],
           [-1.71744627e+00, -1.59926418e+00, -6.96379423e-01],
           [ 7.90821124e-01, -1.94727112e+00, -1.86491593e-01],
           [-2.14832159e+00, -1.50537369e+00, -7.58463596e-01],
           [ 2.77428623e+00, -8.82987672e-01, -1.38814112e+00],
           [-1.21629477e+00, -1.61640464e+00, -5.55769702e-01],
           [ 2.62905084e+00, -8.61237043e-01, -1.42011358e+00],
           [-1.68947038e+00, -1.54542784e+00, -6.55007647e-01],
           [ 1.68582253e+00, -2.02394479e+00,  1.69391673e-02],
           [-1.64607339e+00, -1.52251259e+00, -6.10409943e-01],
           [ 1.17443628e+00, -6.12790961e-01, -1.65373684e+00],
           [-1.32613070e+00, -2.36719149e-01, -2.13541457e+00],
           [ 1.67728253e-02, -1.74344572e+00, -2.66543893e-01],
           [-1.07842454e+00, -2.44715641e-01, -2.05948662e+00],
           [ 1.48758780e+00, -5.72676028e-01, -1.53311653e+00],
           [-1.66373169e+00, -1.43259774e+00, -5.51432037e-01],
           [ 4.88090311e-01, -3.92921145e-01, -1.68967309e+00],
           [-1.01895051e+00, -1.66247511e-01, -1.97897968e+00],
           [ 1.35891492e+00, -1.82866936e+00,  8.74035749e-02],
           [-7.22972722e-01, -1.81687017e-01, -1.90171254e+00],
           [ 1.51315931e+00, -1.76451196e+00,  1.83655917e-01],
           [-1.06241157e+00, -4.31150614e-02, -1.89329142e+00],
           [ 5.88832908e-01, -1.62541614e+00,  9.17212023e-03],
           [-2.06188227e-01, -1.71906970e-01, -1.72925103e+00],
           [ 3.68426440e-01, -1.56300645e+00, -4.87309804e-03],
           [-1.96420414e+00, -1.21211990e+00, -4.51097239e-01],
           [ 2.54759194e+00, -5.27913522e-01, -1.15832312e+00],
           [-1.68983067e+00, -1.22412321e+00, -3.69867308e-01],
           [ 2.20131721e+00, -3.88195647e-01, -1.14609975e+00],
           [-1.87329695e+00,  2.24855270e-01, -1.92008355e+00],
           [ 1.26515693e+00, -1.58125854e+00,  2.62582342e-01],
           [-1.90386557e+00, -1.10444099e+00, -3.38577955e-01],
           [ 8.39344590e-01, -1.48793867e+00,  2.09762947e-01],
           [-1.24644437e+00, -1.17425940e+00, -1.92545586e-01],
           [ 3.02432444e-01, -1.31960089e+00,  1.83051586e-01],
           [-1.79415868e+00, -1.00420352e+00, -2.17275550e-01],
           [ 1.49387594e+00, -1.47013252e+00,  4.14612670e-01],
           [-1.57824777e+00,  3.26252676e-01, -1.74454279e+00],
           [ 1.09972892e+00, -4.75148304e-02, -1.21226839e+00],
           [-8.78229151e-01, -1.08373653e+00,  2.70337008e-03],
           [ 8.90421740e-01, -1.34990772e+00,  3.35765014e-01],
           [-1.33047664e+00, -1.01561907e+00, -7.99864154e-02],
           [ 2.19540681e-01, -1.21995587e+00,  2.25962989e-01],
           [-7.02592333e-01, -1.08085693e+00,  6.63670582e-02],
           [-4.70519161e-01, -1.11577393e+00,  1.10443219e-01],
           [-3.71782513e-01, -1.13065684e+00,  1.28009462e-01],
           [ 2.49709965e-01, -1.16611641e+00,  2.82222631e-01],
           [-2.81507713e-01,  2.47898317e-01, -1.39755760e+00],
           [-6.92486630e-01, -9.94947343e-01,  1.38090881e-01],
           [ 1.00183656e+00,  8.36296961e-02, -1.14462803e+00],
           [ 4.85517270e-01, -1.17239636e+00,  3.52688043e-01],
           [ 9.16095461e-01,  9.67976267e-02, -1.14928780e+00],
           [ 4.01316916e-01, -1.13059450e+00,  3.59529655e-01],
           [ 1.86862991e+00, -1.76710825e-02, -9.58000639e-01],
           [-8.10654470e-01, -8.89574289e-01,  1.92569564e-01],
           [ 1.07827242e+00,  1.59810666e-01, -1.04939377e+00],
           [ 1.65086088e+00,  7.33740960e-02, -9.53100538e-01],
           [-8.40256337e-01,  4.48881521e-01, -1.39573776e+00],
           [ 1.13717903e+00, -1.15401647e+00,  5.63160854e-01],
           [ 3.08719686e-01, -1.02922982e+00,  4.11866388e-01],
           [ 1.43815483e+00,  1.63848653e-01, -9.38935659e-01],
           [-9.96758186e-01,  5.30768704e-01, -1.37659200e+00],
           [ 5.44171232e-03, -9.54225943e-01,  3.87538308e-01],
           [ 1.29298442e+00, -1.14827597e+00,  6.17720316e-01],
           [-9.47389862e-01,  5.23327249e-01, -1.36780888e+00],
           [-4.57607937e-01, -8.84390383e-01,  3.06829920e-01],
           [ 1.67683173e+00,  1.56994962e-01, -8.74093335e-01],
           [ 4.16021294e-01, -9.86837080e-01,  4.89655069e-01],
           [ 8.70906219e-01, -1.02631905e+00,  5.91290173e-01],
           [ 6.84235085e-01, -9.98271327e-01,  5.54176289e-01],
           [ 1.34595784e+00,  3.52842574e-01, -8.05449772e-01],
           [-4.69867989e-01,  6.26458338e-01, -1.13240667e+00],
           [ 2.27835275e-02, -7.81833032e-01,  5.36609041e-01],
           [ 3.81292376e-01,  4.98237051e-01, -9.77631067e-01],
           [-1.03665230e+00, -6.22127824e-01,  3.48682347e-01],
           [ 5.13597177e-01, -8.55673851e-01,  6.30063345e-01],
           [ 1.14055197e+00,  3.83752756e-01, -8.44223949e-01],
           [ 9.58845944e-02,  5.41167687e-01, -1.03231119e+00],
           [ 1.90090826e+00,  2.69270021e-01, -7.03372898e-01],
           [ 3.12157595e-01, -8.25335848e-01,  5.93110013e-01],
           [-1.26872547e+00, -5.87210823e-01,  3.04606187e-01],
           [ 8.29573578e-01,  4.30691769e-01, -8.96762127e-01],
           [ 3.82433784e-01, -7.48585816e-01,  6.71635560e-01],
           [-1.16671899e+00, -5.15038230e-01,  3.97698495e-01],
           [-7.07867782e-01, -5.55092382e-01,  5.01154445e-01],
           [ 4.55367669e-01, -7.30251123e-01,  7.15912160e-01],
           [ 1.10375960e+00, -7.98926738e-01,  8.50858905e-01],
           [-5.19300383e-01,  7.79947493e-01, -1.01090383e+00],
           [ 7.48178745e-01,  6.17967591e-01, -7.65257638e-01],
           [ 1.21223952e-01, -6.21459016e-01,  7.09029657e-01],
           [-3.27057250e-01, -5.53913735e-01,  6.28160716e-01],
           [-5.16662945e-01,  8.08582982e-01, -9.91958508e-01],
           [ 2.80131086e-01, -6.45501482e-01,  7.33397624e-01],
           [-7.87237308e-01, -4.84651914e-01,  5.41829241e-01],
           [ 9.34617726e-01,  6.19129095e-01, -7.03575347e-01],
           [-6.34366480e-01,  8.55537580e-01, -9.86616638e-01],
           [-7.45169167e-01, -4.32528995e-01,  6.03551418e-01],
           [ 3.51736123e-01, -5.97959120e-01,  7.94798697e-01],
           [ 1.41903955e+00,  5.75066566e-01, -6.02261347e-01],
           [-4.98392518e-01,  8.64138980e-01, -9.41161402e-01],
           [ 6.10411629e-01,  6.96992316e-01, -7.44451587e-01],
           [-8.70573120e-01, -4.13639266e-01,  5.80683191e-01],
           [ 1.13743087e+00, -6.87241003e-01,  9.58076206e-01],
           [ 1.12369414e+00,  6.48926087e-01, -6.22947738e-01],
           [ 1.89485259e+00,  5.32725254e-01, -4.84078084e-01],
           [ 1.66277942e+00,  5.67642254e-01, -5.28154244e-01],
           [ 1.50674188e+00,  5.91110981e-01, -5.58145300e-01],
           [-1.28954958e+00, -3.21465312e-01,  5.24061553e-01],
           [-3.17759179e-02, -4.81725557e-01,  7.79133073e-01],
           [-6.04429349e-01,  9.38586267e-01, -9.05788589e-01],
           [-1.17937838e+00, -2.79646307e-01,  5.96227146e-01],
           [-1.18334467e+00, -2.79074127e-01,  5.94406300e-01],
           [ 1.17553272e+00, -6.34545905e-01,  1.01797754e+00],
           [ 5.76993109e-02, -4.66256439e-01,  8.10182100e-01],
           [ 5.82648530e-01, -4.86829980e-01,  9.61716891e-01],
           [ 1.59937938e-01, -4.23293075e-01,  8.78706001e-01],
           [-4.76852264e-01,  1.00675048e+00, -8.14838230e-01],
           [ 2.08622326e-02, -4.02111508e-01,  8.63442260e-01],
           [-3.60876912e-01, -2.86453243e-01,  8.34709615e-01],
           [-8.18252351e-01,  1.11611009e+00, -8.45873378e-01],
           [-4.27018733e-01, -2.46936984e-01,  8.63722473e-01],
           [-1.31210277e+00, -1.14141375e-01,  6.79492241e-01],
           [ 8.72125709e-01,  9.20380930e-01, -4.64693005e-01],
           [-8.76017553e-01,  1.18311437e+00, -8.09161376e-01],
           [ 2.30203421e+00,  7.05154010e-01, -1.96915456e-01],
           [-4.41408104e-01,  1.11786092e+00, -7.20687951e-01],
           [ 1.76294893e+00,  7.86437844e-01, -2.91708949e-01],
           [-3.92039780e-01,  1.11041946e+00, -7.11904829e-01],
           [-4.62082916e-01, -1.83367350e-01,  9.03915612e-01],
           [-1.15256325e+00, -7.97638257e-02,  7.60440955e-01],
           [ 5.36209096e-01,  1.02974833e+00, -4.58508487e-01],
           [-1.70487623e+00,  3.24442165e-02,  6.77309399e-01],
           [ 1.19570032e+00, -4.03729862e-01,  1.23851688e+00],
           [-6.84747745e-01,  1.21291448e+00, -7.13645901e-01],
           [ 3.53272095e-01,  1.08643311e+00, -4.69233119e-01],
           [-9.82221565e-01, -4.71043747e-02,  8.39408271e-01],
           [ 1.89035883e+00, -4.50063966e-01,  1.41243825e+00],
           [-1.21855815e+00,  1.35146929e+00, -7.70548756e-01],
           [-4.74910385e-01, -6.48138980e-02,  9.96726703e-01],
           [-1.53080042e+00,  9.37407158e-02,  7.82665968e-01],
           [ 5.95450970e-01,  1.13742569e+00, -3.53433136e-01],
           [-1.52316497e+00,  1.45568084e+00, -7.77752363e-01],
           [ 1.09090711e+00,  1.06243584e+00, -2.78668940e-01],
           [-1.18435761e+00,  7.08613124e-02,  8.76160159e-01],
           [ 3.00846740e-01, -1.23166897e-01,  1.19399795e+00],
           [-1.04051604e+00,  1.41219361e+00, -6.63371343e-01],
           [ 1.35051212e+00,  1.05272281e+00, -1.97278457e-01],
           [-7.40339814e-01,  1.36697269e+00, -6.08851768e-01],
           [ 8.47864590e-01, -2.05595090e-01,  1.29243313e+00],
           [-9.41714425e-01,  6.34354417e-02,  9.42823329e-01],
           [ 1.10263878e+00, -2.43946728e-01,  1.33999043e+00],
           [-1.83144095e+00,  1.97380086e-01,  7.77283206e-01],
           [ 1.48097906e+00,  1.03326246e+00, -1.65145452e-01],
           [-1.38999563e+00,  1.30980723e-01,  8.61954389e-01],
           [ 1.33287409e+00,  1.05558682e+00, -1.91494817e-01],
           [-1.25665910e+00,  1.10946634e-01,  8.88464306e-01],
           [ 1.04004275e+00, -2.05581064e-01,  1.34286925e+00],
           [-1.57293263e+00,  1.87665496e-01,  8.51229757e-01],
           [ 4.84983547e-01,  1.27086361e+00, -2.70744355e-01],
           [-1.74133333e+00,  2.71269219e-01,  8.64912979e-01],
           [ 1.45445958e+00,  1.24115857e+00, -1.15356094e-02],
           [-1.01540884e+00,  2.78776132e-01,  1.10253792e+00],
           [ 1.21928466e+00,  1.30585854e+00, -2.54202755e-02],
           [-1.69709206e+00,  4.10446083e-01,  9.94717288e-01],
           [ 2.64157486e-01,  1.44867334e-01,  1.40616993e+00],
           [-4.21324839e-01,  1.58177549e+00, -3.16668373e-01],
           [ 1.30698699e+00,  1.32190295e+00,  1.86957719e-02],
           [-9.38805793e-01,  1.65962313e+00, -4.15424661e-01],
           [ 1.18951561e+00,  1.33964832e+00, -5.30764442e-04],
           [-9.97203383e-01,  1.66820738e+00, -4.35293407e-01],
           [ 1.42716078e+00, -1.08217936e-03,  1.64549606e+00],
           [-1.60565560e+00,  4.55127547e-01,  1.06522259e+00],
           [ 2.14205006e+00,  1.22517962e+00,  1.90756402e-01],
           [-8.28402441e-01,  1.67223290e+00, -3.67827476e-01],
           [ 2.21692493e+00,  1.35977748e+00,  3.27683540e-01],
           [-9.94630016e-01,  1.84289057e+00, -2.86062122e-01],
           [ 1.77867756e-01,  4.49552248e-01,  1.63412720e+00],
           [-1.50900833e+00,  7.03127688e-01,  1.30390449e+00],
           [ 1.54730865e+00,  1.60656936e+00,  3.31043008e-01],
           [-1.71068006e+00,  7.62674919e-01,  1.29151957e+00],
           [ 1.88205612e-01,  5.06252606e-01,  1.68128262e+00],
           [-1.42953569e+00,  2.08340094e+00, -2.19681174e-01],
           [ 1.22953909e+00,  4.07829870e-01,  1.92412977e+00],
           [-7.55384802e-01,  2.04050460e+00, -3.43530844e-02],
           [ 7.73957706e-01,  5.34939526e-01,  1.89619988e+00],
           [-1.28626064e+00,  8.44610416e-01,  1.49174940e+00],
           [ 2.68375609e-01,  6.11070624e-01,  1.80290613e+00],
           [-1.05705706e+00,  8.10267154e-01,  1.54144865e+00],
           [ 1.08870927e+00,  2.11339899e+00,  6.22502410e-01],
           [-1.34984935e+00,  1.14586069e+00,  1.72318781e+00],
           [ 1.09288835e+00,  9.82840852e-01,  2.37186351e+00],
           [-1.17957174e+00,  1.32456784e+00,  1.93244109e+00],
           [ 6.72751128e-01,  1.22106095e+00,  2.43808390e+00],
           [-7.23719161e-01,  2.76501038e+00,  5.83177668e-01],
           [ 7.67096226e-01,  2.86193009e+00,  1.15034121e+00],
           [-1.06501524e+00,  3.13725616e+00,  7.88146046e-01]])




```python
scores_pca = pca.transform(segmentation_std)
```

### K-means clustering with PCA


```python
# We fit K means using the transformed data from the PCA.
wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)
```

    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    


```python
# Plot the Within Cluster Sum of Squares for the K-means PCA model. Here we make a decission about the number of clusters.
# Again it looks like four is the best option.
plt.figure(figsize = (10,8))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')
plt.show()
```


    
![png](/assets/images/tsdn2022_files/output_51_0.png)
    



```python
# We have chosen four clusters, so we run K-means with number of clusters equals four. 
# Same initializer and random state as before.
kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
```


```python
# We fit our data with the k-means pca model
kmeans_pca.fit(scores_pca)
```

    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    




<style>#sk-container-id-8 {color: black;background-color: white;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=4, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=4, random_state=42)</pre></div></div></div></div></div>



### K-means clustering with PCA Results


```python
scores_pca
```




    array([[-4.06382715e-01, -5.20713635e-01, -2.07252663e+00],
           [-1.42767287e+00, -3.67310199e-01, -2.27764369e+00],
           [ 5.07605706e-02, -1.89406774e+00, -3.67375232e-01],
           [-1.69451310e+00, -1.63190805e+00, -7.17466691e-01],
           [-3.13108383e-01, -1.81048272e+00, -4.26459924e-01],
           [-1.71744627e+00, -1.59926418e+00, -6.96379423e-01],
           [ 7.90821124e-01, -1.94727112e+00, -1.86491593e-01],
           [-2.14832159e+00, -1.50537369e+00, -7.58463596e-01],
           [ 2.77428623e+00, -8.82987672e-01, -1.38814112e+00],
           [-1.21629477e+00, -1.61640464e+00, -5.55769702e-01],
           [ 2.62905084e+00, -8.61237043e-01, -1.42011358e+00],
           [-1.68947038e+00, -1.54542784e+00, -6.55007647e-01],
           [ 1.68582253e+00, -2.02394479e+00,  1.69391673e-02],
           [-1.64607339e+00, -1.52251259e+00, -6.10409943e-01],
           [ 1.17443628e+00, -6.12790961e-01, -1.65373684e+00],
           [-1.32613070e+00, -2.36719149e-01, -2.13541457e+00],
           [ 1.67728253e-02, -1.74344572e+00, -2.66543893e-01],
           [-1.07842454e+00, -2.44715641e-01, -2.05948662e+00],
           [ 1.48758780e+00, -5.72676028e-01, -1.53311653e+00],
           [-1.66373169e+00, -1.43259774e+00, -5.51432037e-01],
           [ 4.88090311e-01, -3.92921145e-01, -1.68967309e+00],
           [-1.01895051e+00, -1.66247511e-01, -1.97897968e+00],
           [ 1.35891492e+00, -1.82866936e+00,  8.74035749e-02],
           [-7.22972722e-01, -1.81687017e-01, -1.90171254e+00],
           [ 1.51315931e+00, -1.76451196e+00,  1.83655917e-01],
           [-1.06241157e+00, -4.31150614e-02, -1.89329142e+00],
           [ 5.88832908e-01, -1.62541614e+00,  9.17212023e-03],
           [-2.06188227e-01, -1.71906970e-01, -1.72925103e+00],
           [ 3.68426440e-01, -1.56300645e+00, -4.87309804e-03],
           [-1.96420414e+00, -1.21211990e+00, -4.51097239e-01],
           [ 2.54759194e+00, -5.27913522e-01, -1.15832312e+00],
           [-1.68983067e+00, -1.22412321e+00, -3.69867308e-01],
           [ 2.20131721e+00, -3.88195647e-01, -1.14609975e+00],
           [-1.87329695e+00,  2.24855270e-01, -1.92008355e+00],
           [ 1.26515693e+00, -1.58125854e+00,  2.62582342e-01],
           [-1.90386557e+00, -1.10444099e+00, -3.38577955e-01],
           [ 8.39344590e-01, -1.48793867e+00,  2.09762947e-01],
           [-1.24644437e+00, -1.17425940e+00, -1.92545586e-01],
           [ 3.02432444e-01, -1.31960089e+00,  1.83051586e-01],
           [-1.79415868e+00, -1.00420352e+00, -2.17275550e-01],
           [ 1.49387594e+00, -1.47013252e+00,  4.14612670e-01],
           [-1.57824777e+00,  3.26252676e-01, -1.74454279e+00],
           [ 1.09972892e+00, -4.75148304e-02, -1.21226839e+00],
           [-8.78229151e-01, -1.08373653e+00,  2.70337008e-03],
           [ 8.90421740e-01, -1.34990772e+00,  3.35765014e-01],
           [-1.33047664e+00, -1.01561907e+00, -7.99864154e-02],
           [ 2.19540681e-01, -1.21995587e+00,  2.25962989e-01],
           [-7.02592333e-01, -1.08085693e+00,  6.63670582e-02],
           [-4.70519161e-01, -1.11577393e+00,  1.10443219e-01],
           [-3.71782513e-01, -1.13065684e+00,  1.28009462e-01],
           [ 2.49709965e-01, -1.16611641e+00,  2.82222631e-01],
           [-2.81507713e-01,  2.47898317e-01, -1.39755760e+00],
           [-6.92486630e-01, -9.94947343e-01,  1.38090881e-01],
           [ 1.00183656e+00,  8.36296961e-02, -1.14462803e+00],
           [ 4.85517270e-01, -1.17239636e+00,  3.52688043e-01],
           [ 9.16095461e-01,  9.67976267e-02, -1.14928780e+00],
           [ 4.01316916e-01, -1.13059450e+00,  3.59529655e-01],
           [ 1.86862991e+00, -1.76710825e-02, -9.58000639e-01],
           [-8.10654470e-01, -8.89574289e-01,  1.92569564e-01],
           [ 1.07827242e+00,  1.59810666e-01, -1.04939377e+00],
           [ 1.65086088e+00,  7.33740960e-02, -9.53100538e-01],
           [-8.40256337e-01,  4.48881521e-01, -1.39573776e+00],
           [ 1.13717903e+00, -1.15401647e+00,  5.63160854e-01],
           [ 3.08719686e-01, -1.02922982e+00,  4.11866388e-01],
           [ 1.43815483e+00,  1.63848653e-01, -9.38935659e-01],
           [-9.96758186e-01,  5.30768704e-01, -1.37659200e+00],
           [ 5.44171232e-03, -9.54225943e-01,  3.87538308e-01],
           [ 1.29298442e+00, -1.14827597e+00,  6.17720316e-01],
           [-9.47389862e-01,  5.23327249e-01, -1.36780888e+00],
           [-4.57607937e-01, -8.84390383e-01,  3.06829920e-01],
           [ 1.67683173e+00,  1.56994962e-01, -8.74093335e-01],
           [ 4.16021294e-01, -9.86837080e-01,  4.89655069e-01],
           [ 8.70906219e-01, -1.02631905e+00,  5.91290173e-01],
           [ 6.84235085e-01, -9.98271327e-01,  5.54176289e-01],
           [ 1.34595784e+00,  3.52842574e-01, -8.05449772e-01],
           [-4.69867989e-01,  6.26458338e-01, -1.13240667e+00],
           [ 2.27835275e-02, -7.81833032e-01,  5.36609041e-01],
           [ 3.81292376e-01,  4.98237051e-01, -9.77631067e-01],
           [-1.03665230e+00, -6.22127824e-01,  3.48682347e-01],
           [ 5.13597177e-01, -8.55673851e-01,  6.30063345e-01],
           [ 1.14055197e+00,  3.83752756e-01, -8.44223949e-01],
           [ 9.58845944e-02,  5.41167687e-01, -1.03231119e+00],
           [ 1.90090826e+00,  2.69270021e-01, -7.03372898e-01],
           [ 3.12157595e-01, -8.25335848e-01,  5.93110013e-01],
           [-1.26872547e+00, -5.87210823e-01,  3.04606187e-01],
           [ 8.29573578e-01,  4.30691769e-01, -8.96762127e-01],
           [ 3.82433784e-01, -7.48585816e-01,  6.71635560e-01],
           [-1.16671899e+00, -5.15038230e-01,  3.97698495e-01],
           [-7.07867782e-01, -5.55092382e-01,  5.01154445e-01],
           [ 4.55367669e-01, -7.30251123e-01,  7.15912160e-01],
           [ 1.10375960e+00, -7.98926738e-01,  8.50858905e-01],
           [-5.19300383e-01,  7.79947493e-01, -1.01090383e+00],
           [ 7.48178745e-01,  6.17967591e-01, -7.65257638e-01],
           [ 1.21223952e-01, -6.21459016e-01,  7.09029657e-01],
           [-3.27057250e-01, -5.53913735e-01,  6.28160716e-01],
           [-5.16662945e-01,  8.08582982e-01, -9.91958508e-01],
           [ 2.80131086e-01, -6.45501482e-01,  7.33397624e-01],
           [-7.87237308e-01, -4.84651914e-01,  5.41829241e-01],
           [ 9.34617726e-01,  6.19129095e-01, -7.03575347e-01],
           [-6.34366480e-01,  8.55537580e-01, -9.86616638e-01],
           [-7.45169167e-01, -4.32528995e-01,  6.03551418e-01],
           [ 3.51736123e-01, -5.97959120e-01,  7.94798697e-01],
           [ 1.41903955e+00,  5.75066566e-01, -6.02261347e-01],
           [-4.98392518e-01,  8.64138980e-01, -9.41161402e-01],
           [ 6.10411629e-01,  6.96992316e-01, -7.44451587e-01],
           [-8.70573120e-01, -4.13639266e-01,  5.80683191e-01],
           [ 1.13743087e+00, -6.87241003e-01,  9.58076206e-01],
           [ 1.12369414e+00,  6.48926087e-01, -6.22947738e-01],
           [ 1.89485259e+00,  5.32725254e-01, -4.84078084e-01],
           [ 1.66277942e+00,  5.67642254e-01, -5.28154244e-01],
           [ 1.50674188e+00,  5.91110981e-01, -5.58145300e-01],
           [-1.28954958e+00, -3.21465312e-01,  5.24061553e-01],
           [-3.17759179e-02, -4.81725557e-01,  7.79133073e-01],
           [-6.04429349e-01,  9.38586267e-01, -9.05788589e-01],
           [-1.17937838e+00, -2.79646307e-01,  5.96227146e-01],
           [-1.18334467e+00, -2.79074127e-01,  5.94406300e-01],
           [ 1.17553272e+00, -6.34545905e-01,  1.01797754e+00],
           [ 5.76993109e-02, -4.66256439e-01,  8.10182100e-01],
           [ 5.82648530e-01, -4.86829980e-01,  9.61716891e-01],
           [ 1.59937938e-01, -4.23293075e-01,  8.78706001e-01],
           [-4.76852264e-01,  1.00675048e+00, -8.14838230e-01],
           [ 2.08622326e-02, -4.02111508e-01,  8.63442260e-01],
           [-3.60876912e-01, -2.86453243e-01,  8.34709615e-01],
           [-8.18252351e-01,  1.11611009e+00, -8.45873378e-01],
           [-4.27018733e-01, -2.46936984e-01,  8.63722473e-01],
           [-1.31210277e+00, -1.14141375e-01,  6.79492241e-01],
           [ 8.72125709e-01,  9.20380930e-01, -4.64693005e-01],
           [-8.76017553e-01,  1.18311437e+00, -8.09161376e-01],
           [ 2.30203421e+00,  7.05154010e-01, -1.96915456e-01],
           [-4.41408104e-01,  1.11786092e+00, -7.20687951e-01],
           [ 1.76294893e+00,  7.86437844e-01, -2.91708949e-01],
           [-3.92039780e-01,  1.11041946e+00, -7.11904829e-01],
           [-4.62082916e-01, -1.83367350e-01,  9.03915612e-01],
           [-1.15256325e+00, -7.97638257e-02,  7.60440955e-01],
           [ 5.36209096e-01,  1.02974833e+00, -4.58508487e-01],
           [-1.70487623e+00,  3.24442165e-02,  6.77309399e-01],
           [ 1.19570032e+00, -4.03729862e-01,  1.23851688e+00],
           [-6.84747745e-01,  1.21291448e+00, -7.13645901e-01],
           [ 3.53272095e-01,  1.08643311e+00, -4.69233119e-01],
           [-9.82221565e-01, -4.71043747e-02,  8.39408271e-01],
           [ 1.89035883e+00, -4.50063966e-01,  1.41243825e+00],
           [-1.21855815e+00,  1.35146929e+00, -7.70548756e-01],
           [-4.74910385e-01, -6.48138980e-02,  9.96726703e-01],
           [-1.53080042e+00,  9.37407158e-02,  7.82665968e-01],
           [ 5.95450970e-01,  1.13742569e+00, -3.53433136e-01],
           [-1.52316497e+00,  1.45568084e+00, -7.77752363e-01],
           [ 1.09090711e+00,  1.06243584e+00, -2.78668940e-01],
           [-1.18435761e+00,  7.08613124e-02,  8.76160159e-01],
           [ 3.00846740e-01, -1.23166897e-01,  1.19399795e+00],
           [-1.04051604e+00,  1.41219361e+00, -6.63371343e-01],
           [ 1.35051212e+00,  1.05272281e+00, -1.97278457e-01],
           [-7.40339814e-01,  1.36697269e+00, -6.08851768e-01],
           [ 8.47864590e-01, -2.05595090e-01,  1.29243313e+00],
           [-9.41714425e-01,  6.34354417e-02,  9.42823329e-01],
           [ 1.10263878e+00, -2.43946728e-01,  1.33999043e+00],
           [-1.83144095e+00,  1.97380086e-01,  7.77283206e-01],
           [ 1.48097906e+00,  1.03326246e+00, -1.65145452e-01],
           [-1.38999563e+00,  1.30980723e-01,  8.61954389e-01],
           [ 1.33287409e+00,  1.05558682e+00, -1.91494817e-01],
           [-1.25665910e+00,  1.10946634e-01,  8.88464306e-01],
           [ 1.04004275e+00, -2.05581064e-01,  1.34286925e+00],
           [-1.57293263e+00,  1.87665496e-01,  8.51229757e-01],
           [ 4.84983547e-01,  1.27086361e+00, -2.70744355e-01],
           [-1.74133333e+00,  2.71269219e-01,  8.64912979e-01],
           [ 1.45445958e+00,  1.24115857e+00, -1.15356094e-02],
           [-1.01540884e+00,  2.78776132e-01,  1.10253792e+00],
           [ 1.21928466e+00,  1.30585854e+00, -2.54202755e-02],
           [-1.69709206e+00,  4.10446083e-01,  9.94717288e-01],
           [ 2.64157486e-01,  1.44867334e-01,  1.40616993e+00],
           [-4.21324839e-01,  1.58177549e+00, -3.16668373e-01],
           [ 1.30698699e+00,  1.32190295e+00,  1.86957719e-02],
           [-9.38805793e-01,  1.65962313e+00, -4.15424661e-01],
           [ 1.18951561e+00,  1.33964832e+00, -5.30764442e-04],
           [-9.97203383e-01,  1.66820738e+00, -4.35293407e-01],
           [ 1.42716078e+00, -1.08217936e-03,  1.64549606e+00],
           [-1.60565560e+00,  4.55127547e-01,  1.06522259e+00],
           [ 2.14205006e+00,  1.22517962e+00,  1.90756402e-01],
           [-8.28402441e-01,  1.67223290e+00, -3.67827476e-01],
           [ 2.21692493e+00,  1.35977748e+00,  3.27683540e-01],
           [-9.94630016e-01,  1.84289057e+00, -2.86062122e-01],
           [ 1.77867756e-01,  4.49552248e-01,  1.63412720e+00],
           [-1.50900833e+00,  7.03127688e-01,  1.30390449e+00],
           [ 1.54730865e+00,  1.60656936e+00,  3.31043008e-01],
           [-1.71068006e+00,  7.62674919e-01,  1.29151957e+00],
           [ 1.88205612e-01,  5.06252606e-01,  1.68128262e+00],
           [-1.42953569e+00,  2.08340094e+00, -2.19681174e-01],
           [ 1.22953909e+00,  4.07829870e-01,  1.92412977e+00],
           [-7.55384802e-01,  2.04050460e+00, -3.43530844e-02],
           [ 7.73957706e-01,  5.34939526e-01,  1.89619988e+00],
           [-1.28626064e+00,  8.44610416e-01,  1.49174940e+00],
           [ 2.68375609e-01,  6.11070624e-01,  1.80290613e+00],
           [-1.05705706e+00,  8.10267154e-01,  1.54144865e+00],
           [ 1.08870927e+00,  2.11339899e+00,  6.22502410e-01],
           [-1.34984935e+00,  1.14586069e+00,  1.72318781e+00],
           [ 1.09288835e+00,  9.82840852e-01,  2.37186351e+00],
           [-1.17957174e+00,  1.32456784e+00,  1.93244109e+00],
           [ 6.72751128e-01,  1.22106095e+00,  2.43808390e+00],
           [-7.23719161e-01,  2.76501038e+00,  5.83177668e-01],
           [ 7.67096226e-01,  2.86193009e+00,  1.15034121e+00],
           [-1.06501524e+00,  3.13725616e+00,  7.88146046e-01]])




```python
df_segmentation.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We create a new data frame with the original features and add the PCA scores and assigned clusters.
df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
# The last column we add contains the pca k-means clustering labels.
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
```


```python
df_segm_pca_kmeans
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
      <th>Segment K-means PCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
      <td>-0.406383</td>
      <td>-0.520714</td>
      <td>-2.072527</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
      <td>-1.427673</td>
      <td>-0.367310</td>
      <td>-2.277644</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
      <td>0.050761</td>
      <td>-1.894068</td>
      <td>-0.367375</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
      <td>-1.694513</td>
      <td>-1.631908</td>
      <td>-0.717467</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
      <td>-0.313108</td>
      <td>-1.810483</td>
      <td>-0.426460</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>1</td>
      <td>35</td>
      <td>120</td>
      <td>79</td>
      <td>-1.179572</td>
      <td>1.324568</td>
      <td>1.932441</td>
      <td>3</td>
    </tr>
    <tr>
      <th>196</th>
      <td>1</td>
      <td>45</td>
      <td>126</td>
      <td>28</td>
      <td>0.672751</td>
      <td>1.221061</td>
      <td>2.438084</td>
      <td>1</td>
    </tr>
    <tr>
      <th>197</th>
      <td>0</td>
      <td>32</td>
      <td>126</td>
      <td>74</td>
      <td>-0.723719</td>
      <td>2.765010</td>
      <td>0.583178</td>
      <td>2</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0</td>
      <td>32</td>
      <td>137</td>
      <td>18</td>
      <td>0.767096</td>
      <td>2.861930</td>
      <td>1.150341</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199</th>
      <td>0</td>
      <td>30</td>
      <td>137</td>
      <td>83</td>
      <td>-1.065015</td>
      <td>3.137256</td>
      <td>0.788146</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 8 columns</p>
</div>




```python
# We calculate the means by segments.
df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
    </tr>
    <tr>
      <th>Segment K-means PCA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>49.437500</td>
      <td>62.416667</td>
      <td>29.208333</td>
      <td>1.346375</td>
      <td>0.598558</td>
      <td>-0.588323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>47.803571</td>
      <td>58.071429</td>
      <td>34.875000</td>
      <td>0.643591</td>
      <td>-0.756396</td>
      <td>0.757360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>28.250000</td>
      <td>62.000000</td>
      <td>71.675000</td>
      <td>-0.831991</td>
      <td>0.914209</td>
      <td>-1.009810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>28.392857</td>
      <td>60.428571</td>
      <td>68.178571</td>
      <td>-1.203347</td>
      <td>-0.409660</td>
      <td>0.468210</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate the size of each cluster and its proportion to the entire data set.
df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Gender']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Prop Obs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'wise customer', 
                                                                  1:'standard customer',
                                                                  2:'rare customer', 
                                                                  3:'active customer'})
df_segm_pca_kmeans_freq
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
      <th>N Obs</th>
      <th>Prop Obs</th>
    </tr>
    <tr>
      <th>Segment K-means PCA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>wise customer</th>
      <td>0.0</td>
      <td>49.437500</td>
      <td>62.416667</td>
      <td>29.208333</td>
      <td>1.346375</td>
      <td>0.598558</td>
      <td>-0.588323</td>
      <td>48</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>standard customer</th>
      <td>1.0</td>
      <td>47.803571</td>
      <td>58.071429</td>
      <td>34.875000</td>
      <td>0.643591</td>
      <td>-0.756396</td>
      <td>0.757360</td>
      <td>56</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>rare customer</th>
      <td>0.0</td>
      <td>28.250000</td>
      <td>62.000000</td>
      <td>71.675000</td>
      <td>-0.831991</td>
      <td>0.914209</td>
      <td>-1.009810</td>
      <td>40</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>active customer</th>
      <td>1.0</td>
      <td>28.392857</td>
      <td>60.428571</td>
      <td>68.178571</td>
      <td>-1.203347</td>
      <td>-0.409660</td>
      <td>0.468210</td>
      <td>56</td>
      <td>0.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_segm_pca_kmeans_freq[['Gender', 'Age', 'Spending Score', 'N Obs']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Spending Score</th>
      <th>N Obs</th>
    </tr>
    <tr>
      <th>Segment K-means PCA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>wise customer</th>
      <td>0.0</td>
      <td>49.437500</td>
      <td>29.208333</td>
      <td>48</td>
    </tr>
    <tr>
      <th>standard customer</th>
      <td>1.0</td>
      <td>47.803571</td>
      <td>34.875000</td>
      <td>56</td>
    </tr>
    <tr>
      <th>rare customer</th>
      <td>0.0</td>
      <td>28.250000</td>
      <td>71.675000</td>
      <td>40</td>
    </tr>
    <tr>
      <th>active customer</th>
      <td>1.0</td>
      <td>28.392857</td>
      <td>68.178571</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'wise customer', 
                                                                  1:'standard customer',
                                                                  2:'rare customer', 
                                                                  3:'active customer'})
```


```python
df_segm_pca_kmeans['Legend'].value_counts(dropna=False)
```




    standard customer    56
    active customer      56
    wise customer        48
    rare customer        40
    Name: Legend, dtype: int64




```python
# Plot data by PCA components. The Y axis is the first component, X axis is the second.
x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'])
plt.title('Clusters by PCA Components')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\1547245716.py:5: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      sns.scatterplot(x=x_axis, y=y_axis, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'])
    


    
![png](/assets/images/tsdn2022_files/output_64_1.png)
    



```python
# Plot data by PCA components. The Y axis is the first component, X axis is the second.
x_axis_1 = df_segm_pca_kmeans['Component 3']
y_axis_1 = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (10, 8))
sns.scatterplot(x=x_axis_1, y=y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'])
plt.title('Clusters by PCA Components')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\2146061291.py:5: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      sns.scatterplot(x=x_axis_1, y=y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'])
    


    
![png](/assets/images/tsdn2022_files/output_65_1.png)
    



```python
# Plot data by PCA components. The Y axis is the first component, X axis is the second.
x_axis_1 = df_segm_pca_kmeans['Component 2']
y_axis_1 = df_segm_pca_kmeans['Component 3']
plt.figure(figsize = (10, 8))
ax = sns.scatterplot(x=x_axis_1, y=y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'])
plt.title('Clusters by PCA Components')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\3902733220.py:5: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      ax = sns.scatterplot(x=x_axis_1, y=y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'])
    


    
![png](/assets/images/tsdn2022_files/output_66_1.png)
    



```python
# Plot data by PCA components. The Y axis is the first component, X axis is the second.
x_axis_1 = df_segm_pca_kmeans['Component 3']
y_axis_1 = df_segm_pca_kmeans['Component 2']
# plt.figure(figsize = (5, 4))
sns.scatterplot(x=x_axis_1, y=y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'], legend=True)
plt.title('Clusters by PCA Components')
plt.show()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_30264\3084581144.py:5: UserWarning: The palette list has more values (5) than needed (4), which may not be intended.
      sns.scatterplot(x=x_axis_1, y=y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm', 'y'], legend=True)
    


    
![png]/assets/images/tsdn2022_files/(output_67_1.png)
    


### Predict with New Data


```python
df_segm_pca_kmeans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
      <th>Segment K-means PCA</th>
      <th>Legend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
      <td>-0.406383</td>
      <td>-0.520714</td>
      <td>-2.072527</td>
      <td>2</td>
      <td>rare customer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
      <td>-1.427673</td>
      <td>-0.367310</td>
      <td>-2.277644</td>
      <td>2</td>
      <td>rare customer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
      <td>0.050761</td>
      <td>-1.894068</td>
      <td>-0.367375</td>
      <td>1</td>
      <td>standard customer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
      <td>-1.694513</td>
      <td>-1.631908</td>
      <td>-0.717467</td>
      <td>3</td>
      <td>active customer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
      <td>-0.313108</td>
      <td>-1.810483</td>
      <td>-0.426460</td>
      <td>1</td>
      <td>standard customer</td>
    </tr>
  </tbody>
</table>
</div>



#### Create predict function


```python
def predict_segment(data):
    x_scaled = scaler.transform(data)
    x_pca = pca.transform(x_scaled)
    
    segm_kmeans_pcanew =  kmeans_pca.predict(x_pca)
    predict_values = pd.DataFrame({'predict_segment':segm_kmeans_pcanew})
    predict_values['Legend'] = predict_values['predict_segment'].map({0:'wise customer', 
                                                                  1:'standard customer',
                                                                  2:'rare customer', 
                                                                  3:'active customer'})
    return predict_values
```


```python
df_segm_pca_kmeans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Income</th>
      <th>Spending Score</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
      <th>Segment K-means PCA</th>
      <th>Legend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
      <td>-0.406383</td>
      <td>-0.520714</td>
      <td>-2.072527</td>
      <td>2</td>
      <td>rare customer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
      <td>-1.427673</td>
      <td>-0.367310</td>
      <td>-2.277644</td>
      <td>2</td>
      <td>rare customer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
      <td>0.050761</td>
      <td>-1.894068</td>
      <td>-0.367375</td>
      <td>1</td>
      <td>standard customer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
      <td>-1.694513</td>
      <td>-1.631908</td>
      <td>-0.717467</td>
      <td>3</td>
      <td>active customer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
      <td>-0.313108</td>
      <td>-1.810483</td>
      <td>-0.426460</td>
      <td>1</td>
      <td>standard customer</td>
    </tr>
  </tbody>
</table>
</div>




```python
predict_segment([[0, 19, 15, 39],
                [0, 19, 20, 13],
                [0, 19, 100, 15],
                [1,19, 100, 15],
                [1,19, 25, 15],
                [1,19, 40, 15],
                [0,20,15,39]])
```

    C:\Users\ASUS\anaconda3\envs\fachryds\lib\site-packages\sklearn\base.py:420: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
      warnings.warn(
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predict_segment</th>
      <th>Legend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>rare customer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>wise customer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>rare customer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>active customer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>standard customer</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>standard customer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>rare customer</td>
    </tr>
  </tbody>
</table>
</div>



### Export Model


```python
# We save the objects we'll need in the Purchase Analytics part of the course. We export them as pickle objects.
# We need the scaler, pca and kmeans_pca objects to preprocess and segment the purchase data set.
pickle.dump(scaler, open('saved_model2/scaler.pickle', 'wb'))
```


```python
pickle.dump(pca, open('saved_model2/pca.pickle', 'wb'))
```


```python
pickle.dump(kmeans_pca, open('saved_model2/kmeans_pca.pickle', 'wb'))
```

### Import Model


```python
# We load our pickled objects in order to segment the purchase data set.
scaler = pickle.load(open('saved_model2/scaler.pickle', 'rb'))
```


```python
pca = pickle.load(open('saved_model2/pca.pickle', 'rb'))
```


```python
kmeans_pca = pickle.load(open('saved_model2/kmeans_pca.pickle', 'rb'))
```
