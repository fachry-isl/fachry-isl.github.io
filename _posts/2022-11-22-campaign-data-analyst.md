---
title: "Campaign Challenge Analysis"
layout: single
classes: wide
categories:
  - Portfolio
tags:
  - Portfolio
  - Data Analysis
  - Data Visualization
  - Internship
excerpt: "In november 2022, I got an experience to learn how to process Campaign's data and turn it into insight"
# toc: true
---

## Project Report

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vT7BL7WxNaSdbweXdKlF1J-Zfz1umOnRA2ZQx5cCM7qFLZHPSVCOp-45aqqI8VecA/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

## Data Processing

### Import Libraries


```python
# data processing
import pandas as pd
import numpy as np

# statistics
import scipy.stats as stats
from scipy.stats import levene, hypergeom

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### Read Data


```python
dfa_start = pd.read_csv('challenge_a_start.csv')
dfb_start = pd.read_csv('challenge_b_start.csv')

dfa_complete = pd.read_csv("challenge_a_completed.csv")
dfb_complete = pd.read_csv("challenge_b_completed.csv")
```


```python
dfa_start.columns
```




    Index(['No', 'Username', 'Name', 'Total Action', 'Start Date Challenge'], dtype='object')




```python
dfa_start.head()
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
      <th>No</th>
      <th>Username</th>
      <th>Name</th>
      <th>Total Action</th>
      <th>Start Date Challenge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>varaput</td>
      <td>Varissa Rania</td>
      <td>1</td>
      <td>08/26/2022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>enricosihotang</td>
      <td>enricosihotang</td>
      <td>1</td>
      <td>08/25/2022</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>anthonysatya</td>
      <td>anthonysatya</td>
      <td>5</td>
      <td>08/24/2022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>lestiarasepti</td>
      <td>Lestiara Septi</td>
      <td>2</td>
      <td>08/24/2022</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>rein.aldyps</td>
      <td>Reinaldy.P Sarbini</td>
      <td>0</td>
      <td>08/24/2022</td>
    </tr>
  </tbody>
</table>
</div>



### Data Understanding
There are 4 data that indicates:
1. When is the user start or finish the challenge
2. Hom many actions it takes to finish the challenge

#### There are two type of data 

Based on challenge type:
1. challenge a
2. challenge b

And, based on challenge completion
1. challenge start
2. challenge completed

### Identifying challenge conversion
Given the user data, we can sum how many user has join the challenge, and how many user has completed the challenge


```python
def check_conversion(df_start, df_complete):
    name_start = df_start['Username']
    name_complete = df_complete['Username']
    convert_dict = {}
    
    for start in name_start:
        for complete in name_complete:
            if start == complete:
                convert_dict[start] = 1
                break
            else:
                convert_dict[start] = 0
    return pd.DataFrame({'Username': convert_dict.keys(), 'Conversion': convert_dict.values()})
```


```python
dfa = check_conversion(dfa_start, dfa_complete)
```


```python
dfb = check_conversion(dfb_start, dfb_complete)
```


```python
data_a = dfa.Conversion.value_counts().reset_index(name='counts')
data_a
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
      <th>index</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_b = dfb.Conversion.value_counts().reset_index(name='counts')
```

#### Challenge Completion
It seems like challenge B has more completion rate than challenge A


```python
# total = data_a.loc[data_a['index'] == 0]['counts']
# total_complete = data_a.loc[data_a['index'] == 1]['counts']
data_a = pd.DataFrame({'data': 'Challenge A', 'total': [152], 'total_completed': [27]})
data_b = pd.DataFrame({'data': 'Challenge B', 'total': [91], 'total_completed': [367]})
data = data_a.append(data_b)
data
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_20328\2660046288.py:5: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      data = data_a.append(data_b)
    




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
      <th>data</th>
      <th>total</th>
      <th>total_completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Challenge A</td>
      <td>152</td>
      <td>27</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Challenge B</td>
      <td>91</td>
      <td>367</td>
    </tr>
  </tbody>
</table>
</div>



### Visualizaing Data
Now let's visualize the data so it easier to see the pattern


```python
data
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
      <th>data</th>
      <th>total</th>
      <th>total_completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Challenge A</td>
      <td>152</td>
      <td>27</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Challenge B</td>
      <td>91</td>
      <td>367</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots()

# Create stacked bar chart
sns.set_color_codes("muted")
sns.barplot(data=data, x='data', y='total_completed', color='b', label='Completed')
sns.set_color_codes("pastel")
sns.barplot(data=data, x='data', y='total', color='b', bottom=data['total_completed'], label='Not completed')

# Add bar labels
for i in ax.containers:
    ax.bar_label(i, label_type='center')

# Add a legend and informative axis label
ax.legend(ncol=1, loc="upper left", frameon=True)
ax.set_ylabel(None)
ax.set_xlabel(None)
sns.despine(left=True, bottom=True)
```


    
![png](/assets/images/campaign2022_files/output_17_0.png)
    


### Sum of Completed Actions Per Week


```python
data = dfa_complete.copy()
data.loc[data['Total Time (Day)'] <= 7, 'Total Time (Day)'] = 1
data.loc[(data['Total Time (Day)'] > 7) & (data['Total Time (Day)'] <= 14), 'Total Time (Day)'] = 2
data.loc[(data['Total Time (Day)'] > 14) & (data['Total Time (Day)'] <= 21), 'Total Time (Day)'] = 3
data.loc[ data['Total Time (Day)'] > 21, 'Total Time (Day)'] = 4
```


```python
data = data['Total Time (Day)'].value_counts().rename_axis('total weeks').reset_index(name='counts')

f, ax = plt.subplots()
xticks = ['Week 1', 'Week 2', 'Week 3', 'More than 3 Weeks']
x_pos = np.arange(len(xticks))
sns.set_color_codes("muted")
sns.barplot(data=data, x='total weeks', y='counts', color='b')

for i in ax.containers:
    ax.bar_label(i, )

ax.set_xticks(x_pos, labels=xticks)
ax.set_ylim(top=10)
ax.set_ylabel(None)
ax.set_xlabel(None)
plt.title("Sum of Completed Challenges Per Week for Challenge A", fontweight='heavy', ha='center', fontsize=14)
```




    Text(0.5, 1.0, 'Sum of Completed Challenges Per Week for Challenge A')




    
![png](/assets/images/campaign2022_files/output_20_1.png)
    



```python
data = dfb_complete.copy()
data.loc[data['Total Time (Day)'] <= 7, 'Total Time (Day)'] = 1
data.loc[(data['Total Time (Day)'] > 7) & (data['Total Time (Day)'] <= 14), 'Total Time (Day)'] = 2
data.loc[(data['Total Time (Day)'] > 14) & (data['Total Time (Day)'] <= 21), 'Total Time (Day)'] = 3
data.loc[ data['Total Time (Day)'] > 21, 'Total Time (Day)'] = 4
```


```python
data = data['Total Time (Day)'].value_counts().rename_axis('total weeks').reset_index(name='counts')

f, ax = plt.subplots()
xticks = ['Week 1', 'Week 2', 'Week 3', 'More than 3 Weeks']
x_pos = np.arange(len(xticks))
sns.set_color_codes("muted")
sns.barplot(data=data, x='total weeks', y='counts', color='b')

for i in ax.containers:
    ax.bar_label(i, )

ax.set_xticks(x_pos, labels=xticks)
ax.set_ylim(top=400)
ax.set_ylabel(None)
ax.set_xlabel(None)
plt.title("Sum of Completed Challenges Per Week for Challenge B", fontweight='heavy', ha='center', fontsize=14)
```




    Text(0.5, 1.0, 'Sum of Completed Challenges Per Week for Challenge B')




    
![png](/assets/images/campaign2022_files/output_22_1.png)
    


### Correlation


```python
dfa_start.corr()
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_20328\3064303806.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      dfa_start.corr()
    




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
      <th>No</th>
      <th>Total Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>1.000000</td>
      <td>0.376938</td>
    </tr>
    <tr>
      <th>Total Action</th>
      <td>0.376938</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = dfa_start['Total Action'].value_counts().rename_axis('total actions').reset_index(name='counts')

f, ax = plt.subplots()

sns.set_color_codes("muted")
sns.barplot(data=data, x='total actions', y='counts', color='b')

for i in ax.containers:
    ax.bar_label(i, )
    
ax.set_ylabel(None)
ax.set_xlabel(None)
plt.title("Sum of Total Actions for Challenge A", fontweight='heavy', ha='center', fontsize=14)
```




    Text(0.5, 1.0, 'Sum of Total Actions for Challenge A')




    
![png](/assets/images/campaign2022_files/output_25_1.png)
    



```python
data = dfb_start['Total Action'].value_counts().rename_axis('total actions').reset_index(name='counts')

f, ax = plt.subplots()

sns.set_color_codes("muted")
sns.barplot(data=data, x='total actions', y='counts', color='b')

for i in ax.containers:
    ax.bar_label(i, )
    
ax.set_ylabel(None)
ax.set_xlabel(None)
plt.title("Sum of Total Actions for Challenge B", fontweight='heavy', ha='center', fontsize=14)
```




    Text(0.5, 1.0, 'Sum of Total Actions for Challenge B')




    
![png](/assets/images/campaign2022_files/output_26_1.png)
    

