---
title: "BPJS Healthkathon"
categories:
  - Portfolio
tags:
  - Portfolio
  - Machine Learning
  - Supervised Learning
  - Imbalanced Dataset
toc: true
---

### Load Packages


```python
# Data Processing
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Imbalance Class Framework
import imblearn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter


# ML Framework
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve,auc )
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics

# ML Alghoritm
from sklearn.ensemble import RandomForestClassifier
```

### Read Data


```python
df = pd.read_csv('validation.csv')
```


```python
df.columns
```




    Index(['Unnamed: 0', 'id', 'id_peserta', 'dati2', 'typefaskes', 'usia',
           'jenkel', 'pisat', 'tgldatang', 'tglpulang', 'jenispel', 'politujuan',
           'diagfktp', 'biaya', 'jenispulang', 'cbg', 'kelasrawat', 'kdsa', 'kdsp',
           'kdsr', 'kdsi', 'kdsd', 'label', 'diag', 'levelid', 'proc'],
          dtype='object')




```python
df.shape
```




    (11401882, 26)



### Data Preprocessing
This part containts encoding, feature engineering, and removing null. the model only takes 17 columns out of 25 columns available in this dataset, this are the list of the columns that we'll be used:
<ol>
    <li>usia</li>
    <li>jenkel</li>
    <li>pisat</li>
    <li>jenispel</li>
    <li>biaya</li>
    <li>jenispulang</li>
    <li>kelasrawat</li>
    <li><b>date_diff</b> = tgldatang - tglpulang</li>
    <li>typefaskes</li>
    <li>diag</li>
    <li>proc</li>
    <li>kdsa</li>
    <li>kdsr</li>
    <li>politujuan</li>
    <li>dati2</li>
    <li>cbg</li>
    <li>diagfktp</li>
</ol>
While some columns already numeric, we need to convert all categorical columns to numeric to be feed into our model.

#### Categorical list
<ol>
    <li>jenkel</li>
    <li>tgldatang</li>
    <li>tglpulang</li>
    <li>typefaskes</li>
    <li>diag</li>
    <li>proc</li>
    <li>kdsa</li>
    <li>kdsr</li>
    <li>politujuan</li>
    <li>cbg</li>
    <li>diagfktp</li>
</ol>

#### Numerical list

<ol>
    <li>usia</li>
    <li>pisat</li>
    <li>jenispel</li>
    <li>biaya</li>
    <li>jenispulang</li>
    <li>kelasrawat</li>
    <li>dati2</li>
</ol>


```python
df.dati2.value_counts()
```




    113    293414
    217    253418
    135    217038
    38     209744
    90     178315
            ...  
    528       132
    517        61
    366        15
    527         9
    479         1
    Name: dati2, Length: 489, dtype: int64



Drop unused columns


```python
#df.drop(columns=['Unnamed: 0', 'id_peserta', 'kdsp', 'kdsi', 'kdsd', 'levelid'], inplace=True, axis=1)
df.drop(columns=['Unnamed: 0', 'id_peserta'], inplace=True, axis=1)
```


```python
df.isnull().sum()
```




    id                   0
    dati2                0
    typefaskes           0
    usia                 0
    jenkel              49
    pisat              190
    tgldatang            0
    tglpulang            0
    jenispel             0
    politujuan     4041455
    diagfktp          2530
    biaya            57815
    jenispulang         39
    cbg                  0
    kelasrawat           0
    kdsa            198670
    kdsp            197007
    kdsr            198459
    kdsi            198463
    kdsd            195181
    label                0
    diag             23806
    levelid          23806
    proc           4947985
    dtype: int64



#### Convert 'jenkel' to numeric


```python
df.jenkel.value_counts(dropna=False)
```




    P      6161677
    L      5240156
    NaN         49
    Name: jenkel, dtype: int64




```python
df[['jenkel', 'label']].groupby(['jenkel'], as_index=True, dropna=False).mean().sort_values(['jenkel'], ascending=True)
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
      <th>label</th>
    </tr>
    <tr>
      <th>jenkel</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>L</th>
      <td>0.014216</td>
    </tr>
    <tr>
      <th>P</th>
      <td>0.013372</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>0.020408</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['jenkel'] = df['jenkel'].map({'L': 1, 'P': 0})
```

fill na with the most frequent values


```python
df['jenkel'].fillna(0, inplace=True)
```


```python
df.jenkel.value_counts(dropna=False)
```




    0.0    6161726
    1.0    5240156
    Name: jenkel, dtype: int64



#### Feature engineering from 'tgldatang' and 'tglpulang'
While tgldatang and tglpulang potentially has pattern for our model they actually can make a new better pattern if we combine them together. Difference between date will make a feature that more unique and categorized


```python
import re
from datetime import datetime
```


```python
def generate_datediff(tgldatang, tglpulang):
    try:
        m1 = re.match('(^[\d]+-\d+-[\d]+)', str(tgldatang)).group(1)
        m2 = re.match('(^[\d]+-\d+-[\d]+)', str(tglpulang)).group(1)
        d1 = datetime.strptime(m1, '%Y-%m-%d')
        d2 = datetime.strptime(m2, '%Y-%m-%d')
        result = (d2-d1).days
    except AttributeError:
        result = 0
    return result
```


```python
%%time
df['date_diff'] = df.apply(lambda row: generate_datediff(row['tgldatang'], row['tglpulang']), axis=1)
```

    CPU times: total: 8min 32s
    Wall time: 8min 35s
    

%%time
plt.figure(figsize=(7, 6))
sns.distplot(df.date_diff[df.label == 0], bins=3, color='r', label='Non Fraud')
sns.distplot(df.date_diff[df.label == 1], bins=3, color='g', label='Fraud')


```python
df['date_diff_band'] = pd.cut(df.date_diff, bins=10)
df[['date_diff_band', 'label']].groupby(['date_diff_band'], as_index=False).mean().sort_values(by='date_diff_band', ascending=True)
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
      <th>date_diff_band</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-1.431, 143.1]</td>
      <td>0.013757</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(143.1, 286.2]</td>
      <td>0.011658</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(286.2, 429.3]</td>
      <td>0.083942</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(429.3, 572.4]</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(572.4, 715.5]</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(715.5, 858.6]</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(858.6, 1001.7]</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(1001.7, 1144.8]</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(1144.8, 1287.9]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(1287.9, 1431.0]</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['date_diff_band'] = pd.cut(df.date_diff, bins=[0, 131, 262, 393, 524, 1000])
df[['date_diff_band', 'label']].groupby(['date_diff_band'], as_index=False).mean().sort_values(by='date_diff_band', ascending=True)

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
      <th>date_diff_band</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0, 131]</td>
      <td>0.012748</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(131, 262]</td>
      <td>0.010832</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(262, 393]</td>
      <td>0.052117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(393, 524]</td>
      <td>0.130435</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(524, 1000]</td>
      <td>0.085714</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.date_diff_band.value_counts(dropna=False)
```




    (-1.431, 143.1]     11399922
    (143.1, 286.2]          1544
    (286.2, 429.3]           274
    (429.3, 572.4]            90
    (572.4, 715.5]            34
    (715.5, 858.6]            12
    (1001.7, 1144.8]           3
    (858.6, 1001.7]            2
    (1287.9, 1431.0]           1
    (1144.8, 1287.9]           0
    Name: date_diff_band, dtype: int64




```python
df.loc[df['date_diff'] <= 131, 'date_diff'] = 0
df.loc[(df['date_diff'] > 131) & (df['date_diff'] <= 262), 'date_diff'] = 1
df.loc[(df['date_diff'] > 262) & (df['date_diff'] <= 393), 'date_diff'] = 2
df.loc[(df['date_diff'] > 393) & (df['date_diff'] <= 524), 'date_diff'] = 3
df.loc[df['date_diff'] >  524, 'date_diff'] = 4
```


```python
df[['date_diff', 'label']].groupby(['date_diff'], as_index=False).mean().sort_values(by='date_diff', ascending=True)
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
      <th>date_diff</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.013758</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.010832</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.052117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.130435</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.081081</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df[['date_diff_band', 'label']].groupby(['date_diff_band'], as_index=False).mean().sort_values(by='date_diff_band', ascending=True)
```


```python
#df.drop(columns=['date_diff_band'], axis=1, inplace=True)
```

#### Convert 'typefaskes'


```python
df[['typefaskes', 'label']].groupby(['typefaskes'], as_index=False).mean().sort_values(by='typefaskes', ascending=True)
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
      <th>typefaskes</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0.011226</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.014452</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>0.016122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>0.020405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GD</td>
      <td>0.029225</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HD</td>
      <td>0.000973</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I1</td>
      <td>0.018468</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I2</td>
      <td>0.009154</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I3</td>
      <td>0.011775</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I4</td>
      <td>0.013376</td>
    </tr>
    <tr>
      <th>10</th>
      <td>KB</td>
      <td>0.015895</td>
    </tr>
    <tr>
      <th>11</th>
      <td>KC</td>
      <td>0.003597</td>
    </tr>
    <tr>
      <th>12</th>
      <td>KG</td>
      <td>0.008802</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KI</td>
      <td>0.018579</td>
    </tr>
    <tr>
      <th>14</th>
      <td>KJ</td>
      <td>0.006258</td>
    </tr>
    <tr>
      <th>15</th>
      <td>KK</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>KL</td>
      <td>0.014274</td>
    </tr>
    <tr>
      <th>17</th>
      <td>KM</td>
      <td>0.037590</td>
    </tr>
    <tr>
      <th>18</th>
      <td>KO</td>
      <td>0.006951</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KP</td>
      <td>0.007619</td>
    </tr>
    <tr>
      <th>20</th>
      <td>KT</td>
      <td>0.010305</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KU</td>
      <td>0.012201</td>
    </tr>
    <tr>
      <th>22</th>
      <td>SA</td>
      <td>0.007686</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SB</td>
      <td>0.010086</td>
    </tr>
    <tr>
      <th>24</th>
      <td>SC</td>
      <td>0.011294</td>
    </tr>
    <tr>
      <th>25</th>
      <td>SD</td>
      <td>0.015849</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.typefaskes.value_counts()
```




    SC    3165558
    C     2197424
    B     2102708
    SD     825690
    SB     808444
    A      493055
    I3     285780
    D      260327
    I2     226470
    KI     221114
    KM     177017
    KJ     173692
    I4     164106
    KL     101933
    I1      49979
    KC      28917
    KB      25479
    KP      17850
    SA      17174
    HD      14390
    KO      12804
    KG      11134
    GD       9273
    KT       7375
    KU       4180
    KK          9
    Name: typefaskes, dtype: int64




```python
typefaskes_encode = (df.groupby('typefaskes').size())/len(df)
typefaskes_encode
```




    typefaskes
    A     4.324330e-02
    B     1.844176e-01
    C     1.927247e-01
    D     2.283193e-02
    GD    8.132868e-04
    HD    1.262072e-03
    I1    4.383399e-03
    I2    1.986251e-02
    I3    2.506428e-02
    I4    1.439289e-02
    KB    2.234631e-03
    KC    2.536160e-03
    KG    9.765055e-04
    KI    1.939276e-02
    KJ    1.523363e-02
    KK    7.893434e-07
    KL    8.940015e-03
    KM    1.552524e-02
    KO    1.122973e-03
    KP    1.565531e-03
    KT    6.468230e-04
    KU    3.666061e-04
    SA    1.506243e-03
    SB    7.090443e-02
    SC    2.776347e-01
    SD    7.241699e-02
    dtype: float64




```python
len(df.typefaskes.unique())
```




    26




```python
%%time
df['typefaskes'] = df['typefaskes'].apply(lambda x: typefaskes_encode[x])
```

    CPU times: total: 42.9 s
    Wall time: 42.9 s
    


```python
df.typefaskes.value_counts()
```




    2.776347e-01    3165558
    1.927247e-01    2197424
    1.844176e-01    2102708
    7.241699e-02     825690
    7.090443e-02     808444
    4.324330e-02     493055
    2.506428e-02     285780
    2.283193e-02     260327
    1.986251e-02     226470
    1.939276e-02     221114
    1.552524e-02     177017
    1.523363e-02     173692
    1.439289e-02     164106
    8.940015e-03     101933
    4.383399e-03      49979
    2.536160e-03      28917
    2.234631e-03      25479
    1.565531e-03      17850
    1.506243e-03      17174
    1.262072e-03      14390
    1.122973e-03      12804
    9.765055e-04      11134
    8.132868e-04       9273
    6.468230e-04       7375
    3.666061e-04       4180
    7.893434e-07          9
    Name: typefaskes, dtype: int64



#### Convert 'kddiag/diag' : kode diagnosa FKRTL


```python
df.diag.value_counts()
```




    K30       237548
    P03.4     204292
    I10       189563
    D64.9     173226
    Z09.8     164635
               ...  
    I98.0          1
    W57.2          1
    Y31.1          1
    M84.30         1
    V38.0          1
    Name: diag, Length: 12438, dtype: int64




```python
def generate_diag_numeric(diag):
    result = 0
    try:
        # Match diag with regex ex: A21 P01
        alp = re.match('(^[A-Za-z])([0-9]+)', diag).group(1)
        num = re.match('(^[A-Za-z])([0-9]+)', diag).group(2)
        # Categorize using ICD-10
        if alp == 'A' or alp == B:
            result = 1
        elif alp == 'C':
            result = 2
        elif alp == 'D':
            if int(num) <= 48:
                result = 2
            else:
                result = 3
        elif alp == 'E':
            result = 4
        elif alp == 'F':
            result = 5
        elif alp == 'G':
            result = 6
        elif alp == 'H':
            if int(num) <= 59:
                result = 7
            else:
                result = 8
        elif alp == 'I':
            result = 9
        elif alp == 'J':
            result = 10
        elif alp == 'K':
            result = 11
        elif alp == 'L':
            result = 12
        elif alp == 'M':
            result = 13
        elif alp == 'N':
            result = 14
        elif alp == 'O':
            result = 15
        elif alp == 'P':
            result = 16
        elif alp == 'Q':
            result = 17
        elif alp == 'R':
            result = 18
        elif alp == 'S' or alp == 'T':
            result = 19
        elif alp == 'V' or alp == 'W' or alp == 'X' or alp == 'Y':
            result = 20
        elif alp == 'Z':
            result = 21
        elif alp == 'U':
            result = 22
    except:
        result = 0
    return result
```


```python
%%time
df['diag'] = df.apply(lambda row: generate_diag_numeric(row['diag']), axis=1)
```

    CPU times: total: 2min 42s
    Wall time: 2min 42s
    


```python
df.diag.value_counts()
```




    0    10952703
    1      449179
    Name: diag, dtype: int64




```python
df.columns
```




    Index(['id', 'dati2', 'typefaskes', 'usia', 'jenkel', 'pisat', 'tgldatang',
           'tglpulang', 'jenispel', 'politujuan', 'diagfktp', 'biaya',
           'jenispulang', 'cbg', 'kelasrawat', 'kdsa', 'kdsp', 'kdsr', 'kdsi',
           'kdsd', 'label', 'diag', 'levelid', 'proc', 'date_diff',
           'date_diff_band'],
          dtype='object')



#### Convert 'proc' : kode prosedure
Categorize using http://www.icd9data.com/2012/Volume3/default.htm


```python
df.proc.value_counts()
```




    90.59    721461
    99.04    288252
    89.52    250374
    99.18    239962
    39.95    179110
              ...  
    59.03         1
    91.14         1
    09.22         1
    88.46         1
    42.53         1
    Name: proc, Length: 3513, dtype: int64




```python
def generate_proc_numeric(proc):
    result =0
    try:
        m1 = re.match('^([0-9]+)', str(proc)).group(1)
        num = int(m1)
        if num == 0:
            result = 0
        elif num >= 1 and num <= 5:
            result = 1
        elif num >= 6 and num <= 7:
            result = 2
        elif num >= 8 and num <= 16:
            result = 3
        elif num == 17:
            result = 4
        elif num >= 18 and num <= 20:
            result = 5
        elif num >= 21 and num <= 29:
            result = 6
        elif num >= 30 and num <= 34:
            result = 7
        elif num >= 35 and num <= 39:
            result = 8
        elif num >= 40 and num <= 41:
            result = 9
        elif num >= 42 and num <= 54:
            result = 10
        elif num >= 55 and num <= 59:
            result = 11
        elif num >= 60 and num <= 64:
            result = 12
        elif num >= 65 and num <= 71:
            result = 13
        elif num >= 72 and num <= 75:
            result = 14
        elif num >= 76 and num <= 84:
            result = 15
        elif num >= 85 and num <= 86:
            result = 16
        elif num >= 87 and num <= 99:
            result = 17
    except:
        result = 0
    
    return result
```


```python
%%time
df['proc'] = df.apply(lambda row: generate_proc_numeric(row['proc']), axis=1)
```

    CPU times: total: 2min 42s
    Wall time: 2min 42s
    


```python
df.proc.value_counts()
```




    0     5032825
    17    4842583
    16     271302
    14     264325
    8      215122
    6      172023
    3      128108
    15     127133
    13      77609
    10      73211
    11      66891
    5       46461
    12      22319
    1       21281
    7       16630
    9       12437
    2       11394
    4         228
    Name: proc, dtype: int64



#### Convert 'kdsa' : Kode Spesial Sub-Acute Group


```python
df.kdsa.value_counts()
```




    None           11190211
    SF-4-10-II         4457
    SF-4-10-III        3587
    SF-4-10-I          1891
    SF-4-10-X          1142
    SF-4-10-IV          675
    SF-4-13-II          355
    SF-4-16-II          162
    SF-4-16-III         147
    SF-4-13-III         142
    SF-4-13-I           125
    SF-4-15-II           57
    SF-4-16-I            52
    SF-4-16-IV           45
    SF-4-15-III          24
    SF-4-13-IV           24
    SF-4-15-I            20
    SF-4-13-X            18
    SF-4-11-II           12
    SF-4-11-I            11
    SF-4-15-IV           10
    SF-4-16-X            10
    SF-4-19-II            7
    SF-4-11-III           6
    SF-4-19-I             5
    SA-4-14-I             3
    SF-4-19-III           3
    SA-4-14-X             3
    SF-4-19-IV            1
    SA-4-14-II            1
    SF-4-12-II            1
    ST-4-11-II            1
    SF-4-11-IV            1
    ST-1-10-III           1
    SF-4-18-I             1
    SF-4-15-X             1
    Name: kdsa, dtype: int64




```python
def generate_kdsa_numeric(kdsa):
    num = 0
    result = 0
    try:
        num = kdsa.split(sep='-')[2]
        if num == '10':
            result = 1
        elif num == '11':
            result = 2
        elif num == '12':
            result = 3
        elif num == '13':
            result = 4
        elif num == '14':
            result = 5
        elif num == '15':
            result = 6
        elif num == '16':
            result = 7
        else:
            result = 0
    except:
        result = 0
        
    return result
```


```python
%%time
df['kdsa'] = df.apply(lambda row: generate_kdsa_numeric(row['kdsa']), axis=1)
```

    CPU times: total: 2min 11s
    Wall time: 2min 11s
    


```python
df.kdsa.value_counts()
```




    0    11388898
    1       11753
    4         664
    7         416
    6         112
    2          31
    5           7
    3           1
    Name: kdsa, dtype: int64



#### Convert 'kdsr' : Kode Spesial Prothesis


```python
df.kdsr.value_counts()
```




    None         11196767
    RR-04-III        5968
    RR-02-II          651
    RR-05-III          29
    RR-01-II            5
    RR-03-III           3
    Name: kdsr, dtype: int64




```python
def generate_kdsr_numeric(kdsr):
    num = 0
    result = 0
    try:
        num = kdsr.split(sep='-')[1]
        result = int(num)
    except:
        result = 0

    return result
```


```python
%%time
df['kdsr'] = df.apply(lambda row: generate_kdsr_numeric(row['kdsr']), axis=1)
```

    CPU times: total: 2min 7s
    Wall time: 2min 7s
    


```python
df.kdsr.value_counts()
```




    0    11395226
    4        5968
    2         651
    5          29
    1           5
    3           3
    Name: kdsr, dtype: int64



#### Convert 'kdsp' : Kode Spesial Prothesis


```python
df.kdsp.value_counts()
```




    None         11132240
    YY-10-III       51313
    YY-02-III        7337
    YY-01-II         5971
    YY-09-III        4229
    YY-05-III        2985
    YY-11-III         346
    YY-07-III         171
    YY-03-III         130
    YY-06-III          90
    YY-08-III          32
    YY-04-III          27
    YY-12-III           4
    Name: kdsp, dtype: int64




```python
df0 = df.copy()
```


```python
def generate_kdsp_numeric(kdsp):
    num = 0
    result = 0
    try:
        num = kdsp.split(sep='-')[1]
        result = int(num)
    except:
        result = 0

    return result
```


```python
%%time
df['kdsp'] = df0.apply(lambda row: generate_kdsp_numeric(row['kdsp']), axis=1)
```

    CPU times: total: 2min 7s
    Wall time: 2min 7s
    


```python
df.kdsp.value_counts()
```




    0     11329247
    10       51313
    2         7337
    1         5971
    9         4229
    5         2985
    11         346
    7          171
    3          130
    6           90
    8           32
    4           27
    12           4
    Name: kdsp, dtype: int64



#### Convert 'cbg' : Kode Case Base Group


```python
df.cbg.value_counts(dropna=False)
```




    Q-5-44-0      2678848
    Q-5-42-0      1012559
    F-5-14-0       289409
    P-8-17-I       248339
    U-3-15-0       219126
                   ...   
    L-2-21-0            1
    I-2-42-0            1
    M-1-05-III          1
    J-4-10-II           1
    J-2-31-0            1
    Name: cbg, Length: 1034, dtype: int64




```python
encode_cbg = (df.groupby('cbg', dropna=False).size()) / len(df)
encode_cbg
```




    cbg
    A-4-10-I      0.000224
    A-4-10-II     0.000235
    A-4-10-III    0.000178
    A-4-11-I      0.000187
    A-4-11-II     0.000040
                    ...   
    Z-4-11-II     0.000004
    Z-4-11-III    0.000001
    Z-4-12-I      0.000120
    Z-4-12-II     0.000056
    Z-4-12-III    0.000030
    Length: 1034, dtype: float64




```python
df['cbg'] = df['cbg'].apply(lambda x: encode_cbg[x])
```


```python
df.cbg.value_counts()
```




    2.349479e-01    2678848
    8.880630e-02    1012559
    2.538256e-02     289409
    2.178053e-02     248339
    1.921841e-02     219126
                     ...   
    8.770482e-07         30
    1.227867e-06         28
    2.631145e-07         27
    2.017211e-06         23
    8.770482e-08         22
    Name: cbg, Length: 750, dtype: int64




```python
df.shape
```




    (11401882, 26)



#### Convert 'diagfktp'


```python
df.diagfktp
```




    0           L02.8
    1           R23.1
    2           E10.5
    3           H54.2
    4           M54.5
                ...  
    11401877    Z71.8
    11401878    P59.9
    11401879    P24.8
    11401880      P03
    11401881    P21.1
    Name: diagfktp, Length: 11401882, dtype: object




```python
diagfktp_encode = (df.groupby('diagfktp', dropna=False).size()) / len(df)
diagfktp_encode
```




    diagfktp
    -             2.210161e-05
    --            2.806554e-06
    .             2.631145e-07
    0001217347    1.754096e-07
    0001262881    8.770482e-08
                      ...     
    z95           8.770482e-08
    z96.1         3.508193e-07
    z96.6         1.754096e-07
    z98.8         8.770482e-08
    NaN           2.218932e-04
    Length: 17013, dtype: float64




```python
df['diagfktp'] = df['diagfktp'].apply(lambda x: diagfktp_encode[x])
```


```python
df.diagfktp.value_counts()
```




    0.035745    407556
    0.030402    346635
    0.018222    207769
    0.017715    201981
    0.016130    183918
                 ...  
    0.000030       344
    0.000029       333
    0.000028       323
    0.000026       293
    0.000019       211
    Name: diagfktp, Length: 1956, dtype: int64



#### Categorize 'politujuan'


```python
politujuan_encode = df.groupby('politujuan', dropna=False).size() / len(df)
politujuan_encode
```




    politujuan
    004    8.516138e-05
    005    4.176503e-04
    006    1.919858e-04
    007    3.039849e-04
    008    2.386799e-03
               ...     
    tht    1.841801e-06
    tum    7.893434e-07
    ugd    1.578687e-06
    uro    4.385241e-07
    NaN    3.544551e-01
    Length: 262, dtype: float64




```python
df['politujuan'] = df['politujuan'].apply(lambda x: politujuan_encode[x])
```


```python
df.politujuan.value_counts()
```




    3.544551e-01    4041455
    1.376309e-01    1569251
    7.284280e-02     830545
    5.528421e-02     630344
    5.312228e-02     605694
                     ...   
    7.016386e-07         16
    8.770482e-08         16
    1.315572e-06         15
    8.770482e-07         10
    2.631145e-07          9
    Name: politujuan, Length: 178, dtype: int64



#### Categorize 'dati2'


```python
df.dati2.value_counts()
```




    113    293414
    217    253418
    135    217038
    38     209744
    90     178315
            ...  
    528       132
    517        61
    366        15
    527         9
    479         1
    Name: dati2, Length: 489, dtype: int64




```python
dati2_encode = df.groupby('dati2', dropna=False).size() / len(df)
dati2_encode
```




    dati2
    1      1.198311e-03
    2      4.556265e-04
    3      8.705580e-04
    4      8.579285e-04
    5      1.355127e-03
               ...     
    522    1.058597e-04
    524    2.350489e-05
    526    2.525899e-05
    527    7.893434e-07
    528    1.157704e-05
    Length: 489, dtype: float64




```python
df['dati2'] = df['dati2'].apply(lambda x: dati2_encode[x])
```

#### Feature Engineering 'usia'


```python
df.usia.value_counts()
```




    0      853260
    1      200102
    54     193578
    53     190567
    55     189498
            ...  
    104        19
    107        12
    108         8
    110         4
    109         4
    Name: usia, Length: 111, dtype: int64




```python
g = sns.FacetGrid(df, col='label')
g.map(plt.hist, 'usia', bins=20)
```




    <seaborn.axisgrid.FacetGrid at 0x2c3ffe39d30>




    
![png](/assets/images/bpjshealthkathon_files/bpjs1.png)
    



```python
df[['usia', 'label']].groupby(['usia'], as_index=False, dropna=False).mean().sort_values(['usia'], ascending=True)
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
      <th>usia</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.027203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.010799</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.010861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.011280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.011621</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>106</th>
      <td>106</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>107</th>
      <td>107</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>108</th>
      <td>108</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>109</th>
      <td>109</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>110</th>
      <td>110</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>111 rows × 2 columns</p>
</div>



We have so many zero as age which doesnt seems good, but we can guess with the nearest correlated features, in this case im using 'pisat' and 'jenkel'


```python
guess_ages = np.zeros((2,5))
guess_ages
```


```python
%%time
for i in range(0, 2):
        for j in range(0, 5):
            guess_df = df[(df['jenkel'] == i) & (df['pisat'] == j+1)]['usia'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
        for j in range(0, 5):
            df.loc[ (df.usia == 0) & (df.jenkel == i) & (df.pisat == j+1),'usia'] = guess_ages[i,j]

# convert type
df['usia'] = df['usia'].astype(int)
```


```python
g = sns.FacetGrid(df, col='label')
g.map(plt.hist, 'usia', bins=20)
```


```python
df['usiaBand'] = pd.cut(df['usia'], 5)
df[['usiaBand', 'label']].groupby(['usiaBand'], as_index=False).mean().sort_values(['usiaBand'], ascending=True)
```


```python
df.usiaBand.value_counts()
```


```python
df.loc[df['usia'] <= 22, 'usia'] = 0
df.loc[(df['usia'] > 22) & (df['usia'] <= 44), 'usia'] = 1
df.loc[(df['usia'] > 44) & (df['usia'] <= 66), 'usia'] = 2
df.loc[(df['usia'] > 66) & (df['usia'] <= 88), 'usia'] = 3
df.loc[ df['usia'] > 88, 'usia'] = 4
```


```python
df.drop(columns=['usiaBand'], inplace=True, axis=1)
```

#### Feature Engineering 'biaya'


```python
df.isnull().sum()
```




    id                     0
    dati2                  0
    typefaskes             0
    usia                   0
    jenkel                49
    pisat                190
    tgldatang              0
    tglpulang              0
    jenispel               0
    politujuan             0
    diagfktp               0
    biaya              57815
    jenispulang           39
    cbg                    0
    kelasrawat             0
    kdsa                   0
    kdsp                   0
    kdsr                   0
    kdsi              198463
    kdsd              195181
    label                  0
    diag                   0
    levelid            23806
    proc                   0
    date_diff              0
    date_diff_band         0
    dtype: int64




```python
df['biaya'].fillna(value=0, inplace=True)
```


```python
df.isnull().sum()
```




    id               0
    dati2            0
    typefaskes       0
    usia             0
    jenkel          49
    pisat          190
    tgldatang        0
    tglpulang        0
    jenispel         0
    politujuan       0
    diagfktp         0
    biaya            0
    jenispulang     39
    cbg              0
    kelasrawat       0
    kdsa             0
    kdsr             0
    label            0
    diag             0
    proc             0
    date_diff        0
    dtype: int64




```python
#df['biayaBand'] = pd.qcut(df['biaya'], q=5)
```


```python
df['biayaBand'] = pd.cut(df['biaya'], bins=[])
```


```python
df.loc[df['biaya'] <= 200_000, 'biaya'] = 1
df.loc[(df['biaya'] > 200_000) & (df['biaya'] <= 300_000), 'biaya'] = 2
df.loc[(df['biaya'] > 300_000) & (df['biaya'] <= 500_000), 'biaya'] = 3
df.loc[(df['biaya'] > 500_000) & (df['biaya'] <= 1000_000), 'biaya'] = 4
df.loc[(df['biaya'] > 1000_000) & (df['biaya'] <= 5000_000), 'biaya'] = 5
df.loc[(df['biaya'] > 5000_000) & (df['biaya'] <= 10_000_000), 'biaya'] = 6
df.loc[df['biaya'] > 10_000_000, 'biaya'] = 7
```


```python
df[['biaya', 'label']].groupby(['biaya'], as_index=False, dropna=False).mean().sort_values(by='biaya', ascending=True)
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
      <th>biaya</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.005665</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.017510</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.010298</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.009722</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.008371</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>0.010071</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>0.010904</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>0.952521</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(columns=['biayaBand'], axis=1, inplace=True)
```


```python
df.isnull().sum()
```




    id                     0
    dati2                  0
    typefaskes             0
    usia                   0
    jenkel                49
    pisat                190
    tgldatang              0
    tglpulang              0
    jenispel               0
    politujuan             0
    diagfktp               0
    biaya              57815
    jenispulang           39
    cbg                    0
    kelasrawat             0
    kdsa                   0
    kdsp                   0
    kdsr                   0
    kdsi              198463
    kdsd              195181
    label                  0
    diag                   0
    levelid            23806
    proc                   0
    date_diff              0
    date_diff_band         0
    dtype: int64




```python
df.jenkel.value_counts()
```




    0.0    6161677
    1.0    5240156
    Name: jenkel, dtype: int64




```python
df.pisat.value_counts()
```




    1.0    4874164
    4.0    3350737
    3.0    2474862
    5.0     459397
    2.0     242532
    Name: pisat, dtype: int64




```python
df.jenispulang.value_counts()
```




    1.0    10929593
    2.0      185714
    4.0      146116
    5.0       86835
    3.0       53585
    Name: jenispulang, dtype: int64




```python
df.levelid.value_counts()
```




    1.0    9202238
    2.0    2175838
    Name: levelid, dtype: int64




```python
df.groupby('biaya', dropna=False).size()
```




    biaya
    0.0    4309884
    1.0    1515684
    2.0     746629
    3.0     473167
    4.0    2929432
    5.0    1050949
    6.0     318322
    NaN      57815
    dtype: int64




```python
df.pisat.fillna('1', inplace=True)
df.jenispulang.fillna('1', inplace=True)
df.levelid.fillna('1', inplace=True)
df.biaya.fillna('7', inplace=True)
```


```python
df.isnull().sum()
```




    id                     0
    dati2                  0
    typefaskes             0
    usia                   0
    jenkel                 0
    pisat                  0
    tgldatang              0
    tglpulang              0
    jenispel               0
    politujuan             0
    diagfktp               0
    biaya                  0
    jenispulang            0
    cbg                    0
    kelasrawat             0
    kdsa                   0
    kdsp                   0
    kdsr                   0
    kdsi              198463
    kdsd              195181
    label                  0
    diag                   0
    levelid                0
    proc                   0
    date_diff              0
    date_diff_band         0
    dtype: int64




```python
df_save = df.drop(['kdsd', 'kdsi', 'date_diff_band'], axis=1)
```


```python
df_save.columns
```




    Index(['id', 'dati2', 'typefaskes', 'usia', 'jenkel', 'pisat', 'tgldatang',
           'tglpulang', 'jenispel', 'politujuan', 'diagfktp', 'biaya',
           'jenispulang', 'cbg', 'kelasrawat', 'kdsa', 'kdsp', 'kdsr', 'label',
           'diag', 'levelid', 'proc', 'date_diff'],
          dtype='object')




```python
df_save.to_csv('validation_litev3_ready.csv')
```


```python
df.columns
```




    Index(['id', 'dati2', 'typefaskes', 'usia', 'jenkel', 'pisat', 'jenispel',
           'politujuan', 'diagfktp', 'biaya', 'jenispulang', 'cbg', 'kelasrawat',
           'kdsa', 'kdsr', 'label', 'diag', 'proc', 'date_diff'],
          dtype='object')




```python
df.date_diff.value_counts()
```




    0      11399922
    1          1544
    2           274
    3            90
    4            18
    626           3
    655           2
    697           2
    647           2
    615           2
    629           2
    588           1
    657           1
    663           1
    659           1
    684           1
    660           1
    581           1
    671           1
    704           1
    645           1
    636           1
    628           1
    578           1
    604           1
    651           1
    642           1
    605           1
    712           1
    692           1
    579           1
    676           1
    Name: date_diff, dtype: int64



### Balancing Label Distribution Using SMOTE and Undersampling


```python
#df = pd.read_csv("validation_litev2_ready.csv")
df = pd.read_csv("validation_litev4_ready.csv")
```


```python
df.columns
```




    Index(['Unnamed: 0', 'id', 'dati2', 'typefaskes', 'usia', 'jenkel', 'pisat',
           'tgldatang', 'tglpulang', 'jenispel', 'politujuan', 'diagfktp', 'biaya',
           'jenispulang', 'cbg', 'kelasrawat', 'kdsa', 'kdsp', 'kdsr', 'label',
           'diag', 'levelid', 'proc', 'date_diff'],
          dtype='object')




```python
df.shape
```




    (11401882, 24)




```python
df.drop(columns=['Unnamed: 0', 'tgldatang', 'tglpulang'], axis=1, inplace=True)
```


```python
df.columns
```




    Index(['id', 'dati2', 'typefaskes', 'usia', 'jenkel', 'pisat', 'jenispel',
           'politujuan', 'diagfktp', 'biaya', 'jenispulang', 'cbg', 'kelasrawat',
           'kdsa', 'kdsp', 'kdsr', 'label', 'diag', 'levelid', 'proc',
           'date_diff'],
          dtype='object')



### Scaling and Distributing


```python
X = df.drop(['id', 'label'], axis=1)
y = df.label
```


```python
#trans = RobustScaler()
#model = RandomForestClassifier()
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.5)
```


```python
#steps = [('o', over), ('u', under), ('t', trans), ('m', model)]
steps = [('o', over), ('u', under)]
#steps = [('o', over)]
pipeline = Pipeline(steps=steps)

# evaluate the pipeline
#cv = StratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
#n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report pipeline performance
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
X, y = pipeline.fit_resample(X, y)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```


```python
rob_scaler = RobustScaler()

# Fit only to the training data
X_train= rob_scaler.fit_transform(X_train)

X_test = rob_scaler.transform(X_test)
```


```python
counter = Counter(y_train)
counter
```




    Counter({0: 8434814, 1: 4215802})


### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve,auc )

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_predtrain = clf.predict(X_train)
CM_LR = confusion_matrix(y_test,y_pred)
CR_LR = classification_report(y_test,y_pred)
CM_LRtrain = confusion_matrix(y_train,y_predtrain)
CR_LRtrain = classification_report(y_train,y_predtrain)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",CM_LR)
print("Classification Report:\n",CR_LR)
print("Confusion Matrix Train:\n",CM_LRtrain)
print("Classification Report Train:\n",CR_LRtrain)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))
```

    C:\Users\ASUS\anaconda3\envs\tf2.8\lib\site-packages\sklearn\linear_model\_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Accuracy: 0.726171911312461
    Confusion Matrix:
     [[2647869  162309]
     [ 992389  414305]]
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.73      0.94      0.82   2810178
               1       0.72      0.29      0.42   1406694
    
        accuracy                           0.73   4216872
       macro avg       0.72      0.62      0.62   4216872
    weighted avg       0.72      0.73      0.69   4216872
    
    Confusion Matrix Train:
     [[7947663  487151]
     [2975778 1240024]]
    Classification Report Train:
                   precision    recall  f1-score   support
    
               0       0.73      0.94      0.82   8434814
               1       0.72      0.29      0.42   4215802
    
        accuracy                           0.73  12650616
       macro avg       0.72      0.62      0.62  12650616
    weighted avg       0.72      0.73      0.69  12650616
    
    Precision: 0.718513598351757
    Recall: 0.2945238978768659
    F1: 0.4177918911233152
    Area under precision (AUC) Recall: 0.44695713570129525
    


```python
coeff_df = pd.DataFrame(df.drop(["label", "id"], axis=1).columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(clf.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False, key=abs)
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
      <th>Feature</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>biaya</td>
      <td>-2.699715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>jenispel</td>
      <td>-2.287892</td>
    </tr>
    <tr>
      <th>18</th>
      <td>date_diff</td>
      <td>1.484067</td>
    </tr>
    <tr>
      <th>6</th>
      <td>politujuan</td>
      <td>-0.882303</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cbg</td>
      <td>-0.516185</td>
    </tr>
    <tr>
      <th>12</th>
      <td>kdsa</td>
      <td>-0.485156</td>
    </tr>
    <tr>
      <th>13</th>
      <td>kdsp</td>
      <td>0.483613</td>
    </tr>
    <tr>
      <th>15</th>
      <td>diag</td>
      <td>-0.473919</td>
    </tr>
    <tr>
      <th>11</th>
      <td>kelasrawat</td>
      <td>-0.363366</td>
    </tr>
    <tr>
      <th>1</th>
      <td>typefaskes</td>
      <td>-0.312593</td>
    </tr>
    <tr>
      <th>0</th>
      <td>dati2</td>
      <td>-0.291797</td>
    </tr>
    <tr>
      <th>17</th>
      <td>proc</td>
      <td>-0.127218</td>
    </tr>
    <tr>
      <th>14</th>
      <td>kdsr</td>
      <td>-0.124789</td>
    </tr>
    <tr>
      <th>16</th>
      <td>levelid</td>
      <td>0.110650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>jenkel</td>
      <td>0.056143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pisat</td>
      <td>0.053450</td>
    </tr>
    <tr>
      <th>9</th>
      <td>jenispulang</td>
      <td>0.024065</td>
    </tr>
    <tr>
      <th>2</th>
      <td>usia</td>
      <td>0.004810</td>
    </tr>
    <tr>
      <th>7</th>
      <td>diagfktp</td>
      <td>-0.003677</td>
    </tr>
  </tbody>
</table>
</div>


### Random Forest

```python
%%time
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, verbose=1, class_weight='balanced', n_jobs=-1)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
Y_predtrain = random_forest.predict(X_train)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
CM_LR_forest = confusion_matrix(y_test,Y_pred)
CR_LR_forest = classification_report(y_test,Y_pred)
CM_forest_LRtrain = confusion_matrix(y_train,Y_predtrain)
CR_forest_LRtrain = classification_report(y_train,Y_predtrain)
print("Random Forest Score: ", acc_random_forest)
print("Validation Accuracy:", accuracy_score(y_test, Y_pred))
print("Confusion Matrix:\n",CM_LR_forest)
print("Classification Report:\n",CR_LR_forest)
print("Confusion Matrix Train:\n",CM_forest_LRtrain)
print("Classification Report Train:\n",CR_forest_LRtrain)
print("Precision:", precision_score(y_test, Y_pred))
print("Recall:", recall_score(y_test, Y_pred))
print("F1:", f1_score(y_test, Y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, Y_pred))
```

    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  8.5min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed: 21.8min finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:  2.2min finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:  3.3min finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:  3.4min finished
    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:  4.8min finished
    

    Random Forest Score:  99.64
    Validation Accuracy: 0.9782530747909826
    Confusion Matrix:
     [[2777731   32447]
     [  59257 1347437]]
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.98      0.99      0.98   2810178
               1       0.98      0.96      0.97   1406694
    
        accuracy                           0.98   4216872
       macro avg       0.98      0.97      0.98   4216872
    weighted avg       0.98      0.98      0.98   4216872
    
    Confusion Matrix Train:
     [[8400182   34632]
     [  10287 4205515]]
    Classification Report Train:
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00   8434814
               1       0.99      1.00      0.99   4215802
    
        accuracy                           1.00  12650616
       macro avg       1.00      1.00      1.00  12650616
    weighted avg       1.00      1.00      1.00  12650616
    
    Precision: 0.9764857045954588
    Recall: 0.9578749891589784
    F1: 0.9670908189184009
    Area under precision (AUC) Recall: 0.949403592892738
    CPU times: total: 3h 47min 44s
    Wall time: 37min 35s
    


```python
X0 = df.drop(['id', 'label'], axis=1)
Y0 = df.label
id0 = df.id

X0 = rob_scaler.transform(X0)
```


```python
# Random Forest

Y_pred0 = random_forest.predict(X0)
CM_LR = confusion_matrix(Y0,Y_pred0)
CR_LR = classification_report(Y0,Y_pred0)

print("Confusion Matrix:\n",CM_LR)
print("Confusion Matrix:\n",CR_LR)
print("Accuracy:", accuracy_score(Y0, Y_pred0))
print("Precision:", precision_score(Y0, Y_pred0))
print("Recall:", recall_score(Y0, Y_pred0))
print("F1:", f1_score(Y0, Y_pred0))
print("Area under precision (AUC) Recall:", average_precision_score(Y0, Y_pred0))
```

    [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:  3.1min
    [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:  8.7min finished
    

    Confusion Matrix:
     [[11177914    67079]
     [   20725   136164]]
    Confusion Matrix:
                   precision    recall  f1-score   support
    
               0       1.00      0.99      1.00  11244993
               1       0.67      0.87      0.76    156889
    
        accuracy                           0.99  11401882
       macro avg       0.83      0.93      0.88  11401882
    weighted avg       0.99      0.99      0.99  11401882
    
    Accuracy: 0.9922991660499556
    Precision: 0.6699566528736537
    Recall: 0.8679002351981338
    F1: 0.7561893972210191
    Area under precision (AUC) Recall: 0.5832732189807005
    


```python
Y_pred0.shape
```




    (11401882,)




```python
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = CM_LR, display_labels = [False, True])

cm_display.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x21c180a55b0>




    
![png](/assets/images/bpjshealthkathon_files/bpjs2.png)
    



```python
df_id = pd.DataFrame({'id': id0})
df_label = pd.DataFrame({'label': Y_pred0})
```


```python
df_submission = pd.concat([df_id, df_label], axis=1)
```


```python
df_submission
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
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165666</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1010828</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>166042</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>168937</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1005899</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11401877</th>
      <td>9983563</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11401878</th>
      <td>11053870</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11401879</th>
      <td>7461049</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11401880</th>
      <td>1075162</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11401881</th>
      <td>102794</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>11401882 rows × 2 columns</p>
</div>




```python
df_submission.label.value_counts()
```




    0    11198639
    1      203243
    Name: label, dtype: int64




```python
df_submission.label.value_counts()
```




    0    11112451
    1      289431
    Name: label, dtype: int64




```python
df.label.value_counts()
```




    0    11244993
    1      156889
    Name: label, dtype: int64




```python
df_submission.to_csv('submission2.csv', index=False)
```


```python
df_test = pd.read_csv('submission2.csv', index_col='id')
```


```python
df_test.value_counts()
```




    label
    0        11198639
    1          203243
    dtype: int64




```python
df_test.head()
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
      <th>label</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165666</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1010828</th>
      <td>0</td>
    </tr>
    <tr>
      <th>166042</th>
      <td>0</td>
    </tr>
    <tr>
      <th>168937</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1005899</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Submit Answer

```python
df_answer = pd.read_csv('answer.csv')
```


```python
df_answer.drop(['Unnamed: 0'], inplace=True, axis=1)
```


```python
df_answer.to_csv('answer.csv', index=False)
```


```python
df_answer.shape
```




    (11401882, 2)


