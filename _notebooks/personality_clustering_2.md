---
title: "Personality Clustering 2"
excerpt: "Using usupervised clustering to try and cluster personalities based on a personality test"
collection: notebooks
---

# Personality Clustering
**This notebook is merely to practice, the objective is to try and group the participants into some form of cluster given their answer to personality tests. You can get the dataset from Kaggle [here](https://www.kaggle.com/tunguz/big-five-personality-test).  
From the dataset "codebook" we can read:**

This data was collected (2016-2018) through an interactive on-line personality test.
The personality test was constructed with the ["Big-Five Factor Markers" from the IPIP](https://ipip.ori.org/newBigFive5broadKey.htm).
Participants were informed that their responses would be recorded and used for research at the beginning of the test, and asked to confirm their consent at the end of the test.

The following items were presented on one page and each was rated on a five point scale using radio buttons. The order on page was was EXT1, AGR1, CSN1, EST1, OPN1, EXT2, etc.
The scale was labeled 1=Disagree, 3=Neutral, 5=Agree

| Feature | Description |
| ------ | ------ |
| EXT1 | I am the life of the party. |
| EXT2 | I don't talk a lot. |
| EXT3 | I feel comfortable around people. |
| EXT4 | I keep in the background. |
| EXT5 | I start conversations. |
| EXT6 | I have little to say. |
| EXT7 | I talk to a lot of different people at parties. |
| EXT8 | I don't like to draw attention to myself. |
| EXT9 | I don't mind being the center of attention. |
| EXT10 | I am quiet around strangers. |
| EST1 | I get stressed out easily. |
| EST2 | I am relaxed most of the time. |
| EST3 | I worry about things. |
| EST4 | I seldom feel blue. |
| EST5 | I am easily disturbed. |
| EST6 | I get upset easily. |
| EST7 | I change my mood a lot. |
| EST8 | I have frequent mood swings. |
| EST9 | I get irritated easily. |
| EST10 | I often feel blue. |
| AGR1 | I feel little concern for others. |
| AGR2 | I am interested in people. |
| AGR3 | I insult people. |
| AGR4 | I sympathize with others' feelings. |
| AGR5 | I am not interested in other people's problems. |
| AGR6 | I have a soft heart. |
| AGR7 | I am not really interested in others. |
| AGR8 | I take time out for others. |
| AGR9 | I feel others' emotions. |
| AGR10 | I make people feel at ease. |
| CSN1 | I am always prepared. |
| CSN2 | I leave my belongings around. |
| CSN3 | I pay attention to details. |
| CSN4 | I make a mess of things. |
| CSN5 | I get chores done right away. |
| CSN6 | I often forget to put things back in their proper place. |
| CSN7 | I like order. |
| CSN8 | I shirk my duties. |
| CSN9 | I follow a schedule. |
| CSN10 | I am exacting in my work. |
| OPN1 | I have a rich vocabulary. |
| OPN2 | I have difficulty understanding abstract ideas. |
| OPN3 | I have a vivid imagination. |
| OPN4 | I am not interested in abstract ideas. |
| OPN5 | I have excellent ideas. |
| OPN6 | I do not have a good imagination. |
| OPN7 | I am quick to understand things. |
| OPN8 | I use difficult words. |
| OPN9 | I spend time reflecting on things. |
| OPN10 | I am full of ideas. |

The time spent on each question is also recorded in milliseconds. These are the variables ending in _E. This was calculated by taking the time when the button for the question was clicked minus the time of the most recent other button click.

| Feature | Description |
| ------ | ------ |
| dateload | The timestamp when the survey was started. |
| screenw | The width the of user's screen in pixels |
| screenh | The height of the user's screen in pixels |
| introelapse | The time in seconds spent on the landing / intro page |
| testelapse | The time in seconds spent on the page with the survey questions |
| endelapse | The time in seconds spent on the finalization page (where the user was asked to indicate if they has answered accurately and their answers could be stored and used for research. Again: this dataset only includes users who answered "Yes" to this question, users were free to answer no and could still view their results either way) |
| IPC | The number of records from the user's IP address in the dataset. For max cleanliness, only use records where this value is 1. High values can be because of shared networks (e.g. entire universities) or multiple submissions |
| country | The country, determined by technical information (NOT ASKED AS A QUESTION) |
| lat_appx_lots_of_err | approximate latitude of user. determined by technical information, THIS IS NOT VERY ACCURATE. Read the article "How an internet mapping glitch turned a random Kansas farm into a digital hell" https://splinternews.com/how-an-internet-mapping-glitch-turned-a-random-kansas-f-1793856052 to learn about the perils of relying on this information |
| long_appx_lots_of_err | approximate longitude of user |

## The analysis
**Let's import some data and take a look at it**


```python
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score,silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch 
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.utils import resample
from matplotlib import pyplot as plt
from tqdm import tqdm

import seaborn as sb
import pandas as pd
import helpers as h
import numpy as np
import os

%matplotlib inline
```

**we'll set the folder path and seed random processes to get a deterministic environment and allow reproducibility**


```python
os.chdir('D:\Documents\Repos\personality-test')
DATA_FOLDER = 'data'
DATA_FILE = 'data-final.csv'
df = pd.read_csv(os.path.join(DATA_FOLDER,DATA_FILE),sep='\t')

SEED=7
h.seed_everything(SEED)

```

**we'll get the data and follow the advice on the codebook regarding IPC, and GPS info. we'll also delete some other metadata variables that I believe not to be useful**


```python
df = df[df['IPC']==1]
df.drop(columns=['dateload','screenw','screenh','introelapse','lat_appx_lots_of_err','long_appx_lots_of_err','IPC','endelapse'],inplace=True)
```

**let's take a peek**


```python
df.describe()
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
      <th>EXT1</th>
      <th>EXT2</th>
      <th>EXT3</th>
      <th>EXT4</th>
      <th>EXT5</th>
      <th>EXT6</th>
      <th>EXT7</th>
      <th>EXT8</th>
      <th>EXT9</th>
      <th>EXT10</th>
      <th>...</th>
      <th>OPN2_E</th>
      <th>OPN3_E</th>
      <th>OPN4_E</th>
      <th>OPN5_E</th>
      <th>OPN6_E</th>
      <th>OPN7_E</th>
      <th>OPN8_E</th>
      <th>OPN9_E</th>
      <th>OPN10_E</th>
      <th>testelapse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>695704.000000</td>
      <td>...</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
      <td>6.957040e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.577813</td>
      <td>2.826747</td>
      <td>3.221982</td>
      <td>3.194435</td>
      <td>3.230357</td>
      <td>2.414146</td>
      <td>2.703163</td>
      <td>3.442938</td>
      <td>2.940194</td>
      <td>3.591510</td>
      <td>...</td>
      <td>1.254535e+04</td>
      <td>6.691977e+03</td>
      <td>8.483657e+03</td>
      <td>6.140398e+03</td>
      <td>7.432983e+03</td>
      <td>7.915356e+03</td>
      <td>5.020315e+03</td>
      <td>5.712047e+03</td>
      <td>4.968422e+03</td>
      <td>6.502470e+02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.249742</td>
      <td>1.322513</td>
      <td>1.215858</td>
      <td>1.231347</td>
      <td>1.281609</td>
      <td>1.230538</td>
      <td>1.388454</td>
      <td>1.267326</td>
      <td>1.344401</td>
      <td>1.293504</td>
      <td>...</td>
      <td>1.371348e+06</td>
      <td>3.208362e+05</td>
      <td>4.134311e+05</td>
      <td>3.495732e+05</td>
      <td>4.739586e+05</td>
      <td>6.624335e+05</td>
      <td>1.473045e+05</td>
      <td>2.062803e+05</td>
      <td>2.682814e+05</td>
      <td>1.586134e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-2.152050e+05</td>
      <td>-4.170310e+05</td>
      <td>-7.446700e+04</td>
      <td>-7.530000e+04</td>
      <td>-7.125690e+06</td>
      <td>-5.169400e+04</td>
      <td>-1.700700e+04</td>
      <td>-9.598600e+04</td>
      <td>-3.594871e+06</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>3.051000e+03</td>
      <td>1.861000e+03</td>
      <td>2.672000e+03</td>
      <td>1.983000e+03</td>
      <td>2.360000e+03</td>
      <td>2.278000e+03</td>
      <td>2.152000e+03</td>
      <td>2.328000e+03</td>
      <td>1.485000e+03</td>
      <td>1.700000e+02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>4.225000e+03</td>
      <td>2.738000e+03</td>
      <td>3.707000e+03</td>
      <td>2.833000e+03</td>
      <td>3.320000e+03</td>
      <td>3.195000e+03</td>
      <td>3.057000e+03</td>
      <td>3.251000e+03</td>
      <td>2.193000e+03</td>
      <td>2.200000e+02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>6.122000e+03</td>
      <td>4.231000e+03</td>
      <td>5.427000e+03</td>
      <td>4.242000e+03</td>
      <td>4.888000e+03</td>
      <td>4.680000e+03</td>
      <td>4.460000e+03</td>
      <td>4.723000e+03</td>
      <td>3.343000e+03</td>
      <td>3.060000e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>1.026126e+09</td>
      <td>1.244837e+08</td>
      <td>2.015719e+08</td>
      <td>1.626808e+08</td>
      <td>2.435866e+08</td>
      <td>3.891434e+08</td>
      <td>7.803251e+07</td>
      <td>1.138087e+08</td>
      <td>9.048484e+07</td>
      <td>5.372971e+06</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 101 columns</p>
</div>




```python
h.resumetable(df)
```

    Dataset Shape: (696845, 102)
    


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
      <th>Name</th>
      <th>dtypes</th>
      <th>Missing</th>
      <th>Missing %</th>
      <th>Uniques</th>
      <th>Uniques %</th>
      <th>Mean</th>
      <th>Median</th>
      <th>First Value</th>
      <th>Second Value</th>
      <th>Min Value</th>
      <th>Max Value</th>
      <th>Entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EXT1</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.57781</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EXT2</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.82675</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EXT3</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.22198</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EXT4</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.19443</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EXT5</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.23036</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EXT6</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.41415</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>EXT7</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.70316</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EXT8</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.44294</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXT9</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.94019</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>EXT10</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.59151</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EST1</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.28911</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>11</th>
      <td>EST2</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.13997</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>12</th>
      <td>EST3</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.85977</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>13</th>
      <td>EST4</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.63759</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EST5</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.8475</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EST6</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.84609</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>EST7</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.05149</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EST8</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.69076</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EST9</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.08782</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EST10</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.8371</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>20</th>
      <td>AGR1</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.23958</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AGR2</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.82091</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.05</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AGR3</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.25857</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AGR4</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.92139</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AGR5</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.28908</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>25</th>
      <td>AGR6</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.72558</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>26</th>
      <td>AGR7</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.22173</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AGR8</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.65711</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>28</th>
      <td>AGR9</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.77659</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>29</th>
      <td>AGR10</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.57569</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CSN1</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.28096</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>31</th>
      <td>CSN2</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.97932</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.35</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CSN3</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.98052</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>33</th>
      <td>CSN4</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.64195</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CSN5</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.57766</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>35</th>
      <td>CSN6</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.85124</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CSN7</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.69992</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>37</th>
      <td>CSN8</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.47637</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.18</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CSN9</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.13918</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CSN10</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.59561</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>40</th>
      <td>OPN1</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.73453</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OPN2</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.02634</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>42</th>
      <td>OPN3</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>4.03504</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>43</th>
      <td>OPN4</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>1.95589</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>44</th>
      <td>OPN5</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.80586</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>45</th>
      <td>OPN6</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>1.87126</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>46</th>
      <td>OPN7</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>4.01744</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.85</td>
    </tr>
    <tr>
      <th>47</th>
      <td>OPN8</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.24774</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>48</th>
      <td>OPN9</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>4.18217</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th>49</th>
      <td>OPN10</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.97449</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>0.0</td>
      <td>5.000000e+00</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>50</th>
      <td>EXT1_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>70367</td>
      <td>0.100979</td>
      <td>100261</td>
      <td>7322</td>
      <td>9419</td>
      <td>7235</td>
      <td>-42958762.0</td>
      <td>2.147484e+09</td>
      <td>14.44</td>
    </tr>
    <tr>
      <th>51</th>
      <td>EXT2_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>27320</td>
      <td>0.039205</td>
      <td>8437.86</td>
      <td>3434</td>
      <td>5491</td>
      <td>3598</td>
      <td>-75632.0</td>
      <td>2.617734e+08</td>
      <td>13.02</td>
    </tr>
    <tr>
      <th>52</th>
      <td>EXT3_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>29039</td>
      <td>0.041672</td>
      <td>9707.34</td>
      <td>3512</td>
      <td>3959</td>
      <td>3315</td>
      <td>-3593866.0</td>
      <td>6.059057e+08</td>
      <td>13.02</td>
    </tr>
    <tr>
      <th>53</th>
      <td>EXT4_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>31668</td>
      <td>0.045445</td>
      <td>7941.57</td>
      <td>3473</td>
      <td>4821</td>
      <td>2564</td>
      <td>-2494907.0</td>
      <td>1.687112e+08</td>
      <td>13.09</td>
    </tr>
    <tr>
      <th>54</th>
      <td>EXT5_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>26249</td>
      <td>0.037668</td>
      <td>7700.61</td>
      <td>3031</td>
      <td>5611</td>
      <td>2976</td>
      <td>-58566.0</td>
      <td>3.510680e+08</td>
      <td>12.80</td>
    </tr>
    <tr>
      <th>55</th>
      <td>EXT6_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>25316</td>
      <td>0.036329</td>
      <td>7045.08</td>
      <td>3126</td>
      <td>2756</td>
      <td>3050</td>
      <td>-79860.0</td>
      <td>3.164906e+08</td>
      <td>12.85</td>
    </tr>
    <tr>
      <th>56</th>
      <td>EXT7_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30104</td>
      <td>0.043200</td>
      <td>8018.8</td>
      <td>4340</td>
      <td>2388</td>
      <td>4787</td>
      <td>-3594255.0</td>
      <td>9.635879e+07</td>
      <td>13.26</td>
    </tr>
    <tr>
      <th>57</th>
      <td>EXT8_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>29312</td>
      <td>0.042064</td>
      <td>7154.93</td>
      <td>3617</td>
      <td>2113</td>
      <td>3228</td>
      <td>-461138.0</td>
      <td>2.477062e+08</td>
      <td>13.09</td>
    </tr>
    <tr>
      <th>58</th>
      <td>EXT9_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>27498</td>
      <td>0.039461</td>
      <td>6136.92</td>
      <td>3651</td>
      <td>5900</td>
      <td>3465</td>
      <td>-35227.0</td>
      <td>1.803694e+08</td>
      <td>13.05</td>
    </tr>
    <tr>
      <th>59</th>
      <td>EXT10_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>24916</td>
      <td>0.035755</td>
      <td>5305.74</td>
      <td>3224</td>
      <td>4110</td>
      <td>3309</td>
      <td>-142238.0</td>
      <td>1.502521e+08</td>
      <td>12.89</td>
    </tr>
    <tr>
      <th>60</th>
      <td>EST1_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>29436</td>
      <td>0.042242</td>
      <td>8264.63</td>
      <td>3315</td>
      <td>6135</td>
      <td>9036</td>
      <td>-112165.0</td>
      <td>2.403039e+08</td>
      <td>13.08</td>
    </tr>
    <tr>
      <th>61</th>
      <td>EST2_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>29916</td>
      <td>0.042931</td>
      <td>8332.21</td>
      <td>3603</td>
      <td>4150</td>
      <td>2406</td>
      <td>-71572.0</td>
      <td>1.840717e+08</td>
      <td>13.10</td>
    </tr>
    <tr>
      <th>62</th>
      <td>EST3_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>26044</td>
      <td>0.037374</td>
      <td>7350.07</td>
      <td>2787</td>
      <td>5739</td>
      <td>3484</td>
      <td>-24118.0</td>
      <td>5.250724e+08</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>63</th>
      <td>EST4_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>42141</td>
      <td>0.060474</td>
      <td>10851.9</td>
      <td>3575</td>
      <td>6364</td>
      <td>3359</td>
      <td>-3598047.0</td>
      <td>8.800429e+08</td>
      <td>13.31</td>
    </tr>
    <tr>
      <th>64</th>
      <td>EST5_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30183</td>
      <td>0.043314</td>
      <td>7451.72</td>
      <td>3500</td>
      <td>3663</td>
      <td>3061</td>
      <td>-88286.0</td>
      <td>1.947344e+08</td>
      <td>13.05</td>
    </tr>
    <tr>
      <th>65</th>
      <td>EST6_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>28573</td>
      <td>0.041003</td>
      <td>7943.11</td>
      <td>3175</td>
      <td>5070</td>
      <td>2539</td>
      <td>-3574100.0</td>
      <td>3.464129e+08</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>66</th>
      <td>EST7_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>26708</td>
      <td>0.038327</td>
      <td>6540.14</td>
      <td>3176</td>
      <td>5709</td>
      <td>4226</td>
      <td>-2187273.0</td>
      <td>1.016919e+08</td>
      <td>12.92</td>
    </tr>
    <tr>
      <th>67</th>
      <td>EST8_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>29660</td>
      <td>0.042563</td>
      <td>5662.61</td>
      <td>2932</td>
      <td>4285</td>
      <td>2962</td>
      <td>-92455.0</td>
      <td>2.567383e+08</td>
      <td>12.86</td>
    </tr>
    <tr>
      <th>68</th>
      <td>EST9_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>24971</td>
      <td>0.035834</td>
      <td>4969.1</td>
      <td>2791</td>
      <td>2587</td>
      <td>1799</td>
      <td>-79175662.0</td>
      <td>1.838269e+08</td>
      <td>12.76</td>
    </tr>
    <tr>
      <th>69</th>
      <td>EST10_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>27402</td>
      <td>0.039323</td>
      <td>4712.35</td>
      <td>2569</td>
      <td>3997</td>
      <td>1607</td>
      <td>-43558.0</td>
      <td>8.324175e+07</td>
      <td>12.73</td>
    </tr>
    <tr>
      <th>70</th>
      <td>AGR1_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>39391</td>
      <td>0.056528</td>
      <td>17902.4</td>
      <td>4376</td>
      <td>4750</td>
      <td>2158</td>
      <td>-2757521.0</td>
      <td>1.170859e+09</td>
      <td>13.61</td>
    </tr>
    <tr>
      <th>71</th>
      <td>AGR2_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>28221</td>
      <td>0.040498</td>
      <td>8942.01</td>
      <td>3263</td>
      <td>5475</td>
      <td>2090</td>
      <td>-3592606.0</td>
      <td>4.738983e+08</td>
      <td>12.97</td>
    </tr>
    <tr>
      <th>72</th>
      <td>AGR3_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>28381</td>
      <td>0.040728</td>
      <td>6739.65</td>
      <td>3168</td>
      <td>11641</td>
      <td>2143</td>
      <td>-1795552.0</td>
      <td>1.301244e+08</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>73</th>
      <td>AGR4_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30341</td>
      <td>0.043541</td>
      <td>8468.81</td>
      <td>3174</td>
      <td>3115</td>
      <td>2807</td>
      <td>-67786.0</td>
      <td>3.365244e+08</td>
      <td>12.98</td>
    </tr>
    <tr>
      <th>74</th>
      <td>AGR5_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30785</td>
      <td>0.044178</td>
      <td>8746</td>
      <td>4056</td>
      <td>3207</td>
      <td>3422</td>
      <td>-20294.0</td>
      <td>1.563917e+08</td>
      <td>13.18</td>
    </tr>
    <tr>
      <th>75</th>
      <td>AGR6_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>26392</td>
      <td>0.037874</td>
      <td>5877.35</td>
      <td>2880</td>
      <td>3260</td>
      <td>5324</td>
      <td>-247504.0</td>
      <td>1.018158e+08</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>76</th>
      <td>AGR7_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>28487</td>
      <td>0.040880</td>
      <td>7529.42</td>
      <td>3683</td>
      <td>10235</td>
      <td>4494</td>
      <td>-65423.0</td>
      <td>2.518615e+08</td>
      <td>13.05</td>
    </tr>
    <tr>
      <th>77</th>
      <td>AGR8_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>31762</td>
      <td>0.045580</td>
      <td>9288.41</td>
      <td>3844</td>
      <td>5897</td>
      <td>3627</td>
      <td>-764938.0</td>
      <td>1.367497e+09</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>78</th>
      <td>AGR9_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>25205</td>
      <td>0.036170</td>
      <td>5197.64</td>
      <td>3133</td>
      <td>1758</td>
      <td>1850</td>
      <td>-527846.0</td>
      <td>6.275748e+07</td>
      <td>12.85</td>
    </tr>
    <tr>
      <th>79</th>
      <td>AGR10_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30066</td>
      <td>0.043146</td>
      <td>5709.94</td>
      <td>3334</td>
      <td>3081</td>
      <td>1747</td>
      <td>-85674.0</td>
      <td>8.158242e+07</td>
      <td>12.97</td>
    </tr>
    <tr>
      <th>80</th>
      <td>CSN1_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30939</td>
      <td>0.044399</td>
      <td>13067.6</td>
      <td>3569</td>
      <td>6602</td>
      <td>5163</td>
      <td>-3590638.0</td>
      <td>7.726592e+08</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>81</th>
      <td>CSN2_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>34702</td>
      <td>0.049799</td>
      <td>10914.2</td>
      <td>4275</td>
      <td>5457</td>
      <td>5240</td>
      <td>-35996486.0</td>
      <td>2.637374e+08</td>
      <td>13.35</td>
    </tr>
    <tr>
      <th>82</th>
      <td>CSN3_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>27477</td>
      <td>0.039431</td>
      <td>9202.25</td>
      <td>3193</td>
      <td>1569</td>
      <td>7208</td>
      <td>-94464.0</td>
      <td>1.100335e+09</td>
      <td>12.89</td>
    </tr>
    <tr>
      <th>83</th>
      <td>CSN4_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>30055</td>
      <td>0.043130</td>
      <td>7964.61</td>
      <td>3336</td>
      <td>2129</td>
      <td>2783</td>
      <td>-50476.0</td>
      <td>2.690842e+08</td>
      <td>12.99</td>
    </tr>
    <tr>
      <th>84</th>
      <td>CSN5_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>36242</td>
      <td>0.052009</td>
      <td>9445.28</td>
      <td>3585</td>
      <td>3762</td>
      <td>4103</td>
      <td>-3512740.0</td>
      <td>9.586233e+08</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>85</th>
      <td>CSN6_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>31181</td>
      <td>0.044746</td>
      <td>10102.5</td>
      <td>4359</td>
      <td>4420</td>
      <td>3431</td>
      <td>-74245.0</td>
      <td>4.432097e+08</td>
      <td>13.28</td>
    </tr>
    <tr>
      <th>86</th>
      <td>CSN7_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>24481</td>
      <td>0.035131</td>
      <td>5343.7</td>
      <td>2913</td>
      <td>9382</td>
      <td>3347</td>
      <td>-30016.0</td>
      <td>8.482811e+07</td>
      <td>12.75</td>
    </tr>
    <tr>
      <th>87</th>
      <td>CSN8_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>47792</td>
      <td>0.068583</td>
      <td>11106.9</td>
      <td>3733</td>
      <td>5286</td>
      <td>2399</td>
      <td>-177880.0</td>
      <td>2.503232e+08</td>
      <td>13.69</td>
    </tr>
    <tr>
      <th>88</th>
      <td>CSN9_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>24405</td>
      <td>0.035022</td>
      <td>5135.27</td>
      <td>2915</td>
      <td>4983</td>
      <td>3360</td>
      <td>-29167.0</td>
      <td>8.749788e+07</td>
      <td>12.76</td>
    </tr>
    <tr>
      <th>89</th>
      <td>CSN10_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>40494</td>
      <td>0.058110</td>
      <td>9228.05</td>
      <td>3923</td>
      <td>6339</td>
      <td>5595</td>
      <td>-14988.0</td>
      <td>3.380158e+08</td>
      <td>13.45</td>
    </tr>
    <tr>
      <th>90</th>
      <td>OPN1_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>26670</td>
      <td>0.038272</td>
      <td>8881.47</td>
      <td>3023</td>
      <td>3146</td>
      <td>2624</td>
      <td>-53927742.0</td>
      <td>6.750470e+08</td>
      <td>12.87</td>
    </tr>
    <tr>
      <th>91</th>
      <td>OPN2_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>36984</td>
      <td>0.053073</td>
      <td>12545.3</td>
      <td>4225</td>
      <td>4067</td>
      <td>4985</td>
      <td>-215205.0</td>
      <td>1.026126e+09</td>
      <td>13.30</td>
    </tr>
    <tr>
      <th>92</th>
      <td>OPN3_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>31354</td>
      <td>0.044994</td>
      <td>6691.98</td>
      <td>2738</td>
      <td>2959</td>
      <td>1684</td>
      <td>-417031.0</td>
      <td>1.244837e+08</td>
      <td>12.86</td>
    </tr>
    <tr>
      <th>93</th>
      <td>OPN4_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>34083</td>
      <td>0.048910</td>
      <td>8483.66</td>
      <td>3707</td>
      <td>3411</td>
      <td>3026</td>
      <td>-74467.0</td>
      <td>2.015719e+08</td>
      <td>13.13</td>
    </tr>
    <tr>
      <th>94</th>
      <td>OPN5_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>25447</td>
      <td>0.036517</td>
      <td>6140.4</td>
      <td>2833</td>
      <td>2170</td>
      <td>4742</td>
      <td>-75300.0</td>
      <td>1.626808e+08</td>
      <td>12.74</td>
    </tr>
    <tr>
      <th>95</th>
      <td>OPN6_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>27514</td>
      <td>0.039484</td>
      <td>7432.98</td>
      <td>3320</td>
      <td>4920</td>
      <td>3336</td>
      <td>-7125690.0</td>
      <td>2.435866e+08</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>96</th>
      <td>OPN7_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>27040</td>
      <td>0.038803</td>
      <td>7915.36</td>
      <td>3195</td>
      <td>4436</td>
      <td>2718</td>
      <td>-51694.0</td>
      <td>3.891434e+08</td>
      <td>12.86</td>
    </tr>
    <tr>
      <th>97</th>
      <td>OPN8_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>24232</td>
      <td>0.034774</td>
      <td>5020.32</td>
      <td>3057</td>
      <td>3116</td>
      <td>3374</td>
      <td>-17007.0</td>
      <td>7.803251e+07</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>98</th>
      <td>OPN9_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>28092</td>
      <td>0.040313</td>
      <td>5712.05</td>
      <td>3251</td>
      <td>2992</td>
      <td>3096</td>
      <td>-95986.0</td>
      <td>1.138087e+08</td>
      <td>12.89</td>
    </tr>
    <tr>
      <th>99</th>
      <td>OPN10_E</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>23097</td>
      <td>0.033145</td>
      <td>4968.42</td>
      <td>2193</td>
      <td>4354</td>
      <td>3019</td>
      <td>-3594871.0</td>
      <td>9.048484e+07</td>
      <td>12.46</td>
    </tr>
    <tr>
      <th>100</th>
      <td>testelapse</td>
      <td>float64</td>
      <td>1141</td>
      <td>0.001637</td>
      <td>8741</td>
      <td>0.012544</td>
      <td>650.247</td>
      <td>220</td>
      <td>234</td>
      <td>179</td>
      <td>1.0</td>
      <td>5.372971e+06</td>
      <td>8.98</td>
    </tr>
    <tr>
      <th>101</th>
      <td>country</td>
      <td>object</td>
      <td>67</td>
      <td>0.000096</td>
      <td>221</td>
      <td>0.000317</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>MY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
</div>


    100%|████████████████████████████████████████████████████████████████████████████████| 102/102 [07:51<00:00,  4.62s/it]
    


    
![png](output_9_3.png)
    


**for the categorical variables we see different amounts of skewness on each one, usually left-skewed as it seems option 0 is not much used in many of them. Onthe time variables, we seem to have some big outliers that makes the graph look like all data is on 0. However it is good that we only have few outliers and data is concentrated on more normal values. People shouldn't have to take 1e8 seconds to answer a question.**

**we have 1141 nulls that we can drop**


```python
df.dropna(inplace=True)
```

**We also have 221 unique countries, however, the last graph just above this cell already let's us know that US has a desproportionate amount of data, about half the dataset. We could try and balance it out but that would halve our dataset so we'll see what comes out of it.  
Let's try to visualize it better**


```python
TOP_N = 5
cnt = df['country'].value_counts(normalize=True)[:TOP_N]
cnt = cnt.append(pd.Series(df['country'].value_counts(normalize=True)[TOP_N:].sum(),index=['Other']))
print(cnt)
plt.style.use('seaborn-notebook')
cnt.plot(kind='pie',ylabel='')
```

    US       0.496384
    GB       0.071539
    CA       0.062997
    AU       0.049810
    DE       0.017708
    Other    0.301562
    dtype: float64
    




    <AxesSubplot:>




    
![png](output_14_2.png)
    


**from the table we can aslo see that there are negative time amounts, which should't be possible. let's clean those out**


```python
for col in tqdm(df.drop(columns=['country']).columns):
    df = df[df[col]>=0]
```

    100%|████████████████████████████████████████████████████████████████████████████████| 101/101 [00:37<00:00,  2.66it/s]
    


```python
h.resumetable(df,visualize=False)
```

    Dataset Shape: (695225, 102)
    


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
      <th>Name</th>
      <th>dtypes</th>
      <th>Missing</th>
      <th>Missing %</th>
      <th>Uniques</th>
      <th>Uniques %</th>
      <th>Mean</th>
      <th>Median</th>
      <th>First Value</th>
      <th>Second Value</th>
      <th>Min Value</th>
      <th>Max Value</th>
      <th>Entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EXT1</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.5777</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EXT2</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.82684</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EXT3</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.2219</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EXT4</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.19453</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2.27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EXT5</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.23028</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EXT6</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.4142</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>EXT7</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.70305</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EXT8</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.44295</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXT9</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.94016</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>EXT10</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.59162</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EST1</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.2891</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>11</th>
      <td>EST2</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.14004</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>12</th>
      <td>EST3</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.85976</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>13</th>
      <td>EST4</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.63757</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EST5</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.84748</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EST6</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.84613</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>EST7</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.0515</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EST8</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.69082</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EST9</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.08781</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EST10</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.83712</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>20</th>
      <td>AGR1</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.23959</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AGR2</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.82091</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2.05</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AGR3</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.25861</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AGR4</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.92139</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AGR5</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.28908</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>25</th>
      <td>AGR6</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.72551</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>26</th>
      <td>AGR7</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.22175</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AGR8</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.65712</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>28</th>
      <td>AGR9</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.77656</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>29</th>
      <td>AGR10</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.57561</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CSN1</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.2809</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>31</th>
      <td>CSN2</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.97944</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>2.35</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CSN3</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.98043</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>33</th>
      <td>CSN4</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.6421</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CSN5</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.57754</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>35</th>
      <td>CSN6</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.85137</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CSN7</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.69986</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>37</th>
      <td>CSN8</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.47637</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.18</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CSN9</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.13912</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CSN10</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.59556</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>40</th>
      <td>OPN1</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.7346</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OPN2</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>2.02635</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>42</th>
      <td>OPN3</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>4.03509</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>43</th>
      <td>OPN4</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>1.95589</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>44</th>
      <td>OPN5</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.8058</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>45</th>
      <td>OPN6</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>1.8713</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>46</th>
      <td>OPN7</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>4.01746</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1.85</td>
    </tr>
    <tr>
      <th>47</th>
      <td>OPN8</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.24783</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>2.28</td>
    </tr>
    <tr>
      <th>48</th>
      <td>OPN9</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>4.18219</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th>49</th>
      <td>OPN10</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000009</td>
      <td>3.97446</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>50</th>
      <td>EXT1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>70313</td>
      <td>0.101137</td>
      <td>100245</td>
      <td>7323</td>
      <td>9419</td>
      <td>7235</td>
      <td>0</td>
      <td>2.14748e+09</td>
      <td>14.44</td>
    </tr>
    <tr>
      <th>51</th>
      <td>EXT2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>27306</td>
      <td>0.039276</td>
      <td>8440.55</td>
      <td>3434</td>
      <td>5491</td>
      <td>3598</td>
      <td>0</td>
      <td>2.61773e+08</td>
      <td>13.02</td>
    </tr>
    <tr>
      <th>52</th>
      <td>EXT3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>29023</td>
      <td>0.041746</td>
      <td>9715.5</td>
      <td>3512</td>
      <td>3959</td>
      <td>3315</td>
      <td>0</td>
      <td>6.05906e+08</td>
      <td>13.02</td>
    </tr>
    <tr>
      <th>53</th>
      <td>EXT4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>31640</td>
      <td>0.045510</td>
      <td>7946.64</td>
      <td>3473</td>
      <td>4821</td>
      <td>2564</td>
      <td>0</td>
      <td>1.68711e+08</td>
      <td>13.09</td>
    </tr>
    <tr>
      <th>54</th>
      <td>EXT5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>26225</td>
      <td>0.037722</td>
      <td>7700.84</td>
      <td>3032</td>
      <td>5611</td>
      <td>2976</td>
      <td>0</td>
      <td>3.51068e+08</td>
      <td>12.80</td>
    </tr>
    <tr>
      <th>55</th>
      <td>EXT6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>25298</td>
      <td>0.036388</td>
      <td>7046.89</td>
      <td>3126</td>
      <td>2756</td>
      <td>3050</td>
      <td>0</td>
      <td>3.16491e+08</td>
      <td>12.85</td>
    </tr>
    <tr>
      <th>56</th>
      <td>EXT7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30083</td>
      <td>0.043271</td>
      <td>8023.23</td>
      <td>4339</td>
      <td>2388</td>
      <td>4787</td>
      <td>0</td>
      <td>9.63588e+07</td>
      <td>13.25</td>
    </tr>
    <tr>
      <th>57</th>
      <td>EXT8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>29291</td>
      <td>0.042132</td>
      <td>7156.65</td>
      <td>3616</td>
      <td>2113</td>
      <td>3228</td>
      <td>0</td>
      <td>2.47706e+08</td>
      <td>13.09</td>
    </tr>
    <tr>
      <th>58</th>
      <td>EXT9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>27483</td>
      <td>0.039531</td>
      <td>6137.38</td>
      <td>3651</td>
      <td>5900</td>
      <td>3465</td>
      <td>0</td>
      <td>1.80369e+08</td>
      <td>13.05</td>
    </tr>
    <tr>
      <th>59</th>
      <td>EXT10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>24898</td>
      <td>0.035813</td>
      <td>5306.65</td>
      <td>3224</td>
      <td>4110</td>
      <td>3309</td>
      <td>0</td>
      <td>1.50252e+08</td>
      <td>12.88</td>
    </tr>
    <tr>
      <th>60</th>
      <td>EST1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>29421</td>
      <td>0.042319</td>
      <td>8267.4</td>
      <td>3315</td>
      <td>6135</td>
      <td>9036</td>
      <td>0</td>
      <td>2.40304e+08</td>
      <td>13.08</td>
    </tr>
    <tr>
      <th>61</th>
      <td>EST2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>29892</td>
      <td>0.042996</td>
      <td>8333.54</td>
      <td>3602</td>
      <td>4150</td>
      <td>2406</td>
      <td>0</td>
      <td>1.84072e+08</td>
      <td>13.09</td>
    </tr>
    <tr>
      <th>62</th>
      <td>EST3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>26030</td>
      <td>0.037441</td>
      <td>7352.24</td>
      <td>2787</td>
      <td>5739</td>
      <td>3484</td>
      <td>0</td>
      <td>5.25072e+08</td>
      <td>12.77</td>
    </tr>
    <tr>
      <th>63</th>
      <td>EST4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>42116</td>
      <td>0.060579</td>
      <td>10859.1</td>
      <td>3575</td>
      <td>6364</td>
      <td>3359</td>
      <td>0</td>
      <td>8.80043e+08</td>
      <td>13.31</td>
    </tr>
    <tr>
      <th>64</th>
      <td>EST5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30163</td>
      <td>0.043386</td>
      <td>7453.64</td>
      <td>3500</td>
      <td>3663</td>
      <td>3061</td>
      <td>0</td>
      <td>1.94734e+08</td>
      <td>13.05</td>
    </tr>
    <tr>
      <th>65</th>
      <td>EST6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>28561</td>
      <td>0.041082</td>
      <td>7950.72</td>
      <td>3175</td>
      <td>5070</td>
      <td>2539</td>
      <td>0</td>
      <td>3.46413e+08</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>66</th>
      <td>EST7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>26686</td>
      <td>0.038385</td>
      <td>6543.86</td>
      <td>3176</td>
      <td>5709</td>
      <td>4226</td>
      <td>0</td>
      <td>1.01692e+08</td>
      <td>12.92</td>
    </tr>
    <tr>
      <th>67</th>
      <td>EST8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>29639</td>
      <td>0.042632</td>
      <td>5663.89</td>
      <td>2932</td>
      <td>4285</td>
      <td>2962</td>
      <td>0</td>
      <td>2.56738e+08</td>
      <td>12.86</td>
    </tr>
    <tr>
      <th>68</th>
      <td>EST9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>24948</td>
      <td>0.035885</td>
      <td>5083.97</td>
      <td>2791</td>
      <td>2587</td>
      <td>1799</td>
      <td>0</td>
      <td>1.83827e+08</td>
      <td>12.76</td>
    </tr>
    <tr>
      <th>69</th>
      <td>EST10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>27390</td>
      <td>0.039397</td>
      <td>4712.96</td>
      <td>2569</td>
      <td>3997</td>
      <td>1607</td>
      <td>0</td>
      <td>8.32418e+07</td>
      <td>12.73</td>
    </tr>
    <tr>
      <th>70</th>
      <td>AGR1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>39360</td>
      <td>0.056615</td>
      <td>17899.3</td>
      <td>4376</td>
      <td>4750</td>
      <td>2158</td>
      <td>0</td>
      <td>1.17086e+09</td>
      <td>13.61</td>
    </tr>
    <tr>
      <th>71</th>
      <td>AGR2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>28202</td>
      <td>0.040565</td>
      <td>8950.46</td>
      <td>3263</td>
      <td>5475</td>
      <td>2090</td>
      <td>0</td>
      <td>4.73898e+08</td>
      <td>12.97</td>
    </tr>
    <tr>
      <th>72</th>
      <td>AGR3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>28363</td>
      <td>0.040797</td>
      <td>6742.53</td>
      <td>3168</td>
      <td>11641</td>
      <td>2143</td>
      <td>0</td>
      <td>1.30124e+08</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>73</th>
      <td>AGR4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30313</td>
      <td>0.043602</td>
      <td>8471</td>
      <td>3174</td>
      <td>3115</td>
      <td>2807</td>
      <td>0</td>
      <td>3.36524e+08</td>
      <td>12.98</td>
    </tr>
    <tr>
      <th>74</th>
      <td>AGR5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30764</td>
      <td>0.044250</td>
      <td>8746.91</td>
      <td>4056</td>
      <td>3207</td>
      <td>3422</td>
      <td>0</td>
      <td>1.56392e+08</td>
      <td>13.18</td>
    </tr>
    <tr>
      <th>75</th>
      <td>AGR6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>26377</td>
      <td>0.037940</td>
      <td>5878.7</td>
      <td>2880</td>
      <td>3260</td>
      <td>5324</td>
      <td>0</td>
      <td>1.01816e+08</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>76</th>
      <td>AGR7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>28472</td>
      <td>0.040954</td>
      <td>7531.02</td>
      <td>3683</td>
      <td>10235</td>
      <td>4494</td>
      <td>0</td>
      <td>2.51861e+08</td>
      <td>13.05</td>
    </tr>
    <tr>
      <th>77</th>
      <td>AGR8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>31739</td>
      <td>0.045653</td>
      <td>9292.07</td>
      <td>3844</td>
      <td>5897</td>
      <td>3627</td>
      <td>0</td>
      <td>1.3675e+09</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>78</th>
      <td>AGR9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>25186</td>
      <td>0.036227</td>
      <td>5198.95</td>
      <td>3133</td>
      <td>1758</td>
      <td>1850</td>
      <td>0</td>
      <td>6.27575e+07</td>
      <td>12.85</td>
    </tr>
    <tr>
      <th>79</th>
      <td>AGR10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30053</td>
      <td>0.043228</td>
      <td>5711.06</td>
      <td>3334</td>
      <td>3081</td>
      <td>1747</td>
      <td>0</td>
      <td>8.15824e+07</td>
      <td>12.97</td>
    </tr>
    <tr>
      <th>80</th>
      <td>CSN1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30923</td>
      <td>0.044479</td>
      <td>13083.1</td>
      <td>3569</td>
      <td>6602</td>
      <td>5163</td>
      <td>0</td>
      <td>7.72659e+08</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>81</th>
      <td>CSN2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>34681</td>
      <td>0.049885</td>
      <td>10969.1</td>
      <td>4274</td>
      <td>5457</td>
      <td>5240</td>
      <td>0</td>
      <td>2.63737e+08</td>
      <td>13.35</td>
    </tr>
    <tr>
      <th>82</th>
      <td>CSN3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>27458</td>
      <td>0.039495</td>
      <td>9205.65</td>
      <td>3193</td>
      <td>1569</td>
      <td>7208</td>
      <td>0</td>
      <td>1.10033e+09</td>
      <td>12.89</td>
    </tr>
    <tr>
      <th>83</th>
      <td>CSN4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>30033</td>
      <td>0.043199</td>
      <td>7958.02</td>
      <td>3336</td>
      <td>2129</td>
      <td>2783</td>
      <td>0</td>
      <td>2.69084e+08</td>
      <td>12.99</td>
    </tr>
    <tr>
      <th>84</th>
      <td>CSN5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>36227</td>
      <td>0.052108</td>
      <td>9452.12</td>
      <td>3584</td>
      <td>3762</td>
      <td>4103</td>
      <td>0</td>
      <td>9.58623e+08</td>
      <td>13.17</td>
    </tr>
    <tr>
      <th>85</th>
      <td>CSN6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>31161</td>
      <td>0.044821</td>
      <td>10104.1</td>
      <td>4359</td>
      <td>4420</td>
      <td>3431</td>
      <td>0</td>
      <td>4.4321e+08</td>
      <td>13.28</td>
    </tr>
    <tr>
      <th>86</th>
      <td>CSN7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>24468</td>
      <td>0.035194</td>
      <td>5344.83</td>
      <td>2913</td>
      <td>9382</td>
      <td>3347</td>
      <td>0</td>
      <td>8.48281e+07</td>
      <td>12.75</td>
    </tr>
    <tr>
      <th>87</th>
      <td>CSN8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>47771</td>
      <td>0.068713</td>
      <td>11109.6</td>
      <td>3733</td>
      <td>5286</td>
      <td>2399</td>
      <td>0</td>
      <td>2.50323e+08</td>
      <td>13.69</td>
    </tr>
    <tr>
      <th>88</th>
      <td>CSN9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>24389</td>
      <td>0.035081</td>
      <td>5135.82</td>
      <td>2915</td>
      <td>4983</td>
      <td>3360</td>
      <td>0</td>
      <td>8.74979e+07</td>
      <td>12.76</td>
    </tr>
    <tr>
      <th>89</th>
      <td>CSN10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>40470</td>
      <td>0.058211</td>
      <td>9229.25</td>
      <td>3922</td>
      <td>6339</td>
      <td>5595</td>
      <td>0</td>
      <td>3.38016e+08</td>
      <td>13.44</td>
    </tr>
    <tr>
      <th>90</th>
      <td>OPN1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>26653</td>
      <td>0.038337</td>
      <td>8962.5</td>
      <td>3023</td>
      <td>3146</td>
      <td>2624</td>
      <td>0</td>
      <td>6.75047e+08</td>
      <td>12.87</td>
    </tr>
    <tr>
      <th>91</th>
      <td>OPN2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>36963</td>
      <td>0.053167</td>
      <td>12550.2</td>
      <td>4225</td>
      <td>4067</td>
      <td>4985</td>
      <td>0</td>
      <td>1.02613e+09</td>
      <td>13.30</td>
    </tr>
    <tr>
      <th>92</th>
      <td>OPN3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>31331</td>
      <td>0.045066</td>
      <td>6693.28</td>
      <td>2738</td>
      <td>2959</td>
      <td>1684</td>
      <td>0</td>
      <td>1.24484e+08</td>
      <td>12.86</td>
    </tr>
    <tr>
      <th>93</th>
      <td>OPN4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>34061</td>
      <td>0.048993</td>
      <td>8482.78</td>
      <td>3707</td>
      <td>3411</td>
      <td>3026</td>
      <td>0</td>
      <td>2.01572e+08</td>
      <td>13.13</td>
    </tr>
    <tr>
      <th>94</th>
      <td>OPN5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>25434</td>
      <td>0.036584</td>
      <td>6132.35</td>
      <td>2833</td>
      <td>2170</td>
      <td>4742</td>
      <td>0</td>
      <td>1.62681e+08</td>
      <td>12.74</td>
    </tr>
    <tr>
      <th>95</th>
      <td>OPN6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>27489</td>
      <td>0.039540</td>
      <td>7445.71</td>
      <td>3320</td>
      <td>4920</td>
      <td>3336</td>
      <td>0</td>
      <td>2.43587e+08</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>96</th>
      <td>OPN7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>27023</td>
      <td>0.038869</td>
      <td>7917.11</td>
      <td>3195</td>
      <td>4436</td>
      <td>2718</td>
      <td>0</td>
      <td>3.89143e+08</td>
      <td>12.86</td>
    </tr>
    <tr>
      <th>97</th>
      <td>OPN8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>24220</td>
      <td>0.034838</td>
      <td>5020.94</td>
      <td>3057</td>
      <td>3116</td>
      <td>3374</td>
      <td>0</td>
      <td>7.80325e+07</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>98</th>
      <td>OPN9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>28075</td>
      <td>0.040383</td>
      <td>5713.16</td>
      <td>3251</td>
      <td>2992</td>
      <td>3096</td>
      <td>0</td>
      <td>1.13809e+08</td>
      <td>12.89</td>
    </tr>
    <tr>
      <th>99</th>
      <td>OPN10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>23083</td>
      <td>0.033202</td>
      <td>4975</td>
      <td>2193</td>
      <td>4354</td>
      <td>3019</td>
      <td>0</td>
      <td>9.04848e+07</td>
      <td>12.46</td>
    </tr>
    <tr>
      <th>100</th>
      <td>testelapse</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>8735</td>
      <td>0.012564</td>
      <td>650.277</td>
      <td>220</td>
      <td>234</td>
      <td>179</td>
      <td>1</td>
      <td>5.37297e+06</td>
      <td>8.98</td>
    </tr>
    <tr>
      <th>101</th>
      <td>country</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>221</td>
      <td>0.000318</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>MY</td>
      <td>AD</td>
      <td>ZW</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
</div>


## We'll create a couple of datasets so we can run them through our model and see what is convenient

### First dataset will have all the variables, we'll scale *everything* (even though scaling the categorical features can be kind of pointless, we'll give it a go) and we'll one hot encode the countries


```python
# We have some big outliers on time variables so we could use robust scaling for those and normal for the other ones
# set the structure
df_all_scaled = pd.get_dummies(df['country'].to_frame())
# go through each column and scale
for col in tqdm(df.columns.values):
    if df[col].dtype != 'object':
        if col.endswith('_E') | col.endswith('e'):
            df_all_scaled[col] = RobustScaler().fit_transform(df[col].to_numpy().reshape(-1,1))
        else:
            df_all_scaled[col] = StandardScaler().fit_transform(df[col].to_numpy().reshape(-1,1))
```

    100%|████████████████████████████████████████████████████████████████████████████████| 102/102 [00:03<00:00, 31.56it/s]
    


```python
# let's check the shape and see if we didn't introduce any erros like nulls just in case, and let's take a look at the robust scaled time features
display(df_all_scaled.shape)
display(df_all_scaled.isna().sum().sum())
```


    (695225, 322)



    0


### Second dataset will also have all the variables, we'll scale *everything*, but! we'll label encode country


```python
df_lbl_enc_scaled = pd.DataFrame(LabelEncoder().fit_transform(df['country']),columns=['country'])
for col in tqdm(df.columns.values):
    if df[col].dtype != 'object':
        if col.endswith('_E') | col.endswith('e'):
            df_lbl_enc_scaled[col] = RobustScaler().fit_transform(df[col].to_numpy().reshape(-1,1))
        else:
            df_lbl_enc_scaled[col] = StandardScaler().fit_transform(df[col].to_numpy().reshape(-1,1))

display(df_lbl_enc_scaled.shape)
display(df_lbl_enc_scaled.isna().sum().sum())
```

    100%|████████████████████████████████████████████████████████████████████████████████| 102/102 [00:03<00:00, 31.81it/s]
    


    (695225, 102)



    0


### Third dataset will also have all the variables, but we'll forego scaling non-time variables. After all scaling doesn't make that much sense since they are actually label encoded categorical variables. we'll label encode country as well. And we'll replace time outliers with median value before scaling.


```python
# first the imputations
time_cols = [x for x in df.columns.values if x.endswith('_E') | x.endswith('e')]
q75 = df[time_cols].quantile(.75)
med = df[time_cols].median()
imputed = pd.DataFrame()
for col in time_cols:
    imputed[col] = np.where(df[col] > q75[col], med[col], df[col])
display(imputed.shape)
display(imputed.isna().sum().sum())
```


    (695225, 51)



    0



```python
df_not_all_scaled = pd.DataFrame(LabelEncoder().fit_transform(df['country']),columns=['country'])
for col in tqdm(df.columns.values):
    if df[col].dtype != 'object':
        if col.endswith('_E') | col.endswith('e'):
            df_not_all_scaled[col] = RobustScaler().fit_transform(imputed[col].to_numpy().reshape(-1,1))
        else:
            df_not_all_scaled[col] = df[col].to_numpy().reshape(-1,1)

display(df_not_all_scaled.shape)
display(df_not_all_scaled.isna().sum().sum())
h.resumetable(df_not_all_scaled[time_cols])
```

    100%|████████████████████████████████████████████████████████████████████████████████| 102/102 [00:02<00:00, 44.25it/s]
    


    (695225, 102)



    0


    Dataset Shape: (695225, 51)
    


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
      <th>Name</th>
      <th>dtypes</th>
      <th>Missing</th>
      <th>Missing %</th>
      <th>Uniques</th>
      <th>Uniques %</th>
      <th>Mean</th>
      <th>Median</th>
      <th>First Value</th>
      <th>Second Value</th>
      <th>Min Value</th>
      <th>Max Value</th>
      <th>Entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EXT1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>11484</td>
      <td>0.016518</td>
      <td>-0.325244</td>
      <td>0.0</td>
      <td>0.839407</td>
      <td>-0.035242</td>
      <td>-2.932719</td>
      <td>1.840208</td>
      <td>10.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EXT2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4839</td>
      <td>0.006960</td>
      <td>-0.372262</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.158607</td>
      <td>-3.321083</td>
      <td>1.594778</td>
      <td>9.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EXT3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4914</td>
      <td>0.007068</td>
      <td>-0.370333</td>
      <td>0.0</td>
      <td>0.436098</td>
      <td>-0.192195</td>
      <td>-3.426341</td>
      <td>1.609756</td>
      <td>9.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EXT4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4967</td>
      <td>0.007144</td>
      <td>-0.367894</td>
      <td>0.0</td>
      <td>1.264540</td>
      <td>-0.852720</td>
      <td>-3.257974</td>
      <td>1.618199</td>
      <td>9.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EXT5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4265</td>
      <td>0.006135</td>
      <td>-0.365339</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.063063</td>
      <td>-3.414414</td>
      <td>1.665541</td>
      <td>9.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EXT6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4387</td>
      <td>0.006310</td>
      <td>-0.366777</td>
      <td>0.0</td>
      <td>-0.396146</td>
      <td>-0.081370</td>
      <td>-3.346895</td>
      <td>1.597430</td>
      <td>9.61</td>
    </tr>
    <tr>
      <th>6</th>
      <td>EXT7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6006</td>
      <td>0.008639</td>
      <td>-0.388224</td>
      <td>0.0</td>
      <td>-1.518288</td>
      <td>0.348638</td>
      <td>-3.376654</td>
      <td>1.484825</td>
      <td>9.95</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EXT8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5124</td>
      <td>0.007370</td>
      <td>-0.369155</td>
      <td>0.0</td>
      <td>-1.385253</td>
      <td>-0.357604</td>
      <td>-3.332719</td>
      <td>1.606452</td>
      <td>9.79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXT9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5109</td>
      <td>0.007349</td>
      <td>-0.375779</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.176471</td>
      <td>-3.463947</td>
      <td>1.611954</td>
      <td>9.77</td>
    </tr>
    <tr>
      <th>9</th>
      <td>EXT10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4491</td>
      <td>0.006460</td>
      <td>-0.376442</td>
      <td>0.0</td>
      <td>0.919087</td>
      <td>0.088174</td>
      <td>-3.344398</td>
      <td>1.543568</td>
      <td>9.65</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EST1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4889</td>
      <td>0.007032</td>
      <td>-0.354286</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.055300</td>
      <td>1.670968</td>
      <td>9.75</td>
    </tr>
    <tr>
      <th>11</th>
      <td>EST2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5151</td>
      <td>0.007409</td>
      <td>-0.370035</td>
      <td>0.0</td>
      <td>0.492363</td>
      <td>-1.074573</td>
      <td>-3.236298</td>
      <td>1.607367</td>
      <td>9.77</td>
    </tr>
    <tr>
      <th>12</th>
      <td>EST3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>3990</td>
      <td>0.005739</td>
      <td>-0.356719</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.816159</td>
      <td>-3.263466</td>
      <td>1.663934</td>
      <td>9.52</td>
    </tr>
    <tr>
      <th>13</th>
      <td>EST4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5380</td>
      <td>0.007739</td>
      <td>-0.353125</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.192000</td>
      <td>-3.177778</td>
      <td>1.808889</td>
      <td>9.81</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EST5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4936</td>
      <td>0.007100</td>
      <td>-0.366418</td>
      <td>0.0</td>
      <td>0.157640</td>
      <td>-0.424565</td>
      <td>-3.384913</td>
      <td>1.626692</td>
      <td>9.74</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EST6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4585</td>
      <td>0.006595</td>
      <td>-0.361394</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.654995</td>
      <td>-3.269825</td>
      <td>1.685891</td>
      <td>9.63</td>
    </tr>
    <tr>
      <th>16</th>
      <td>EST7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4504</td>
      <td>0.006478</td>
      <td>-0.365225</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.084711</td>
      <td>-3.280992</td>
      <td>1.608471</td>
      <td>9.65</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EST8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4194</td>
      <td>0.006033</td>
      <td>-0.360998</td>
      <td>0.0</td>
      <td>1.571429</td>
      <td>0.034843</td>
      <td>-3.405343</td>
      <td>1.727062</td>
      <td>9.55</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EST9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>3988</td>
      <td>0.005736</td>
      <td>-0.354944</td>
      <td>0.0</td>
      <td>-0.239156</td>
      <td>-1.162954</td>
      <td>-3.271981</td>
      <td>1.665885</td>
      <td>9.52</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EST10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>3749</td>
      <td>0.005392</td>
      <td>-0.345658</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-1.196517</td>
      <td>-3.195274</td>
      <td>1.730100</td>
      <td>9.44</td>
    </tr>
    <tr>
      <th>20</th>
      <td>AGR1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6597</td>
      <td>0.009489</td>
      <td>-0.364847</td>
      <td>0.0</td>
      <td>0.255814</td>
      <td>-1.517100</td>
      <td>-2.993160</td>
      <td>1.687415</td>
      <td>10.14</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AGR2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4689</td>
      <td>0.006745</td>
      <td>-0.361778</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-1.159091</td>
      <td>-3.224308</td>
      <td>1.645257</td>
      <td>9.67</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AGR3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4480</td>
      <td>0.006444</td>
      <td>-0.362733</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-1.099785</td>
      <td>-3.399142</td>
      <td>1.672747</td>
      <td>9.63</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AGR4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4601</td>
      <td>0.006618</td>
      <td>-0.356156</td>
      <td>0.0</td>
      <td>-0.060327</td>
      <td>-0.375256</td>
      <td>-3.245399</td>
      <td>1.696319</td>
      <td>9.66</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AGR5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5617</td>
      <td>0.008079</td>
      <td>-0.380758</td>
      <td>0.0</td>
      <td>-0.748677</td>
      <td>-0.559083</td>
      <td>-3.576720</td>
      <td>1.598765</td>
      <td>9.86</td>
    </tr>
    <tr>
      <th>25</th>
      <td>AGR6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4113</td>
      <td>0.005916</td>
      <td>-0.361704</td>
      <td>0.0</td>
      <td>0.439815</td>
      <td>0.000000</td>
      <td>-3.333333</td>
      <td>1.696759</td>
      <td>9.51</td>
    </tr>
    <tr>
      <th>26</th>
      <td>AGR7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5100</td>
      <td>0.007336</td>
      <td>-0.379039</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.785092</td>
      <td>-3.565344</td>
      <td>1.599226</td>
      <td>9.75</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AGR8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5445</td>
      <td>0.007832</td>
      <td>-0.372206</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.187069</td>
      <td>-3.313793</td>
      <td>1.587069</td>
      <td>9.84</td>
    </tr>
    <tr>
      <th>28</th>
      <td>AGR9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4397</td>
      <td>0.006325</td>
      <td>-0.364632</td>
      <td>0.0</td>
      <td>-1.506024</td>
      <td>-1.405257</td>
      <td>-3.431544</td>
      <td>1.642935</td>
      <td>9.61</td>
    </tr>
    <tr>
      <th>29</th>
      <td>AGR10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4640</td>
      <td>0.006674</td>
      <td>-0.378221</td>
      <td>0.0</td>
      <td>-0.267725</td>
      <td>-1.679365</td>
      <td>-3.528042</td>
      <td>1.632804</td>
      <td>9.65</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CSN1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5273</td>
      <td>0.007585</td>
      <td>-0.361782</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.370593</td>
      <td>-3.068788</td>
      <td>1.676698</td>
      <td>9.81</td>
    </tr>
    <tr>
      <th>31</th>
      <td>CSN2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6029</td>
      <td>0.008672</td>
      <td>-0.383626</td>
      <td>0.0</td>
      <td>0.917766</td>
      <td>0.749418</td>
      <td>-3.315749</td>
      <td>1.553918</td>
      <td>9.97</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CSN3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4465</td>
      <td>0.006422</td>
      <td>-0.364557</td>
      <td>0.0</td>
      <td>-1.755676</td>
      <td>0.000000</td>
      <td>-3.451892</td>
      <td>1.643243</td>
      <td>9.62</td>
    </tr>
    <tr>
      <th>33</th>
      <td>CSN4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4693</td>
      <td>0.006750</td>
      <td>-0.369237</td>
      <td>0.0</td>
      <td>-1.216734</td>
      <td>-0.557460</td>
      <td>-3.362903</td>
      <td>1.611895</td>
      <td>9.68</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CSN5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5167</td>
      <td>0.007432</td>
      <td>-0.362599</td>
      <td>0.0</td>
      <td>0.168880</td>
      <td>0.492410</td>
      <td>-3.400380</td>
      <td>1.725806</td>
      <td>9.77</td>
    </tr>
    <tr>
      <th>35</th>
      <td>CSN6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5983</td>
      <td>0.008606</td>
      <td>-0.398443</td>
      <td>0.0</td>
      <td>0.048644</td>
      <td>-0.740032</td>
      <td>-3.476077</td>
      <td>1.485646</td>
      <td>9.96</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CSN7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4088</td>
      <td>0.005880</td>
      <td>-0.359631</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.503480</td>
      <td>-3.379350</td>
      <td>1.641531</td>
      <td>9.52</td>
    </tr>
    <tr>
      <th>37</th>
      <td>CSN8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>6657</td>
      <td>0.009575</td>
      <td>-0.296346</td>
      <td>0.0</td>
      <td>1.215180</td>
      <td>-1.043818</td>
      <td>-2.920970</td>
      <td>2.460876</td>
      <td>10.02</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CSN9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4095</td>
      <td>0.005890</td>
      <td>-0.368232</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.505682</td>
      <td>-3.312500</td>
      <td>1.587500</td>
      <td>9.54</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CSN10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5830</td>
      <td>0.008386</td>
      <td>-0.361019</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.328832</td>
      <td>-3.115171</td>
      <td>1.695790</td>
      <td>9.94</td>
    </tr>
    <tr>
      <th>40</th>
      <td>OPN1_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4315</td>
      <td>0.006207</td>
      <td>-0.369087</td>
      <td>0.0</td>
      <td>0.129747</td>
      <td>-0.420886</td>
      <td>-3.188819</td>
      <td>1.603376</td>
      <td>9.60</td>
    </tr>
    <tr>
      <th>41</th>
      <td>OPN2_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5871</td>
      <td>0.008445</td>
      <td>-0.384133</td>
      <td>0.0</td>
      <td>-0.134583</td>
      <td>0.647359</td>
      <td>-3.598807</td>
      <td>1.615843</td>
      <td>9.91</td>
    </tr>
    <tr>
      <th>42</th>
      <td>OPN3_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4025</td>
      <td>0.005789</td>
      <td>-0.352065</td>
      <td>0.0</td>
      <td>0.251995</td>
      <td>-1.201824</td>
      <td>-3.122007</td>
      <td>1.701254</td>
      <td>9.53</td>
    </tr>
    <tr>
      <th>43</th>
      <td>OPN4_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>5195</td>
      <td>0.007472</td>
      <td>-0.377924</td>
      <td>0.0</td>
      <td>-0.285990</td>
      <td>-0.657971</td>
      <td>-3.581643</td>
      <td>1.661836</td>
      <td>9.77</td>
    </tr>
    <tr>
      <th>44</th>
      <td>OPN5_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4014</td>
      <td>0.005774</td>
      <td>-0.358547</td>
      <td>0.0</td>
      <td>-0.779083</td>
      <td>0.000000</td>
      <td>-3.329025</td>
      <td>1.655699</td>
      <td>9.50</td>
    </tr>
    <tr>
      <th>45</th>
      <td>OPN6_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4653</td>
      <td>0.006693</td>
      <td>-0.373083</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.016667</td>
      <td>-3.458333</td>
      <td>1.633333</td>
      <td>9.66</td>
    </tr>
    <tr>
      <th>46</th>
      <td>OPN7_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4436</td>
      <td>0.006381</td>
      <td>-0.376083</td>
      <td>0.0</td>
      <td>1.353326</td>
      <td>-0.520174</td>
      <td>-3.484188</td>
      <td>1.619411</td>
      <td>9.61</td>
    </tr>
    <tr>
      <th>47</th>
      <td>OPN8_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4232</td>
      <td>0.006087</td>
      <td>-0.374869</td>
      <td>0.0</td>
      <td>0.065193</td>
      <td>0.350276</td>
      <td>-3.377901</td>
      <td>1.550276</td>
      <td>9.57</td>
    </tr>
    <tr>
      <th>48</th>
      <td>OPN9_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>4497</td>
      <td>0.006468</td>
      <td>-0.380296</td>
      <td>0.0</td>
      <td>-0.280607</td>
      <td>-0.167931</td>
      <td>-3.522210</td>
      <td>1.594800</td>
      <td>9.62</td>
    </tr>
    <tr>
      <th>49</th>
      <td>OPN10_E</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>3143</td>
      <td>0.004521</td>
      <td>-0.344548</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.165021</td>
      <td>-3.093089</td>
      <td>1.622003</td>
      <td>9.27</td>
    </tr>
    <tr>
      <th>50</th>
      <td>testelapse</td>
      <td>float64</td>
      <td>0</td>
      <td>0.0</td>
      <td>306</td>
      <td>0.000440</td>
      <td>-0.339571</td>
      <td>0.0</td>
      <td>0.274510</td>
      <td>-0.803922</td>
      <td>-4.294118</td>
      <td>1.686275</td>
      <td>6.56</td>
    </tr>
  </tbody>
</table>
</div>


    100%|██████████████████████████████████████████████████████████████████████████████████| 51/51 [02:41<00:00,  3.17s/it]
    


    
![png](output_26_6.png)
    


**well, that doesn't look that good, but I don't know if there is a good way to handle that many outliers on the time variables. Each one has plenty of high outliers, removing that much data would eliminate too much of our dataset.**

### Last and fourth dataset, no scaling, only non-time variables. Countries label encoded


```python
df_simple = df.drop(columns=['country'] + time_cols)
df_simple['country'] = LabelEncoder().fit_transform(df['country'])

display(df_simple.shape)
display(df_simple.isna().sum().sum())
```


    (695225, 51)



    0


## The modeling
**We'll start modeling now. I wanted to try a bunch of unsupervised clustering models (MeanShift, AffinityPropagation, AgglomerativeClustering, SpectralClustering, DBSCAN, OPTICS, Birch) but sadly my computer does not have the necessary specs to try more than KMeans with this many observations.  
We'll check various performance measures, sum of squared errors, davies-bouldin, calinski-harabasz**


```python
#k-means args
kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 100,
    "random_state": SEED,
}

sse = {'df_all_scaled':[], 'df_lbl_enc_scaled':[], 'df_not_all_scaled':[], 'df_simple':[]} # A dict holds the SSE values for each k and each dataset
dbs = {'df_all_scaled':[], 'df_lbl_enc_scaled':[], 'df_not_all_scaled':[], 'df_simple':[]} # A dict holds the DBS values for each k and each dataset
chs = {'df_all_scaled':[], 'df_lbl_enc_scaled':[], 'df_not_all_scaled':[], 'df_simple':[]} # A dict holds the CHS values for each k and each dataset
rng = range(2, 18) # a range of clusters to go through
df_list = [df_all_scaled,df_lbl_enc_scaled,df_not_all_scaled,df_simple] #list of datasets to iterate through
df_names_list = ['df_all_scaled','df_lbl_enc_scaled','df_not_all_scaled','df_simple'] #list of dataset names to iterate through
```


```python
for i in range(len(df_list)):
    for k in tqdm(rng):
        clusterer = MiniBatchKMeans(n_clusters=k, **kmeans_kwargs)
        clusterer.fit(df_list[i])
        labels = clusterer.labels_
        sse[df_names_list[i]].append(clusterer.inertia_)
        dbs[df_names_list[i]].append(davies_bouldin_score(df_list[i],labels))
        chs[df_names_list[i]].append(calinski_harabasz_score(df_list[i],labels))
    print('{} done'.format(i))
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [03:28<00:00, 13.03s/it]
      0%|                                                                                           | 0/16 [00:00<?, ?it/s]

    0 done
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [01:24<00:00,  5.29s/it]
      0%|                                                                                           | 0/16 [00:00<?, ?it/s]

    1 done
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [01:39<00:00,  6.22s/it]
      0%|                                                                                           | 0/16 [00:00<?, ?it/s]

    2 done
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [01:02<00:00,  3.88s/it]

    3 done
    

    
    


```python
plt.style.use('seaborn-notebook')
display(pd.DataFrame(sse,index=rng).plot(xlabel='Number of Clusters',ylabel='SSE'))
# we had to separate this one because the scale is so different we couldn't see it's value!!
display(pd.DataFrame(sse,index=rng)[['df_simple']].plot(xlabel='Number of Clusters',ylabel='SSE'))
```


    <AxesSubplot:xlabel='Number of Clusters', ylabel='SSE'>



    <AxesSubplot:xlabel='Number of Clusters', ylabel='SSE'>



    
![png](output_33_2.png)
    



    
![png](output_33_3.png)
    


### Davies-Bouldin Index
**This index signifies the average ‘similarity’ between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves. Zero is the lowest possible score. Values closer to zero indicate a better partition**


```python
plt.style.use('seaborn-notebook')
display(pd.DataFrame(dbs,index=rng).plot(xlabel='Number of Clusters',ylabel='DBS'))
display(pd.DataFrame(dbs,index=rng)[['df_simple']].plot(xlabel='Number of Clusters',ylabel='DBS'))
```


    <AxesSubplot:xlabel='Number of Clusters', ylabel='DBS'>



    <AxesSubplot:xlabel='Number of Clusters', ylabel='DBS'>



    
![png](output_35_2.png)
    



    
![png](output_35_3.png)
    


**Apparently the fewer the clusters the better?**

###  Calinski-Harabasz Index
**A higher Calinski-Harabasz score relates to a model with better defined clusters.  
The index is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters**


```python
plt.style.use('seaborn-notebook')
display(pd.DataFrame(chs,index=rng).plot(xlabel='Number of Clusters',ylabel='CHS'))
display(pd.DataFrame(chs,index=rng)[['df_all_scaled']].plot(xlabel='Number of Clusters',ylabel='CHS'))
display(pd.DataFrame(chs,index=rng)[['df_lbl_enc_scaled']].plot(xlabel='Number of Clusters',ylabel='CHS'))
```


    <AxesSubplot:xlabel='Number of Clusters', ylabel='CHS'>



    <AxesSubplot:xlabel='Number of Clusters', ylabel='CHS'>



    <AxesSubplot:xlabel='Number of Clusters', ylabel='CHS'>



    
![png](output_38_3.png)
    



    
![png](output_38_4.png)
    



    
![png](output_38_5.png)
    


### Some observations from the error metrics
- SSE shows a rapid decrease until about 4 clusters where it starts to smooth off.  
- Davies-Bouldin indicates the lowest the number of clusters the better.
- Calinski-Harabasz shows 4 clusters generate better defined clusters.
- The simple dataset shows the best scores across all tests

**We've used MiniBatchKmeans to get an overall picture since it trains much faster without losing that much precision. However, given the above observations, we'll now use K-means to train with the simple dataframe and 4 clusters, trying to get the maximum amount of precision. Yes I know its "the big 5" and why not use 5 clusters? Well data is saying otherwise, who is to say the big 5 is not actually the big 4?** 


```python
clusterer = KMeans(n_clusters=4, **kmeans_kwargs)
clusterer.fit(df_simple)
labels = clusterer.labels_
```

## Let's look at the clusters data to see if we can distinguish what separates each cluster from one another


```python
df_cl = df_simple.copy()
df_cl['cluster'] = labels
```


```python
a = []
a.append('1')
a.append('2')
a
```




    ['1', '2']




```python
def boxplots(df_cl, cl_col):
    cl_nm = len(np.unique(df_cl[cl_col]))
    for col in df_cl.drop(columns=[cl_col]).columns.values:

#         data = [df_cl[df_cl['cluster']==0][col]
#                 ,df_cl[df_cl['cluster']==1][col]
#                 ,df_cl[df_cl['cluster']==2][col]
#                 ,df_cl[df_cl['cluster']==3][col]] 
        data = []
        for i in range(cl_nm):
            data.append(df_cl[df_cl[cl_col]==i][col])

        fig = plt.figure(figsize =(10, 7)) 
        ax = fig.add_subplot(111) 

        # Creating axes instance 
        bp = ax.boxplot(data, patch_artist = True, 
                        notch ='True', vert = 0) 

        colors = ['#0000FF', '#00FF00',  
                  '#FFFF00', '#FF00FF'] 

        for patch, color in zip(bp['boxes'], colors): 
            patch.set_facecolor(color) 

        # changing color and linewidth of 
        # whiskers 
        for whisker in bp['whiskers']: 
            whisker.set(color ='#8B008B', 
                        linewidth = 1.5, 
                        linestyle =":") 

        # changing color and linewidth of 
        # caps 
        for cap in bp['caps']: 
            cap.set(color ='#8B008B', 
                    linewidth = 2) 

        # changing color and linewidth of 
        # medians 
        for median in bp['medians']: 
            median.set(color ='red', 
                       linewidth = 3) 

        # changing style of fliers 
        for flier in bp['fliers']: 
            flier.set(marker ='D', 
                      color ='#e7298a', 
                      alpha = 0.5) 

        # x-axis labels 
        yticks = []
        for i in range(cl_nm):
            yticks.append('cluster {}'.format(i))
#         ax.set_yticklabels(['cluster 0', 'cluster 1',  
#                             'cluster 2', 'cluster 3']) 
        ax.set_yticklabels(yticks) 
        # Adding title  
        plt.title(col) 

        # Removing top axes and right axes 
        # ticks 
        ax.get_xaxis().tick_bottom() 
        ax.get_yaxis().tick_left() 

        # show plot 
        plt.show(bp)
```


```python
boxplots(df_cl,'cluster')
```


    
![png](output_46_0.png)
    



    
![png](output_46_1.png)
    



    
![png](output_46_2.png)
    



    
![png](output_46_3.png)
    



    
![png](output_46_4.png)
    



    
![png](output_46_5.png)
    



    
![png](output_46_6.png)
    



    
![png](output_46_7.png)
    



    
![png](output_46_8.png)
    



    
![png](output_46_9.png)
    



    
![png](output_46_10.png)
    



    
![png](output_46_11.png)
    



    
![png](output_46_12.png)
    



    
![png](output_46_13.png)
    



    
![png](output_46_14.png)
    



    
![png](output_46_15.png)
    



    
![png](output_46_16.png)
    



    
![png](output_46_17.png)
    



    
![png](output_46_18.png)
    



    
![png](output_46_19.png)
    



    
![png](output_46_20.png)
    



    
![png](output_46_21.png)
    



    
![png](output_46_22.png)
    



    
![png](output_46_23.png)
    



    
![png](output_46_24.png)
    



    
![png](output_46_25.png)
    



    
![png](output_46_26.png)
    



    
![png](output_46_27.png)
    



    
![png](output_46_28.png)
    



    
![png](output_46_29.png)
    



    
![png](output_46_30.png)
    



    
![png](output_46_31.png)
    



    
![png](output_46_32.png)
    



    
![png](output_46_33.png)
    



    
![png](output_46_34.png)
    



    
![png](output_46_35.png)
    



    
![png](output_46_36.png)
    



    
![png](output_46_37.png)
    



    
![png](output_46_38.png)
    



    
![png](output_46_39.png)
    



    
![png](output_46_40.png)
    



    
![png](output_46_41.png)
    



    
![png](output_46_42.png)
    



    
![png](output_46_43.png)
    



    
![png](output_46_44.png)
    



    
![png](output_46_45.png)
    



    
![png](output_46_46.png)
    



    
![png](output_46_47.png)
    



    
![png](output_46_48.png)
    



    
![png](output_46_49.png)
    



    
![png](output_46_50.png)
    


### Some observations after looking at the graphs
 1. Featues that seem to vary the most for each cluster, meaning they might have an important role defining them, are CSN5 and Country
 2. Cluster 0 sets itself appart the most from the others, it shows small differences in features EXT1, EXT8, EST1, EST4, EST8, ARG4, ARG8, CSN1, CSN3, CSN7, CSN8. Note it does not distinguish itself in any OPN feature. Having that much variance might mean it holds the most amount of observation, meaning it's the biggest cluster and most heterogeneous.
 3. Bar point 1, Cluster 1 shows some differences in features EXT6 and EXT7 only
 4. Bar point 1, Cluster 2 shows some differences in features AGR1, OPN5 and OPN7 only
 5. Bar point 1, Cluster 3 shows some differences in features AGR1 and OPN5 only
 

Let's check our guess on obs. nº 2


```python
df_cl['cluster'].value_counts()
```




    0    357937
    1    120737
    3    115597
    2    100954
    Name: cluster, dtype: int64



It seems we were right, cluster 0 has over 50% of the data. Might it have US observations?


```python
for i in range(4):
    print('Cluster {} top 3:'.format(i), df_cl[df_cl['cluster']==i]['country'].value_counts()[0:3], sep='\n')
```

    Cluster 0 top 3:
    206    345069
    218      2830
    200      2421
    Name: country, dtype: int64
    Cluster 1 top 3:
    35    43801
    12    34631
    51    12309
    Name: country, dtype: int64
    Cluster 2 top 3:
    160    11126
    142     9777
    151     9606
    Name: country, dtype: int64
    Cluster 3 top 3:
    69    49744
    94    12207
    90     5604
    Name: country, dtype: int64
    


```python
cntry = df[['country']].copy()
cntry['enc'] = LabelEncoder().fit_transform(df['country'])
print('Cluster 0''s top country is ' + pd.unique(cntry[cntry['enc']==206]['country'])[0]
     ,'Cluster 1''s top country is ' + pd.unique(cntry[cntry['enc']==35]['country'])[0]
     ,'Cluster 2''s top country is ' + pd.unique(cntry[cntry['enc']==160]['country'])[0]
     ,'Cluster 3''s top country is ' + pd.unique(cntry[cntry['enc']==69]['country'])[0]
     ,sep='\n')
```

    Cluster 0s top country is US
    Cluster 1s top country is CA
    Cluster 2s top country is PH
    Cluster 3s top country is GB
    

Another observation, Cluster 1 seems perhaps the most evenly distributed.

## Let's do some PCA and see if we can draw a nice 2D graph


```python
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df_simple)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = labels
df_pca.head()
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
      <th>PCA1</th>
      <th>PCA2</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.446001</td>
      <td>-5.779149</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.441659</td>
      <td>0.155677</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.445594</td>
      <td>-0.931190</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77.460765</td>
      <td>0.808060</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-30.551297</td>
      <td>-2.700953</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,10))
sb.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
plt.title('Personality Clusters after PCA');
```


    
![png](output_56_0.png)
    


That doesn't look so good though...

## What if we removed the country and did this again?

(just with df_simple this time around)


```python
sse = {'df_all_scaled':[], 'df_lbl_enc_scaled':[], 'df_not_all_scaled':[], 'df_simple':[]} # A dict holds the SSE values for each k and each dataset
dbs = {'df_all_scaled':[], 'df_lbl_enc_scaled':[], 'df_not_all_scaled':[], 'df_simple':[]} # A dict holds the DBS values for each k and each dataset
chs = {'df_all_scaled':[], 'df_lbl_enc_scaled':[], 'df_not_all_scaled':[], 'df_simple':[]} # A dict holds the CHS values for each k and each dataset
```


```python
for k in tqdm(rng):
    clusterer = MiniBatchKMeans(n_clusters=k, **kmeans_kwargs)
    clusterer.fit(df_simple.drop(columns=['country']))
    labels = clusterer.labels_
    sse[df_names_list[i]].append(clusterer.inertia_)
    dbs[df_names_list[i]].append(davies_bouldin_score(df_list[i],labels))
    chs[df_names_list[i]].append(calinski_harabasz_score(df_list[i],labels))
print('{} done'.format(i))
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [01:04<00:00,  4.01s/it]

    3 done
    

    
    


```python
plt.style.use('seaborn-notebook')
print('Low is good')
display(pd.DataFrame(sse['df_simple'],index=rng).plot(xlabel='Number of Clusters',ylabel='SSE'))
```

    Low is good
    


    <AxesSubplot:xlabel='Number of Clusters', ylabel='SSE'>



    
![png](output_62_2.png)
    



```python
plt.style.use('seaborn-notebook')
print('Low is good')
display(pd.DataFrame(dbs['df_simple'],index=rng).plot(xlabel='Number of Clusters',ylabel='DBS'))
```

    Low is good
    


    <AxesSubplot:xlabel='Number of Clusters', ylabel='DBS'>



    
![png](output_63_2.png)
    



```python
plt.style.use('seaborn-notebook')
print('High is good')
display(pd.DataFrame(chs['df_simple'],index=rng).plot(xlabel='Number of Clusters',ylabel='CHS'))
```

    High is good
    


    <AxesSubplot:xlabel='Number of Clusters', ylabel='CHS'>



    
![png](output_64_2.png)
    


### Observations
 - SSE looking better, its in 1e7 instead of 1e8
 - SSE doesn't have that pronounced break like last time, you could kind of see it at 8, perhaps, any value between 4 and 8 should be ok
 - DBS still says 2 is best, next best thing is 6
 - CHS, similarly to DBS, points to 2 and 5 for best number of clusters

**Okey let's this time assume "The big 5" thing works and do 5 clusters with K means, see if we can spot its characteristics**


```python
clusterer = KMeans(n_clusters=5, **kmeans_kwargs)
clusterer.fit(df_simple.drop(columns=['country']))
labels = clusterer.labels_
```


```python
df_cl = df_simple.drop(columns=['country']).copy()
df_cl['cluster'] = labels
```


```python
boxplots(df_cl,'cluster')
```


    
![png](output_69_0.png)
    



    
![png](output_69_1.png)
    



    
![png](output_69_2.png)
    



    
![png](output_69_3.png)
    



    
![png](output_69_4.png)
    



    
![png](output_69_5.png)
    



    
![png](output_69_6.png)
    



    
![png](output_69_7.png)
    



    
![png](output_69_8.png)
    



    
![png](output_69_9.png)
    



    
![png](output_69_10.png)
    



    
![png](output_69_11.png)
    



    
![png](output_69_12.png)
    



    
![png](output_69_13.png)
    



    
![png](output_69_14.png)
    



    
![png](output_69_15.png)
    



    
![png](output_69_16.png)
    



    
![png](output_69_17.png)
    



    
![png](output_69_18.png)
    



    
![png](output_69_19.png)
    



    
![png](output_69_20.png)
    



    
![png](output_69_21.png)
    



    
![png](output_69_22.png)
    



    
![png](output_69_23.png)
    



    
![png](output_69_24.png)
    



    
![png](output_69_25.png)
    



    
![png](output_69_26.png)
    



    
![png](output_69_27.png)
    



    
![png](output_69_28.png)
    



    
![png](output_69_29.png)
    



    
![png](output_69_30.png)
    



    
![png](output_69_31.png)
    



    
![png](output_69_32.png)
    



    
![png](output_69_33.png)
    



    
![png](output_69_34.png)
    



    
![png](output_69_35.png)
    



    
![png](output_69_36.png)
    



    
![png](output_69_37.png)
    



    
![png](output_69_38.png)
    



    
![png](output_69_39.png)
    



    
![png](output_69_40.png)
    



    
![png](output_69_41.png)
    



    
![png](output_69_42.png)
    



    
![png](output_69_43.png)
    



    
![png](output_69_44.png)
    



    
![png](output_69_45.png)
    



    
![png](output_69_46.png)
    



    
![png](output_69_47.png)
    



    
![png](output_69_48.png)
    



    
![png](output_69_49.png)
    


**That looks much better! a lot more variance, meaning perhaps each cluster is more uniquely identified**

### PCA time!


```python
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df_simple.drop(columns=['country']))

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = labels
df_pca.head()
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
      <th>PCA1</th>
      <th>PCA2</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-5.582889</td>
      <td>-1.514488</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.136778</td>
      <td>3.014162</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.762445</td>
      <td>2.069605</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000176</td>
      <td>0.085359</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.774470</td>
      <td>2.415694</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,10))
sb.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
plt.title('Personality Clusters after PCA');
```


    
![png](output_73_0.png)
    


**Beautiful!!**

# Wrap up

**There are more things we can do and try, like try and use neural networks to find the clusters. But I believe this dataset has it's limitations and we'll leave those adventures for more interesting datasets. This was a simple exercise of data analysis and some clustering for practice.  
As a conclusion, it seems 5 clusters do work best for "The big 5", nothing wrong with some confirmative evidence right?**
