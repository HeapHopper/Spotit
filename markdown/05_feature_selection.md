# Feature Selection




```python
# DS18 ML Essentials project
# Module 5: Feature Selection

# Submitted by: Tzvi Eliezer Nir
# mail: tzvienir@gmail.com
# First submission: 29/03/2025
```

## In this Notebook

In the previous chapter, we created new columns using various *Feature Engineering* techniques. However, having a wide set of columns does not necessarily lead to better results. We need to ensure that all features provided to the model are meaningful; otherwise, the model may become vulnerable to overfitting.

Fortunately, there are *Regularization* methods designed specifically for this purpose. We will use **Lasso** and **Ridge**, along with other regression models, to **select a subset of features** to be used by the model.

But before, we need to perform some additional formatting, such as encoding categorical variables and explicitly specifying that the dummy columns we created along the way are boolean.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```

## Load the dataset


```python
df = pd.read_pickle('pickle/04_feature_engineering/feature_engineering.pkl')
df.head()
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
      <th>track_id</th>
      <th>track_popularity</th>
      <th>track_album_id</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>...</th>
      <th>month</th>
      <th>day</th>
      <th>decade</th>
      <th>released_in_internet_era</th>
      <th>feat</th>
      <th>Remix</th>
      <th>Love</th>
      <th>Radio Edit</th>
      <th>Remastered</th>
      <th>track_artist_followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0017A6SJgTbfQVU2EtsPNo</td>
      <td>41</td>
      <td>1srJQ0njEQgd8w4XSqI4JQ</td>
      <td>0.682</td>
      <td>0.401</td>
      <td>2</td>
      <td>-10.068</td>
      <td>1</td>
      <td>0.0236</td>
      <td>0.279000</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>103090.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>002xjHwzEx66OWFV2IP9dk</td>
      <td>15</td>
      <td>1ficfUnZMaY1QkNp15Slzm</td>
      <td>0.582</td>
      <td>0.704</td>
      <td>5</td>
      <td>-6.242</td>
      <td>1</td>
      <td>0.0347</td>
      <td>0.065100</td>
      <td>...</td>
      <td>1</td>
      <td>26</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>366482.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>004s3t0ONYlzxII9PLgU6z</td>
      <td>28</td>
      <td>3z04Lb9Dsilqw68SHt6jLB</td>
      <td>0.303</td>
      <td>0.880</td>
      <td>9</td>
      <td>-4.739</td>
      <td>1</td>
      <td>0.0442</td>
      <td>0.011700</td>
      <td>...</td>
      <td>11</td>
      <td>21</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4132.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>008MceT31RotUANsKuzy3L</td>
      <td>24</td>
      <td>1Z4ANBVuhTlS6DprlP0m1q</td>
      <td>0.659</td>
      <td>0.794</td>
      <td>10</td>
      <td>-5.644</td>
      <td>0</td>
      <td>0.0540</td>
      <td>0.000761</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>557.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>008rk8F6ZxspZT4bUlkIQG</td>
      <td>38</td>
      <td>2BuYm9UcKvI0ydXs5JKwt0</td>
      <td>0.662</td>
      <td>0.838</td>
      <td>1</td>
      <td>-6.300</td>
      <td>1</td>
      <td>0.0499</td>
      <td>0.114000</td>
      <td>...</td>
      <td>11</td>
      <td>16</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2913.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28356 entries, 0 to 28355
    Data columns (total 33 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   track_id                  28356 non-null  object 
     1   track_popularity          28356 non-null  int64  
     2   track_album_id            28356 non-null  object 
     3   danceability              28356 non-null  float64
     4   energy                    28356 non-null  float64
     5   key                       28356 non-null  int64  
     6   loudness                  28356 non-null  float64
     7   mode                      28356 non-null  int64  
     8   speechiness               28356 non-null  float64
     9   acousticness              28356 non-null  float64
     10  instrumentalness          28356 non-null  float64
     11  liveness                  28356 non-null  float64
     12  valence                   28356 non-null  float64
     13  tempo                     28356 non-null  float64
     14  duration_ms               28356 non-null  int64  
     15  playlist_count            28356 non-null  int64  
     16  edm                       28356 non-null  int64  
     17  latin                     28356 non-null  int64  
     18  pop                       28356 non-null  int64  
     19  r&b                       28356 non-null  int64  
     20  rap                       28356 non-null  int64  
     21  rock                      28356 non-null  int64  
     22  year                      28356 non-null  int32  
     23  month                     28356 non-null  int32  
     24  day                       28356 non-null  int32  
     25  decade                    28356 non-null  int32  
     26  released_in_internet_era  28356 non-null  int64  
     27  feat                      28356 non-null  int64  
     28  Remix                     28356 non-null  int64  
     29  Love                      28356 non-null  int64  
     30  Radio Edit                28356 non-null  int64  
     31  Remastered                28356 non-null  int64  
     32  track_artist_followers    28356 non-null  float64
    dtypes: float64(10), int32(4), int64(17), object(2)
    memory usage: 6.7+ MB


## Category encoding

### Cast dummy columns as `bool`

Right above 2hen using `df.info()`, we can see that the "dummy" categories we created like: genre, hot-word in the track name, released in the internet era etc. are all treated as `int64`. As those columns have just 0 and 1, it will be wise to just cast them into a boolean type:


```python
bool_columns = ['released_in_internet_era', 'feat', 'Remix', 'Love', 'Radio Edit', 'Remastered', 'edm', 'latin', 'pop', 'r&b', 'rap', 'rock']
df[bool_columns] = df[bool_columns].astype(bool)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28356 entries, 0 to 28355
    Data columns (total 33 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   track_id                  28356 non-null  object 
     1   track_popularity          28356 non-null  int64  
     2   track_album_id            28356 non-null  object 
     3   danceability              28356 non-null  float64
     4   energy                    28356 non-null  float64
     5   key                       28356 non-null  int64  
     6   loudness                  28356 non-null  float64
     7   mode                      28356 non-null  int64  
     8   speechiness               28356 non-null  float64
     9   acousticness              28356 non-null  float64
     10  instrumentalness          28356 non-null  float64
     11  liveness                  28356 non-null  float64
     12  valence                   28356 non-null  float64
     13  tempo                     28356 non-null  float64
     14  duration_ms               28356 non-null  int64  
     15  playlist_count            28356 non-null  int64  
     16  edm                       28356 non-null  bool   
     17  latin                     28356 non-null  bool   
     18  pop                       28356 non-null  bool   
     19  r&b                       28356 non-null  bool   
     20  rap                       28356 non-null  bool   
     21  rock                      28356 non-null  bool   
     22  year                      28356 non-null  int32  
     23  month                     28356 non-null  int32  
     24  day                       28356 non-null  int32  
     25  decade                    28356 non-null  int32  
     26  released_in_internet_era  28356 non-null  bool   
     27  feat                      28356 non-null  bool   
     28  Remix                     28356 non-null  bool   
     29  Love                      28356 non-null  bool   
     30  Radio Edit                28356 non-null  bool   
     31  Remastered                28356 non-null  bool   
     32  track_artist_followers    28356 non-null  float64
    dtypes: bool(12), float64(10), int32(4), int64(5), object(2)
    memory usage: 4.4+ MB


### Label encoding `df.decade`

Another casting I would liked to do, is to set the `df.decade` to be a `category`.

Unfortunately, some regression models like `XGBoost` have a problem with categories, and so I will skip this step. 


```python
#df.decade = df.decade.astype('category')
```

## Multivariable Analysis

Time for **Feature Selection**!

This is how its gonna work: we will choose a handful of regression models, among them Lasso and Ridge, and fit those models on our data. Our WHOLE data (we will not split it into train and test).

Than we will look into the feature importance of each model (or the coefficients value, in the regularization methods) and see which features should be selected to be provided to the model.

This process should help us avoid overfitting in the model-selection part.


```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
```

Define the target variable y, and the feature set X:


```python
y=df['track_popularity']
X = df.drop(columns=['track_popularity','track_id', 'track_album_id'])
```

Fitting the models, and finding for each feature, for each model, if it should be selected:


```python
# Fit models and determine if a feature is selected (1) or not (0)
lasso = Lasso(alpha=0.1).fit(X, y)
lasso_selected = (np.abs(lasso.coef_) > 0).astype(int)

ridge = Ridge(alpha=0.1).fit(X, y)
ridge_selected = (np.abs(ridge.coef_) > 0).astype(int)

svm = LinearSVR(C=0.01, max_iter=5000).fit(X, y)
svm_selected = (np.abs(svm.coef_) > 0).astype(int)

gb = GradientBoostingRegressor().fit(X, y)
gb_selected = (gb.feature_importances_ > 0).astype(int)

rf = RandomForestRegressor().fit(X, y)
rf_selected = (rf.feature_importances_ > 0).astype(int)

# Create a DataFrame to store results
selection_df = pd.DataFrame({
    'Feature': X.columns,
    'Lasso': lasso_selected, 
    'SVR': svm_selected,
    'GradientBoost': gb_selected,
    'RandomForest': rf_selected,
    'Ridge': ridge_selected
})

# Sum the number of selections for each feature
selection_df['Sum'] = selection_df[['Lasso', 'SVR', 'GradientBoost', 'RandomForest','Ridge']].sum(axis=1)

# Output the results
selection_df
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
      <th>Lasso</th>
      <th>SVR</th>
      <th>GradientBoost</th>
      <th>RandomForest</th>
      <th>Ridge</th>
      <th>Sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>danceability</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>energy</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>key</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loudness</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mode</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>speechiness</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>acousticness</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>instrumentalness</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>liveness</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>valence</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tempo</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>duration_ms</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>playlist_count</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>edm</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>latin</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>pop</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>r&amp;b</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>rap</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>rock</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>year</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>20</th>
      <td>month</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21</th>
      <td>day</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>decade</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>released_in_internet_era</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>feat</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Remix</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Love</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Radio Edit</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Remastered</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>track_artist_followers</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Our goal should be having between 15-to-30 selected features, seems that if we choose all features that are "selected" in more than 4 models we get a good subset:


```python
selection_df[selection_df['Sum'] > 4].count()
```




    Feature          22
    Lasso            22
    SVR              22
    GradientBoost    22
    RandomForest     22
    Ridge            22
    Sum              22
    dtype: int64



Select the chosen features - and don't forget the target variable! :-)


```python
final_var = selection_df[selection_df['Sum'] > 4]['Feature'].tolist()
df_model = df[final_var].copy()
df_model['track_popularity'] = df['track_popularity'].copy()
```

Lets have a look at the dataset after the feature selection:


```python
df_model.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28356 entries, 0 to 28355
    Data columns (total 23 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   danceability            28356 non-null  float64
     1   energy                  28356 non-null  float64
     2   key                     28356 non-null  int64  
     3   loudness                28356 non-null  float64
     4   acousticness            28356 non-null  float64
     5   instrumentalness        28356 non-null  float64
     6   liveness                28356 non-null  float64
     7   tempo                   28356 non-null  float64
     8   duration_ms             28356 non-null  int64  
     9   playlist_count          28356 non-null  int64  
     10  edm                     28356 non-null  bool   
     11  pop                     28356 non-null  bool   
     12  r&b                     28356 non-null  bool   
     13  rap                     28356 non-null  bool   
     14  rock                    28356 non-null  bool   
     15  year                    28356 non-null  int32  
     16  month                   28356 non-null  int32  
     17  day                     28356 non-null  int32  
     18  decade                  28356 non-null  int32  
     19  feat                    28356 non-null  bool   
     20  Remix                   28356 non-null  bool   
     21  track_artist_followers  28356 non-null  float64
     22  track_popularity        28356 non-null  int64  
    dtypes: bool(7), float64(8), int32(4), int64(4)
    memory usage: 3.2 MB


## Save for the next chapter


```python
df_model.to_csv('./data/05_feature_selection/feature_selection.csv', index=False)

df_model.to_pickle('./pickle/05_feature_selection/feature_selection.pkl')
```

![tobecontinued.jpg](05_feature_selection_files/tobecontinued.jpg)
