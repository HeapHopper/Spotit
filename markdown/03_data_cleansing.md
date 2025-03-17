# Data Cleansing


```python
# DS18 ML Essentials project
# Module 3: Data Cleansing

# Submitted by: Tzvi Eliezer Nir
# mail: tzvienir@gmail.com
# First submission: 17/03/2025
```

## In this notebook

The data cleansing process involves two main steps:

1. Detecting outliers - using statistical methods such as Z-Score and IQR
2. Filling missing data - either with constant values (e.g., mean) or by using a smart unsupervised model approach

It is important to first detect outliers and only then deal with the missing data, since outliers will be set to null.

I will use IQR for outliers detection and the MICE imputation for filling missing data.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_pickle("./pickle/01_data_preparation/data_preparation.pkl")
```

## Outliers detection

First lets have a visual representation of the outliers using Box Plots:


```python
# Select only numerical columns (int, float, and optionally bool)
df_num = df.select_dtypes(include=["number"]).copy()

# Drop the dummy columns
df_num = df_num.drop(columns=['mode', 'edm', 'latin', 'pop', 'r&b', 'rap', 'rock','playlist_count'])

con_col = df_num.columns
```


```python
## Using box (Wiskers) plot 
plt.figure(figsize=(20,200))

def outliers_boxplot(df_num):
    for i, col in enumerate(df_num.columns):
        
            ax = plt.subplot(60, 3, i+1)
            sns.boxplot(data=df_num, x=col, ax=ax)
            plt.subplots_adjust(hspace = 0.7)
            plt.title('Box Plot: {}'.format(col), fontsize=15)
            plt.xlabel('{}'.format(col), fontsize=14)
        
outliers_boxplot(df_num)
plt.show()
```


    
![png](03_data_cleansing_files/03_data_cleansing_6_0.png)
    


### Using IQR score to determine outliers

It is important to label outlier values based on a designated metric and not just by relying on a visual tool.

In this project I decided to use the IQR method for finding outlier values. I will set lower and upper bound from the IQR value, computed by: $IQR = Q3 - Q1$. All values outside those bounds will be set to Null.

First, lets see how many outlier values we have in each column:


```python
def get_outliers_df(df):
    total_outliers = pd.DataFrame(columns=['Outlier count', 'Percent'])

    for col in df.columns:  # Ensure processing numeric columns only
        temp = pd.DataFrame(df[col])
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        # Filter rows that are outliers in either direction
        temp_outliers = temp[(temp[col] > upper_limit) | (temp[col] < lower_limit)]
        num_outliers = len(temp_outliers)
        total_outliers.loc[col] = [num_outliers, num_outliers / len(df) * 100]

    return total_outliers[total_outliers['Percent'] > 0]

# Assume con_df is your DataFrame
# Call the function and sort results
outliers_df = get_outliers_df(df_num).sort_values('Percent', ascending=False)
outliers_df
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
      <th>Outlier count</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>instrumentalness</th>
      <td>6085.0</td>
      <td>21.459303</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>2725.0</td>
      <td>9.609959</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>1911.0</td>
      <td>6.739314</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>1623.0</td>
      <td>5.723656</td>
    </tr>
    <tr>
      <th>duration_ms</th>
      <td>1197.0</td>
      <td>4.221329</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>837.0</td>
      <td>2.951756</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>494.0</td>
      <td>1.742136</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>257.0</td>
      <td>0.906334</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>222.0</td>
      <td>0.782903</td>
    </tr>
  </tbody>
</table>
</div>



#### Set outilers to `NaN`


```python
def replace_outliers_with_nan(df):
    label_out_df = df.copy()
    outliers_df_result = get_outliers_df(df)  # Call the outliers_df function
    for col in label_out_df:
        if col in outliers_df_result.index:
            Q1 = label_out_df[col].quantile(0.25)
            Q3 = label_out_df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q1 + 1.5 * IQR
            lower_limit = Q3 - 1.5 * IQR
            label_out_df[col] = np.where((label_out_df[col] > upper_limit) | (label_out_df[col] < lower_limit), np.nan, label_out_df[col])
    return label_out_df

# Example usage
# Assume df is your DataFrame
df_num_nan = replace_outliers_with_nan(df_num)
display(df_num_nan.shape)
df_num_nan.head(10)
```


    (28356, 12)





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
      <th>track_popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>0.682</td>
      <td>NaN</td>
      <td>2</td>
      <td>-10.068</td>
      <td>0.0236</td>
      <td>0.279000</td>
      <td>NaN</td>
      <td>0.0887</td>
      <td>0.566</td>
      <td>97.091</td>
      <td>235440.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>0.582</td>
      <td>0.704</td>
      <td>5</td>
      <td>-6.242</td>
      <td>0.0347</td>
      <td>0.065100</td>
      <td>0.000000</td>
      <td>0.2120</td>
      <td>0.698</td>
      <td>150.863</td>
      <td>197286.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>NaN</td>
      <td>0.880</td>
      <td>9</td>
      <td>-4.739</td>
      <td>0.0442</td>
      <td>0.011700</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.404</td>
      <td>135.225</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>0.659</td>
      <td>0.794</td>
      <td>10</td>
      <td>-5.644</td>
      <td>0.0540</td>
      <td>0.000761</td>
      <td>NaN</td>
      <td>0.3220</td>
      <td>0.852</td>
      <td>128.041</td>
      <td>228565.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38</td>
      <td>0.662</td>
      <td>0.838</td>
      <td>1</td>
      <td>-6.300</td>
      <td>0.0499</td>
      <td>0.114000</td>
      <td>0.000697</td>
      <td>0.0881</td>
      <td>0.496</td>
      <td>129.884</td>
      <td>236308.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12</td>
      <td>0.836</td>
      <td>0.799</td>
      <td>7</td>
      <td>-4.247</td>
      <td>0.0873</td>
      <td>0.187000</td>
      <td>0.000000</td>
      <td>0.0920</td>
      <td>0.772</td>
      <td>94.033</td>
      <td>217653.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>41</td>
      <td>NaN</td>
      <td>0.616</td>
      <td>1</td>
      <td>-8.747</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.716</td>
      <td>145.461</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>52</td>
      <td>0.764</td>
      <td>0.594</td>
      <td>6</td>
      <td>-10.050</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.1450</td>
      <td>0.695</td>
      <td>87.261</td>
      <td>286441.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>36</td>
      <td>0.743</td>
      <td>0.860</td>
      <td>5</td>
      <td>-6.346</td>
      <td>0.0445</td>
      <td>0.226000</td>
      <td>0.000422</td>
      <td>0.0513</td>
      <td>0.687</td>
      <td>102.459</td>
      <td>259267.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>42</td>
      <td>0.573</td>
      <td>0.746</td>
      <td>10</td>
      <td>-4.894</td>
      <td>0.0421</td>
      <td>0.024900</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.134</td>
      <td>130.001</td>
      <td>188000.0</td>
    </tr>
  </tbody>
</table>
</div>



## Missing Values Imputation

The next step in our data cleansing task will be to fill the missing values.

I will be using **MICE - Multiple Imputation by Chained Equations**, a data imputation technique which uses regression model to predict missing values. MICE is the most relevant technique here since we are dealing with missing continuous values, and there are other continuous features that can be used as predictors for the regression.

But first, lets visualize the missing values in each column - using the `missingno` package:


```python
import missingno as msno

### plot the missingness (nullity) matrix
missingdata_df = df_num_nan.columns[df_num_nan.isnull().any()].tolist()
msno.matrix(df_num_nan[missingdata_df])
plt.show()
```


    
![png](03_data_cleansing_files/03_data_cleansing_12_0.png)
    


Another interesting thing to check is to see if there is a correlation between the missing values in each column. Meaning, if a missing value in a column A can indicate a missing value in column B:


```python
## missingness correlation heatmap

msno.heatmap(df_num_nan[missingdata_df], figsize=(5,5))
plt.show()
```


    
![png](03_data_cleansing_files/03_data_cleansing_14_0.png)
    


### Using MICE to impute missing values


```python
import fancyimpute
df_num_mice = fancyimpute.IterativeImputer(max_iter=15).fit_transform(df_num_nan)
df_num_mice = pd.DataFrame(df_num_mice, columns=df_num_nan.columns)
df_num_mice.head(10)
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
      <th>track_popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41.0</td>
      <td>0.68200</td>
      <td>0.475927</td>
      <td>2.0</td>
      <td>-10.068</td>
      <td>0.023600</td>
      <td>0.279000</td>
      <td>0.000548</td>
      <td>0.088700</td>
      <td>0.566</td>
      <td>97.091</td>
      <td>235440.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>0.58200</td>
      <td>0.704000</td>
      <td>5.0</td>
      <td>-6.242</td>
      <td>0.034700</td>
      <td>0.065100</td>
      <td>0.000000</td>
      <td>0.212000</td>
      <td>0.698</td>
      <td>150.863</td>
      <td>197286.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.0</td>
      <td>0.61656</td>
      <td>0.880000</td>
      <td>9.0</td>
      <td>-4.739</td>
      <td>0.044200</td>
      <td>0.011700</td>
      <td>0.000876</td>
      <td>0.145307</td>
      <td>0.404</td>
      <td>135.225</td>
      <td>215036.905676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>0.65900</td>
      <td>0.794000</td>
      <td>10.0</td>
      <td>-5.644</td>
      <td>0.054000</td>
      <td>0.000761</td>
      <td>0.000599</td>
      <td>0.322000</td>
      <td>0.852</td>
      <td>128.041</td>
      <td>228565.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38.0</td>
      <td>0.66200</td>
      <td>0.838000</td>
      <td>1.0</td>
      <td>-6.300</td>
      <td>0.049900</td>
      <td>0.114000</td>
      <td>0.000697</td>
      <td>0.088100</td>
      <td>0.496</td>
      <td>129.884</td>
      <td>236308.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.0</td>
      <td>0.83600</td>
      <td>0.799000</td>
      <td>7.0</td>
      <td>-4.247</td>
      <td>0.087300</td>
      <td>0.187000</td>
      <td>0.000000</td>
      <td>0.092000</td>
      <td>0.772</td>
      <td>94.033</td>
      <td>217653.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>41.0</td>
      <td>0.69130</td>
      <td>0.616000</td>
      <td>1.0</td>
      <td>-8.747</td>
      <td>0.058062</td>
      <td>0.112745</td>
      <td>0.000000</td>
      <td>0.126999</td>
      <td>0.716</td>
      <td>145.461</td>
      <td>219510.717176</td>
    </tr>
    <tr>
      <th>7</th>
      <td>52.0</td>
      <td>0.76400</td>
      <td>0.594000</td>
      <td>6.0</td>
      <td>-10.050</td>
      <td>0.062445</td>
      <td>0.144667</td>
      <td>0.000000</td>
      <td>0.145000</td>
      <td>0.695</td>
      <td>87.261</td>
      <td>286441.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>36.0</td>
      <td>0.74300</td>
      <td>0.860000</td>
      <td>5.0</td>
      <td>-6.346</td>
      <td>0.044500</td>
      <td>0.226000</td>
      <td>0.000422</td>
      <td>0.051300</td>
      <td>0.687</td>
      <td>102.459</td>
      <td>259267.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>42.0</td>
      <td>0.57300</td>
      <td>0.746000</td>
      <td>10.0</td>
      <td>-4.894</td>
      <td>0.042100</td>
      <td>0.024900</td>
      <td>0.000000</td>
      <td>0.145187</td>
      <td>0.134</td>
      <td>130.001</td>
      <td>188000.000000</td>
    </tr>
  </tbody>
</table>
</div>



Lets see the status of missing values after applying MICE:


```python
msno.matrix(df_num_mice)
plt.show()
```


    
![png](03_data_cleansing_files/03_data_cleansing_18_0.png)
    



```python
df_num_mice.isnull().sum()
```




    track_popularity    0
    danceability        0
    energy              0
    key                 0
    loudness            0
    speechiness         0
    acousticness        0
    instrumentalness    0
    liveness            0
    valence             0
    tempo               0
    duration_ms         0
    dtype: int64



### Visualizing imputed values

Lets have some fun. Currently we have three data sets:

1. `df_num` - original dataframe with no missing values.
2. `df_num_nan` - dataframe with some of its values set to `np.nan`, due to being outlier values.
3. `df_num_mice` - no missing values because the MICE imputation technique was used to fill them from `df_num_nan`

Lets visualize this! I want to see the original distribution (with outliers), then the kept values (those who were inside the IQR based bounds), and the imputed data - all on the same plot.

Lets start with the `energy` column:


```python
# Define the mask for imputed values (where df_num_nan had NaNs)
imputed_mask = df_num_nan["energy"].isna()

plt.figure(figsize=(10, 5))
dot_size = 4  # Smaller dots to reduce clutter

# Plot original values (grey, transparent)
plt.scatter(df_num.index, df_num["energy"], color='grey', alpha=0.5, s=dot_size, label="Original (df_num)")

# Plot df_num_nan values (blue)
plt.scatter(df_num_nan.index, df_num_nan["energy"], color='blue', s=dot_size, label="With NaNs (df_num_nan)")

# Plot only imputed values (orange)
plt.scatter(df_num.index[imputed_mask], df_num_mice.loc[imputed_mask, "energy"], 
            color='orange', s=dot_size, label="Imputed (df_num_mice)")

plt.xlabel("Index")
plt.ylabel("Energy")
plt.legend()
plt.title("Energy Column Comparison")

plt.show()

```


    
![png](03_data_cleansing_files/03_data_cleansing_21_0.png)
    


So cool. Lets fo this for each column in `df_num`:


```python
import numpy as np
import matplotlib.pyplot as plt

# List of columns to visualize
columns = ["instrumentalness", "speechiness", "acousticness", "liveness", 
           "duration_ms", "loudness", "tempo", "danceability", "energy"]

num_cols = len(columns)

# Create subplots (adjusting the figure size dynamically)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, num_cols * 3), sharex=True)

dot_size = 4  # Smaller dots to reduce clutter

for ax, col in zip(axes, columns):
    # Identify the imputed values for the current column
    imputed_mask = df_num_nan[col].isna()

    # Plot original values (grey, transparent)
    ax.scatter(df_num.index, df_num[col], color='grey', alpha=0.5, s=dot_size, label="Original (df_num)")

    # Plot df_num_nan values (blue)
    ax.scatter(df_num_nan.index, df_num_nan[col], color='blue', s=dot_size, label="With NaNs (df_num_nan)")

    # Plot only imputed values (orange)
    ax.scatter(df_num.index[imputed_mask], df_num_mice.loc[imputed_mask, col], 
               color='orange', s=dot_size, label="Imputed (df_num_mice)")

    # Formatting
    ax.set_ylabel(col)
    ax.legend()

# Add common X label
axes[-1].set_xlabel("Index")

plt.suptitle("Comparison of Original, NaN, and Imputed Values Across Features")
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to fit title
plt.show()

```


    
![png](03_data_cleansing_files/03_data_cleansing_23_0.png)
    


### Saving imputed data

Finally lets "merge" the imputed data back into the original dataframe, so we can use it in the next chapter!


```python
df_final = df.copy()
for column in df_num_mice.columns:
    df_final[column] = df_num_mice[column]

df_final.head()
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
      <th>track_artist</th>
      <th>track_popularity</th>
      <th>track_album_id</th>
      <th>track_album_release_date</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>...</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>playlist_count</th>
      <th>edm</th>
      <th>latin</th>
      <th>pop</th>
      <th>r&amp;b</th>
      <th>rap</th>
      <th>rock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0017A6SJgTbfQVU2EtsPNo</td>
      <td>Barbie's Cradle</td>
      <td>41.0</td>
      <td>1srJQ0njEQgd8w4XSqI4JQ</td>
      <td>2001-01-01</td>
      <td>0.68200</td>
      <td>0.475927</td>
      <td>2.0</td>
      <td>-10.068</td>
      <td>1</td>
      <td>...</td>
      <td>0.566</td>
      <td>97.091</td>
      <td>235440.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>002xjHwzEx66OWFV2IP9dk</td>
      <td>RIKA</td>
      <td>15.0</td>
      <td>1ficfUnZMaY1QkNp15Slzm</td>
      <td>2018-01-26</td>
      <td>0.58200</td>
      <td>0.704000</td>
      <td>5.0</td>
      <td>-6.242</td>
      <td>1</td>
      <td>...</td>
      <td>0.698</td>
      <td>150.863</td>
      <td>197286.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>004s3t0ONYlzxII9PLgU6z</td>
      <td>Steady Rollin</td>
      <td>28.0</td>
      <td>3z04Lb9Dsilqw68SHt6jLB</td>
      <td>2017-11-21</td>
      <td>0.61656</td>
      <td>0.880000</td>
      <td>9.0</td>
      <td>-4.739</td>
      <td>1</td>
      <td>...</td>
      <td>0.404</td>
      <td>135.225</td>
      <td>215036.905676</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>008MceT31RotUANsKuzy3L</td>
      <td>The.madpix.project</td>
      <td>24.0</td>
      <td>1Z4ANBVuhTlS6DprlP0m1q</td>
      <td>2015-08-07</td>
      <td>0.65900</td>
      <td>0.794000</td>
      <td>10.0</td>
      <td>-5.644</td>
      <td>0</td>
      <td>...</td>
      <td>0.852</td>
      <td>128.041</td>
      <td>228565.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>008rk8F6ZxspZT4bUlkIQG</td>
      <td>YOSA &amp; TAAR</td>
      <td>38.0</td>
      <td>2BuYm9UcKvI0ydXs5JKwt0</td>
      <td>2018-11-16</td>
      <td>0.66200</td>
      <td>0.838000</td>
      <td>1.0</td>
      <td>-6.300</td>
      <td>1</td>
      <td>...</td>
      <td>0.496</td>
      <td>129.884</td>
      <td>236308.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
df_final.shape
```




    (28356, 24)



### Handling missing artist names

A missing track artist name cannot be predicted sing regression model - however it does not need to.
In the next chapter Feature Engineering I will use label encoding for the artist name, so for now lets just fill missing names as an "Anonymous" artist. There are only four of those tracks with no artist, so it can be considered its own group in future encoding.


```python
df_final['track_artist'].fillna('Anonymous Artist', inplace=True)
df_final.head()
```

    /tmp/ipykernel_13544/1595701375.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df_final['track_artist'].fillna('Anonymous Artist', inplace=True)





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
      <th>track_artist</th>
      <th>track_popularity</th>
      <th>track_album_id</th>
      <th>track_album_release_date</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>...</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>playlist_count</th>
      <th>edm</th>
      <th>latin</th>
      <th>pop</th>
      <th>r&amp;b</th>
      <th>rap</th>
      <th>rock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0017A6SJgTbfQVU2EtsPNo</td>
      <td>Barbie's Cradle</td>
      <td>41.0</td>
      <td>1srJQ0njEQgd8w4XSqI4JQ</td>
      <td>2001-01-01</td>
      <td>0.68200</td>
      <td>0.475927</td>
      <td>2.0</td>
      <td>-10.068</td>
      <td>1</td>
      <td>...</td>
      <td>0.566</td>
      <td>97.091</td>
      <td>235440.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>002xjHwzEx66OWFV2IP9dk</td>
      <td>RIKA</td>
      <td>15.0</td>
      <td>1ficfUnZMaY1QkNp15Slzm</td>
      <td>2018-01-26</td>
      <td>0.58200</td>
      <td>0.704000</td>
      <td>5.0</td>
      <td>-6.242</td>
      <td>1</td>
      <td>...</td>
      <td>0.698</td>
      <td>150.863</td>
      <td>197286.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>004s3t0ONYlzxII9PLgU6z</td>
      <td>Steady Rollin</td>
      <td>28.0</td>
      <td>3z04Lb9Dsilqw68SHt6jLB</td>
      <td>2017-11-21</td>
      <td>0.61656</td>
      <td>0.880000</td>
      <td>9.0</td>
      <td>-4.739</td>
      <td>1</td>
      <td>...</td>
      <td>0.404</td>
      <td>135.225</td>
      <td>215036.905676</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>008MceT31RotUANsKuzy3L</td>
      <td>The.madpix.project</td>
      <td>24.0</td>
      <td>1Z4ANBVuhTlS6DprlP0m1q</td>
      <td>2015-08-07</td>
      <td>0.65900</td>
      <td>0.794000</td>
      <td>10.0</td>
      <td>-5.644</td>
      <td>0</td>
      <td>...</td>
      <td>0.852</td>
      <td>128.041</td>
      <td>228565.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>008rk8F6ZxspZT4bUlkIQG</td>
      <td>YOSA &amp; TAAR</td>
      <td>38.0</td>
      <td>2BuYm9UcKvI0ydXs5JKwt0</td>
      <td>2018-11-16</td>
      <td>0.66200</td>
      <td>0.838000</td>
      <td>1.0</td>
      <td>-6.300</td>
      <td>1</td>
      <td>...</td>
      <td>0.496</td>
      <td>129.884</td>
      <td>236308.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
null_values = df_final.isnull().sum()
print(null_values)

```

    track_id                    0
    track_artist                0
    track_popularity            0
    track_album_id              0
    track_album_release_date    0
    danceability                0
    energy                      0
    key                         0
    loudness                    0
    mode                        0
    speechiness                 0
    acousticness                0
    instrumentalness            0
    liveness                    0
    valence                     0
    tempo                       0
    duration_ms                 0
    playlist_count              0
    edm                         0
    latin                       0
    pop                         0
    r&b                         0
    rap                         0
    rock                        0
    dtype: int64


## Save as pickle and csv

We finished the data preparation part! It is time to store the df as a pickle file for the next chapter :-)


```python
df.to_pickle('pickle/03_data_cleansing/data_data_cleansing.pkl')
```


```python
df.to_csv('data/03_data_cleansing/data_cleansing.csv')
```

### Processed Dataset Overview


```python
display(df.shape)

df.info()
```


    (28356, 24)


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28356 entries, 0 to 28355
    Data columns (total 24 columns):
     #   Column                    Non-Null Count  Dtype         
    ---  ------                    --------------  -----         
     0   track_id                  28356 non-null  object        
     1   track_artist              28352 non-null  object        
     2   track_popularity          28356 non-null  int64         
     3   track_album_id            28356 non-null  object        
     4   track_album_release_date  28356 non-null  datetime64[ns]
     5   danceability              28356 non-null  float64       
     6   energy                    28356 non-null  float64       
     7   key                       28356 non-null  int64         
     8   loudness                  28356 non-null  float64       
     9   mode                      28356 non-null  int64         
     10  speechiness               28356 non-null  float64       
     11  acousticness              28356 non-null  float64       
     12  instrumentalness          28356 non-null  float64       
     13  liveness                  28356 non-null  float64       
     14  valence                   28356 non-null  float64       
     15  tempo                     28356 non-null  float64       
     16  duration_ms               28356 non-null  int64         
     17  playlist_count            28356 non-null  int64         
     18  edm                       28356 non-null  int64         
     19  latin                     28356 non-null  int64         
     20  pop                       28356 non-null  int64         
     21  r&b                       28356 non-null  int64         
     22  rap                       28356 non-null  int64         
     23  rock                      28356 non-null  int64         
    dtypes: datetime64[ns](1), float64(9), int64(11), object(3)
    memory usage: 5.2+ MB


![tobecontinued.jpg](03_data_cleansing_files/tobecontinued.jpg)
