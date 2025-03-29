# Feature Engineering


```python
# DS18 ML Essentials project
# Module 4: Feature Engineering

# Submitted by: Tzvi Eliezer Nir
# mail: tzvienir@gmail.com
# First submission: 27/03/2025
```

## In this Notebook

In this notebook, we will apply some chosen feature enrichment methods to our data. This will provide our model with additional information to work with — hopefully leading to better results.

We will also engage in feature engineering, which is crucial as it transforms "raw" data into meaningful features that enhance the performance of machine learning models.

### Chosen Features

The new features can be classified into three groups:

#### Parsing the `track_album_release_date` column into more meaningful features:
1. A textbook feature engineering method is to split a datetime into `Year`, `Month`, and `Day`.
2. Additionally, we are working with music! As we all know, music has **decades** (personally, I'm crazy for 2010s EDM and 1990s rock). Categorizing the tracks by their decades could be meaningful.
3. The internet and the streaming industry have drastically changed the way we consume music, which may affect popularity patterns as well. An additional binary column indicating whether or not the track was released in the internet era will be created.

#### Analyzing the `track_name` textual column:

Remember when we moved all free-text columns into `df_text` back in [creating the flat file](01_data_preparation.ipynb)? Now it's the last chance to put it to use before we say goodbye for good. Using **`wordcloud`**, we will analyze the top-mentioned words in the `track_name` column and create dummy (binary) columns to indicate whether the current track contains the frequent word in its name.

#### Considering Artist Followers

The original dataset has no unique identifier for the artists; it simply provides their names as strings. This presents a challenge because there is not much we can do with the names: encoding the artist name will not be meaningful since there are too many of them (10k unique values), and raw text would not help our future model.

So, I took a different approach — the `track_artist` column will be replaced with the number of artist followers, a new feature derived from an external dataset!


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_pickle("./pickle/03_data_cleansing/data_cleansing.pkl")
```

## Parsing datetime into multiple new features

### Year, Month, Day & Decade

Lets start with the trivial step of separating the datetime to its components + decade. We can drop the original `track_album_release_date` column once we finished:


```python
# Extract year, month, and day from track_album_release_date
df['year'] = df['track_album_release_date'].dt.year
df['month'] = df['track_album_release_date'].dt.month
df['day'] = df['track_album_release_date'].dt.day

# Calculate the decade
df['decade'] = (df['year'] // 10) * 10

# drop the track_album_release_date column
df = df.drop('track_album_release_date', axis=1)

# Display the updated dataframe
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
      <th>track_artist</th>
      <th>track_popularity</th>
      <th>track_album_id</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>...</th>
      <th>edm</th>
      <th>latin</th>
      <th>pop</th>
      <th>r&amp;b</th>
      <th>rap</th>
      <th>rock</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0017A6SJgTbfQVU2EtsPNo</td>
      <td>Barbie's Cradle</td>
      <td>41</td>
      <td>1srJQ0njEQgd8w4XSqI4JQ</td>
      <td>0.682</td>
      <td>0.401</td>
      <td>2</td>
      <td>-10.068</td>
      <td>1</td>
      <td>0.0236</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2001</td>
      <td>1</td>
      <td>1</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>002xjHwzEx66OWFV2IP9dk</td>
      <td>RIKA</td>
      <td>15</td>
      <td>1ficfUnZMaY1QkNp15Slzm</td>
      <td>0.582</td>
      <td>0.704</td>
      <td>5</td>
      <td>-6.242</td>
      <td>1</td>
      <td>0.0347</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2018</td>
      <td>1</td>
      <td>26</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>004s3t0ONYlzxII9PLgU6z</td>
      <td>Steady Rollin</td>
      <td>28</td>
      <td>3z04Lb9Dsilqw68SHt6jLB</td>
      <td>0.303</td>
      <td>0.880</td>
      <td>9</td>
      <td>-4.739</td>
      <td>1</td>
      <td>0.0442</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2017</td>
      <td>11</td>
      <td>21</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>008MceT31RotUANsKuzy3L</td>
      <td>The.madpix.project</td>
      <td>24</td>
      <td>1Z4ANBVuhTlS6DprlP0m1q</td>
      <td>0.659</td>
      <td>0.794</td>
      <td>10</td>
      <td>-5.644</td>
      <td>0</td>
      <td>0.0540</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>8</td>
      <td>7</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>008rk8F6ZxspZT4bUlkIQG</td>
      <td>YOSA &amp; TAAR</td>
      <td>38</td>
      <td>2BuYm9UcKvI0ydXs5JKwt0</td>
      <td>0.662</td>
      <td>0.838</td>
      <td>1</td>
      <td>-6.300</td>
      <td>1</td>
      <td>0.0499</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2018</td>
      <td>11</td>
      <td>16</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



Having the decade in a separate column is nice, but does it really helps? I mean does the mean popularity really varies between decades? Using an ANOVA test answers this exact question:


```python
import scipy.stats as stats

decades = df['decade'].unique()
popularity_by_decade = {decade: df[df['decade'] == decade]['track_popularity'] for decade in decades}

f_statistic, p_value = stats.f_oneway(*popularity_by_decade.values())

print(f"F-statistic: {f_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference between the group means")
else:
    print("No significant difference between the group means")
```

    F-statistic: 71.52541039234148
    p-value: 4.879105730122917e-103
    There is a significant difference between the group means


Lets visualize it!


```python
plt.figure(figsize=(14, 3))

# Define a color palette
palette = sns.color_palette("husl", len(decades))

# # Plot histograms for each decade
# for i, decade in enumerate(decades):
#     sns.histplot(popularity_by_decade[decade], kde=False, label=f'{decade}', bins=30, alpha=0.5, color=palette[i])

# Add vertical lines for the mean popularity of each decade
for i, decade in enumerate(decades):
    mean_popularity = popularity_by_decade[decade].mean()
    plt.axvline(mean_popularity, linestyle='--', label=f'{decade} mean', color=palette[i])

plt.title('Track Popularity Distribution by Decade')
plt.xlabel('Track Popularity')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```


    
![png](04_feature_engineering_files/04_feature_engineering_9_0.png)
    


### The internet era

The internet have changed the way we consume music forever, so popularity patterns might have changed as well. But, when exactly shall we draw the line for the internet to take over our life? A good split point for "before/after the internet era" in music could be based on key technological and industry shifts. Here are some possible points:

    1999 – Napster's Launch: Marked the start of widespread digital music sharing, disrupting traditional music distribution.

    2003 – iTunes Store Launch: Made legal digital music downloads mainstream.

    2005 – YouTube's Launch: Video-based music discovery exploded.

    2008 – Spotify's Launch: Streaming became a dominant distribution model.

    2010s – Streaming Overtakes Downloads & CDs: The music industry fully transitioned to streaming as the primary consumption method.

Since this project is about Spotify, I decided out of respect to draw the line at 2008 - Spotify launch year:


```python
df['released_in_internet_era'] = (df['year'] > 2008).astype(int)

popularity_by_internet_era = {era: df[df['released_in_internet_era'] == era]['track_popularity'] for era in df['released_in_internet_era'].unique()}

f_statistic, p_value = stats.f_oneway(*popularity_by_internet_era.values())

print(f"F-statistic: {f_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference between the group means")
else:
    print("No significant difference between the group means")

plt.figure(figsize=(14, 8))

# Define a color palette
palette = sns.color_palette("husl", len(popularity_by_internet_era))

# Plot histograms for each era
for i, (era, popularity) in enumerate(popularity_by_internet_era.items()):
    sns.histplot(popularity, kde=False, label=f'{"Internet Era" if era == 1 else "Pre-Internet Era"}', bins=30, alpha=0.5, color=palette[i])

# Add vertical lines for the mean popularity of each era
for i, (era, popularity) in enumerate(popularity_by_internet_era.items()):
    mean_popularity = popularity.mean()
    plt.axvline(mean_popularity, linestyle='--', label=f'{"Internet Era" if era == 1 else "Pre-Internet Era"} mean', color=palette[i])

plt.title('Track Popularity Distribution by Internet Era')
plt.xlabel('Track Popularity')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

    F-statistic: 102.82085018856533
    p-value: 4.033459112969597e-24
    There is a significant difference between the group means



    
![png](04_feature_engineering_files/04_feature_engineering_11_1.png)
    


It seems that track popularity does vary before/after the internet era.

But wait, isn't this contribution redundant since we already created a `Decade` column? well.. yeah it kind of is. But we will let the feature selection step to compare and decide which of the methods to keep, so it won't create bias in the final model.

## Analyzing `track_name` text attributes

Back in [data preparation](01_data_preparation.ipynb), we moved all text-based features into a separate dataframe called `df_text`. This is our last chance to extract some useful insights from these text columns before we drop them for good.

![My_Time_Has_Come.jpg](04_feature_engineering_files/My_Time_Has_Come.jpg)

first lets load it, and drop the `playlist_name` we won't need it.


```python
df_text = pd.read_pickle("./pickle/01_data_preparation/df_text.pkl")
df_text = df_text.drop(columns=['playlist_name'])
df_text = df_text.drop_duplicates(subset='track_id')

df_text.describe()
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
      <th>track_name</th>
      <th>track_album_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28356</td>
      <td>28352</td>
      <td>28352</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>28356</td>
      <td>23449</td>
      <td>19743</td>
    </tr>
    <tr>
      <th>top</th>
      <td>6f807x0ima9a1j3VPbc7VN</td>
      <td>Breathe</td>
      <td>Greatest Hits</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>18</td>
      <td>135</td>
    </tr>
  </tbody>
</table>
</div>



A good feature engineering will be to analyze popular words in the `track_name`. We will choose the top-5 words, and those words will be used to create five new binary columns - indicating if the current track name contain the popular word.

We will start our analysis using the `wordcloud` package. In addition to get us the answers we want - it also creates nice visualization:


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all track names into a single string
text = ' '.join(df_text['track_name'].dropna().values)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Get the frequencies of the words
word_frequencies = wordcloud.words_

# Print the top 20 words
top_20_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
print(top_20_words)
```


    
![png](04_feature_engineering_files/04_feature_engineering_18_0.png)
    


    [('feat', 1.0), ('Remix', 0.8182837913628715), ('Love', 0.5339315759955132), ('Radio Edit', 0.2641615255187886), ('Remastered', 0.19125070106561975), ('Original Mix', 0.15872125630959058), ('Remaster', 0.1441390914189568), ('One', 0.14133482893998878), ('U', 0.12226584408300617), ('Back', 0.12114413909141895), ('Way', 0.12002243409983174), ('Night', 0.11721817162086372), ('Time', 0.11497476163768929), ('Go', 0.1076836791923724), ('Life', 0.10600112170499158), ('Let', 0.10487941671340438), ('Girl', 0.10431856421761077), ('Edit', 0.0975883342680875), ('Live', 0.09590577678070668), ('Know', 0.09478407178911946)]


To be hones I'm a bit disappointed, I hoped for more meaningful words. *Remix* and *Love* are cool, but the other 3 are kinda boring. But the data is what the data is, so lets continue as planned:


```python
# Create new features based on the top 5 words
for word, _ in top_20_words[:5]:
    df_text[word] = df_text['track_name'].apply(lambda x: 0 if pd.isna(x) else int(word.lower() in x.lower()))

df_text.sample(20)

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
      <th>track_name</th>
      <th>track_album_name</th>
      <th>feat</th>
      <th>Remix</th>
      <th>Love</th>
      <th>Radio Edit</th>
      <th>Remastered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15138</th>
      <td>4XETUJmrPQnmPd9elRiRE9</td>
      <td>Rorschach</td>
      <td>Ataraxis</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>927</th>
      <td>1bSO9nzT5h55OrJO7BbHL2</td>
      <td>One Life - VIP Mix</td>
      <td>One Life (VIP Mix)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29304</th>
      <td>1qZMPmpD1jDcOA7gZ6TCde</td>
      <td>One (Your Name) - Radio Edit</td>
      <td>One (Your Name)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4372</th>
      <td>6TSlbHp7Vx4wZ0Rqciwn5v</td>
      <td>Til I Don't</td>
      <td>Til I Don't</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30868</th>
      <td>1HfPcQ3c2HyGeID3u1lmCa</td>
      <td>She Wolf (Falling to Pieces) (feat. Sia)</td>
      <td>Nothing but the Beat 2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28764</th>
      <td>2LFlu4XDVjO8czYT2pdf3q</td>
      <td>Tarantella</td>
      <td>Tarantella</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23577</th>
      <td>2dcZFrC8XkdxhVxmb8p2kO</td>
      <td>Brain Damage</td>
      <td>Lizard Vision</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25370</th>
      <td>224EQMt8CpNPV98j9L4iMJ</td>
      <td>No Lie (feat. WizKid)</td>
      <td>Diaspora</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26452</th>
      <td>7zL1R4SiaifOwieMmxsG7i</td>
      <td>Who</td>
      <td>Simply a Vessel, Vol 3: Surrender All</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3307</th>
      <td>1foeacjwgWD6UMmirTXwL5</td>
      <td>Por Ti</td>
      <td>Dulce Beat</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17577</th>
      <td>4ZmAMOU0bcmrwwOvEK8aDT</td>
      <td>Quién Diría</td>
      <td>Canciones De Amor</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30836</th>
      <td>7pr168TcvrcajZoKC0qxi7</td>
      <td>Tokyo</td>
      <td>Tokyo</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15711</th>
      <td>4BSjvDSi3ZtrSavQqgu6jO</td>
      <td>Got Your Six</td>
      <td>Got Your Six (Deluxe)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9746</th>
      <td>3Jw774vf185xkKIUZWySx5</td>
      <td>Asi Son Mis Dias</td>
      <td>Mucho Barato</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24938</th>
      <td>765k9tDIFOnoOfkO2cgitB</td>
      <td>Take Me with U</td>
      <td>Purple Rain</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27284</th>
      <td>29T5wjKGRJ2JAgWodMQGyf</td>
      <td>Be My Oxygen</td>
      <td>Be My Oxygen</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15916</th>
      <td>5zsgyaahJyDW91qyvzSG3Y</td>
      <td>Guilty All the Same (feat. Rakim)</td>
      <td>Guilty All the Same (feat. Rakim)</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10093</th>
      <td>7JCp9oPXLc8Akz9bZpYZ09</td>
      <td>Vamos Pa' Encimota</td>
      <td>Vamos Pa' Encimota</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32309</th>
      <td>3sz8Gn0fOmXmyXOaPSWBJ9</td>
      <td>Sweet Escape</td>
      <td>Forever</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29260</th>
      <td>07KQ3yYHQz6nfx05dLDMY6</td>
      <td>We Remember</td>
      <td>We Remember</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select the relevant columns from df_text
df_text_features = df_text[['track_id', 'feat', 'Remix', 'Love', 'Radio Edit', 'Remastered']]

# Merge the features into df
df = df.merge(df_text_features, on='track_id', how='left')

```


```python
# Display the updated dataframe
df.sample(10)
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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>...</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>decade</th>
      <th>released_in_internet_era</th>
      <th>feat</th>
      <th>Remix</th>
      <th>Love</th>
      <th>Radio Edit</th>
      <th>Remastered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17048</th>
      <td>4hFV8xf1MD6Tmg2g0XltTY</td>
      <td>Slim Thug</td>
      <td>52</td>
      <td>6ZmRtymFNo8pU2PQno73lv</td>
      <td>0.781</td>
      <td>0.710</td>
      <td>1</td>
      <td>-3.683</td>
      <td>0</td>
      <td>0.0903</td>
      <td>...</td>
      <td>2009</td>
      <td>3</td>
      <td>24</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27752</th>
      <td>7qDV8EUMjj5sJdholWKUqk</td>
      <td>Martin Garrix</td>
      <td>42</td>
      <td>2XXJOpnawvArFxXTu2A1VM</td>
      <td>0.610</td>
      <td>0.907</td>
      <td>2</td>
      <td>-5.224</td>
      <td>1</td>
      <td>0.1540</td>
      <td>...</td>
      <td>2017</td>
      <td>8</td>
      <td>18</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15815</th>
      <td>4N1r70StM4WeymDNajIJQ6</td>
      <td>bad nelson</td>
      <td>22</td>
      <td>4t4YCMTeNolYI0RziTnUnO</td>
      <td>0.634</td>
      <td>0.906</td>
      <td>5</td>
      <td>-5.106</td>
      <td>0</td>
      <td>0.0492</td>
      <td>...</td>
      <td>2019</td>
      <td>10</td>
      <td>4</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19325</th>
      <td>5KVbaJP4IOZmZZlj5v4jp2</td>
      <td>Marsha Ambrosius</td>
      <td>48</td>
      <td>3l9KeT7TXfQKg8RhzoC6DI</td>
      <td>0.433</td>
      <td>0.668</td>
      <td>9</td>
      <td>-6.544</td>
      <td>1</td>
      <td>0.2110</td>
      <td>...</td>
      <td>2011</td>
      <td>2</td>
      <td>25</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6024</th>
      <td>1epsUiCmy5UvIIeN5QpKwP</td>
      <td>2nd Life</td>
      <td>49</td>
      <td>6N0Kl9owvaSy8JRnPFDfBb</td>
      <td>0.526</td>
      <td>0.773</td>
      <td>2</td>
      <td>-7.790</td>
      <td>0</td>
      <td>0.0513</td>
      <td>...</td>
      <td>2019</td>
      <td>10</td>
      <td>11</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26874</th>
      <td>7cG3wfohoNDSp2M8FWrgTg</td>
      <td>Paradisio</td>
      <td>59</td>
      <td>3iJwt0Sq44ZBzR7kNCxf0y</td>
      <td>0.649</td>
      <td>0.955</td>
      <td>8</td>
      <td>-7.817</td>
      <td>0</td>
      <td>0.0388</td>
      <td>...</td>
      <td>1996</td>
      <td>1</td>
      <td>1</td>
      <td>1990</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20796</th>
      <td>5jmyckW7sSzXcGrP1AI1Zs</td>
      <td>Stoltenhoff</td>
      <td>34</td>
      <td>1WlA9JfBZnrp6K2LCEVvbD</td>
      <td>0.694</td>
      <td>0.991</td>
      <td>7</td>
      <td>-3.045</td>
      <td>1</td>
      <td>0.1290</td>
      <td>...</td>
      <td>2019</td>
      <td>8</td>
      <td>5</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>0NIkKL6wsLK5V2vX97zOTr</td>
      <td>Five Finger Death Punch</td>
      <td>51</td>
      <td>3Ey9TgEz0LdFKFKKftpkN1</td>
      <td>0.564</td>
      <td>0.978</td>
      <td>1</td>
      <td>-4.294</td>
      <td>0</td>
      <td>0.1320</td>
      <td>...</td>
      <td>2013</td>
      <td>7</td>
      <td>30</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>406</th>
      <td>07AIGejOb0kM3S7mMHi7pm</td>
      <td>BROHUG</td>
      <td>44</td>
      <td>1ZSGPXoqQonfIn9FWWdhq9</td>
      <td>0.729</td>
      <td>0.915</td>
      <td>9</td>
      <td>-4.925</td>
      <td>1</td>
      <td>0.0376</td>
      <td>...</td>
      <td>2019</td>
      <td>4</td>
      <td>12</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6926</th>
      <td>1tpbAGsGpZee5jDHMcLrUJ</td>
      <td>Dalex</td>
      <td>82</td>
      <td>3RWeME5ryDw9wxO99OoDgP</td>
      <td>0.584</td>
      <td>0.698</td>
      <td>8</td>
      <td>-3.961</td>
      <td>1</td>
      <td>0.1060</td>
      <td>...</td>
      <td>2019</td>
      <td>9</td>
      <td>6</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>



## Consider artist followers

Having the `track_artist` name as a string is not meaningful in the context of classic ML models. Even encoding it will not help - there are 10K different artists!

So lets drop it.

Sorry, I meant lets solve the issue and then drop it. I got an external dataset of Spotify artists, lets see what it can do for us.


```python
df_artist = pd.read_csv('data/04_feature_engineering/external/artists.csv')
df_artist
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
      <th>followers</th>
      <th>genres</th>
      <th>name</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0DheY5irMjBUeLybbCUEZ2</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Armid &amp; Amir Zare Pashai feat. Sara Rouzbehani</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0DlhY15l3wsrnlfGio2bjU</td>
      <td>5.0</td>
      <td>[]</td>
      <td>ปูนา ภาวิณี</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0DmRESX2JknGPQyO15yxg7</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Sadaa</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0DmhnbHjm1qw6NCYPeZNgJ</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Tra'gruda</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0Dn11fWM7vHQ3rinvWEl4E</td>
      <td>2.0</td>
      <td>[]</td>
      <td>Ioannis Panoutsopoulos</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1162090</th>
      <td>3cOzi726Iav1toV2LRVEjp</td>
      <td>4831.0</td>
      <td>['black comedy']</td>
      <td>Ali Siddiq</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1162091</th>
      <td>6LogY6VMM3jgAE6fPzXeMl</td>
      <td>46.0</td>
      <td>[]</td>
      <td>Rodney Laney</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1162092</th>
      <td>19boQkDEIay9GaVAWkUhTa</td>
      <td>257.0</td>
      <td>[]</td>
      <td>Blake Wexler</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1162093</th>
      <td>5nvjpU3Y7L6Hpe54QuvDjy</td>
      <td>2357.0</td>
      <td>['black comedy']</td>
      <td>Donnell Rawlings</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1162094</th>
      <td>2bP2cNhNBdKXHC6AnqgyVp</td>
      <td>40.0</td>
      <td>['new comedy']</td>
      <td>Gabe Kea</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>1162095 rows × 5 columns</p>
</div>



I will **not** use artist popularity, this feels like cheating.

However the `followers` column is interesting... For each `track_artist` in `df` lets find it number of followers in `df_artist`:


```python
df, df_artist

# Create a dictionary for quick lookup of artist followers
artist_followers_dict = df_artist.set_index('name')['followers'].to_dict()

# Replace df['track_artist'] with corresponding followers
df['track_artist_followers'] = df['track_artist'].map(artist_followers_dict)

# Display the updated dataframe
df_sorted = df.sort_values(by='track_artist_followers', ascending=False)
df_sorted.head()


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
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
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
      <th>21453</th>
      <td>5ug4vqGZ3eisGhY1IsziNX</td>
      <td>Ed Sheeran</td>
      <td>74</td>
      <td>6Z5DhADmyybfKNdymaPLjB</td>
      <td>0.565</td>
      <td>0.242</td>
      <td>2</td>
      <td>-8.367</td>
      <td>1</td>
      <td>0.0318</td>
      <td>...</td>
      <td>7</td>
      <td>5</td>
      <td>2010</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78900234.0</td>
    </tr>
    <tr>
      <th>19955</th>
      <td>5Va44ERiHtC2tfBD53OApA</td>
      <td>Ed Sheeran</td>
      <td>59</td>
      <td>0kHdDpkkSlesh2UMKhF20G</td>
      <td>0.747</td>
      <td>0.649</td>
      <td>10</td>
      <td>-6.218</td>
      <td>0</td>
      <td>0.2190</td>
      <td>...</td>
      <td>7</td>
      <td>11</td>
      <td>2010</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78900234.0</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>0jmTp8jaCOm78zrdzdvmaq</td>
      <td>Ed Sheeran</td>
      <td>0</td>
      <td>2DL4AOHc4MKvCHbUr544aU</td>
      <td>0.733</td>
      <td>0.821</td>
      <td>4</td>
      <td>-4.982</td>
      <td>1</td>
      <td>0.0407</td>
      <td>...</td>
      <td>2</td>
      <td>16</td>
      <td>2010</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78900234.0</td>
    </tr>
    <tr>
      <th>25430</th>
      <td>70eFcWOvlMObDhURTqT4Fv</td>
      <td>Ed Sheeran</td>
      <td>85</td>
      <td>3oIFxDIo2fwuk4lwCmFZCx</td>
      <td>0.640</td>
      <td>0.648</td>
      <td>5</td>
      <td>-8.113</td>
      <td>0</td>
      <td>0.1870</td>
      <td>...</td>
      <td>7</td>
      <td>12</td>
      <td>2010</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78900234.0</td>
    </tr>
    <tr>
      <th>4150</th>
      <td>19TOAlTFq0NDHvUPQR0tkr</td>
      <td>Ed Sheeran</td>
      <td>69</td>
      <td>3BjxjIkTZKUpeZ6n5MYMNx</td>
      <td>0.845</td>
      <td>0.766</td>
      <td>2</td>
      <td>-5.727</td>
      <td>0</td>
      <td>0.0658</td>
      <td>...</td>
      <td>9</td>
      <td>27</td>
      <td>2010</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78900234.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



There is a little issue to solve here - unlike us the creator of the artist dataset didn't fill all null values.

I will fix it using the median as a constant replacement for missing values:


```python
df['track_artist_followers'] = df['track_artist_followers'].fillna(df['track_artist_followers'].median())

```


```python
nan_count = df['track_artist_followers'].isna().sum()
print(f"Number of NaN values in 'track_artist_followers': {nan_count}")
```

    Number of NaN values in 'track_artist_followers': 0


We can drop `track_artist` now, we got all the data we can from it, and the original strings are not meaningful for our model:


```python
df = df.drop(columns=['track_artist'])
```

## Summary

Lets have a look at our dataset, now that we have added all of those cool features!


```python
df.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>track_popularity</th>
      <td>28356.0</td>
      <td>3.932977e+01</td>
      <td>2.370238e+01</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>42.000000</td>
      <td>5.800000e+01</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>28356.0</td>
      <td>6.533723e-01</td>
      <td>1.457853e-01</td>
      <td>0.000000</td>
      <td>0.561000</td>
      <td>0.670000</td>
      <td>7.600000e-01</td>
      <td>9.830000e-01</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>28356.0</td>
      <td>6.983875e-01</td>
      <td>1.835030e-01</td>
      <td>0.000175</td>
      <td>0.579000</td>
      <td>0.722000</td>
      <td>8.430000e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>key</th>
      <td>28356.0</td>
      <td>5.368000e+00</td>
      <td>3.613904e+00</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>9.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>28356.0</td>
      <td>-6.817696e+00</td>
      <td>3.036243e+00</td>
      <td>-46.448000</td>
      <td>-8.309250</td>
      <td>-6.261000</td>
      <td>-4.709000e+00</td>
      <td>1.275000e+00</td>
    </tr>
    <tr>
      <th>mode</th>
      <td>28356.0</td>
      <td>5.654888e-01</td>
      <td>4.957014e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>28356.0</td>
      <td>1.079536e-01</td>
      <td>1.025562e-01</td>
      <td>0.000000</td>
      <td>0.041000</td>
      <td>0.062600</td>
      <td>1.330000e-01</td>
      <td>9.180000e-01</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>28356.0</td>
      <td>1.771759e-01</td>
      <td>2.228029e-01</td>
      <td>0.000000</td>
      <td>0.014375</td>
      <td>0.079700</td>
      <td>2.600000e-01</td>
      <td>9.940000e-01</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>28356.0</td>
      <td>9.111682e-02</td>
      <td>2.325484e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000021</td>
      <td>6.570000e-03</td>
      <td>9.940000e-01</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>28356.0</td>
      <td>1.909582e-01</td>
      <td>1.558943e-01</td>
      <td>0.000000</td>
      <td>0.092600</td>
      <td>0.127000</td>
      <td>2.490000e-01</td>
      <td>9.960000e-01</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>28356.0</td>
      <td>5.103866e-01</td>
      <td>2.343399e-01</td>
      <td>0.000000</td>
      <td>0.329000</td>
      <td>0.512000</td>
      <td>6.950000e-01</td>
      <td>9.910000e-01</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>28356.0</td>
      <td>1.209562e+02</td>
      <td>2.695456e+01</td>
      <td>0.000000</td>
      <td>99.972000</td>
      <td>121.993000</td>
      <td>1.339990e+02</td>
      <td>2.394400e+02</td>
    </tr>
    <tr>
      <th>duration_ms</th>
      <td>28356.0</td>
      <td>2.265760e+05</td>
      <td>6.107845e+04</td>
      <td>4000.000000</td>
      <td>187742.000000</td>
      <td>216933.000000</td>
      <td>2.549752e+05</td>
      <td>5.178100e+05</td>
    </tr>
    <tr>
      <th>playlist_count</th>
      <td>28356.0</td>
      <td>1.157885e+00</td>
      <td>5.438341e-01</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <th>edm</th>
      <td>28356.0</td>
      <td>2.400903e-01</td>
      <td>4.271456e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>28356.0</td>
      <td>1.636691e-01</td>
      <td>3.699815e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>28356.0</td>
      <td>2.940824e-01</td>
      <td>4.556372e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>r&amp;b</th>
      <td>28356.0</td>
      <td>1.811962e-01</td>
      <td>3.851875e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>rap</th>
      <td>28356.0</td>
      <td>2.848780e-01</td>
      <td>4.513643e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>rock</th>
      <td>28356.0</td>
      <td>1.569685e-01</td>
      <td>3.637775e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>year</th>
      <td>28356.0</td>
      <td>2.011054e+03</td>
      <td>1.122922e+01</td>
      <td>1957.000000</td>
      <td>2008.000000</td>
      <td>2016.000000</td>
      <td>2.019000e+03</td>
      <td>2.020000e+03</td>
    </tr>
    <tr>
      <th>month</th>
      <td>28356.0</td>
      <td>6.101813e+00</td>
      <td>3.841027e+00</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>1.000000e+01</td>
      <td>1.200000e+01</td>
    </tr>
    <tr>
      <th>day</th>
      <td>28356.0</td>
      <td>1.341427e+01</td>
      <td>9.666976e+00</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>2.200000e+01</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <th>decade</th>
      <td>28356.0</td>
      <td>2.004914e+03</td>
      <td>1.048412e+01</td>
      <td>1950.000000</td>
      <td>2000.000000</td>
      <td>2010.000000</td>
      <td>2.010000e+03</td>
      <td>2.020000e+03</td>
    </tr>
    <tr>
      <th>released_in_internet_era</th>
      <td>28356.0</td>
      <td>7.374101e-01</td>
      <td>4.400492e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>feat</th>
      <td>28356.0</td>
      <td>7.871350e-02</td>
      <td>2.692958e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>Remix</th>
      <td>28356.0</td>
      <td>6.259698e-02</td>
      <td>2.422409e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>Love</th>
      <td>28356.0</td>
      <td>3.918042e-02</td>
      <td>1.940274e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>Radio Edit</th>
      <td>28356.0</td>
      <td>1.653971e-02</td>
      <td>1.275411e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>Remastered</th>
      <td>28356.0</td>
      <td>1.445902e-02</td>
      <td>1.193753e-01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>track_artist_followers</th>
      <td>28356.0</td>
      <td>2.812550e+06</td>
      <td>7.002592e+06</td>
      <td>0.000000</td>
      <td>28162.250000</td>
      <td>366482.000000</td>
      <td>2.108973e+06</td>
      <td>7.890023e+07</td>
    </tr>
  </tbody>
</table>
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


Just making sure that there are no missing values in the dataset:


```python
nan_counts = df.isna().sum()
nan_counts[nan_counts > 0]
```




    Series([], dtype: int64)



### Save it to the next chapter


```python
df.to_csv('data/04_feature_engineering/feature_engineering.csv', index=False)
```


```python
df.to_pickle('pickle/04_feature_engineering/feature_engineering.pkl')   # Save the dataframe to a pickle file
```

![tobecontinued.jpg](04_feature_engineering_files/tobecontinued.jpg)
