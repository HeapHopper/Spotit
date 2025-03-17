# Exploratory Data Analysis


```python
# DS18 ML Essentials project
# Module 2: Exploratory Data Analysis

# Submitted by: Tzvi Eliezer Nir
# mail: tzvienir@gmail.com
# First submission: 17/03/2025

```

## IMPORTANT MESSAGE

This notebook is long due to automatically generated plots.
It is highly recommended to use the [markdown version](markdown/02_exploratory_data_analysis.md) instead of the GitHub preview, as the latter can show partial notebook.

## In this notebook

In this notebook we will do an Exploratory Data Analysis for the Spotify dataset. Using the Flat File from previous chapter we will explore the data to better understand its structure, quality and patterns before moving on to modeling.

We will start with the Data Protocol, move on to reating data-relations reports automatically using AutoViz, and finish with applying descriptive statistics tools such as Skewness, ANOVA, correlations etc. to learn more about the data and its internal relations.

The analysis will be done for the target value, dummy columns, categorical columns and continous columns. Each type of data with its own unique analysis tools.


 

## Data Protocol

The Data Protocol is used for project documentation and knowledge-preservation. It can be used like a pocket guide to the dataset description.

First lets import the data from the previous notebook:


```python
import pandas as pd

df = pd.read_pickle("./pickle/01_data_preparation/data_preparation.pkl")
```

Now lets create the Data Protocol by printing the dataset characteristics into `.xlsx` files.

Notice that since the dataset includes also `object` columns, when getting the `min` and `max` properties I use a clone dataset containing only numerical data:


```python
data_protocol_files_path = "./data/02_exploratory_data_analysis/data_protocol"

#Type of data 
df.dtypes.to_excel (f"{data_protocol_files_path}/df_datatype.xlsx", sheet_name='data_types')

#Isnull values 
df.isnull().to_excel(f"{data_protocol_files_path}/df_isnull.xlsx", sheet_name='isnull')

#unique values 
df.nunique().to_excel(f"{data_protocol_files_path}/df_nunique.xlsx", sheet_name='nunique')

# Select only numerical columns (int, float, and optionally bool)
df_numeric_copy = df.select_dtypes(include=["number"]).copy()

#Maximum points 
df_numeric_copy.max().to_excel(f"{data_protocol_files_path}/df_max.xlsx", sheet_name='max')

#Minimum points 
df_numeric_copy.min().to_excel(f"{data_protocol_files_path}/df_min.xlsx", sheet_name='min')
        
```

## Automated Reports

Using the `autoviz` package we can generate tabular and visual reports automatically:


```python
from autoviz.AutoViz_Class import AutoViz_Class
%matplotlib inline

AV = AutoViz_Class()
AV.AutoViz('data/01_data_preparation/data_preparation.csv')
```

    Shape of your Data Set loaded: (28356, 25)
    #######################################################################################
    ######################## C L A S S I F Y I N G  V A R I A B L E S  ####################
    #######################################################################################
    Classifying variables in data set...
        Number of Numeric Columns =  9
        Number of Integer-Categorical Columns =  4
        Number of String-Categorical Columns =  0
        Number of Factor-Categorical Columns =  0
        Number of String-Boolean Columns =  0
        Number of Numeric-Boolean Columns =  7
        Number of Discrete String Columns =  0
        Number of NLP String Columns =  3
        Number of Date Time Columns =  0
        Number of ID Columns =  2
        Number of Columns to Delete =  0
        25 Predictors classified...
            2 variable(s) removed since they were ID or low-information variables
            List of variables removed: ['Unnamed: 0', 'track_id']
    To fix these data quality issues in the dataset, import FixDQ from autoviz...
        All variables classified into correct types.



<style type="text/css">
#T_2662f_row0_col0, #T_2662f_row0_col2, #T_2662f_row0_col3, #T_2662f_row0_col4, #T_2662f_row0_col5, #T_2662f_row1_col0, #T_2662f_row1_col2, #T_2662f_row1_col3, #T_2662f_row1_col4, #T_2662f_row1_col5, #T_2662f_row2_col0, #T_2662f_row2_col2, #T_2662f_row2_col3, #T_2662f_row2_col4, #T_2662f_row2_col5, #T_2662f_row3_col0, #T_2662f_row3_col2, #T_2662f_row3_col3, #T_2662f_row3_col4, #T_2662f_row3_col5, #T_2662f_row4_col0, #T_2662f_row4_col2, #T_2662f_row4_col3, #T_2662f_row4_col4, #T_2662f_row4_col5, #T_2662f_row5_col0, #T_2662f_row5_col2, #T_2662f_row5_col3, #T_2662f_row5_col4, #T_2662f_row5_col5, #T_2662f_row6_col0, #T_2662f_row6_col2, #T_2662f_row6_col3, #T_2662f_row6_col4, #T_2662f_row6_col5, #T_2662f_row7_col0, #T_2662f_row7_col2, #T_2662f_row7_col3, #T_2662f_row7_col4, #T_2662f_row7_col5, #T_2662f_row8_col0, #T_2662f_row8_col2, #T_2662f_row8_col3, #T_2662f_row8_col4, #T_2662f_row8_col5, #T_2662f_row9_col0, #T_2662f_row9_col2, #T_2662f_row9_col3, #T_2662f_row9_col4, #T_2662f_row9_col5, #T_2662f_row10_col0, #T_2662f_row10_col2, #T_2662f_row10_col3, #T_2662f_row10_col4, #T_2662f_row10_col5, #T_2662f_row11_col0, #T_2662f_row11_col2, #T_2662f_row11_col3, #T_2662f_row11_col4, #T_2662f_row11_col5, #T_2662f_row12_col0, #T_2662f_row12_col2, #T_2662f_row12_col3, #T_2662f_row12_col4, #T_2662f_row12_col5, #T_2662f_row13_col0, #T_2662f_row13_col2, #T_2662f_row13_col3, #T_2662f_row13_col4, #T_2662f_row13_col5, #T_2662f_row14_col0, #T_2662f_row14_col2, #T_2662f_row14_col3, #T_2662f_row14_col4, #T_2662f_row14_col5, #T_2662f_row15_col0, #T_2662f_row15_col2, #T_2662f_row15_col3, #T_2662f_row15_col4, #T_2662f_row15_col5, #T_2662f_row16_col0, #T_2662f_row16_col2, #T_2662f_row16_col3, #T_2662f_row16_col4, #T_2662f_row16_col5, #T_2662f_row17_col0, #T_2662f_row17_col2, #T_2662f_row17_col3, #T_2662f_row17_col4, #T_2662f_row17_col5, #T_2662f_row18_col0, #T_2662f_row18_col2, #T_2662f_row18_col3, #T_2662f_row18_col4, #T_2662f_row18_col5, #T_2662f_row19_col0, #T_2662f_row19_col2, #T_2662f_row19_col3, #T_2662f_row19_col4, #T_2662f_row19_col5, #T_2662f_row20_col0, #T_2662f_row20_col2, #T_2662f_row20_col3, #T_2662f_row20_col4, #T_2662f_row20_col5, #T_2662f_row21_col0, #T_2662f_row21_col2, #T_2662f_row21_col3, #T_2662f_row21_col4, #T_2662f_row21_col5, #T_2662f_row22_col0, #T_2662f_row22_col2, #T_2662f_row22_col3, #T_2662f_row22_col4, #T_2662f_row22_col5, #T_2662f_row23_col0, #T_2662f_row23_col2, #T_2662f_row23_col3, #T_2662f_row23_col4, #T_2662f_row23_col5, #T_2662f_row24_col0, #T_2662f_row24_col2, #T_2662f_row24_col3, #T_2662f_row24_col4, #T_2662f_row24_col5 {
  font-family: Segoe UI;
}
#T_2662f_row0_col1, #T_2662f_row1_col1, #T_2662f_row3_col1, #T_2662f_row4_col1, #T_2662f_row5_col1, #T_2662f_row6_col1, #T_2662f_row7_col1, #T_2662f_row8_col1, #T_2662f_row9_col1, #T_2662f_row10_col1, #T_2662f_row11_col1, #T_2662f_row12_col1, #T_2662f_row13_col1, #T_2662f_row14_col1, #T_2662f_row15_col1, #T_2662f_row16_col1, #T_2662f_row17_col1, #T_2662f_row18_col1, #T_2662f_row19_col1, #T_2662f_row20_col1, #T_2662f_row21_col1, #T_2662f_row22_col1, #T_2662f_row23_col1, #T_2662f_row24_col1 {
  background-color: #fff5f0;
  color: #000000;
  font-family: Segoe UI;
}
#T_2662f_row2_col1 {
  background-color: #67000d;
  color: #f1f1f1;
  font-family: Segoe UI;
}
</style>
<table id="T_2662f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2662f_level0_col0" class="col_heading level0 col0" >Data Type</th>
      <th id="T_2662f_level0_col1" class="col_heading level0 col1" >Missing Values%</th>
      <th id="T_2662f_level0_col2" class="col_heading level0 col2" >Unique Values%</th>
      <th id="T_2662f_level0_col3" class="col_heading level0 col3" >Minimum Value</th>
      <th id="T_2662f_level0_col4" class="col_heading level0 col4" >Maximum Value</th>
      <th id="T_2662f_level0_col5" class="col_heading level0 col5" >DQ Issue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2662f_level0_row0" class="row_heading level0 row0" >Unnamed: 0</th>
      <td id="T_2662f_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_2662f_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_2662f_row0_col2" class="data row0 col2" >100</td>
      <td id="T_2662f_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_2662f_row0_col4" class="data row0 col4" >28355.000000</td>
      <td id="T_2662f_row0_col5" class="data row0 col5" >Possible ID column: drop before modeling step.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row1" class="row_heading level0 row1" >track_id</th>
      <td id="T_2662f_row1_col0" class="data row1 col0" >object</td>
      <td id="T_2662f_row1_col1" class="data row1 col1" >0.000000</td>
      <td id="T_2662f_row1_col2" class="data row1 col2" >100</td>
      <td id="T_2662f_row1_col3" class="data row1 col3" ></td>
      <td id="T_2662f_row1_col4" class="data row1 col4" ></td>
      <td id="T_2662f_row1_col5" class="data row1 col5" >Possible ID column: drop before modeling step.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row2" class="row_heading level0 row2" >track_artist</th>
      <td id="T_2662f_row2_col0" class="data row2 col0" >object</td>
      <td id="T_2662f_row2_col1" class="data row2 col1" >0.014106</td>
      <td id="T_2662f_row2_col2" class="data row2 col2" >37</td>
      <td id="T_2662f_row2_col3" class="data row2 col3" ></td>
      <td id="T_2662f_row2_col4" class="data row2 col4" ></td>
      <td id="T_2662f_row2_col5" class="data row2 col5" >4 missing values. Impute them with mean, median, mode, or a constant value such as 123., Mixed dtypes: has 2 different data types:  object, float,</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row3" class="row_heading level0 row3" >track_popularity</th>
      <td id="T_2662f_row3_col0" class="data row3 col0" >int64</td>
      <td id="T_2662f_row3_col1" class="data row3 col1" >0.000000</td>
      <td id="T_2662f_row3_col2" class="data row3 col2" >0</td>
      <td id="T_2662f_row3_col3" class="data row3 col3" >0.000000</td>
      <td id="T_2662f_row3_col4" class="data row3 col4" >100.000000</td>
      <td id="T_2662f_row3_col5" class="data row3 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row4" class="row_heading level0 row4" >track_album_id</th>
      <td id="T_2662f_row4_col0" class="data row4 col0" >object</td>
      <td id="T_2662f_row4_col1" class="data row4 col1" >0.000000</td>
      <td id="T_2662f_row4_col2" class="data row4 col2" >79</td>
      <td id="T_2662f_row4_col3" class="data row4 col3" ></td>
      <td id="T_2662f_row4_col4" class="data row4 col4" ></td>
      <td id="T_2662f_row4_col5" class="data row4 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row5" class="row_heading level0 row5" >track_album_release_date</th>
      <td id="T_2662f_row5_col0" class="data row5 col0" >object</td>
      <td id="T_2662f_row5_col1" class="data row5 col1" >0.000000</td>
      <td id="T_2662f_row5_col2" class="data row5 col2" >15</td>
      <td id="T_2662f_row5_col3" class="data row5 col3" ></td>
      <td id="T_2662f_row5_col4" class="data row5 col4" ></td>
      <td id="T_2662f_row5_col5" class="data row5 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row6" class="row_heading level0 row6" >danceability</th>
      <td id="T_2662f_row6_col0" class="data row6 col0" >float64</td>
      <td id="T_2662f_row6_col1" class="data row6 col1" >0.000000</td>
      <td id="T_2662f_row6_col2" class="data row6 col2" >NA</td>
      <td id="T_2662f_row6_col3" class="data row6 col3" >0.000000</td>
      <td id="T_2662f_row6_col4" class="data row6 col4" >0.983000</td>
      <td id="T_2662f_row6_col5" class="data row6 col5" >Column has 257 outliers greater than upper bound (1.06) or lower than lower bound(0.26). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row7" class="row_heading level0 row7" >energy</th>
      <td id="T_2662f_row7_col0" class="data row7 col0" >float64</td>
      <td id="T_2662f_row7_col1" class="data row7 col1" >0.000000</td>
      <td id="T_2662f_row7_col2" class="data row7 col2" >NA</td>
      <td id="T_2662f_row7_col3" class="data row7 col3" >0.000175</td>
      <td id="T_2662f_row7_col4" class="data row7 col4" >1.000000</td>
      <td id="T_2662f_row7_col5" class="data row7 col5" >Column has 222 outliers greater than upper bound (1.24) or lower than lower bound(0.18). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row8" class="row_heading level0 row8" >key</th>
      <td id="T_2662f_row8_col0" class="data row8 col0" >int64</td>
      <td id="T_2662f_row8_col1" class="data row8 col1" >0.000000</td>
      <td id="T_2662f_row8_col2" class="data row8 col2" >0</td>
      <td id="T_2662f_row8_col3" class="data row8 col3" >0.000000</td>
      <td id="T_2662f_row8_col4" class="data row8 col4" >11.000000</td>
      <td id="T_2662f_row8_col5" class="data row8 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row9" class="row_heading level0 row9" >loudness</th>
      <td id="T_2662f_row9_col0" class="data row9 col0" >float64</td>
      <td id="T_2662f_row9_col1" class="data row9 col1" >0.000000</td>
      <td id="T_2662f_row9_col2" class="data row9 col2" >NA</td>
      <td id="T_2662f_row9_col3" class="data row9 col3" >-46.448000</td>
      <td id="T_2662f_row9_col4" class="data row9 col4" >1.275000</td>
      <td id="T_2662f_row9_col5" class="data row9 col5" >Column has 837 outliers greater than upper bound (0.69) or lower than lower bound(-13.71). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row10" class="row_heading level0 row10" >mode</th>
      <td id="T_2662f_row10_col0" class="data row10 col0" >int64</td>
      <td id="T_2662f_row10_col1" class="data row10 col1" >0.000000</td>
      <td id="T_2662f_row10_col2" class="data row10 col2" >0</td>
      <td id="T_2662f_row10_col3" class="data row10 col3" >0.000000</td>
      <td id="T_2662f_row10_col4" class="data row10 col4" >1.000000</td>
      <td id="T_2662f_row10_col5" class="data row10 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row11" class="row_heading level0 row11" >speechiness</th>
      <td id="T_2662f_row11_col0" class="data row11 col0" >float64</td>
      <td id="T_2662f_row11_col1" class="data row11 col1" >0.000000</td>
      <td id="T_2662f_row11_col2" class="data row11 col2" >NA</td>
      <td id="T_2662f_row11_col3" class="data row11 col3" >0.000000</td>
      <td id="T_2662f_row11_col4" class="data row11 col4" >0.918000</td>
      <td id="T_2662f_row11_col5" class="data row11 col5" >Column has 2725 outliers greater than upper bound (0.27) or lower than lower bound(-0.10). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row12" class="row_heading level0 row12" >acousticness</th>
      <td id="T_2662f_row12_col0" class="data row12 col0" >float64</td>
      <td id="T_2662f_row12_col1" class="data row12 col1" >0.000000</td>
      <td id="T_2662f_row12_col2" class="data row12 col2" >NA</td>
      <td id="T_2662f_row12_col3" class="data row12 col3" >0.000000</td>
      <td id="T_2662f_row12_col4" class="data row12 col4" >0.994000</td>
      <td id="T_2662f_row12_col5" class="data row12 col5" >Column has 1911 outliers greater than upper bound (0.63) or lower than lower bound(-0.35). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row13" class="row_heading level0 row13" >instrumentalness</th>
      <td id="T_2662f_row13_col0" class="data row13 col0" >float64</td>
      <td id="T_2662f_row13_col1" class="data row13 col1" >0.000000</td>
      <td id="T_2662f_row13_col2" class="data row13 col2" >NA</td>
      <td id="T_2662f_row13_col3" class="data row13 col3" >0.000000</td>
      <td id="T_2662f_row13_col4" class="data row13 col4" >0.994000</td>
      <td id="T_2662f_row13_col5" class="data row13 col5" >Column has 6085 outliers greater than upper bound (0.02) or lower than lower bound(-0.01). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row14" class="row_heading level0 row14" >liveness</th>
      <td id="T_2662f_row14_col0" class="data row14 col0" >float64</td>
      <td id="T_2662f_row14_col1" class="data row14 col1" >0.000000</td>
      <td id="T_2662f_row14_col2" class="data row14 col2" >NA</td>
      <td id="T_2662f_row14_col3" class="data row14 col3" >0.000000</td>
      <td id="T_2662f_row14_col4" class="data row14 col4" >0.996000</td>
      <td id="T_2662f_row14_col5" class="data row14 col5" >Column has 1623 outliers greater than upper bound (0.48) or lower than lower bound(-0.14). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row15" class="row_heading level0 row15" >valence</th>
      <td id="T_2662f_row15_col0" class="data row15 col0" >float64</td>
      <td id="T_2662f_row15_col1" class="data row15 col1" >0.000000</td>
      <td id="T_2662f_row15_col2" class="data row15 col2" >NA</td>
      <td id="T_2662f_row15_col3" class="data row15 col3" >0.000000</td>
      <td id="T_2662f_row15_col4" class="data row15 col4" >0.991000</td>
      <td id="T_2662f_row15_col5" class="data row15 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row16" class="row_heading level0 row16" >tempo</th>
      <td id="T_2662f_row16_col0" class="data row16 col0" >float64</td>
      <td id="T_2662f_row16_col1" class="data row16 col1" >0.000000</td>
      <td id="T_2662f_row16_col2" class="data row16 col2" >NA</td>
      <td id="T_2662f_row16_col3" class="data row16 col3" >0.000000</td>
      <td id="T_2662f_row16_col4" class="data row16 col4" >239.440000</td>
      <td id="T_2662f_row16_col5" class="data row16 col5" >Column has 494 outliers greater than upper bound (185.04) or lower than lower bound(48.93). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row17" class="row_heading level0 row17" >duration_ms</th>
      <td id="T_2662f_row17_col0" class="data row17 col0" >int64</td>
      <td id="T_2662f_row17_col1" class="data row17 col1" >0.000000</td>
      <td id="T_2662f_row17_col2" class="data row17 col2" >69</td>
      <td id="T_2662f_row17_col3" class="data row17 col3" >4000.000000</td>
      <td id="T_2662f_row17_col4" class="data row17 col4" >517810.000000</td>
      <td id="T_2662f_row17_col5" class="data row17 col5" >Column has 1197 outliers greater than upper bound (355825.12) or lower than lower bound(86892.12). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row18" class="row_heading level0 row18" >playlist_count</th>
      <td id="T_2662f_row18_col0" class="data row18 col0" >int64</td>
      <td id="T_2662f_row18_col1" class="data row18 col1" >0.000000</td>
      <td id="T_2662f_row18_col2" class="data row18 col2" >0</td>
      <td id="T_2662f_row18_col3" class="data row18 col3" >1.000000</td>
      <td id="T_2662f_row18_col4" class="data row18 col4" >10.000000</td>
      <td id="T_2662f_row18_col5" class="data row18 col5" >Column has 3166 outliers greater than upper bound (1.00) or lower than lower bound(1.00). Cap them or remove them.</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row19" class="row_heading level0 row19" >edm</th>
      <td id="T_2662f_row19_col0" class="data row19 col0" >int64</td>
      <td id="T_2662f_row19_col1" class="data row19 col1" >0.000000</td>
      <td id="T_2662f_row19_col2" class="data row19 col2" >0</td>
      <td id="T_2662f_row19_col3" class="data row19 col3" >0.000000</td>
      <td id="T_2662f_row19_col4" class="data row19 col4" >1.000000</td>
      <td id="T_2662f_row19_col5" class="data row19 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row20" class="row_heading level0 row20" >latin</th>
      <td id="T_2662f_row20_col0" class="data row20 col0" >int64</td>
      <td id="T_2662f_row20_col1" class="data row20 col1" >0.000000</td>
      <td id="T_2662f_row20_col2" class="data row20 col2" >0</td>
      <td id="T_2662f_row20_col3" class="data row20 col3" >0.000000</td>
      <td id="T_2662f_row20_col4" class="data row20 col4" >1.000000</td>
      <td id="T_2662f_row20_col5" class="data row20 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row21" class="row_heading level0 row21" >pop</th>
      <td id="T_2662f_row21_col0" class="data row21 col0" >int64</td>
      <td id="T_2662f_row21_col1" class="data row21 col1" >0.000000</td>
      <td id="T_2662f_row21_col2" class="data row21 col2" >0</td>
      <td id="T_2662f_row21_col3" class="data row21 col3" >0.000000</td>
      <td id="T_2662f_row21_col4" class="data row21 col4" >1.000000</td>
      <td id="T_2662f_row21_col5" class="data row21 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row22" class="row_heading level0 row22" >r&b</th>
      <td id="T_2662f_row22_col0" class="data row22 col0" >int64</td>
      <td id="T_2662f_row22_col1" class="data row22 col1" >0.000000</td>
      <td id="T_2662f_row22_col2" class="data row22 col2" >0</td>
      <td id="T_2662f_row22_col3" class="data row22 col3" >0.000000</td>
      <td id="T_2662f_row22_col4" class="data row22 col4" >1.000000</td>
      <td id="T_2662f_row22_col5" class="data row22 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row23" class="row_heading level0 row23" >rap</th>
      <td id="T_2662f_row23_col0" class="data row23 col0" >int64</td>
      <td id="T_2662f_row23_col1" class="data row23 col1" >0.000000</td>
      <td id="T_2662f_row23_col2" class="data row23 col2" >0</td>
      <td id="T_2662f_row23_col3" class="data row23 col3" >0.000000</td>
      <td id="T_2662f_row23_col4" class="data row23 col4" >1.000000</td>
      <td id="T_2662f_row23_col5" class="data row23 col5" >No issue</td>
    </tr>
    <tr>
      <th id="T_2662f_level0_row24" class="row_heading level0 row24" >rock</th>
      <td id="T_2662f_row24_col0" class="data row24 col0" >int64</td>
      <td id="T_2662f_row24_col1" class="data row24 col1" >0.000000</td>
      <td id="T_2662f_row24_col2" class="data row24 col2" >0</td>
      <td id="T_2662f_row24_col3" class="data row24 col3" >0.000000</td>
      <td id="T_2662f_row24_col4" class="data row24 col4" >1.000000</td>
      <td id="T_2662f_row24_col5" class="data row24 col5" >No issue</td>
    </tr>
  </tbody>
</table>



    Number of All Scatter Plots = 45



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_3.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_4.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_5.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_6.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_7.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_8.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_9.png)
    


    [nltk_data] Downloading collection 'popular'
    [nltk_data]    | 
    [nltk_data]    | Downloading package cmudict to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package cmudict is already up-to-date!
    [nltk_data]    | Downloading package gazetteers to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package gazetteers is already up-to-date!
    [nltk_data]    | Downloading package genesis to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package genesis is already up-to-date!
    [nltk_data]    | Downloading package gutenberg to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package gutenberg is already up-to-date!
    [nltk_data]    | Downloading package inaugural to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package inaugural is already up-to-date!
    [nltk_data]    | Downloading package movie_reviews to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package movie_reviews is already up-to-date!
    [nltk_data]    | Downloading package names to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package names is already up-to-date!
    [nltk_data]    | Downloading package shakespeare to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package shakespeare is already up-to-date!
    [nltk_data]    | Downloading package stopwords to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package stopwords is already up-to-date!
    [nltk_data]    | Downloading package treebank to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package treebank is already up-to-date!
    [nltk_data]    | Downloading package twitter_samples to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package twitter_samples is already up-to-date!
    [nltk_data]    | Downloading package omw to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package omw is already up-to-date!
    [nltk_data]    | Downloading package omw-1.4 to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package omw-1.4 is already up-to-date!
    [nltk_data]    | Downloading package wordnet to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package wordnet is already up-to-date!
    [nltk_data]    | Downloading package wordnet2021 to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package wordnet2021 is already up-to-date!
    [nltk_data]    | Downloading package wordnet31 to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package wordnet31 is already up-to-date!
    [nltk_data]    | Downloading package wordnet_ic to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package wordnet_ic is already up-to-date!
    [nltk_data]    | Downloading package words to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package words is already up-to-date!
    [nltk_data]    | Downloading package maxent_ne_chunker to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package maxent_ne_chunker is already up-to-date!
    [nltk_data]    | Downloading package punkt to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package punkt is already up-to-date!
    [nltk_data]    | Downloading package snowball_data to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package snowball_data is already up-to-date!
    [nltk_data]    | Downloading package averaged_perceptron_tagger to
    [nltk_data]    |     /home/heaphopper/nltk_data...
    [nltk_data]    |   Package averaged_perceptron_tagger is already up-
    [nltk_data]    |       to-date!
    [nltk_data]    | 
    [nltk_data]  Done downloading collection popular



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_11.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_9_12.png)
    


    Could not draw wordcloud plot for track_album_release_date. We need at least 1 word to plot a word cloud, got 0.
    All Plots done
    Time to run AutoViz = 30 seconds 
    
     ###################### AUTO VISUALIZATION Completed ########################





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
    <tr style="text-align: center;">
      <th></th>
      <th>Unnamed: 0</th>
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
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
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
      <td>0</td>
      <td>0017A6SJgTbfQVU2EtsPNo</td>
      <td>Barbie's Cradle</td>
      <td>41</td>
      <td>1srJQ0njEQgd8w4XSqI4JQ</td>
      <td>2001-01-01</td>
      <td>0.682</td>
      <td>0.401</td>
      <td>2</td>
      <td>-10.068</td>
      <td>1</td>
      <td>0.0236</td>
      <td>0.279000</td>
      <td>0.011700</td>
      <td>0.0887</td>
      <td>0.566</td>
      <td>97.091</td>
      <td>235440</td>
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
      <td>1</td>
      <td>002xjHwzEx66OWFV2IP9dk</td>
      <td>RIKA</td>
      <td>15</td>
      <td>1ficfUnZMaY1QkNp15Slzm</td>
      <td>2018-01-26</td>
      <td>0.582</td>
      <td>0.704</td>
      <td>5</td>
      <td>-6.242</td>
      <td>1</td>
      <td>0.0347</td>
      <td>0.065100</td>
      <td>0.000000</td>
      <td>0.2120</td>
      <td>0.698</td>
      <td>150.863</td>
      <td>197286</td>
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
      <td>2</td>
      <td>004s3t0ONYlzxII9PLgU6z</td>
      <td>Steady Rollin</td>
      <td>28</td>
      <td>3z04Lb9Dsilqw68SHt6jLB</td>
      <td>2017-11-21</td>
      <td>0.303</td>
      <td>0.880</td>
      <td>9</td>
      <td>-4.739</td>
      <td>1</td>
      <td>0.0442</td>
      <td>0.011700</td>
      <td>0.009940</td>
      <td>0.3470</td>
      <td>0.404</td>
      <td>135.225</td>
      <td>373512</td>
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
      <td>3</td>
      <td>008MceT31RotUANsKuzy3L</td>
      <td>The.madpix.project</td>
      <td>24</td>
      <td>1Z4ANBVuhTlS6DprlP0m1q</td>
      <td>2015-08-07</td>
      <td>0.659</td>
      <td>0.794</td>
      <td>10</td>
      <td>-5.644</td>
      <td>0</td>
      <td>0.0540</td>
      <td>0.000761</td>
      <td>0.132000</td>
      <td>0.3220</td>
      <td>0.852</td>
      <td>128.041</td>
      <td>228565</td>
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
      <td>4</td>
      <td>008rk8F6ZxspZT4bUlkIQG</td>
      <td>YOSA &amp; TAAR</td>
      <td>38</td>
      <td>2BuYm9UcKvI0ydXs5JKwt0</td>
      <td>2018-11-16</td>
      <td>0.662</td>
      <td>0.838</td>
      <td>1</td>
      <td>-6.300</td>
      <td>1</td>
      <td>0.0499</td>
      <td>0.114000</td>
      <td>0.000697</td>
      <td>0.0881</td>
      <td>0.496</td>
      <td>129.884</td>
      <td>236308</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>28351</th>
      <td>28351</td>
      <td>7zxRMhXxJMQCeDDg0rKAVo</td>
      <td>NAV</td>
      <td>72</td>
      <td>4JR29CNW14Zc4Z5vflxkoJ</td>
      <td>2017-02-24</td>
      <td>0.744</td>
      <td>0.715</td>
      <td>0</td>
      <td>-6.103</td>
      <td>1</td>
      <td>0.3510</td>
      <td>0.101000</td>
      <td>0.000000</td>
      <td>0.0919</td>
      <td>0.340</td>
      <td>154.962</td>
      <td>179773</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28352</th>
      <td>28352</td>
      <td>7zyLObYw4QUKQDyZOb4J0Y</td>
      <td>Father MC</td>
      <td>36</td>
      <td>14HYMxFhpgDIr9cci1u0kt</td>
      <td>2010-10-01</td>
      <td>0.832</td>
      <td>0.666</td>
      <td>1</td>
      <td>-4.920</td>
      <td>0</td>
      <td>0.0633</td>
      <td>0.143000</td>
      <td>0.000000</td>
      <td>0.0720</td>
      <td>0.810</td>
      <td>109.536</td>
      <td>223890</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28353</th>
      <td>28353</td>
      <td>7zycSpvjDcqh6YT1FEl2kY</td>
      <td>Nicki Minaj</td>
      <td>49</td>
      <td>5qs8T6ZHSrnllnOuUk6muC</td>
      <td>2014-12-15</td>
      <td>0.963</td>
      <td>0.603</td>
      <td>2</td>
      <td>-6.224</td>
      <td>1</td>
      <td>0.1800</td>
      <td>0.067300</td>
      <td>0.000006</td>
      <td>0.2140</td>
      <td>0.647</td>
      <td>129.990</td>
      <td>260240</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28354</th>
      <td>28354</td>
      <td>7zye9v6B785eFWEFYs13C2</td>
      <td>Ponderosa Twins Plus One</td>
      <td>40</td>
      <td>1xdgLmTFMSyJyI5DJOOX7T</td>
      <td>2013-07-09</td>
      <td>0.458</td>
      <td>0.540</td>
      <td>5</td>
      <td>-6.457</td>
      <td>0</td>
      <td>0.0270</td>
      <td>0.715000</td>
      <td>0.000428</td>
      <td>0.1150</td>
      <td>0.657</td>
      <td>142.218</td>
      <td>191205</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28355</th>
      <td>28355</td>
      <td>7zzZmpw8L66ZPjH1M6qmOs</td>
      <td>Moonstar88</td>
      <td>61</td>
      <td>4t3FtECyV1gClHmpBhXSfB</td>
      <td>2008-07-28</td>
      <td>0.664</td>
      <td>0.537</td>
      <td>2</td>
      <td>-6.000</td>
      <td>1</td>
      <td>0.0290</td>
      <td>0.217000</td>
      <td>0.000003</td>
      <td>0.1180</td>
      <td>0.221</td>
      <td>115.049</td>
      <td>267960</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>28356 rows × 25 columns</p>
</div>



## Descriptive Statistics

We will use statistical tools to better understand and explore the data. We will use descriptive tools for the following four cases:

1. Exploring the **target value**
2. Exploring **dummy** columns
3. Exploring **category** columns
4. Exploring **continous** columns

Each of the four types of data requires their own type of tools and statistical tests, as we will see shortly.

Later in this section, we will look for **correlations** in the dataset.

### Target Value


```python
import seaborn as sns

import matplotlib.pyplot as plt

sns.histplot(df['track_popularity'], kde=True, bins=100)
plt.title('Distribution of Track Popularity')
plt.xlabel('Track Popularity')
plt.ylabel('Density')
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_12_0.png)
    


Notice the high number of 0 popularity tracks. Spoiler alert: this data is valid, from recent years - and was not considerd as an outlier when looking for outliers.

For now lets keep using the statistical tools, starting with skewness:


```python
skewness = df['track_popularity'].skew()
print(f"Skewness of track_popularity: {skewness}")
```

    Skewness of track_popularity: -0.23134997680726382


It seems that we can treat the target value column as parametric distributed!

### Dummy columns


```python
df_dummy = df[['mode','edm','latin','pop','r&b','rap','rock']]
df_dummy
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
    <tr style="text-align: center;">
      <th></th>
      <th>mode</th>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>28351</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28352</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28353</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28354</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28355</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>28356 rows × 7 columns</p>
</div>




```python
for column in df_dummy.columns:
    sns.countplot(x=column, data=df_dummy, palette="viridis")
    plt.title(f'Bar Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_0.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_1.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_2.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_3.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_4.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_5.png)
    



    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_18_6.png)
    


Im also trying to look for a correlation that might tell us of some dependency between the genres, but as the heatmap below shows - not even a single positive correlation can be found, and negative correlation are not very meaningful in the context of genre correlation.


```python
sns.heatmap(df_dummy.corr(), annot=True, cmap='viridis')
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_20_0.png)
    


### Categorical columns

Apperantly there is only one (non-dummy) categorical column - `key`:


```python
df_cat = df[['key']]

sns.countplot(x='key', data=df_cat, palette="viridis")
plt.title(f'Bar Plot for key')
plt.xlabel('key')
plt.ylabel('Count')
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_22_0.png)
    


### Continues (numeric) column

With continous data we have many statistical tools we can use. especially if the data is parametric - and fortunately there are parametric distributed features in our data..


```python
df_num=df._get_numeric_data().dropna()
df_num = df_num.drop(columns=['key','mode','edm','latin','pop','r&b','rap','rock'])

df_num.shape
```




    (28356, 12)




```python
df_num.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28356 entries, 0 to 28355
    Data columns (total 12 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   track_popularity  28356 non-null  int64  
     1   danceability      28356 non-null  float64
     2   energy            28356 non-null  float64
     3   loudness          28356 non-null  float64
     4   speechiness       28356 non-null  float64
     5   acousticness      28356 non-null  float64
     6   instrumentalness  28356 non-null  float64
     7   liveness          28356 non-null  float64
     8   valence           28356 non-null  float64
     9   tempo             28356 non-null  float64
     10  duration_ms       28356 non-null  int64  
     11  playlist_count    28356 non-null  int64  
    dtypes: float64(9), int64(3)
    memory usage: 2.6 MB



```python
df_num.hist(figsize=(35, 35), bins=35, xlabelsize=8, ylabelsize=8, color = "cornflowerblue")
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_27_0.png)
    


#### Skewness

Skewness is a statistical metric used to assess the asymmetry of the distribution of values in a dataset column. It helps determine whether the data can be considered parametric.

Values between -1 and 1 are generally considered to indicate a parametric distribution, while values outside this range suggest a non-parametric distribution.


```python
def highlight(cell_value):
    highlight = 'background-color: mediumspringgreen;'
    default = ''
    negative = 'background-color: hotpink;'
    if cell_value > 1:
        return highlight
    elif cell_value < -1:
        return negative
    #else
       # return default  
pd.DataFrame(df_num.skew(),columns=['skewness']).sort_values(by='skewness', ascending=False).style.applymap(highlight)
```




<style type="text/css">
#T_0eecb_row0_col0, #T_0eecb_row1_col0, #T_0eecb_row2_col0, #T_0eecb_row3_col0, #T_0eecb_row4_col0, #T_0eecb_row5_col0 {
  background-color: mediumspringgreen;
}
#T_0eecb_row11_col0 {
  background-color: hotpink;
}
</style>
<table id="T_0eecb">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0eecb_level0_col0" class="col_heading level0 col0" >skewness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0eecb_level0_row0" class="row_heading level0 row0" >playlist_count</th>
      <td id="T_0eecb_row0_col0" class="data row0 col0" >5.481869</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row1" class="row_heading level0 row1" >instrumentalness</th>
      <td id="T_0eecb_row1_col0" class="data row1 col0" >2.624985</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row2" class="row_heading level0 row2" >liveness</th>
      <td id="T_0eecb_row2_col0" class="data row2 col0" >2.081481</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row3" class="row_heading level0 row3" >speechiness</th>
      <td id="T_0eecb_row3_col0" class="data row3 col0" >1.964975</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row4" class="row_heading level0 row4" >acousticness</th>
      <td id="T_0eecb_row4_col0" class="data row4 col0" >1.576720</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row5" class="row_heading level0 row5" >duration_ms</th>
      <td id="T_0eecb_row5_col0" class="data row5 col0" >1.115164</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row6" class="row_heading level0 row6" >tempo</th>
      <td id="T_0eecb_row6_col0" class="data row6 col0" >0.513949</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row7" class="row_heading level0 row7" >valence</th>
      <td id="T_0eecb_row7_col0" class="data row7 col0" >-0.007334</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row8" class="row_heading level0 row8" >track_popularity</th>
      <td id="T_0eecb_row8_col0" class="data row8 col0" >-0.231350</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row9" class="row_heading level0 row9" >danceability</th>
      <td id="T_0eecb_row9_col0" class="data row9 col0" >-0.505789</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row10" class="row_heading level0 row10" >energy</th>
      <td id="T_0eecb_row10_col0" class="data row10 col0" >-0.645440</td>
    </tr>
    <tr>
      <th id="T_0eecb_level0_row11" class="row_heading level0 row11" >loudness</th>
      <td id="T_0eecb_row11_col0" class="data row11 col0" >-1.359303</td>
    </tr>
  </tbody>
</table>




### Correlations

Correlations can tell us about relationship in the dataset, however correlation do not imply causation! still it provides valuable insights. 

Lets start with the Pearson correlation for linear relationship:


```python
df2 = df._get_numeric_data().dropna()
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 5})
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_32_0.png)
    


Using the Spearman correlation gave us about the same results:


```python
sns.heatmap(df2.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 5})
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_34_0.png)
    


### ANOVA

Our data have a textbook use-case for the ANOVA statistical tool. 
The genres are separate groups each with its own `track_popularity` distribution. Due to the large sample size and the fact that popularity distribution was already found to be parametric, we can use ANOVA to analyze the variance in popularitiy for each of the six main genres!


```python
import scipy.stats as stats
import seaborn as sns
import numpy as np
```


```python
pop_track_popularity = df[df['pop'] == 1]['track_popularity']
rap_track_popularity = df[df['rap'] == 1]['track_popularity']
rock_track_popularity = df[df['rock'] == 1]['track_popularity']
edm_track_popularity = df[df['edm'] == 1]['track_popularity']
latin_track_popularity = df[df['latin'] == 1]['track_popularity']
rnb_track_popularity = df[df['r&b'] == 1]['track_popularity']

# performe ANOVA
f_statistic, p_value = stats.f_oneway(  pop_track_popularity, 
                                        rap_track_popularity,
                                        rock_track_popularity,
                                        edm_track_popularity,
                                        latin_track_popularity,
                                        rnb_track_popularity)

print(f"F-statistic: {f_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference between the group means")
else:
    print("No significant difference between the group means")
```

    F-statistic: 157.26337523426938
    p-value: 5.993999686772797e-166
    There is a significant difference between the group means


The high F-statistics and tiny P-value - both indicates that at least one group mean is significantly different from the others.

Lets visualize it!


```python
# Create a density plot
plt.figure(figsize=(12, 6))

# Plot the density of each group
sns.histplot(pop_track_popularity, kde=True, label='pop', color='red', bins=50, stat="density", alpha=0.3)
sns.histplot(rap_track_popularity, kde=True, label='rap', color='blue', bins=50, stat="density", alpha=0.3)
sns.histplot(rock_track_popularity, kde=True, label='rock', color='orange', bins=50, stat="density", alpha=0.3)
sns.histplot(edm_track_popularity, kde=True, label='edm', color='green', bins=50, stat="density", alpha=0.3)
sns.histplot(latin_track_popularity, kde=True, label='latin', color='purple', bins=50, stat="density", alpha=0.3)
sns.histplot(rnb_track_popularity, kde=True, label='r&b', color='brown', bins=50, stat="density", alpha=0.3)

# Add lines for the means of each group
plt.axvline(np.mean(pop_track_popularity), color='red', linestyle='--', linewidth=1)
plt.axvline(np.mean(rap_track_popularity), color='blue', linestyle='--', linewidth=1)
plt.axvline(np.mean(rock_track_popularity), color='orange', linestyle='--', linewidth=1)
plt.axvline(np.mean(edm_track_popularity), color='green', linestyle='--', linewidth=1)
plt.axvline(np.mean(latin_track_popularity), color='purple', linestyle='--', linewidth=1)
plt.axvline(np.mean(rnb_track_popularity), color='brown', linestyle='--', linewidth=1)

# Set titles and labels with smaller font size
plt.title('Density Plot of each Genre within-group Variance', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)

# Show the plot
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_39_0.png)
    


### Addressing the zero-popularity tracks

Having a so many tracks with zero popularity is alarming. I just want to make sure that this data is valid. For example, if all `0` values are old songs - maybe the popularity was not measured yet. So i decided to plot the distribution of zero popularity songs over the years:


```python
# Create a new instance to store the years
release_years = df['track_album_release_date'].dt.year

# Filter tracks with popularity equal to 0
df_popularity_zero = df[df['track_popularity'] == 0]

# Create a histogram of the number of tracks with popularity = 0 by years
plt.figure(figsize=(10, 6))
sns.histplot(release_years[df['track_popularity'] == 0], bins=30, kde=False)
plt.title('Number of Tracks with Popularity = 0 by Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Tracks')
plt.show()
```


    
![png](02_exploratory_data_analysis_files/02_exploratory_data_analysis_41_0.png)
    


It seems that the zero poppularity is actually a late years trend, this makes me more comfortable working with this data. 

![tobecontinued.jpg](02_exploratory_data_analysis_files/tobecontinued.jpg)
