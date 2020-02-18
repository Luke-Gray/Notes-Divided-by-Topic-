

#fb coding challenge
(data.groupby('referredID')
 .apply(lambda x: (x.type=='Comment').sum())
 .value_counts()
 .to_frame()
 .reset_index()
 .rename(columns = {'index':'Comments', 0:'Posts'}))

data[data.type == "Comment"]['referredID'].value_counts().to_dict()
data[data['referredID'] == 1001]['type']=='Comment'

data.isin({'referredID':[1001],'actionID':[1001]})

#coding challenge counting how many days of the week a user uses an app
(q2.groupby('userID')
 .apply(lambda x: len((x[x.equipment =='mobile']['time'].unique())))
 .reset_index()
 .rename(columns = {0:'day_count'}))

#using apply then lambda to grab last of a list
tags = hn_df['tags']
len_tags = tags.apply(len) == 4
cleaned_tags = tags.apply(lambda l: l[-1] if len(l) == 4 else None)

#sorted and then using lambda to sort by a certain column
hn_sorted_points = sorted(hn_clean, key = lambda d: d['points'], reverse=True)
top_5_titles = [d['title'] for d in hn_sorted_points[:5]]
	
#using regex, extracting, and then concat values of two date strings together
pattern = r"(?P<First_Year>[1-2][0-9]{3})/?(?P<Second_Year>[0-9]{2})?"
years = merged['IESurvey'].str.extractall(pattern)
first_two_year = years['First_Year'].str[0:2]
years['Second_Year'] = first_two_year + years['Second_Year']

#Using str.split and get to grab last value
merged['Currency Vectorized'] = merged['CurrencyUnit'].str.split().str.get(-1)
merged['Currency Vectorized'].head()


#Pandas summing columns based on value of another column
df['total'] = df.loc[df['A'] > 0,['A','B']].sum(axis=1)
df['total'].fillna(0, inplace=True)
df

Out[73]:
          A         B         C     total
0  0.197334  0.707852 -0.443475  0.905186
1 -1.063765 -0.914877  1.585882  0.000000
2  0.899477  1.064308  1.426789  1.963785
3 -0.556486 -0.150080 -0.149494  0.000000
4 -0.035858  0.777523 -0.453747  0.000000


#regex to grab groups from url
pattern = r"(?P<protocol>.+)://(?P<domain>[\w\.]+)/?(?P<path>.*)"
url_parts = hn['url'].str.extract(pattern, flags=re.I)

#grabbing all iterations of email
pattern = r"e[\-\s]?mail"
email_uniform = email_variations.str.replace(pattern, "email", flags=re.I)

#repeated words
pattern = r"\b(\w+)\s\1\b"

#Lookarounds -- matches when not preceded by and not followed by:
pattern = r"(?<!Series\s)\b[Cc]\b(?![\+\.])"
c_mentions = titles.str.contains(pattern).sum()


#will only grab the 't' if an 's' is not in front of it
(?<!s)t
ssssttreetst

#list comp grabbing columns
garage_list = [col for col in house.columns if 'Garage' in col]

#number of distinct places
ratings.userID.unique().shape

#create dataframe out of groupby and directly get the percentage/ratio of count
ratings.groupby('rating')[['food_rating']].count().apply(lambda t:t/t.sum())

#label encoding, appending, and dropping in one loop -- ML midterm
result  = []
classes = {}
for name in final.drop('rating',axis=1).columns:
    if name != ['BMI', 'Upayment', 'Rpayment', 'Rcuisine']:
             result.append(pd.Series(le.fit_transform(final[name])))
             classes[name] = le.classes_
    else:
             result.append(final.fillna({name:final[name].mean()})) 

###Pandas str.contains
pattern = r"[Jj]ava[^Ss]"
java_titles = titles[titles.str.contains(pattern)]

##binning with pd.cut
housing['income_cat']=pd.cut(housing['median_income'],
 bins=[0,1.5,3,4.5,np.inf],
 labels = [1,2,3,4,5])

##correlations
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

#scatter matrix

from pandas.plotting import scatter_matrix

attributes = ['median_house_value','median_income','total_rooms']
scatter_matrix(housing[attributes], figzise=(12,8))


###using nunique to check the importance of a column
def check_state():
    if LOCAL_TEST:
        bad_uids = full_df.groupby(['uid'])['isFraud'].agg(['nunique', 'count'])
        bad_uids = bad_uids[(bad_uids['nunique']==2)]
        print('Inconsistent groups',len(bad_uids))

    print('Cleaning done...')
    print('Total groups:', len(full_df['uid'].unique()), 
          '| Total items:', len(full_df),
          '| Total fraud', full_df['isFraud'].sum())


###using .clip to create a boundary and shift() to
for col in ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']:
        temp_df['sanity_check'] = temp_df.groupby(['uid'])[col].shift()
        temp_df['sanity_check'] = (temp_df[col]-temp_df['sanity_check']).fillna(0).clip(None,0)

 col_0  col_1
0      9     -2
1     -3     -7
2      0      6
3     -1      8
4      5     -5

 df.clip(-4, 6) #reassigns if outside boundary
   col_0  col_1
0      6     -2
1     -3     -4
2      0      6
3     -1      6
4      5     -4

###used floor so that after dividing by (24*60*60), you are just left with the day
full_df['DT_day'] = np.floor(full_df['TransactionDT']/(24*60*60)) + 1000

##after grabbing a subset of columns to drop
to_drop = (train_transaction.isna().sum()/len(train_transaction)>.60)
train_transaction = train_transaction.loc[:, ~to_drop]

########################### Single Transaction
# Let's filter single card apearence card1/D1 -> single transaction per card
full_df['count'] = full_df.groupby(['card1','uid_td_D1'])['TransactionID'].transform('count')
single_items = full_df[full_df['count']==1]
single_items['uid'] = single_items['TransactionID']
del full_df, single_items['count']

all_items = all_items[~all_items['TransactionID'].isin(single_items['TransactionID'])]
print('Single transaction',len(single_items))

##creatiing a new column according to config name using categorical index.codes
# Use CategoricalIndex with CategoricalIndex.codes:
###groupby straight to new feature
df['config_version_count'] = (df.groupby('config_name')['config_version']
                                .transform(lambda x: pd.CategoricalIndex(x).codes))

print (df)
   ID config_name  config_version  config_version_count
0  aa           A               0                     0
1  ab           A               7                     1
2  ad           A               7                     1
3  ad           A              27                     2
4  bb           B               0                     0
5  cc           C               0                     0
6  cd           C               8                     1


##Using boolean mask
exists = house['PoolQC'].notnull()
l = house[exists]['MiscFeature']
l.iloc[5]

l.iloc[:5] = 'Pool' 
l.iloc[-1] = 'Pool'

house['MiscFeature'] = house['MiscFeature'].mask(exists, l)

house['MiscFeature'] = house['MiscFeature'].fillna('None')
MiscF = house.pivot_table('SalePrice', 'MiscFeature', aggfunc = [np.size, np.mean])


#select and heatmap of only strong correlations
strong_corrs = sorted_corrs[sorted_corrs>0.3]

corrmat = train_subset[strong_corrs.index].corr()
sns.heatmap(corrmat)

##min-max scaling, then sorting features by variance
unit_train = train[features].apply(lambda x:((x-min(x))/(max(x)-min(x))))

sorted_vars = unit_train.var().sort_values()

#between dates pandas
df.loc[df['date'].between('2018-10-2','2018-10-31')]

##Selecting columns without missing values
no_null = [col for col in train if train[col].isna().sum()==0]
df_no_mv = train[no_null]

all_data_na = (house.isnull().sum() / len(house)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Percent Missing' :all_data_na})

##check len of unique values to determine if they are categorical. Use cat.codes to retrieve numerical value of cats.
for col in text_cols:
    print(col+":", len(train[col].unique()))
    train[col] = train[col].astype('category')
    
train['Utilities'].cat.codes.value_counts()

##printing specific columns of a df
point_guards[['pts', 'g', 'ppg']].head(5)

##pd.index.difference()
idx = pd.Index([17, 69, 33, 15, 19, 74, 10, 5]) 
idx2 = pd.Index(([69, 33, 15, 74, 19]))
idx.difference(idx2)



#iterating over a dict using dict.items()
for key, value in dict.items():
	print(key, value)



    ##pd.resample use for frequency conversion; resampling 'H'-hour 'D'-day, 'M'- month from datetime object
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();

##Using np.select to change 'conditions' to 'solutions'
conditions = [
    [train['air_temperature_f']<=50.0, train['wind_speed_mph']>3.0],
    [train['air_temperature_f']>50.0 , train['wind_speed_mph']<=3.0],
]
solutions = [
    wind_chill(train['air_temperature_f'], train['wind_speed_mph']),
    -999.0,
]

train['wind_chill'] = np.select(conditions, outputs, 'Other')


###using zfill to fill string series with zeros in the front
data["Salary"]= data["Salary"].astype(str) 
width = 10
data["Salary"]= data["Salary"].str.zfill(width) 

##importing multiple data files at once and reading each as df['data file name']
import pandas as pd
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]
data = {}
for f in data_files:
    d = pd.read_csv("schools/{0}".format(f))
    key_name = f.replace(".csv", "")
    data[key_name] = d

##combining columns after changing to numeric
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]
print(data['sat_results']['sat_score'].head())


## Random permutation and splitting train/test 80/20
# Set a random seed so the shuffle is the same every time
numpy.random.seed(1)

# Shuffle the rows  
# This permutes the index randomly using numpy.random.permutation
# Then, it reindexes the dataframe with the result
# The net effect is to put the rows into random order
income = income.reindex(numpy.random.permutation(income.index))

train_max_row = math.floor(income.shape[0] * .8)

train = income.iloc[:train_max_row]
test = income.iloc[train_max_row:]

###function to count count number of deaths superhero had based on 5 columns and .apply to new column
def clean_deaths(row):
    num_deaths = 0
    columns = ['Death1', 'Death2', 'Death3', 'Death4', 'Death5']
    
    for c in columns:
        death = row[c]
        if pd.isnull(death) or death == 'NO':
            continue
        elif death == 'YES':
            num_deaths += 1
    return num_deaths

true_avengers['Deaths'] = true_avengers.apply(clean_deaths, axis=1)




## Adding lag features
def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]
#Pandas dataframe.rolling() function provides the feature of rolling window calculations. The concept
#of rolling window calculation is most primarily used in signal processing and time series data. In a 
#very simple words we take a window size of k at a time and perform some desired mathematical operation
#on it. A window of size k means k consecutive values at a time. In a very simple case all the ‘k’ values
#are equally weighted.

### Using interpolate to fill nan values - various methods to do it
>>> s = pd.Series([np.nan, "single_one", np.nan,
...                "fill_two_more", np.nan, np.nan, np.nan,
...                4.71, np.nan])
>>> s
0              NaN
1       single_one
2              NaN
3    fill_two_more
4              NaN
5              NaN
6              NaN
7             4.71
8              NaN
dtype: object
>>> s.interpolate(method='pad', limit=2)
0              NaN
1       single_one
2       single_one
3    fill_two_more
4    fill_two_more
5    fill_two_more
6              NaN
7             4.71
8             4.71 


##dropping all non-numeric rows with apply() function
df[df.id.apply(lambda x: x.isnumeric())] 


###reduce_mem_usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
   
    return df

train = reduce_mem_usage(train)


##DF Summary
'''Variable Description'''
def description(df):
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.iloc[0].values
    summary['Second Value'] = df.iloc[1].values
    summary['Third Value'] = df.iloc[2].values
    return summary

    #using query to filter out certain rows
    train = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')


##group by without using object as index
df.groupby('Id', as_index=False).agg(lambda x: set(x))

##clean group_by and agg code
def compute_game_time_stats(group, col):
    return group[
        ['installation_id', col, 'event_count', 'game_time']
    ].groupby(['installation_id', col]).agg(
        [np.mean, np.sum, np.std]
    ).reset_index().pivot(
        columns=col,
        index='installation_id'
    )

# group3, group4 are grouped by installation_id 
# and reduced using summation and other summary stats
title_group = (
    pd.get_dummies(
        group_game_time.drop(columns=['game_session', 'event_count', 'game_time']),
        columns=['title', 'type', 'world'])
    .groupby(['installation_id'])
    .sum()
)

## looping through df series using iloc
for e in range(len(train[:20])):
    x = train['event_data'].iloc[e]
    y = json.loads(x)

## retrieving value from json dicts
cnt = 0
k = 'misses'
for e in range(len(train)):
    x = train['event_data'].iloc[e]
    y = json.loads(x)
    for key, value in y.items():
        if type(value)==dict:
            for key, val in value.items():
                if k in key:
                    print(key)
            

##qcut binning example wfor multiple columns
def group_data(data_frame):
    columns_group = get_predict_columns(data_frame)
    df_ranged = data_frame.copy()
    # For every numeric column, we create 5 (or less) ranges
    for column in df_ranged[columns_group].select_dtypes(include=numpy.number).columns:
        column_bin = pandas.qcut(df_ranged[column], 5, duplicates='drop')
        df_ranged[column] = column_bin

##dropping a particular col
reduce_train.drop(reduce_train.columns[-16], axis=1,inplace=True)
reduce_train = reduce_train.iloc[:, [j for j, c in enumerate(reduce_train.columns) if j != -16]]
##get index of col with Nan values
reduce_train.columns.get_loc('NaN')

### using .resampling on minute timestamps
>>> index = pd.date_range('1/1/2000', periods=9, freq='T')
>>> series = pd.Series(range(9), index=index)
>>> series
2000-01-01 00:00:00    0
2000-01-01 00:01:00    1
2000-01-01 00:02:00    2
2000-01-01 00:03:00    3
2000-01-01 00:04:00    4
2000-01-01 00:05:00    5
2000-01-01 00:06:00    6
2000-01-01 00:07:00    7
2000-01-01 00:08:00    8
Freq: T, dtype: int64

>>> series.resample('3T').sum()
2000-01-01 00:00:00     3
2000-01-01 00:03:00    12
2000-01-01 00:06:00    21

# .apply() see if all sample_submission installation_ids is in test set
n = len(sample_submission.apply(lambda x: x.installation_id in test.installation_id, axis=1))

##group_by df without having to agg()
grouped = train.groupby('installation_id',as_index=False).last()

#group by immediately to_frame()
generation = (global_power_plants.groupby(['primary_fuel'])['estimated_generation_gwh'].sum()).to_frame()

##unique way to use list comprehension for Series
years = [2013, 2014, 2015, 2016, 2017]
print([(gpp_df[f'generation_gwh_{x}'].nunique()) for x in years])

 df['dog2_sum'] = df1.apply(lambda row: (row['A']+row['B']+row['C']) if df['dog'] == 'dog2'))

##.fillna(method = 'bfill') --  fills Nan with item after missing value. 'ffill' fills with Nan with previous val
