

# Common Library Import

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Steps

Import Dataset
Perform Exploratory Data Analysis
Perform Data Visualization
Create Training and Testing Dataset
Train Model
Deploy Model


## Pandas

### Reading / Loading Files

insurance_df = pd.read_csv('case-02/insurance.csv') -> DataFrame

### DataFrame - Return First N Rows

insurance_df.head(5)

```
	age	sex	bmi	children	smoker	region	charges
0	19	female	27.900	0	yes	southwest	16884.92400
1	18	male	33.770	1	no	southeast	1725.55230
2	28	male	33.000	3	no	southeast	4449.46200
3	33	male	22.705	0	no	northwest	21984.47061
4	32	male	28.880	0	no	northwest	3866.85520
```

### DataFrame - Return Last N Rows

insurance_df.tail(5)

```
	age	sex	bmi	children	smoker	region	charges
1333	50	male	30.97	3	no	northwest	10600.5483
1334	18	female	31.92	0	no	northeast	2205.9808
1335	18	female	36.85	0	no	southeast	1629.8335
1336	21	female	25.80	0	no	southwest	2007.9450
1337	61	female	29.07	0	yes	northwest	29141.3603
```

### DataFrame - Check for null values

insurance_df.isnull().sum()

```
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
dtype: int64
```

### DataFrame - Info / DataTypes

insurance_df.info()

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
dtypes: float64(2), int64(2), object(3)
memory usage: 73.3+ KB
```

### DataFrame - Gives all the statistical 

insurance_df.describe()

```
	    age	        sex	        bmi	        children	smoker	    charges
count  1338.0000	1338.0000	1338.0000	1338.0000	1338.0000	1338.0000
mean   39.207025    0.505232	30.663397	1.094918	0.204783	13270.422265
std	   14.049960	0.500160	6.098187	1.205493	0.403694	12110.011237
min	   18.000000	0.000000	15.960000	0.000000	0.000000	1121.873900
25%	   27.000000	0.000000	26.296250	0.000000	0.000000	4740.287150
50%	   39.000000	1.000000	30.400000	1.000000	0.000000	9382.033000
75%	   51.000000	1.000000	34.693750	2.000000	0.000000	16639.912515
max	   64.000000	1.000000	53.130000	5.000000	1.000000	63770.428010
```

### DataFrame - Check Unique Value for Column

unique_sex = insurance_df['sex'].unique()



### DataFrame - Mapping Function

insurance_df['sex'] = insurance_df['sex'].apply(lambda x: 0 if x == 'female' else 1)
insurance_df['smoker'] = insurance_df['smoker'].apply(lambda x: 0 if x == 'no' else 1)

### DataFrame - Expand to Multiple Colums

region_dummies = pd.get_dummies(insurance_df['region'], drop_first = False)

```
	northeast	northwest	southeast	southwest
0	False	False	False	True
1	False	False	True	False
2	False	False	True	False
3	False	True	False	False
4	False	True	False	False
```

### DataFrame - Select Columns

insurance_df[['age', 'sex', 'bmi', 'children', 'smoker', 'charges']]

```
	age	sex	bmi	children	smoker	charges
0	19	0	27.900	0	1	16884.92400
1	18	1	33.770	1	0	1725.55230
2	28	1	33.000	3	0	4449.46200
3	33	1	22.705	0	0	21984.47061
4	32	1	28.880	0	0	3866.85520
```

### DataFrame - Visualize 

insurance_df[['age', 'sex', 'bmi', 'children', 'smoker', 'charges']].hist(bins = 30, figsize = (20,20), color = 'r')



### DataFrame - Group By

df_region = insurance_df.groupby(by='region').mean()

df_age = insurance_df.groupby(by='age').mean()

```
	sex	bmi	children	smoker	charges	northeast	northwest	southeast	southwest
age									
18	0.521739	31.326159	0.449275	0.173913	7086.217556	0.463768	0.000000	0.536232	0.000000
19	0.514706	28.596912	0.426471	0.264706	9747.909335	0.000000	0.500000	0.044118	0.455882
20	0.517241	30.632759	0.862069	0.310345	10159.697736	0.206897	0.241379	0.275862	0.275862
21	0.535714	28.185714	0.785714	0.071429	4730.464330	0.250000	0.250000	0.250000	0.250000
```