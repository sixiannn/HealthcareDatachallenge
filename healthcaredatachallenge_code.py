# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:49:59 2021

@author: Sixian
"""
import pandas as pd
import os
import glob

# Load dataframes
os.chdir('C:/Users/Sixia/downloads/HealthcareDataChalenge/Healthcare Data Challenge Data')
csv_list = glob.glob('*.csv')
print(csv_list)

# append datasets to the list
dfs = {}
for file in csv_list:
    dfs[file] = pd.read_csv(file)
  

pd.set_option("display.max.columns", None)

# Reassign dfs variable
print(dfs.keys())
demo = dfs['demographics.csv']
bill_amount = dfs['bill_amount.csv']
clinical_data = dfs['clinical_data.csv']
bill_id = dfs['bill_id.csv']

# Clean up dataframe for demographics df
demo.info()
demo.describe()

for col in demo:
    print(col, ":", demo[col].unique())

demo.replace({'gender': {'f':'Female',
                         'm':'Male'},
              'race': {'India':'Indian', 'chinese':'Chinese'},
              'resident_status' : {'Singapore citizen':'Singaporean'}
              }, inplace=True)
# check that it is cleaned up
for col in demo:
    print(col, ":", demo[col].unique())

# demo countplot
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x = 'resident_status', data=demo, palette=['#7fc97f','#beaed4','#fdc086'])
figure = plt.gcf()
plt.savefig('resident_status_demo.tiff', dpi=199, bbox_inches = "tight")
sns.countplot(x = 'race', data=demo, palette=['#7fc97f','#beaed4','#fdc086', '#ffff99'])
plt.gcf()
plt.savefig('race.tiff', dpi=199, bbox_inches = "tight")

# Clean up dataframe for bill_id
bill_id.info()
bill_id['date_of_admission'] = pd.to_datetime(bill_id['date_of_admission'], format="%Y-%m-%d")
bill_id['bill_id'] = bill_id['bill_id'].astype('object')
bill_id.info()

# Clean up dataframe for clinical_data
for col in clinical_data:
    print(col, ":", clinical_data[col].unique())
clinical_data.replace({'medical_history_3':{'No':'0', 'Yes':'1'}}, inplace=True)
clinical_data['medical_history_3'].unique() #check if replaced properly
clinical_data['medical_history_3'] = pd.to_numeric(clinical_data['medical_history_3']).astype('category')
clinical_data.iloc[:, 3:21] = clinical_data.iloc[:, 3:21].astype('object')
clinical_data.info()

# Check for missing values
clinical_data.isna().sum()
import missingno as msno
msno.bar(clinical_data)
msno.matrix(clinical_data)

# Check if missingno related to dates
clinical_data_sorted = clinical_data.sort_values(['date_of_admission', 'date_of_discharge']) # missingno are random

# Add new column length of stay(LOS)
clinical_data[['date_of_discharge', 'date_of_admission']] = clinical_data[['date_of_discharge', 'date_of_admission']].apply(pd.to_datetime)
clinical_data['LOS'] = clinical_data['date_of_discharge'] - clinical_data['date_of_admission']
clinical_data.info()
clinical_data.describe(include=['object'])
#check = clinical_data[clinical_data['id'] == '4e46fddfa404b306809c350aecbf0f6a']
# change LOS to numbers only
clinical_data['LOS'] = clinical_data['LOS'].astype(str)
clinical_data['LOS'] = clinical_data['LOS'].apply(lambda x: x.split(' ')[0])
clinical_data['LOS'] = clinical_data['LOS'].astype(int)


# Join bill_id with bill_amount
bill_join = pd.merge(bill_amount, bill_id, on = 'bill_id')
# check duplicated patient_ID
len(bill_join['patient_id'].unique()) #3000
bill_join['patient_id'][bill_join['patient_id'].duplicated()].count()
bill_join_sum = bill_join.groupby(['patient_id', 'date_of_admission']).agg(sum).reset_index()
duplicated = bill_join_sum[bill_join_sum['patient_id'].duplicated(keep=False)].sort_values('patient_id') #just to check duplicated id entries are from different admission dates
# join bill_id_sum with clinical data
clinical_data_joined = pd.merge(clinical_data, bill_join_sum, how='left', left_on=['id', 'date_of_admission'], right_on=['patient_id', 'date_of_admission'])
sum(clinical_data_joined['patient_id'] != clinical_data_joined['id']) #0
# drop one column since it's the same
clinical_data_joined.drop('patient_id', inplace=True, axis=1)

# fill up missing value by checking if there is any other entries of same patient
clinical_data.isna().sum() # both are medical history
missing_med_his2_id = clinical_data['id'][clinical_data['medical_history_2'].isna()]
med_hist_2 = {}
for a in missing_med_his2_id:
    for c in clinical_data_joined['id']:
        if a == c:
            b = list(clinical_data_joined[clinical_data_joined.id == a]['medical_history_2'].dropna())
            med_hist_2[a] = b

# remove empty values from dictionary and drop those with conflicting values
med_hist_2_tofill = {}
for k,v in med_hist_2.items(): 
    if bool(med_hist_2[k]) == True: #remove empty keys
        med_hist_2_tofill[k] = v
        if len(v) >1 and v[0] != v[1]: #if values conflicting then dropped from fillin
            med_hist_2_tofill.pop(k)
        elif len(v) >1 and v[0] == v[1]:
            med_hist_2[k] = v[0]

# to unlist all values in dictionary   
for k,v in med_hist_2_tofill.items():
    if isinstance(v, list):
        med_hist_2_tofill[k] = v[0]
            
missing_med_his5_id = clinical_data['id'][clinical_data['medical_history_5'].isna()]
med_hist_5 = {}
for a in missing_med_his5_id:
    for c in clinical_data_joined['id']:
        if a == c:
            b = list(clinical_data_joined[clinical_data_joined.id == a]['medical_history_5'].dropna())
            med_hist_5[a] = b
        else:
            continue

# Remove empty values from dictionary and drop those with conflicting values
med_hist_5_tofill = {}
for k,v in med_hist_5.items(): 
    if bool(med_hist_5[k]) == True: #remove empty keys
        med_hist_5_tofill[k] = v
        if len(v) >1 and v[0] != v[1]: #if values conflicting then dropped from fillin
            med_hist_5_tofill.pop(k)
        elif len(v) > 1 and v[0] == v[1]:
            med_hist_5_tofill[k] = v[0]


# to unlist all values in dictionary   
for k,v in med_hist_5_tofill.items():
    if isinstance(v, list):
        med_hist_5_tofill[k] = v[0]
    
# Fill up missing values
clinical_data_joined.medical_history_5 = clinical_data_joined.medical_history_5.fillna(clinical_data_joined.id.map(med_hist_5_tofill))
clinical_data_joined.medical_history_2 = clinical_data_joined.medical_history_2.fillna(clinical_data_joined.id.map(med_hist_2_tofill))

# check how many NA filled 
x = clinical_data[['medical_history_2', 'medical_history_5']].isna().sum()
y = clinical_data_joined[['medical_history_2', 'medical_history_5']].isna().sum()
a = (x-y)[0]
b = (x-y)[1]
print('{} of medical_history_2 missing values was filled and {} of medical_history_5 missing values was filled'.format(a,b))

# Check for inconsistency in medical history columns of the same ID
clinical_data_joined.set_index('id', inplace=True)
all_med_hist_only = clinical_data_joined.loc[:, clinical_data_joined.columns.str.startswith('medical_history')]
all_med_hist_only = all_med_hist_only[all_med_hist_only.columns].reset_index()
dup_1 = all_med_hist_only.drop_duplicates(keep='first') 
dup_1_value_counts = dup_1['id'].value_counts()
id_to_keep = dup_1_value_counts.index[dup_1_value_counts == 1] #2670
clean_df = clinical_data_joined.loc[id_to_keep].dropna(how='any', axis=0) #2330

# Further split up the date of admission by year
clean_df['year_of_admission'] = clean_df.date_of_admission.dt.year
clean_df.drop(['date_of_admission', 'date_of_discharge'], axis=1, inplace=True)

# merge clean_df with demographics
clean_df = clean_df.join(demo.set_index('patient_id'))
# change DOB to age
from datetime import date
def calculate_age(birthday):
    today = date.today()
    return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
clean_df['age'] = clean_df['date_of_birth'].apply(pd.to_datetime).apply(calculate_age)
clean_df.drop('date_of_birth', axis=1, inplace=True)


# Check distribution of data
summary_table = clinical_data_joined.describe()
summary_table.to_csv("c:/users/sixia/downloads/HealthcareDataChalenge/summary_table.csv")
cat_var = clinical_data_joined.loc[:,clinical_data_joined.columns[clinical_data_joined.columns.str.contains('medica|symptom')]].astype('category')
cat_var_count = cat_var.apply(lambda x: x.value_counts()).transpose()
cat_var_count.plot.bar(rot=0)
plt.xticks(rotation=90)


# EDA plot
colors=["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"]
a=0
b=0
c=0
subplots=(1,2)
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
#fig.delaxes(axs[1,2])
for x in clinical_data_joined.iloc[:,20:27].select_dtypes(include='float64'):
    sns.histplot(data=clinical_data_joined.reset_index(), x=x, kde=True, color=colors[c], ax=axs[a,b])
    c +=1
    if b < subplots[1]:
        b +=1
    else: 
        a +=1
        b = 0

# correlation analysis
import numpy as np
clean_df.loc[:,clean_df.columns[clean_df.columns.str.contains('medica|symptom')]] = clean_df.loc[:,clean_df.columns[clean_df.columns.str.contains('medica|symptom')]].astype("category")
from scipy import stats
spearman_corr_df = clean_df.select_dtypes(include=['float64','int32', 'int64'])
spear_corr_corr, spear_corr_pvalue = stats.spearmanr(spearman_corr_df)
spear_corr_corr = pd.DataFrame(spear_corr_corr)
spear_corr_pvalue = pd.DataFrame(spear_corr_pvalue)
spear_corr_corr.columns = spearman_corr_df.columns
spear_corr_corr.index = spearman_corr_df.columns
spear_corr_pvalue.columns = spearman_corr_df.columns
spear_corr_pvalue.index = spearman_corr_df.columns

mask = np.triu(np.ones_like(spear_corr_corr, dtype=bool))
mask_2 = spear_corr_pvalue > 0.05
cmap = sns.diverging_palette(210,20, as_cmap=True)
fig, axs = plt.subplots(figsize=(11, 9))
hm = sns.heatmap(spear_corr_corr, mask=mask|mask_2, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, annot_kws={'fontsize':18}, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, linecolor='grey')
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 18, rotation = 90)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 18, rotation = 0)
figure = plt.gcf()
plt.savefig('spearman_corrplot.tiff', dpi=199, bbox_inches = "tight")

# correlation between dichotomous variable and continuous variable using Point biserial’s correlation
# get continuous and dichotomous data
continuous_variable = clean_df['amount']
clean_df['gender'] = clean_df['gender'].astype('category')
list_dicho = clean_df.select_dtypes(include='category').columns
# convert gender to numerical values
clean_df_pointbiserialr = clean_df.replace({'gender':{'Female':'0', 'Male':'1'}})
clean_df_pointbiserialr['gender'] = clean_df_pointbiserialr['gender'].astype('float64')
from scipy.stats import pointbiserialr
x_list = []
corr_list = []
p_value_list = []
for x in list(list_dicho):
    corr = pointbiserialr(clean_df_pointbiserialr[x], continuous_variable)
    x_list.append(x)
    corr_list.append(corr[0])
    p_value_list.append(corr[1])
pointbiserialrResult = pd.DataFrame(list(zip(x_list, corr_list, p_value_list)), columns=['x', 'amount', 'p-value'])
pointbiserialrResult_1 = pointbiserialrResult.iloc[:,0:2]
pointbiserialrResult_1 = pointbiserialrResult_1.set_index('x')

mask = np.array(pointbiserialrResult['p-value'] > 0.05).reshape((19,1))
fig, ax = plt.subplots(1,1, figsize=(2,8))
corr_plot = sns.heatmap(pointbiserialrResult_1, cmap=cmap, vmin=-1, vmax=1, center=0, mask=mask,
            square=False, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, annot_kws={'fontsize':12}, fmt='.2f').tick_params(left=False, bottom=False)
ax.set_ylabel('') 
figure = plt.gcf()
plt.savefig('pointbiserialr_corrplot.tiff', dpi=199, bbox_inches = "tight")


cat_datasubset_aft_clean = clean_df.loc[:,clean_df.columns[clean_df.columns.str.contains('medica|symptom')]].astype('category')
cat_summary_aft_clean = cat_datasubset_aft_clean.apply(lambda x: x.value_counts())

# Kruskal wallis test
KW_df = clean_df.loc[:,['race','resident_status', 'amount']]
test = KW_df.pivot_table(columns='race', values='amount', index=KW_df.index)
resident_test = KW_df.pivot_table(columns='resident_status', values='amount', index=KW_df.index)
from scipy.stats import kruskal
stat, p = kruskal(test.Chinese.dropna(), test.Malay.dropna(), test.Indian.dropna(), test.Others.dropna())
if p < 0.05:
    print("Statistic: {:.3f}, p-value:{:.3f} ".format(stat,p))
    print("Null hypothesis reject: At least one group is significantly different")
else:
    print("Statistic: {:3f}, p-value:{:.3f} ".format(stat,p))
    print("The mean for each population is equal")
    
stat_2, p_2 = kruskal(resident_test.Foreigner.dropna(), resident_test.Singaporean.dropna(), resident_test.PR.dropna())
if p_2 < 0.05:
    print("Statistic: {:.3f}, p-value:{:.3f} ".format(stat_2,p_2))
    print("Null hypothesis reject: At least one group is significantly different")
else:
    print("Statistic: {:3f}, p-value:{:.3f} ".format(stat_2,p_2))
    print("The mean for each population is equal")

# posthoc dscf
import scikit_posthocs as sp 
x = pd.DataFrame({'chinese': test.Chinese, 'malay': test.Malay, 'indian': test.Indian, 'others':test.Others})
x= x.melt(var_name="groups", value_name='values')
x.dropna(inplace=True)
sp.posthoc_dscf(x, val_col='values', group_col='groups')

z = pd.DataFrame({'foreigner': resident_test.Foreigner, 'singaporean': resident_test.Singaporean, 'PR':resident_test.PR})
z = z.melt(var_name="groups", value_name='values')
z.dropna(inplace=True)
sp.posthoc_dscf(z, val_col='groups', group_col='groups')
# Model selection using LazyRegressor

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor


X = clean_df.drop('amount', axis=1)
y = clean_df.amount
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# fit all models
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)


models['R-Squared']= [0 if i <0 else i for i in models.iloc[:,0]]
plt.figure(figsize=(10,5))
sns.set_theme(style='whitegrid')
ax = sns.barplot(x=models.index, y="R-Squared", data=models)
plt.xticks(rotation=90)

# GradientBosstingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Encode categorical features 
X = pd.get_dummies(X, columns=['gender', 'race', 'resident_status'])
y = clean_df.amount
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)
regressor = GradientBoostingRegressor(
    max_depth=2, 
    n_estimators=2, 
    learning_rate=1.0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mean_absolute_error(y_test, y_pred)

# Find best parameters with GridSearchCV
from sklearn.model_selection import GridSearchCV, ShuffleSplit

regressor_2 = GradientBoostingRegressor(
    max_depth=2,
    n_estimators = 199,
    learning_rate=1.0
    )
regressor_2.fit(X_train, y_train)
y_pred = regressor_2.predict(X_test)
mean_absolute_error(y_test, y_pred)

model = GradientBoostingRegressor()
n_estimators = range(50, 400, 50)
param_grid = {
    'learning_rate': np.arange(0.1,1.0,0.1),
    'max_depth': [1,2,3,4,5,6],
    'n_estimators': list(range(10,400,10))}
shuffle_split = ShuffleSplit(n_splits=1,test_size=.25)
grid_search = GridSearchCV(model, param_grid, verbose=1, cv=shuffle_split, scoring='neg_mean_squared_error')
grid_result = grid_search.fit(X_train, y_train)
print("The best parameters are:", grid_search.best_params_)

regressor_2 = GradientBoostingRegressor(
    max_depth=3,
    n_estimators = 350,
    learning_rate=0.1,
    )
regressor_2.fit(X_train, y_train)
y_pred = regressor_2.predict(X_test)
mean_absolute_error(y_test, y_pred)
importance = regressor_2.feature_importances_

# summarize feature importance
Features = []
Scores = []
for features, importance in zip(X.columns, importance):
    Features.append(features)
    Scores.append(importance)
    print('Feature: {}, Score: {}'.format(features, importance))
Important_features = pd.DataFrame(list(zip(Features, Scores)), columns=['Features', 'Scores'])
Important_features.sort_values(by="Scores", ascending=False)

# plot feature importance

sns.set(rc={"figure.figsize":(7, 6)})
sns.set_style("ticks")
sns.barplot(x='Features', y='Scores', data=Important_features, palette="ch:.25")
plt.xticks(rotation=90)
plt.tight_layout()
figure = plt.gcf()
plt.savefig('GBR.tiff', dpi=199, bbox_inches = "tight")

# define the evaluation procedure
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
n_scores = cross_val_score(regressor_2, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# LightGBM for regression
import lightgbm
from lightgbm import LGBMRegressor
model = LGBMRegressor()
shuffle_split = ShuffleSplit(n_splits=1,test_size=.25)
grid_search = GridSearchCV(model, param_grid, verbose=1, cv=shuffle_split, scoring='neg_mean_squared_error')
grid_result = grid_search.fit(X_train, y_train)
print("The best parameters are:", grid_search.best_params_)

regressor_3 = LGBMRegressor(max_depth=2,
n_estimators = 390,
learning_rate=0.8)
n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


model = regressor_3.fit(X_train,y_train, eval_set=[(X_test, y_test), (X_train,y_train)])
print('Training accuracy {:.4f}'.format(model.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(X_test,y_test)))

lightgbm.plot_importance(model)
figure = plt.gcf()
plt.savefig('LGBM.tiff', dpi=199, bbox_inches = "tight")


# chisquare test - compare between each categorical value
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder
cat_df = clean_df.select_dtypes(include=['category','object'])
cat_df.loc[:, ['gender','race','resident_status']] = OrdinalEncoder().fit_transform(cat_df.loc[:, ['gender','race','resident_status']])

resultant = pd.DataFrame(data=[(0 for i in range(len(cat_df.columns))) for i in range(len(cat_df.columns))], 
                         columns=list(cat_df.columns))
resultant.set_index(pd.Index(list(cat_df.columns)), inplace = True)

# Finding p_value for all columns and putting them in the resultant matrix
for i in list(cat_df.columns):
    for j in list(cat_df.columns):
        if i != j:
            chi2_val, p_val = chi2(np.array(cat_df[i]).reshape(-1, 1), np.array(cat_df[j]).reshape(-1, 1))
            resultant.loc[i,j] = p_val
print(resultant)
mask_3 = np.array(resultant > 0.05)
mask_4 = np.triu(np.ones_like(resultant, dtype=bool))

fig = plt.figure(figsize=(15,15))
sns.heatmap(resultant, annot=True, cmap='Blues', mask=mask_3|mask_4, linecolor='grey', linewidths=1)
plt.title('Chi-Square Test Results (p-value)')
figure = plt.gcf
plt.savefig('Chi-Square.tiff', dpi=199, bbox_inches = "tight")
