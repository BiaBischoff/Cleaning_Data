#Importing the packages:
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, skew
from scipy import stats

# sklearn modules for data preprocessing:
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# sklearn modules for Model Selection:
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# sklearn modules for Model Evaluation & Improvement:

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score

# Standard libraries for data visualization:
import seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
pd.options.display.max_columns = None
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve

#Loading the Churn Dataset
churn_df=pd.read_csv('/Users/bia/Desktop/churn_raw_data.csv', encoding = "ISO-8859-1")
print(churn_df)

#Return the columns names
df=churn_df.columns
print(df)

#Return all the data as a numpy array
data=churn_df.values

#Start cleaning up the data, getting rid of irrelevant data
df = churn_df.drop(churn_df.columns[0], axis = 1)
df.head()

#Removing the 9 columns: CaseOrder, Interaction, City, Zip, Lat and Lng
df_clean = df.drop(columns=['CaseOrder', 'Interaction', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Population'])
print(df_clean)

#Renaming the last 8 survey columns for a more descriptive value
df_clean.rename(columns = {'item1':'CS Responses', 'item2':'CS Fixes', 'item3':'CS Replacements', 'item4':'CS Reliability', 'item5':'CS Options',
                           'item6':'CS Respectfulness', 'item7':'CS Courteous', 'item8':'CS Listening'},
inplace=True)
print(df_clean)

# #How big my dataset currently is
# df_clean.shape
#
# df_stats = df_clean.describe()
#
# #Displaying Basic Statistics of Tenure
# df_stats_tenure = df_clean['Tenure'].describe()
# print(df_stats_tenure)
#
# #Displaying Basic Statistics of Monthly Charges
# df_stats_charge = df_clean['MonthlyCharge'].describe()
# print(df_stats_charge)
#
# #Calculating the Churn Rate
# churn_rate=df.Churn.value_counts() / len(df)
# print(churn_rate)

# #Creating a PieChart to Visualize the Churn rate
# plt.figure(figsize=(5,5))
# labels = ["Yes", "No"]
# values = [26.5, 73.5]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# #*********************************************************************************************************************
#A little bit more details about customers profile:
# #Gender Dist
# print(df_clean['Gender'].unique())
#
# male_count= df_clean['Gender'].value_counts()['Male']
# female_count = df_clean['Gender'].value_counts()['Female']
# none_count = df_clean['Gender'].value_counts()['Prefer not to answer']
#
# gender_total = female_count + male_count + none_count
#
# pct_female = (female_count / gender_total) * 100
# pct_male = (male_count / gender_total) * 100
# pct_none = (none_count / gender_total) * 100
# #Pie Chart of Gender
# plt.figure(figsize=(5,5))
# labels = ["Female", "Male", "Prefer not to answer"]
# values = [pct_female, pct_male, pct_none]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ##********************************************************************************************************************
# #Marital Values
# df_clean['Marital'].unique()
#
# #Marital Status Calcs for simplicity I am going to consider either with a partner or without one
#
# def partner_status(df_clean):
#     if df_clean['Marital'] == 'Married':
#         return "Yes"
#     else:
#         return "No"
#
# df_clean_9 = df_clean.copy()
# df_clean_9['partner'] = df_clean_9.apply(lambda df_clean_9:partner_status(df_clean_9), axis=1)
#
# with_count= df_clean_9['partner'].value_counts()['Yes']
# without_count = df_clean_9['partner'].value_counts()['No']
#
#
# divorced_count= df_clean['Marital'].value_counts()['Divorced']
# widowed_count = df_clean['Marital'].value_counts()['Widowed']
# separated_count = df_clean['Marital'].value_counts()['Separated']
# never_married_count = df_clean['Marital'].value_counts()['Never Married']
# married_count = df_clean['Marital'].value_counts()['Married']
#
# marital_total = divorced_count + widowed_count + separated_count + never_married_count + married_count
#
# pct_divorced = (divorced_count / marital_total) * 100
# pct_widowed = (widowed_count / marital_total) * 100
# pct_separated = (separated_count / marital_total) * 100
# pct_never_married = (never_married_count / marital_total) * 100
# pct_married = (married_count / marital_total) * 100
#
# pct_with_partner = pct_married
#
# pct_no_partner = pct_divorced + pct_widowed + pct_separated + pct_never_married


# #Pie Chart of Marital Status
#
# plt.figure(figsize=(5,5))
# labels = ["With Partner", "With No Partner"]
# values = [pct_with_partner, pct_no_partner]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# #*********************************************************************************************************************
# #Area Values
# df_clean['Area'].unique()
#
# #Area Distribution
#
# suburban_count= df_clean['Area'].value_counts()['Suburban']
# urban_count = df_clean['Area'].value_counts()['Urban']
# rural_count = df_clean['Area'].value_counts()['Rural']
#
# area_total = suburban_count + urban_count + rural_count
#
# pct_suburban = (suburban_count / area_total) * 100
# pct_urban = (urban_count / area_total) * 100
# pct_rural = (rural_count / area_total) * 100

# #Pie Chart of Area Distribution
#
# plt.figure(figsize=(5,5))
# labels = ["Suburban", "Urban", "Rural"]
# values = [pct_suburban, pct_urban, pct_rural]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Customers age profile
# print(df_clean['Age'].unique())
#
# def age_intervals(df_clean):
#     if df_clean['Age'] <= 30:
#         return "Young Adult"
#     elif (df_clean['Age'] > 31) & (df_clean['Age'] <= 50):
#         return "Adult"
#     elif (df_clean['Age'] > 51) & (df_clean['Age'] <= 65):
#         return "Older Adult"
#     elif (df_clean['Age'] > 65):
#         return "Senior"
#
# df_clean_4 = df_clean.copy()
# df_clean_4['age_group'] = df_clean_4.apply(lambda df_clean_4:age_intervals(df_clean_4), axis=1)
#
# age1_count= df_clean_4['age_group'].value_counts()['Young Adult']
# age2_count = df_clean_4['age_group'].value_counts()['Adult']
# age3_count= df_clean_4['age_group'].value_counts()['Older Adult']
# age4_count= df_clean_4['age_group'].value_counts()['Senior']
#
# age_total = age1_count + age2_count + age3_count + age4_count
#
# pct_age1 = (age1_count / age_total) * 100
# pct_age2 = (age2_count / age_total) * 100
# pct_age3 = (age3_count / age_total) * 100
# pct_age4 = (age4_count / age_total) * 100

# #Pie Chart of Customers age
#
# plt.figure(figsize=(5,5))
# labels = ["Younger Adult", "Adult", "Older Adult", "Senior"]
# values = [pct_age1, pct_age2, pct_age3, pct_age4]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Unique values for dependents
# kids = print(df_clean['Children'].unique())
#
# #Customers Dependents
# def dep(df_clean):
#     if df_clean['Children'] >= 1:
#        return "Dependents"
#     elif df_clean['Children'] == 0:
#         return "No Dependents"
#     else: return "N/A"
#
# df_clean_5 = df_clean.copy()
# df_clean_5['dependents_group'] = df_clean_5.apply(lambda df_clean_5:dep(df_clean_5), axis=1)
#
# dep_count = df_clean_5['dependents_group'].value_counts()['Dependents']
# no_dep_count = df_clean_5['dependents_group'].value_counts()['No Dependents']
# na_count = df_clean_5['dependents_group'].value_counts()['N/A']
#
# dep_total = dep_count + no_dep_count + na_count
#
# pct_dep = (dep_count / dep_total) * 100
# pct_no_dep = (no_dep_count / dep_total) * 100
# pct_na = (na_count / dep_total) * 100
#
# #Pie Chart of customers dependents
# plt.figure(figsize=(5,5))
# labels = ["With Dependents", "No Dependents", "N/A"]
# values = [pct_dep, pct_no_dep, pct_na]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
##Employment Profile
# df_clean['Employment'].unique()
#
# #Customer Employment Profile
# fulltime_count= df_clean['Employment'].value_counts()['Full Time']
# parttime_count = df_clean['Employment'].value_counts()['Part Time']
# student_count = df_clean['Employment'].value_counts()['Student']
# retired_count = df_clean['Employment'].value_counts()['Retired']
# unemployed_count = df_clean['Employment'].value_counts()['Unemployed']
#
# employment_total = fulltime_count + parttime_count + student_count + retired_count + unemployed_count
#
# pct_fulltime = (fulltime_count / employment_total) * 100
# pct_parttime = (parttime_count / employment_total) * 100
# pct_student = (student_count / employment_total) * 100
# pct_retired = (retired_count / employment_total) * 100
# pct_unemployed = (unemployed_count / employment_total) * 100
#
# #Pie Chart of Employment Distribution
# plt.figure(figsize=(5,5))
# labels = ["Full Time", "Part Time", "Student", "Retired", "Unemployed"]
# values = [pct_fulltime, pct_parttime, pct_student, pct_retired, pct_unemployed ]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Payment Method
# eletronic_check_count= df_clean['PaymentMethod'].value_counts()['Electronic Check']
# mailed_check_count = df_clean['PaymentMethod'].value_counts()['Mailed Check']
# bank_transfer_count = df_clean['PaymentMethod'].value_counts()['Bank Transfer(automatic)']
# cc_count = df_clean['PaymentMethod'].value_counts()['Credit Card (automatic)']
#
# payment_total = eletronic_check_count + mailed_check_count + bank_transfer_count + cc_count
#
# pct_eletronic_check = (eletronic_check_count / payment_total) * 100
# pct_mailed_check = (mailed_check_count / payment_total) * 100
# pct_bank_transfer = (bank_transfer_count / payment_total) * 100
# pct_cc = (cc_count / payment_total) * 100

# #Pie Chart of Payment Method
#
# plt.figure(figsize=(5,5))
# labels = ["Electronic Check", "Mailed Check", "Bank Transfer (Auto)", "CC (Auto)"]
# values = [pct_eletronic_check, pct_mailed_check, pct_bank_transfer, pct_cc ]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Tenure Evaluation: I am going to create tenure intervals (0-12 months, 13-24 months, 25-48 months, 49-60 months,
# #greater than 60 months)
#
# def tenure_bands(df_clean):
#     if df_clean['Tenure'] <= 12:
#         return "Tenure_0-12"
#     elif (df_clean['Tenure'] > 12) & (df_clean['Tenure'] <= 24):
#         return "Tenure_13-24"
#     elif (df_clean['Tenure'] > 24) & (df_clean['Tenure'] <= 48):
#         return "Tenure_25-48"
#     elif (df_clean['Tenure'] > 48) & (df_clean['Tenure'] <= 60):
#         return "Tenure_49-60"
#     elif df_clean['Tenure'] > 60:
#         return "Tenure_gt_60"
#
# df_clean_2 = df_clean.copy()
# df_clean_2['tenure_group'] = df_clean_2.apply(lambda df_clean_2:tenure_bands(df_clean_2), axis=1)
#
# band1_count= df_clean_2['tenure_group'].value_counts()['Tenure_0-12']
# band2_count = df_clean_2['tenure_group'].value_counts()['Tenure_13-24']
# band3_count= df_clean_2['tenure_group'].value_counts()['Tenure_25-48']
# band4_count= df_clean_2['tenure_group'].value_counts()['Tenure_49-60']
# band5_count= df_clean_2['tenure_group'].value_counts()['Tenure_gt_60']
#
# band_total = band1_count + band2_count + band3_count + band4_count + band5_count
#
# pct_band1 = (band1_count / band_total) * 100
# pct_band2 = (band2_count / band_total) * 100
# pct_band3 = (band3_count / band_total) * 100
# pct_band4 = (band4_count / band_total) * 100
# pct_band5 = (band5_count / band_total) * 100
#
# #Pie Chart of Tenure Time
# plt.figure(figsize=(5,5))
# labels = ["0-12 Months", "13-24 Months", "25-48 Months", "48-60 Months", "Longer than 60 months"]
# values = [pct_band1, pct_band2, pct_band3, pct_band4, pct_band5]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Income Evaluation
# def income(df_clean):
#     if df_clean['Income'] <= 40000:
#         return "Income Level 1"
#     elif (df_clean['Income'] > 40001) & (df_clean['Income'] <= 80000):
#         return "Income Level 2"
#     elif (df_clean['Income'] > 80001) & (df_clean['Income'] <= 150000):
#         return "Income Level 3"
#     elif (df_clean['Income'] > 150001):
#         return "Income Level 4"
#
# df_clean_3 = df_clean.copy()
# df_clean_3['income_level'] = df_clean_3.apply(lambda df_clean_3:income(df_clean_3), axis=1)
#
# inc1_count= df_clean_3['income_level'].value_counts()['Income Level 1']
# inc2_count = df_clean_3['income_level'].value_counts()['Income Level 2']
# inc3_count= df_clean_3['income_level'].value_counts()['Income Level 3']
# inc4_count= df_clean_3['income_level'].value_counts()['Income Level 4']
#
# inc_total = inc1_count + inc2_count + inc3_count + inc4_count
#
# pct_inc1 = (inc1_count / inc_total) * 100
# pct_inc2 = (inc2_count / inc_total) * 100
# pct_inc3 = (inc3_count / inc_total) * 100
# pct_inc4 = (inc4_count / inc_total) * 100
#
# #Pie Chart of Income
#
# plt.figure(figsize=(5,5))
# labels = ["Below $40K", "Above $40K below $80K", "Above $80K below $150K", "Above $150K"]
# values = [pct_inc1, pct_inc2, pct_inc3, pct_inc4]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Monthly Charge Evaluation
# def monthly_charge(df_clean):
#     if df_clean['MonthlyCharge'] <= 100:
#         return "Charge Level 1"
#     elif (df_clean['MonthlyCharge'] > 100) & (df_clean['MonthlyCharge'] <= 150):
#         return "Charge Level 2"
#     elif (df_clean['MonthlyCharge'] > 150) & (df_clean['MonthlyCharge'] <= 200):
#         return "Charge Level 3"
#     elif (df_clean['MonthlyCharge'] > 200):
#         return "Charge Level 4"
#
# df_clean_6 = df_clean.copy()
# df_clean_6['Monthly_Charge'] = df_clean_6.apply(lambda df_clean_6:monthly_charge(df_clean_6), axis=1)
#
# charge1_count = df_clean_6['Monthly_Charge'].value_counts()['Charge Level 1']
# charge2_count = df_clean_6['Monthly_Charge'].value_counts()['Charge Level 2']
# charge3_count = df_clean_6['Monthly_Charge'].value_counts()['Charge Level 3']
# charge4_count = df_clean_6['Monthly_Charge'].value_counts()['Charge Level 4']
#
# charge_total = charge1_count + charge2_count + charge3_count + charge4_count
#
# pct_charge1 = (charge1_count / charge_total) * 100
# pct_charge2 = (charge2_count / charge_total) * 100
# pct_charge3 = (charge3_count / charge_total) * 100
# pct_charge4 = (charge4_count / charge_total) * 100
#
# #Pie Chart of Monthly Charge
#
# plt.figure(figsize=(5,5))
# labels = ["Monthly Charge below $100", "Monthly Charge in between $101 and $150", "Monthly Charge in between $150 and $200", "Monthly Charge above $201"]
# values = [pct_charge1, pct_charge2, pct_charge3, pct_charge4]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Number of times a customer had to call CS: "Contacts"
# def number_contacts(df_clean):
#     if df_clean['Contacts'] == 0:
#         return "Never Contacted"
#     elif (df_clean['Contacts'] == 1):
#         return "Contacted once"
#     elif (df_clean['Contacts'] ==2):
#         return "Contacted twice"
#     elif (df_clean['Contacts'] >= 3):
#         return "Contacted 3 or more times"
#
# df_clean_7 = df_clean.copy()
# df_clean_7['Number_Contacts'] = df_clean_7.apply(lambda df_clean_7:number_contacts(df_clean_7), axis=1)
#
# contact1_count = df_clean_7['Number_Contacts'].value_counts()['Never Contacted']
# contact2_count = df_clean_7['Number_Contacts'].value_counts()['Contacted once']
# contact3_count = df_clean_7['Number_Contacts'].value_counts()['Contacted twice']
# contact4_count = df_clean_7['Number_Contacts'].value_counts()['Contacted 3 or more times']
#
# contact_total = contact1_count + contact2_count + contact3_count + contact4_count
#
# contact1_pct = (contact1_count / contact_total) * 100
# contact2_pct = (contact2_count / contact_total) * 100
# contact3_pct = (contact3_count / contact_total) * 100
# contact4_pct = (contact4_count / contact_total) * 100

# #Pie Chart of how many times customer contacted CS
# plt.figure(figsize=(5,5))
# labels = ["Never Contacted", "Contacted once", "Contacted twice", "Contacted 3 or more times"]
# values = [contact1_pct, contact2_pct, contact3_pct, contact4_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Type of contract: monthly, one year or two year contract
# print(df_clean['Contract'].unique())
#
# month_to_month_count= df_clean['Contract'].value_counts()['Month-to-month']
# one_year_count= df_clean['Contract'].value_counts()['One year']
# two_year_count = df_clean['Contract'].value_counts()['Two Year']
#
# contract_total = month_to_month_count + one_year_count + two_year_count
#
# contract1_pct = (month_to_month_count / contract_total) * 100
# contract2_pct = (one_year_count / contract_total) * 100
# contract3_pct = (two_year_count / contract_total) * 100

# #Pie Chart of how many times customer contacted CS
#
# plt.figure(figsize=(5,5))
# labels = ["Month-to-month", "1 year Contract", "2 year Contract"]
# values = [contract1_pct, contract2_pct, contract3_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
#*********************************************************************************************************************
# #Type of Internet Service
# print(df_clean['InternetService'].unique())
#
# fiber_optic_count = df_clean['InternetService'].value_counts()['Fiber Optic']
# dsl_count = df_clean['InternetService'].value_counts()['DSL']
# none_count = df_clean['InternetService'].value_counts()['None']
#
# service_total = fiber_optic_count + dsl_count + none_count
#
# fiber_pct = (fiber_optic_count / service_total) * 100
# dsl_pct = (dsl_count / service_total) * 100
# none_pct = (none_count / service_total) * 100

# #Pie Chart of type of Internet Service
#
# plt.figure(figsize=(5,5))
# labels = ["Fiber Optic", "DSL", "None"]
# values = [fiber_pct, dsl_pct, none_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ##*********************************************************************************************************************
# #Number of services a customer signed up for
# print(df_clean['OnlineSecurity'].unique())
# print(df_clean['OnlineBackup'].unique())
# print(df_clean['DeviceProtection'].unique())
# print(df_clean['TechSupport'].unique())
# print(df_clean['StreamingTV'].unique())
# print(df_clean['StreamingMovies'].unique())
# ##*********************************************************************************************************************
# #Online Security
# os_yes_count = df_clean['OnlineSecurity'].value_counts()['Yes']
# os_no_count = df_clean['OnlineSecurity'].value_counts()['No']
#
# onlinesec_total = os_yes_count + os_no_count
#
# os_yes_pct = (os_yes_count / onlinesec_total) * 100
# os_no_pct = (os_no_count / onlinesec_total) * 100
#
# #Pie Chart of type of Online Security
#
# plt.figure(figsize=(5,5))
# labels = ["Yes for Online Security", "No for Online Security"]
# values = [os_yes_pct, os_no_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ###*********************************************************************************************************************
# #Online Backup
# ob_yes_count = df_clean['OnlineBackup'].value_counts()['Yes']
# ob_no_count = df_clean['OnlineBackup'].value_counts()['No']
#
# onlinebackup_total = ob_yes_count + ob_no_count
#
# ob_yes_pct = (ob_yes_count / onlinebackup_total) * 100
# ob_no_pct = (ob_no_count / onlinebackup_total) * 100
#
# #Pie Chart of type of Online Backup
#
# plt.figure(figsize=(5,5))
# labels = ["Yes for Online Backup", "No for Online Backup"]
# values = [ob_yes_pct, ob_no_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ##*********************************************************************************************************************
# #DeviceProtection
# devprotection_yes_count = df_clean['DeviceProtection'].value_counts()['Yes']
# devprotection_no_count = df_clean['DeviceProtection'].value_counts()['No']
#
# devprotection_total = devprotection_yes_count + devprotection_no_count
#
# devprotection_yes_pct = (devprotection_yes_count / devprotection_total) * 100
# devprotection_no_pct = (devprotection_no_count / devprotection_total) * 100
#
# #Pie Chart of type of Device Protection
#
# plt.figure(figsize=(5,5))
# labels = ["Yes for Device Protection", "No for Device Protection"]
# values = [devprotection_yes_pct, devprotection_no_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ##*********************************************************************************************************************
# #Tech Support
# techsupport_yes_count = df_clean['TechSupport'].value_counts()['Yes']
# techsupport_no_count = df_clean['TechSupport'].value_counts()['No']
#
# techsupport_nan_count = 10000 - techsupport_yes_count - techsupport_no_count
#
# techsupport_total = techsupport_yes_count + techsupport_no_count + techsupport_nan_count
#
# techsupport_yes_pct = (techsupport_yes_count / techsupport_total) * 100
# techsupport_no_pct = (techsupport_no_count / techsupport_total) * 100
# techsupport_nan_pct = (techsupport_nan_count / techsupport_total) * 100
#
# #Pie Chart of type of Tech Support
#
# plt.figure(figsize=(5,5))
# labels = ["Yes for Tech Support", "No for tech Support", "N/A"]
# values = [techsupport_yes_pct, techsupport_no_pct, techsupport_nan_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ##*********************************************************************************************************************
# #Streaming TV
# st_yes_count = df_clean['StreamingTV'].value_counts()['Yes']
# st_no_count = df_clean['StreamingTV'].value_counts()['No']
#
# streamingtv_total = st_yes_count + st_no_count
#
# st_yes_pct = (st_yes_count / streamingtv_total) * 100
# st_no_pct = (st_no_count / streamingtv_total) * 100
#
# #Pie Chart of type of Streaming TV
#
# plt.figure(figsize=(5,5))
# labels = ["Yes for Streaming TV", "No for Streaming TV"]
# values = [st_yes_pct, st_no_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()
# ##*********************************************************************************************************************
# #Streaming Movies
# sm_yes_count = df_clean['StreamingMovies'].value_counts()['Yes']
# sm_no_count = df_clean['StreamingMovies'].value_counts()['No']
#
# streamingm_total = sm_yes_count + sm_no_count
#
# sm_yes_pct = (sm_yes_count / streamingm_total) * 100
# sm_no_pct = (sm_no_count / streamingm_total) * 100
#
# #Pie Chart of type of Streaming Movies
#
# plt.figure(figsize=(5,5))
# labels = ["Yes for Streaming Movies", "No for Streaming Movies"]
# values = [st_yes_pct, st_no_pct]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()

# #Bandwidth_GB_Year Aalysis
# print(df_clean['Bandwidth_GB_Year'].unique())
# def bandwidth(df_clean):
#     if df_clean['Bandwidth_GB_Year'] <= 500:
#         return "Bandwidth Below 500"
#     elif (df_clean['Bandwidth_GB_Year'] > 500) & (df_clean['Bandwidth_GB_Year'] <= 1000):
#         return "Bandwidth 500 - 1000"
#     elif (df_clean['Bandwidth_GB_Year'] > 1000) & (df_clean['Bandwidth_GB_Year'] <= 2000):
#         return "Bandwidth 1000 - 2000"
#     elif (df_clean['Bandwidth_GB_Year'] > 2000):
#         return "Bandwidth Above 2000"
#
# df_clean_10 = df_clean.copy()
# df_clean_10['Bandwidth_GB_Year'] = df_clean_10.apply(lambda df_clean_10:bandwidth(df_clean_10), axis=1)
#
# bd1_count = df_clean_10['Bandwidth_GB_Year'].value_counts()['Bandwidth Below 500']
# bd2_count = df_clean_10['Bandwidth_GB_Year'].value_counts()['Bandwidth 500 - 1000']
# bd3_count = df_clean_10['Bandwidth_GB_Year'].value_counts()['Bandwidth 1000 - 2000']
# bd4_count = df_clean_10['Bandwidth_GB_Year'].value_counts()['Bandwidth Above 2000']
#
# bd_total = bd1_count + bd2_count + bd3_count + bd4_count
#
# pct_bd1 = (bd1_count / bd_total) * 100
# pct_bd2 = (bd2_count / bd_total) * 100
# pct_bd3 = (bd3_count / bd_total) * 100
# pct_bd4 = (bd4_count / bd_total) * 100
#
# #Pie Chart of Monthly Charge
#
# plt.figure(figsize=(5,5))
# labels = ["Bandwidth below 500", "Bandwidth 500 to 1000", "Bandwidth 1000 to 2000", "Bandwidth Above 2000"]
# values = [pct_bd1, pct_bd2, pct_bd3, pct_bd4]
# plt.pie(values, labels=labels, autopct="%.1f%%")
# plt.show()

# #**********************************************************************************************************************
# #Data Distribution by Churn Rate:
# #**********************************************************************************************************************
# #1. Gender
# #Lets not forget that Gender also has "Prefer not to answer" which were not considered in the analysis
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['Gender'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['Gender'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['Gender'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['Gender'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# male = [no_churn['Male'], yes_churn['Male']]
# female = [no_churn['Female'], yes_churn['Female']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, male, color='#00BFFF', label='Male', edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, female, bottom=male, color='#FF9999', label='Female', edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by Gender')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Partner against Churn Rate
# no_churn = ((df_clean_9[df_clean_9['Churn'] == 'No']['partner'].value_counts()) / (df_clean_9[df_clean_9['Churn'] == 'No']['partner'].value_counts().sum()))
# yes_churn = ((df_clean_9[df_clean_9['Churn'] == 'Yes']['partner'].value_counts()) / (df_clean_9[df_clean_9['Churn'] == 'Yes']['partner'].value_counts().sum()))
#
# # # Getting values from the group and categories
# x_labels = df_clean_9['Churn'].value_counts().keys().tolist()
# y_var = [no_churn['Yes'], yes_churn['Yes']]
# n_var = [no_churn['No'], yes_churn['No']]
# #
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, y_var, color='#00BFFF', label='Yes', edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, n_var, bottom= n_var, color='#FF9999', label='No', edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by Partner')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Churn Rate against Yes and No variables will be analyzed now
# #**********************************************************************************************************************
# #Phone Service
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['Phone'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['Phone'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['Phone'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['Phone'].value_counts().sum()))
#
# # Getting values from the group and categories
# #Churn is my X axis
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# #Y axis will be my other variables
# n_var = [no_churn['No'], yes_churn['No']]
# y_var = [no_churn['Yes'], yes_churn['Yes']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, n_var, color='#00BFFF', label=('No Phone'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, y_var, bottom=n_var, color='lightgreen', label=('Yes Phone'), edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by Phone')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Online Security
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['OnlineSecurity'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['OnlineSecurity'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['OnlineSecurity'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['OnlineSecurity'].value_counts().sum()))
#
# # Getting values from the group and categories
# #Churn is my X axis
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# #Y axis will be my other variables
# n_var = [no_churn['No'], yes_churn['No']]
# y_var = [no_churn['Yes'], yes_churn['Yes']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, n_var, color='#00BFFF', label=('No OnlineSecurity'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, y_var, bottom=n_var, color='lightgreen', label=('Yes OnlineSecurity'), edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by OnlineSecurity')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Online Backup
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['OnlineBackup'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['OnlineBackup'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['OnlineBackup'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['OnlineBackup'].value_counts().sum()))
#
# # Getting values from the group and categories
# #Churn is my X axis
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# #Y axis will be my other variables
# n_var = [no_churn['No'], yes_churn['No']]
# y_var = [no_churn['Yes'], yes_churn['Yes']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, n_var, color='#00BFFF', label=('No OnlineBackup'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, y_var, bottom=n_var, color='lightgreen', label=('Yes OnlineBackup'), edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by Online Backup')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Device Protection
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['DeviceProtection'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['DeviceProtection'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['DeviceProtection'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['DeviceProtection'].value_counts().sum()))
#
# # Getting values from the group and categories
# #Churn is my X axis
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# #Y axis will be my other variables
# n_var = [no_churn['No'], yes_churn['No']]
# y_var = [no_churn['Yes'], yes_churn['Yes']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, n_var, color='#00BFFF', label=('No DeviceProtection'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, y_var, bottom=n_var, color='lightgreen', label=('Yes DeviceProtection'), edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by DeviceProtection')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Tech Support
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['TechSupport'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['TechSupport'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['TechSupport'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['TechSupport'].value_counts().sum()))
#
# # Getting values from the group and categories
# #Churn is my X axis
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# #Y axis will be my other variables
# n_var = [no_churn['No'], yes_churn['No']]
# y_var = [no_churn['Yes'], yes_churn['Yes']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, n_var, color='#00BFFF', label=('No TechSupport'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, y_var, bottom=n_var, color='lightgreen', label=('Yes TechSupport'), edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by TechSupport')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Multiple lines service
# no_churn = ((df_clean[df_clean['Churn'] == 'No']['Multiple'].value_counts()) / (df_clean[df_clean['Churn'] == 'No']['Multiple'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn'] == 'Yes']['Multiple'].value_counts()) / (df_clean[df_clean['Churn'] == 'Yes']['Multiple'].value_counts().sum()))
#
# # Getting values from the group and categories
# #Churn is my X axis
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# #Y axis will be my other variables
# n_var = [no_churn['No'], yes_churn['No']]
# y_var = [no_churn['Yes'], yes_churn['Yes']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7, 7))
# ax1 = plt.bar(x_labels, n_var, color='#00BFFF', label=('No Multiple'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, y_var, bottom=n_var, color='lightgreen', label=('Yes Multiple'), edgecolor='white', width=barWidth)
# plt.legend()
# plt.title('Churn Distribution by Multiple')
#
# for r1, r2 in zip(ax1, ax2):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Tenure and Churn Rate
# no_churn = ((df_clean_2[df_clean_2['Churn']=='No']['tenure_group'].value_counts()) /(df_clean_2[df_clean_2['Churn']=='No']['tenure_group'].value_counts().sum()))
# yes_churn = ((df_clean_2[df_clean_2['Churn']=='Yes']['tenure_group'].value_counts()) /(df_clean_2[df_clean_2['Churn']=='Yes']['tenure_group'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# t_0_12 = [no_churn['Tenure_0-12'], yes_churn['Tenure_0-12']]
# t_13_24 = [no_churn['Tenure_13-24'], yes_churn['Tenure_13-24']]
# t_25_48 = [no_churn['Tenure_25-48'], yes_churn['Tenure_25-48']]
# t_49_60 = [no_churn['Tenure_49-60'], yes_churn['Tenure_49-60']]
# t_gt_60 = [no_churn['Tenure_gt_60'], yes_churn['Tenure_gt_60']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, t_0_12, color='#00BFFF', label=('Below 12M'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, t_13_24, bottom=t_0_12, color='lightgreen', label=('13 to 24'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, t_25_48, bottom=np.array(t_0_12) + np.array(t_13_24), color='#FF9999',  label=('25 to 48'), edgecolor='white', width=barWidth)
# ax4 = plt.bar(x_labels, t_49_60, bottom=np.array(t_0_12) + np.array(t_13_24) + np.array(t_25_48), color='#FFA07A', label=('48 to 60'), edgecolor='white', width=barWidth)
# ax5 = plt.bar(x_labels, t_gt_60, bottom=np.array(t_0_12) + np.array(t_13_24) + np.array(t_25_48) + np.array(t_49_60), color='#F0E68C', label=('> 60'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Tenure Group')
#
# for r1, r2, r3, r4, r5 in zip(ax1, ax2, ax3, ax4, ax5):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     h4 = r4.get_height()
#     h5 = r5.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r4.get_x() + r4.get_width() / 2., h1 + h2 + h3 + h4 / 2., '{:.2%}'.format(h4), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r5.get_x() + r5.get_width() / 2., h1 + h2 + h3 + h4 + h5 / 2., '{:.2%}'.format(h5), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Area and Churn Rate
# no_churn = ((df_clean[df_clean['Churn']=='No']['Area'].value_counts()) /(df_clean[df_clean['Churn']=='No']['Area'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn']=='Yes']['Area'].value_counts()) /(df_clean[df_clean['Churn']=='Yes']['Area'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# rural = [no_churn['Rural'], yes_churn['Rural']]
# suburban = [no_churn['Suburban'], yes_churn['Suburban']]
# urban = [no_churn['Urban'], yes_churn['Urban']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, rural, color='#00BFFF', label=('Rural'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, suburban, bottom=rural, color='lightgreen', label=('Suburban'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, urban, bottom=np.array(rural) + np.array(suburban), color='#FF9999',  label=('Urban'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Area')
#
# for r1, r2, r3 in zip(ax1, ax2, ax3):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Payment type and Churn Rate
# no_churn = ((df_clean[df_clean['Churn']=='No']['PaymentMethod'].value_counts()) /(df_clean[df_clean['Churn']=='No']['PaymentMethod'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn']=='Yes']['PaymentMethod'].value_counts()) /(df_clean[df_clean['Churn']=='Yes']['PaymentMethod'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# electornic_check = [no_churn['Electronic Check'], yes_churn['Electronic Check']]
# mailed_check = [no_churn['Mailed Check'], yes_churn['Mailed Check']]
# bank_transfer = [no_churn['Bank Transfer(automatic)'], yes_churn['Bank Transfer(automatic)']]
# credit_card = [no_churn['Credit Card (automatic)'], yes_churn['Credit Card (automatic)']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, electornic_check, color='#00BFFF', label=('Electronic Check'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, mailed_check, bottom=electornic_check, color='lightgreen', label=('Mailed Check'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, bank_transfer, bottom=np.array(electornic_check) + np.array(mailed_check), color='#FF9999', label=('Bank Transfer(automatic)'), edgecolor='white', width=barWidth)
# ax4 = plt.bar(x_labels, credit_card, bottom=np.array(electornic_check) + np.array(mailed_check) + np.array(bank_transfer), color='#FFA07A', label=('Credit Card (automatic)'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Payment Method')
#
# for r1, r2, r3, r4 in zip(ax1, ax2, ax3, ax4):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     h4 = r4.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r4.get_x() + r4.get_width() / 2., h1 + h2 + h3 + h4 / 2., '{:.2%}'.format(h4), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Employment type and Churn Rate
# no_churn = ((df_clean[df_clean['Churn']=='No']['Employment'].value_counts()) /(df_clean[df_clean['Churn']=='No']['Employment'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn']=='Yes']['Employment'].value_counts()) /(df_clean[df_clean['Churn']=='Yes']['Employment'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# student = [no_churn['Student'], yes_churn['Student']]
# full_time = [no_churn['Full Time'], yes_churn['Full Time']]
# part_time = [no_churn['Part Time'], yes_churn['Part Time']]
# retired = [no_churn['Retired'], yes_churn['Retired']]
# unemployed = [no_churn['Unemployed'], yes_churn['Unemployed']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, student, color='#00BFFF', label=('Student'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, full_time, bottom=student, color='lightgreen', label=('Full Time'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, part_time, bottom=np.array(full_time) + np.array(student), color='#FF9999', label=('Part Time'), edgecolor='white', width=barWidth)
# ax4 = plt.bar(x_labels, retired, bottom=np.array(student) + np.array(full_time) + np.array(part_time), color='#FFA07A', label=('Retired'), edgecolor='white', width=barWidth)
# ax5 = plt.bar(x_labels, unemployed, bottom=np.array(student) + np.array(full_time) + np.array(part_time) + np.array(retired), color='#F0E68C', label=('Unemployed'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Employment')
#
# for r1, r2, r3, r4, r5 in zip(ax1, ax2, ax3, ax4, ax5):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     h4 = r4.get_height()
#     h5 = r5.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r4.get_x() + r4.get_width() / 2., h1 + h2 + h3 + h4 / 2., '{:.2%}'.format(h4), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r5.get_x() + r5.get_width() / 2., h1 + h2 + h3 + h4 + h5 / 2., '{:.2%}'.format(h5), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Monthly Charge and Churn Rate
# no_churn = ((df_clean_6[df_clean['Churn']=='No']['Monthly_Charge'].value_counts()) /(df_clean_6[df_clean_6['Churn']=='No']['Monthly_Charge'].value_counts().sum()))
# yes_churn = ((df_clean_6[df_clean['Churn']=='Yes']['Monthly_Charge'].value_counts()) /(df_clean_6[df_clean_6['Churn']=='Yes']['Monthly_Charge'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean_6['Churn'].value_counts().keys().tolist()
# level1 = [no_churn['Charge Level 1'], yes_churn['Charge Level 1']]
# level2 = [no_churn['Charge Level 2'], yes_churn['Charge Level 2']]
# level3 = [no_churn['Charge Level 3'], yes_churn['Charge Level 3']]
# level4 = [no_churn['Charge Level 4'], yes_churn['Charge Level 4']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, level1, color='#00BFFF', label=('Below $100'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, level2, bottom=level1, color='lightgreen', label=('Between $101 - $150'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, level3, bottom=np.array(level2) + np.array(level1), color='#FF9999', label=('Between $151 - $200'), edgecolor='white', width=barWidth)
# ax4 = plt.bar(x_labels, level4, bottom=np.array(level1) + np.array(level2) + np.array(level3), color='#FFA07A', label=('Above $201'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Monthly Charge')
#
# for r1, r2, r3, r4 in zip(ax1, ax2, ax3, ax4):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     h4 = r4.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r4.get_x() + r4.get_width() / 2., h1 + h2 + h3 + h4 / 2., '{:.2%}'.format(h4), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# # #Contract Type and Churn Rate
# no_churn = ((df_clean[df_clean['Churn']=='No']['Contract'].value_counts()) /(df_clean[df_clean['Churn']=='No']['Contract'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn']=='Yes']['Contract'].value_counts()) /(df_clean[df_clean['Churn']=='Yes']['Contract'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# monthly = [no_churn['Month-to-month'], yes_churn['Month-to-month']]
# one_year = [no_churn['One year'], yes_churn['One year']]
# two_year = [no_churn['Two Year'], yes_churn['Two Year']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, monthly, color='#00BFFF', label=('Monthly'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, one_year, bottom=monthly, color='lightgreen', label=('One Year'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, two_year, bottom=np.array(one_year) + np.array(monthly), color='#FF9999', label=('Two Year'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Contract Type')
#
# for r1, r2, r3 in zip(ax1, ax2, ax3):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
# plt.show()
# #**********************************************************************************************************************
# #Internet Service and Churn Rate
# no_churn = ((df_clean[df_clean['Churn']=='No']['InternetService'].value_counts()) /(df_clean[df_clean['Churn']=='No']['InternetService'].value_counts().sum()))
# yes_churn = ((df_clean[df_clean['Churn']=='Yes']['InternetService'].value_counts()) /(df_clean[df_clean['Churn']=='Yes']['InternetService'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# dsl = [no_churn['DSL'], yes_churn['DSL']]
# fiber_optic = [no_churn['Fiber Optic'], yes_churn['Fiber Optic']]
# none = [no_churn['None'], yes_churn['None']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, dsl, color='#00BFFF', label=('DSL'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, fiber_optic, bottom=dsl, color='lightgreen', label=('Fiber Optic'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, none, bottom=np.array(fiber_optic) + np.array(dsl), color='#FF9999', label=('None'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Internet Service')
#
# for r1, r2, r3 in zip(ax1, ax2, ax3):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
# plt.show()

# #Bandwidth against churn rate
#
# #Bandwidth
# no_churn = ((df_clean_10[df_clean_10['Churn']=='No']['Bandwidth_GB_Year'].value_counts()) /(df_clean_10[df_clean_10['Churn']=='No']['Bandwidth_GB_Year'].value_counts().sum()))
# yes_churn = ((df_clean_10[df_clean_10['Churn']=='Yes']['Bandwidth_GB_Year'].value_counts()) /(df_clean_10[df_clean_10['Churn']=='Yes']['Bandwidth_GB_Year'].value_counts().sum()))
#
# # Getting values from the group and categories
# x_labels = df_clean['Churn'].value_counts().keys().tolist()
# b_500 = [no_churn['Bandwidth Below 500'], yes_churn['Bandwidth Below 500']]
# b_500_1000 = [no_churn['Bandwidth 500 - 1000'], yes_churn['Bandwidth 500 - 1000']]
# b_1000_2000 = [no_churn['Bandwidth 1000 - 2000'], yes_churn['Bandwidth 1000 - 2000']]
# b_gt_2000 = [no_churn['Bandwidth Above 2000'], yes_churn['Bandwidth Above 2000']]
#
# # Plotting bars
# barWidth = 0.8
# plt.figure(figsize=(7,7))
# ax1 = plt.bar(x_labels, b_500, color='#00BFFF', label=('Below 500 GB'), edgecolor='white', width=barWidth)
# ax2 = plt.bar(x_labels, b_500_1000, bottom=b_500, color='lightgreen', label=('Bandwidth 500 - 1000'), edgecolor='white', width=barWidth)
# ax3 = plt.bar(x_labels, b_1000_2000, bottom=np.array(b_500) + np.array(b_500_1000), color='#FF9999',  label=('Bandwidth 1000 - 2000'), edgecolor='white', width=barWidth)
# ax4 = plt.bar(x_labels, b_gt_2000, bottom=np.array(b_500) + np.array(b_500_1000) + np.array(b_1000_2000), color='#FFA07A', label=('Bandwidth Above 2000'), edgecolor='white', width=barWidth)
#
# plt.legend(loc='lower left', bbox_to_anchor=(1,0))
# plt.title('Churn Distribution by Tenure Group')
#
# for r1, r2, r3, r4 in zip(ax1, ax2, ax3, ax4):
#     h1 = r1.get_height()
#     h2 = r2.get_height()
#     h3 = r3.get_height()
#     h4 = r4.get_height()
#     plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., '{:.2%}'.format(h1), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., '{:.2%}'.format(h2), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., '{:.2%}'.format(h3), ha='center', va='center', color='black', fontweight='bold')
#     plt.text(r4.get_x() + r4.get_width() / 2., h1 + h2 + h3 + h4 / 2., '{:.2%}'.format(h4), ha='center', va='center', color='black', fontweight='bold')
#
# plt.show()




#**********************************************************************************************************************
# #Created a list for each attribute
# all_att = list(df_clean.columns)
# num_att = list(df_clean._get_numeric_data().columns)
# cat_att = list(df_clean.select_dtypes(include=['object']).columns)
# print('Numerical Attributes:' + '\n {}'.format(num_att))
# print('Categorical Attributes:' + '\n {}'.format(cat_att))
# # #Removing Churn and CustomerID
# id_col = ['customer_id']
# target = ['Churn']
# cat_att = [x for x in cat_att if x not in id_col + target]
# print('Numerical Attributes:' + '\n {}'.format(num_att))
# print('Categorical Attributes:' + '\n {}'.format(cat_att))
#
# #Lets see the unique values for each Categorical Att columns
# for col in cat_att:
#    print(col + ': ')
#    print('Unique values: {}'.format(df_clean[col].nunique()))
#    print(df_clean[col].value_counts())
#    print('\n')

#Finding missing values in my dataset
df_clean.isnull().any(axis=1)
null_values = df_clean.isna().any()
print(null_values)
#How many rows of data are we missing?
data_null_sum = df.isnull().sum()
print(data_null_sum)
#Filling the missing data with the median of each variable
#We saw that the columns Children, Age, Income, Tenure and Bandwidth_GB_Year have missing values
na_cols = df_clean.isna().any()
na_cols = na_cols[na_cols == True].reset_index()
na_cols = na_cols["index"].tolist()
for col in df_clean.columns[1:]:
     if col in na_cols:
        if df_clean[col].dtype != 'object':
             df_clean[col] = df_clean[col].fillna(df_clean[col].median()).round(0)

#Phone and Techie Columns are categorical with missing values as well
print(df_clean['Phone'].unique())
print(df_clean['Techie'].unique())
#Since We have "YES" "NO" and "NAN" we will need to replace the nan values for something
df_stats_phone = df_clean['Phone'].describe()
print(df_stats_phone)

df_stats_phone = df_clean['Techie'].describe()
print(df_stats_phone)

#Since these are categorical columns, I am going to replace the "NAN" values for whatever shows more
#Phone --> "YES"
#Techie --> "NO"

df_clean = df_clean.fillna(df.mode().iloc[0])

# #Making sure all values were replaced by the median (num) and mode (cat), so we check against missing data again
print(df_clean)
missing_data_clean = df_clean.isna().any()
print(missing_data_clean)
# #Extracting the clean dataset
df_clean.to_csv('churn_clean.csv')
churn_user = pd.read_csv('churn_clean.csv')
result = churn_user.isna().any()
print(result)
#
#**********************************************************************************************************************
# #Creating a label encoder object
# le = LabelEncoder()
# df_clean['Churn'] = le.fit_transform(df_clean['Churn'])
# # Label Encoding will be used for columns with 2 or less unique values at this time
# le_count = 0
# for col in df_clean.columns[1:]:
#     if df_clean[col].dtype == 'object':
#         if len(list(df_clean[col].unique())) <= 2:
#             le.fit(df_clean[col])
#             df_clean[col] = le.transform(df_clean[col])
#             le_count += 1
# print('{} columns were label encoded.'.format(le_count))
##**********************************************************************************************************************
# # Creating histograms of Numerical Data
# dataset = df_clean[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'CS Responses', 'CS Fixes', 'CS Replacements', 'CS Reliability', 'CS Options', 'CS Respectfulness', 'CS Courteous', 'CS Listening']]
# #Histogram
# fig = plt.figure(figsize=(15, 12))
# plt.suptitle('Histograms of Numerical Columns\n', horizontalalignment="center", fontstyle="normal", fontsize=24, fontfamily="sans-serif")
# for i in range(dataset.shape[1]):
#     plt.subplot(6, 3, i + 1)
#     f = plt.gca()
#     f.set_title(dataset.columns.values[i])
#     vals = np.size(dataset.iloc[:, i].unique())
#     if vals >= 100:
#         vals = 100
#     plt.hist(dataset.iloc[:, i], bins=vals, color='#ec838a')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
#
# #Independent Variables and Churn Rate Positive and Negative Correlations
# correlations = dataset.corrwith(df_clean.Churn)
# correlations = correlations[correlations!=1]
# positive_correlations = correlations[correlations >0].sort_values(ascending = False)
# negative_correlations = correlations[correlations<0].sort_values(ascending = False)
# print('Most Positive Correlations: \n', positive_correlations)
# print('\nMost Negative Correlations: \n', negative_correlations)
#
#
# # #Creating Correlations
# correlations = dataset.corrwith(df_clean.Churn)
# correlations = correlations[correlations!=1]
# correlations.plot.bar(figsize = (18, 10), fontsize = 15, color = '#ec838a', rot = 45, grid = True)
# plt.title('Correlation with Churn Rate \n', horizontalalignment="center", fontstyle = "normal", fontsize = "22", fontfamily = "sans-serif")
#
# df_clean_dummies = pd.get_dummies(df_clean[num_att + cat_att + target])
# df_clean_dummies.head()
#
# #Identify the standard deviation of every numeric column in the dataset
# data_std = df_clean.std()
# print(data_std)
#
# # Create a boxplot of tenure, monthly charge & usage variables
# df_clean.boxplot(['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year'])
# plt.savefig('churn_boxplots.png')
# plt.show()

# #Boxplot Monthly Charge
# df_stats.boxplot(['MonthlyCharge'])
# plt.show()
# #Slice off all but last eleven service related variables
#data = churn_user.loc[:, 'Tenure':'CS Listening']
#data = churn_user.loc[:, 'Tenure':'Bandwidth_GB_Year']
# print(data.head())

######PCA ANALYSIS#########
##Normalize the data
# churn_normalized = (data - data.mean()) / data.std()
# pca = PCA(n_components = data.shape[1])
# churn_numeric = data[['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year','CS Responses','CS Fixes', 'CS Replacements', 'CS Reliability', 'CS Options', 'CS Respectfulness', 'CS Courteous', 'CS Listening']]
# #churn_numeric = data[['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']]
# pcs_names = []
# for i, col in enumerate(churn_numeric.columns):
#     pcs_names.append('PC' + str(i + 1))
# print(pcs_names)
#
# pca.fit(churn_normalized)
# churn_pca = pd.DataFrame(pca.transform(churn_normalized), columns = pcs_names)
#
# plt.plot(pca.explained_variance_ratio_)
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.show()
#
# #Extract the eigenvalues
# cov_matrix = np.dot(churn_normalized.T, churn_normalized) / data.shape[0]
# eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]
#
# print(eigenvalues)
#
# #Plot the eigenvalues
# plt.plot(eigenvalues)
# plt.xlabel('Number of Components')
# plt.ylabel('Eigenvalue')
# plt.show()

# #Select the fewest components
# for pc, var in zip(pcs_names, np.cumsum(pca.explained_variance_ratio_)):
#     print(pc, var)
# #Creating Rotation
# rotation = pd.DataFrame(pca.components_.T, columns = pcs_names, index = churn_numeric.columns)
# print(rotation)
#
# churn_reduced = churn_pca.iloc[ : , 0:4]
# print(churn_reduced)