####################################################################################
#
# Kaggle Competition: https://www.kaggle.com/c/msk-redefining-cancer-treatment
# Sponsor : Memorial Sloan Kettering Cancer Center (MSKCC)
# Author: Amrut Shintre
#
####################################################################################

#####################
# Importing Libraries
#####################
import numpy as np
import pandas as pd
import matplotlib as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gc
import random

####################
# Importing datasets
####################

# Training Dataset
train_df = pd.read_csv('training_text', sep = '\|\|', engine = 'python', names = ['ID', 'Text'],
                       header = None)
train_df = train_df.iloc[1:,:]
train_df.index = range(len(train_df))
train_var = pd.read_csv('training_variants')

# Testing Dataset
test_df = pd.read_csv('test_text', sep = '\|\|', engine = 'python', names = ['ID', 'Text'],
                      header = None)
test_var = pd.read_csv('test_variants')

# --------------------------------------------TEXT ---------------------------------------------

##############
# TextCleaning
##############

def text_cleaning(text_df):
    corpus = []
    for i in range(len(text_df)):
        text = re.sub('[^a-zA-Z]', ' ', text_df['Text'][i]) # Removing punctuation marks,
        #numbers, etc and returning only letters
        text = text.lower() # Converting all the uppercase letters to lowercase
        text = text.split() # Splitting a sentence into a list of strings containing a single word.
        ps = PorterStemmer() # Stemming e.g. lovely -> love
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text) # Joining the cleaned words
        corpus.append(text) # Appending it to the new list.
    return (corpus)

# Training Text Data
corpus_train = text_cleaning(train_df)

# Testing Text Data
corpus_test = text_cleaning(test_df)

#############################################
# Term Frequency - Inverse Document Frequency
#############################################

tfidf = TfidfVectorizer()
tfidf_tr = tfidf.fit_transform(corpus_train).toarray()
tfidf_test = tfidf.transform(corpus_test).toarray()

##############################
# Singular Value Decomposition
##############################

svd = TruncatedSVD(n_components = 1000) # considering 98% variance in the Data
svd_tr = svd.fit_transform(tfidf_tr) # Fitting on cleaned training text data
svd_train = svd.transform(tfidf_test) # Transforming on cleaned testing text data
svd_tr = pd.DataFrame(svd_tr)
svd_test = pd.DataFrame(svd_train)
#explainedvar = svd.explained_variance_ratio_
#exp_var = explainedvar.cumsum()

# -------------------------------------------- VARIANTS ---------------------------------------------

####################
# Dependent Variable
####################

y = train_var['Class'].values
y = y-1

#################
# Merging Dataset
#################

# Merging the dataset for data preparation and feature engineering

df = pd.concat([train_var, test_var], axis = 0)
df = df.drop(['ID'], axis = 1)
df['ID'] = range(df.shape[0])
df.index = range(df.shape[0])
df_text = pd.concat([train_df, test_df], axis = 0)
df_text = df_text.drop('ID', axis = 1)
df_text['ID'] = range(df_text.shape[0])
df_text.index = range(df_text.shape[0])
df_all = pd.merge(df, df_text, how = 'left', on = 'ID')


################
# Missing Values
################

# Checking for missing values

column_list = train_var.columns.values.tolist()
missing_values = pd.DataFrame()
missing_values['Columns'] = column_list
for i in column_list:
    missing_values['No. of missing values'] = train_var[i].isnull().values.ravel().sum()

# There are no missing values.

#######################
# Categorical Variables
#######################

# Extracting the columns having categorical Variables.

column_list = df.columns
categorical_columns = []
for i in column_list:
    if df[i].dtype == 'O':
        categorical_columns.append(i)

# Encoding the columns with categorical variables

# Label Encoding

for i in categorical_columns:
    le = LabelEncoder()
    df[i + '_le'] = le.fit_transform(df[i])
    df[i + '_length'] = df[i].map(lambda x: len(str(x)))

# Feature Engineering

df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)
  
###################
# Splitting Dataset
###################  

train = df_all.iloc[:len(train_var), :]
test = df_all.iloc[len(train_var):,:]
test.index = range(len(test_var))
train = train.drop(['Gene', 'Variation', 'ID', 'Text', 'Class'], axis = 1)
test = test.drop(['Gene', 'Variation', 'Text', 'ID', 'Class'], axis = 1)

train_final = pd.concat([train, svd_tr], axis = 1)
test_final = pd.concat([test, svd_test], axis = 1)

#################
# Standardization
#################

sc = StandardScaler()
train_final = sc.fit_transform(train_final)
test_final = sc.transform(test_final)
train_final = pd.DataFrame(train_final)
test_final = pd.DataFrame(test_final) 

# -------------------------------------------- MODEL ---------------------------------------------

##################
# XGBoost Matrix 
##################

dtrain = xgb.DMatrix(train_final, y)
dtest = xgb.DMatrix(test_final)

##################
# Cross-Validation  
##################

def docv(param, iterations, nfold):
    model_CV = xgb.cv(
            params = param,
            num_boost_round = iterations,
            nfold = nfold,
            dtrain = dtrain,
            seed = random.randint(1, 10000),
            early_stopping_rounds = 100,
            maximize = False,
            verbose_eval = 50)
    gc.collect()
    best = min(model_CV['test-mlogloss-mean'])
    best_iter = model_CV.shape[0]
    print (best)
    return (best_iter)

#########
# Testing  
#########

def doTest(param, iteration):
    X_tr, X_val, y_tr, y_val = train_test_split(train_final, y, test_size = 0.2, random_state = random.randint(1,1000))
    watchlist = [(xgb.DMatrix(X_tr, y_tr), 'train'), (xgb.DMatrix(X_val, y_val), 'validation')]
    model = xgb.train(
            params = param,
            dtrain = xgb.DMatrix(X_tr, y_tr),
            num_boost_round = iteration,
            evals = watchlist,
            verbose_eval = 50,
            early_stopping_rounds = 100)
    score = metrics.log_loss(y_val, model.predict(xgb.DMatrix(X_val)), labels = range(9))
    predicted_class = model.predict(dtest)
    print (score)
    return (predicted_class)

#########
# Bagging
#########

def Bagging(N, params, best_iter):
    for i in range(N):
        param = params
        p = doTest(param, best_iter)
        if i == 0:
            preds = p.copy()
        else:
            preds = preds + p
    predictions = preds/N
    predictions = pd.DataFrame(predictions)
    return (predictions)

###################
# Running the Model
###################

params = {
        'eta': 0.02,
        'max_depth': 6,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'silent': False,
        'seed': random.randint(1,100),
        'num_class': 9
        }

cross_vali = docv(params, 10000, 5)

predicted_class = Bagging(5, params, cross_vali)


# -------------------------------------------- SUBMISSION ---------------------------------------------

sub_file = pd.DataFrame()
sub_file['ID'] = test_var['ID'].values
Sub_File = pd.concat([sub_file, predicted_class], axis = 1)
Sub_File.columns = ['ID', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 
                    'Class8', 'Class9']
Sub_File.to_csv("submission33.csv", index = False)

# -------------------------------------------- Project Layout ---------------------------------------------

# 1) Text Cleaning
# 2) TFIDF Vectorizer and Singular Value Decomposition
# 3) Feature Engineering
# 4) Building a Model and trying out different models
# 5) Parameter Tuning
# 6) Bagged Boosting        