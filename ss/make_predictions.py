
# coding: utf-8

# In[1]:

import pandas as pd
from numpy import divide, log, log1p
from sklearn.externals import joblib
import datetime as dt
import os
import time
import jaydebeapi
today_string          = dt.date.today().strftime("%Y%m%d")
yesterday_string      = (dt.date.today() - dt.timedelta(1)).strftime("%Y%m%d")
yesterday_date_string = (dt.date.today() - dt.timedelta(1)).strftime("%Y-%m-%d")
exec_start_time       = time.time()

from SSv2_feature_spaces import Wide_x_, Case3_x_
import model_engine


#%pylab inline


# ####Design:
# 
# 1. Add the following features {relative_item_rank: item_rank/agg1, brandshare: agg5/agg2}
# 2. Filter out all non-zero c30s, run them through:
#   - p(c30==0|D1):D1=allcomments, is_in_stock, item_rank, agg7, agg8, 
#   - p(q30==0|D2):D2=p0c30, allcomments, is_in_stock, relative_item_rank, is_promo
# 3. Generate a case number for each row, based on rules in documentation
# 
# 

# In[3]:

from jaydebeapi import _DEFAULT_CONVERTERS, _java_to_py
_DEFAULT_CONVERTERS.update({'BIGINT':_java_to_py('longValue')})

conn = jaydebeapi.connect('com.ingres.jdbc.IngresDriver', 
                          ['jdbc:ingres://192.168.6.199:vw7/compass_v1', 'mintel', 'mintel'], 
                          "lib/iijdbc.jar")
curs = conn.cursor()
curs.execute('''
             SELECT 
             a.egoodsid,
             a.goodsid,
             a.currentprice, 
             a.allcomments,
             a.c34, 
             a.c1, 
             a.item_rank, 
             a.agg1, 
             a.agg2, 
             a.agg3, 
             a.agg5, 
             a.agg7, 
             a.is_promo
             FROM mv_simtxn a
             WHERE a.etype='JD' AND a.ref_date='%s'
             ''' % yesterday_string)
rows = curs.fetchall()


# In[4]:

#df = pd.read_csv('data/MV_Simtxn_20160701_5000sample.csv', delimiter=',')
print "query read time: %.1f seconds" % (time.time() - exec_start_time)
columnNames = [curs.description[i][0] for i in range(len(curs.description))]
df = pd.DataFrame(data=rows, columns=columnNames)
df.index = df['egoodsid'].values


# Next, engineer the following features:
# 
#   - relative_item_rank = item_rank / number of products in the category
#   - brand_share = count of products under the brand in the category / number of products in the category
#   - log(c34), log(relative_item_rank), log(allcomments)
#   
#   - (possible) AC_updated = {1: allcomments was updated in last 30 days, 0: otherwise}
#   

# In[5]:

df['relative_item_rank']= divide(df['item_rank'],df['agg1'])
df['brand_share']       = divide(df['agg5'],df['agg2'])
df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1'])) #log(df['relative_item_rank'])
df['log(c34)']          = log(df['c34'])
df['log(allcomments)']  = log(df['allcomments'])
df['is_promo']          = pd.to_numeric(df['is_promo'])
df['c1']                = pd.to_numeric(df['c1'])
#df['AC_updated']        = df['agg7']>0


# Then we can label the cases (in order of desirability):
# 
# 
# 1. item_rank >0, c30>0
# 2. item_rank >0, c30==0
# 3. item_rank==0, c30>0
# 4. item_rank==0, c30==0
# 
# 2016-08-01: Case 2 has been removed to allow for a larger training set for the models.

# In[6]:

df["case_label"] = 0
df.loc[df.item_rank.notnull(),"case_label"]=1
#df.loc[df.item_rank.notnull() & df.c34.isnull(),"case_label"]=2
df.loc[df.item_rank.isnull() & df.c34.notnull(),"case_label"]=3
df.loc[df.item_rank.isnull() & df.c34.isnull(),"case_label"]=4


# ###Case 1

# Features defined:
# 
# - features_thin=("log(34)","log1p(relative_item_rank)","log(q30)") 
# - features_poly=("log(34)","log1p(relative_item_rank)","log(q30)")
# - features_wide=('currentprice','allcomments', 'c1', 'agg1', 'agg2', 'agg3', 'is_promo', 'brand_share', 'log(c34)', 'log1p(RIR)', 'log(q30)']

# #####prediction of q30
# 
# This step picks up multiple models for different algorithms, and then another model is trained on more training data outside of the sample from what was used in the upstream modeling process, to produce a final prediction based on model weights.

# In[7]:

#2016-07-28 models trained on rows where item_rank is not null, NOT case 1.
modelinput = df.loc[df.case_label==1, Wide_x_[:-1]]
if len(modelinput)>0:
    modelinput["q30"] = model_engine.predictq30(modelinput)
    df.loc[df.case_label==1,"q30"] = modelinput["q30"]


# ###Case 3
# 
# NOTE: 2016-07-29: Not enough data to train this on case 3/4, so combining these cases for now.

# In[8]:

modelinput = df.loc[df.case_label==3, Case3_x_[:-1]]
if len(modelinput)>0:
    modelinput["q30"] = model_engine.predictq30(modelinput, responseColumn="log(q30)",modelname="3.x-Ensemble")
    df.loc[df.case_label==3,"q30"] =  modelinput["q30"]


# ###Case 4
# 
# Handle this later-- it is possible to use a combination of the aggX features and allcomments, but that also depends on their data quality.  Those predictions would only act as dummies. 

# In[8]:

df.loc[df.case_label==4,"q30"] = 0


# ###Write output to database
# 
# This will be handled by a vwload process outside of this script.  output results to a file in the $PROJECT/output/ folder, with filename given by the ref_date in the original query (this defaults to yesterday, as this is to be scheduled to run every day).

# In[25]:

df["etype"]                 = "JD"
df["ref_date"]              = int(yesterday_string)
df["prediction_confidence"] = df["case_label"]
df["confidence_interval"]   = None
df["update_date"]           = yesterday_date_string
outputCSV = df.loc[:,["goodsid", "etype","ref_date","q30","prediction_confidence", "confidence_interval","update_date"]]
outputCSV.to_csv("output/%s.csv" % yesterday_string, index=True, header=False)
print "\nPrediction job complete!\nExecute time: %.1f seconds" % (time.time() - exec_start_time)


# ##Archived Code

# In[26]:

#%pylab inline
#df
#reload(model_engine)
#%pylab inline
#from matplotlib import pyplot as plt
#plt.hist(df.loc[df.item_rank.isnull(),"q30"], range=(00,10), bins=10)
#df.loc[df.item_rank.isnull(),Case3_x_+["q30", "c34"]]
#hist(df.loc[df.item_rank.notnull(),"q30"], range(100,100000))#["q30","is_promo","c34", "item_rank"]]
#scatter(y=df.loc[df.case_label==3,"q30"], x=df.loc[df.case_label==3,"c34"])
#df.loc[df.item_rank.isnull(),Case3_x_[:-1]+["q30","c34"]]


# In[27]:

#model = joblib.load("models/3.01-MARS/20160801/model.pkl")
#model = joblib.load("models/3.05-RFR/20160801/model.pkl")
#modelinput['log(q30)'] = model.predict(model_engine.replace_nans_infs(modelinput))
#modelinput['q30'] = model.predict(model_engine.replace_nans_infs(modelinput))
#from numpy import exp
#df.loc[df.case_label==3, "q30"] = exp(modelinput["log(q30)"])


# In[28]:

#piethis = df.case_label.value_counts()
#pie(piethis, labels=piethis.index, autopct="%.1f")
#title=title("Case distribution")
#%pylab inline
#piethis = df.is_promo.value_counts(dropna=False)
#pie(piethis, labels=piethis.index, autopct="%.1f")
#title=title("is_promo distribution")


# ###Case 2(archived... change when models are trained on case 1 data, instead of when item_rank is null)
# 
# First, predict whether the item is q30=0, based totally on the relationship between 0 values of q30 and high values of  relative_item_rank (illustrated in the image below):
# 
# ![x: q30==0, y: relative_item_rank](https://localhost:9999/files/project/models/2.1-item_rank_logit/0-images/20160725violin.png)
# 
# 
# 
# ####Step 1: pick up the threshold pretrained in an earlier Logistic Regression model

# sorted(os.listdir("models/2.1-item_rank_logit/"))[-1]

# mostRecentModelDate["2.1-IRLogit"] = sorted(os.listdir("models/2.1-item_rank_logit/"))[-1]
# clflogit = joblib.load("models/2.1-item_rank_logit/%s/model.pkl" % mostRecentModelDate["2.1-IRLogit"])
# 
# #Check that it works
# #clflogit.predict(arange(0,1,.1)[:,newaxis])

# filteredInput = df.loc[df.case_label==2,"relative_item_rank"]
# df.loc[df.case_label==2,"q30"] = 1-pd.Series(clflogit.predict(filteredInput.reshape(-1,1)), index=filteredInput.index)

# Note: classifier 1.1-item_rank_logit returns True if it thinks q30==0.  In python, True/False is equivalent to 1/0, so we take 1 minus the prediction to give q30 a starting value of 0 or 1.  Then, if it's 1, we will proceed to predictions using step 2.
# 
# ####Step 2: interpolate based on prediction from case 4
# 
# 2 possible methodologies:
# - sklearn.neighbors.KNeighborsRegressor
# - scipy.stats.hmean

# In[29]:

#df.loc[df.case_label==3,Wide_x_[:-1]+["q30","c34","AC_updated"]]
#df.loc[df.q30>400000,"q30"] = 29299
#df.loc[df.case_label==3,["q30","c34","AC_updated", "allcomments", "is_promo"]]
#df.loc[2792916,Wide_x_[:-1]+["q30","AC_updated"]]
#df.loc[df.case_label==2,["q30", "relative_item_rank"]].loc[df.q30>0,:].sort_values(by="relative_item_rank")


# In[69]:

#figsize(4,4)
##subplot(131)
#a = df.AC_updated.value_counts()
#b = df.allcomments.notnull().value_counts()
#piethis = pd.Series([b[False],b[True]-a[True],a[True]], index=["False","True (but old)","True & updated in last 30 days"])
#pie(piethis, labels=piethis.index, autopct="%.1f", colors=("w","yellow","green"))
#title("allcomments is nonnull")
##subplot(132)
##pie(df.allcomments.notnull().value_counts(), labels=df.allcomments.notnull().value_counts().index,autopct="%.2f", colors=("g","w"))
##title("allcomments is nonnull")
##subplot(133)
##hist(df.c1.fillna(0).values, range=(0,20), bins=20)
##xlim(0);ylim(0)



# In[14]:

###Make model directory tree

#os.mkdir("models/4.01-MARS")#/000000")
#os.mkdir("models/4.02-poly2")#/000000")
#os.mkdir("models/4.03-poly3")#/000000")
#os.mkdir("models/4.04-GBTR")#/000000")
#os.mkdir("models/4.05-RFR")#/000000")
#os.mkdir("models/4.01-MARS/000000")
#os.mkdir("models/4.02-poly2/000000")
#os.mkdir("models/4.03-poly3/000000")
#os.mkdir("models/4.04-GBTR/000000")
#os.mkdir("models/4.05-RFR/000000")
#os.mkdir("models/4.x-Ensemble")#/000000")
#os.mkdir("models/4.x-Ensemble/000000")


# In[17]:

#case2df = df[df['item_rank'].notnull()].copy()
#print len(case2df['item_rank'])
#case2df['over_threshold'] = clflogit.predict(case2df[['relative_item_rank']].values)
#case2df.loc[:,"relative_item_rank","c30"]


# In[56]:

#case2df.loc[:,("relative_item_rank", "c30", "over_threshold")]
#import seaborn as sns
#figsize(12.5,4)
#subplot(131)
#sns.violinplot(data=case2df, x="over_threshold", y="c30")
#ylim(0,100)
#subplot(132)
#plt.hist(case2df.loc[case2df['over_threshold']==True,"c30"], range=(0,10), bins=10)
#subplot(133)
#plt.hist(case2df.loc[case2df['over_threshold']==False,"c30"], range=(0,1000), bins=100)


# In[15]:

#piethis = df[df["item_rank"].isnull()]['c30'].isnull().value_counts()
#pie((piethis[True],piethis[False]), labels=("Null","Nonnull"), colors=("w", "g"))
#title("c30 values when item_rank is null")


# In[16]:




# In[ ]:



