import pandas as pd
from numpy import divide, log, log1p
from sklearn.externals import joblib
import datetime as dt
import os
import time
import jaydebeapi
import lag_query
import emailnotice
import lag_query
from model_engine_bigstack import Wide_x_, Case3_x_,getlastmonth,getlastbutone,generate_Bigstack,bigstack_predictq30,find_or_generate
from preprocess import preprocess,replace_nans_infs,comment_log

today_string          = dt.date.today().strftime("%Y%m%d")
yesterday_string      = (dt.date.today() - dt.timedelta(1)).strftime("%Y%m%d")
yesterday_date_string = (dt.date.today() - dt.timedelta(1)).strftime("%Y-%m-%d")
exec_start_time       = time.time()

df_norm, df_lag = lag_query.lag()  
df = df_lag

month = find_or_generate(max_look_back=3)
df.index = df['goodsid'].values
df = preprocess(df)



###Case 1
modelinput = df.loc[df.case_label==1, Wide_x_[:-1]]
if len(modelinput)>0:
    modelinput["q30"] = bigstack_predictq30(modelinput,month)
    df.loc[df.case_label==1,"q30"] = modelinput["q30"]

###Case 3
modelinput = df.loc[df.case_label==3, Case3_x_[:-1]]
if len(modelinput)>0:
    modelinput["q30"] = bigstack_predictq30(modelinput,month, responseColumn="log(q30)",modelname="3.y-BigStack")
    df.loc[df.case_label==3,"q30"] =  modelinput["q30"]

###Case 4    
df.loc[df.case_label==4,"q30"] = 0

df["etype"]                 = "JD"
df["ref_date"]              = int(yesterday_string)
df["prediction_confidence"] = df["case_label"]
df["confidence_interval"]   = None
df["update_date"]           = yesterday_date_string

outputCSV = df.loc[:,["egoodsid", "etype","ref_date","q30","prediction_confidence", "confidence_interval","update_date"]]
if not os.path.isdir("output/bigstack/lag/"): os.makedirs("output/bigstack/lag/")
outputCSV.to_csv("output/bigstack/lag/%s.csv" % yesterday_string, index=True, header=False)
print "\nData has been saved in output/bigstack/%s.csv" %yesterday_string

import re
tmp=[]
for data in os.listdir("data"):
    if re.match(r'[0-9]',data[:6]):
        tmp.append(data)
filename=sorted(tmp)[-1]
if filename.endswith("(custcode).csv"): filename=sorted(tmp)[-2]
try:
    with open("metadata/latestdatainfo.txt",'r') as fh:
        fname = fh.read().strip()
except:
    if os.path.exists("matadata"): 
        continue
    else: 
        os.mkdir("metadata"); 
    fname = None
    print "No latestdatainfo.txt there!!"
if filename!=fname:
    massage_text = "New model has been trained in the project!!\nplease see the diagnose picture and the models results"
    attachment1='models/4.y-BigStack/bigstacks/%s/4.y-BigStack/0-images/diagnostics.png' %(month)
    #/opt/jupyter/project/models/4.y-BigStack/bigstacks/201609/4.y-BigStack/0-images
    attachment2='models/4.y-BigStack/bigstacks/%s/4.y-BigStack/model.pkl' %(month) ###need to verify
    #models/4.y-BigStack/bigstacks/201609/4.y-BigStack 
    attachments=[attachment1,attachment2]
    emailnotice.mailtx(massage_text,attachments=[attachment for attachment in attachments])
    fname=filename
    with open("metadata/latestdatainfo.txt", 'w') as file1:
        file1.write("{}\n".format(filename))
else: 
    with open("metadata/latestdatainfo.txt", 'w') as file1:
        file1.write("{}\n".format(filename))

print "\nPrediction job complete!\nExecute time: %.1f seconds" % (time.time() - exec_start_time)