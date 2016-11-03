import pandas as pd
from numpy import divide, log, log1p
from sklearn.externals import joblib
import datetime as dt
import os
import time
import jaydebeapi
from datetime import datetime

from model_engine_bigstack import Wide_x_, Case3_x_, find_or_generate
import model_engine_bigstack
from preprocess import preprocess

today_string          = dt.date.today().strftime("%Y%m%d")
yesterday_string      = (dt.date.today() - dt.timedelta(1)).strftime("%Y%m%d")
yesterday_date_string = (dt.date.today() - dt.timedelta(1)).strftime("%Y-%m-%d")
exec_start_time       = time.time()


###DELETE?#####
### lag_query.lag()
###############

### ADD: ###
### df = ???
############


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
             a.c30, 
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


#df = pd.read_csv('data/MV_Simtxn_20160701_5000sample.csv', delimiter=',')
print "query read time: %.1f seconds" % (time.time() - exec_start_time)
columnNames = [curs.description[i][0] for i in range(len(curs.description))]


### Checking if a bigstack model is already generated

    
month = find_or_generate(max_look_back=3)


df = pd.DataFrame(data=rows, columns=columnNames)
    
df.index = df['goodsid'].values

## preprocess() assumes that df has no q30/sum column.
df = preprocess(df)

# ###Case 1
# 
modelinput = df.loc[df.case_label==1, Wide_x_[:-1]]
if len(modelinput)>0:
    modelinput["q30"] = model_engine_bigstack.bigstack_predictq30(modelinput,month)
    df.loc[df.case_label==1,"q30"] = modelinput["q30"]


### Case 3
modelinput = df.loc[df.case_label==3, Case3_x_[:-1]]
if len(modelinput)>0:
    modelinput["q30"] = model_engine_bigstack.bigstack_predictq30(modelinput,month, responseColumn="log(q30)",modelname="3.y-BigStack")
    df.loc[df.case_label==3,"q30"] =  modelinput["q30"]

### Case 4
### We should consider randomly generating this value by using c30 (lognormal) and r (beta dist)
df.loc[df.case_label==4,"q30"] = 0


### parameterize etype
df["etype"]                 = "JD"
df["ref_date"]              = int(yesterday_string)
df["prediction_confidence"] = df["case_label"]
df["confidence_interval"]   = None
df["update_date"]           = yesterday_date_string


outputCSV = df.loc[:,["egoodsid", "etype","ref_date","q30","prediction_confidence", "confidence_interval","update_date"]]
if not os.path.isdir("output/bigstack/"): os.mkdir("output/bigstack/")
outputCSV.to_csv("output/bigstack/output.csv", index=True, header=False)
print "\nPrediction job complete!\nExecute time: %.1f seconds" % (time.time() - exec_start_time)
