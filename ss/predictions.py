import model_engine
import pandas as pd
from numpy import divide, log, log1p, arange
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from SSv2_feature_spaces import Thin_x_, Poly_x_, Wide_x_, Case3_x_
import model_engine
import datetime as dt
import os
import time



def trainingmodel(filename="201609_JD_sales_plus_compass_joined(SKUcustcode).csv"):
    df_train=pd.read_csv("data/" + filename)
    df_train['brand_share']       = divide(1.0*df_train['agg5'],df_train['agg2'])
    df_train['log(c34)']          = log(df_train['c30'])
    df_train['log1p(RIR)']        = log1p(divide(1.0*df_train['item_rank'],df_train['agg1'])) #log(df['relative_item_rank'])
    df_train['log(q30)']          = log(df_train['sum'])
    df_train['is_promo']          = pd.to_numeric(df_train['is_promo'])

    modelinput = df_train.loc[df_train["item_rank"].notnull(), Wide_x_]
    modelinput, ensembleinput = train_test_split(modelinput,train_size = 0.68)
    treecountlowlvl, treecount = 105, 225

    model_engine.generate_MARS(modelinput, predictorColumns=Thin_x_[:-1], trainingSplitRatio=0.89, verbose=False)
    model_engine.generate_PolyR(modelinput, predictorColumns=Poly_x_[:-1], poly_degree=2, trainingSplitRatio=0.89, verbose=False)
    model_engine.generate_PolyR(modelinput, predictorColumns=Poly_x_[:-1], poly_degree=3, trainingSplitRatio=0.89, verbose=False)
    model_engine.generate_GBTR(modelinput,lossfctn="ls",n_trees=117, trainingSplitRatio=0.89, verbose=False)
    model_engine.generate_RFR(modelinput, n_trees=treecountlowlvl, trainingSplitRatio=0.89, verbose=False)
    model_engine.generate_XGBR(modelinput,n_trees=treecount, trainingSplitRatio=0.89, verbose=False)
    yhat,ytest,Xtest = model_engine.generate_Ensemble(ensembleinput, modelChoices=None,returnTestSetResults=True,trainingSplitRatio=0.78, verbose=False)


# In[5]:

def generatedata(lag_day=7):
    exec_start_time=time.time()
    today_string          = dt.date.today().strftime("%Y%m%d")
    yesterday_string      = (dt.date.today() - dt.timedelta(1)).strftime("%Y%m%d")
    n_date_string = (dt.date.today() - dt.timedelta(lag_day)).strftime("%Y%m%d")
    
    from jaydebeapi import _DEFAULT_CONVERTERS, _java_to_py
    import jaydebeapi
    _DEFAULT_CONVERTERS.update({'BIGINT':_java_to_py('longValue')})

    conn = jaydebeapi.connect('com.ingres.jdbc.IngresDriver', 
                              ['jdbc:ingres://192.168.6.199:vw7/compass_v1', 'mintel', 'mintel'], 
                              "lib/iijdbc.jar")
    curs = conn.cursor()
    curs.execute('''
                 select 
                 TOP 10000
                 a.egoodsid,
                 a.c34 as c34a,
                 b.c34 as c34b,
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
                 from mv_simtxn a, mv_simtxn b
                 where a.etype='JD' AND a.ref_date='%s'
                 AND b.etype='JD' AND b.ref_date='%s'
                 AND a.egoodsid = b.egoodsid
                 ''' % (yesterday_string,n_date_string))
    rows = curs.fetchall()
    print "query read time: %.1f seconds" % (time.time() - exec_start_time)
    columnNames = [curs.description[i][0] for i in range(len(curs.description))]
    df_test = pd.DataFrame(data=rows, columns=columnNames)
    df_test.index = df_test['egoodsid'].values

    df_test['relative_item_rank']= divide(df_test['item_rank'],df_test['agg1'])
    df_test['brand_share']       = divide(df_test['agg5'],df_test['agg2'])
    df_test['log1p(RIR)']        = log(divide(df_test['item_rank'],df_test['agg1'])) #log(df_test['relative_item_rank'])
    df_test['log(c34)']          = log(df_test['c34b'])
    df_test['log(allcomments)']  = log(df_test['allcomments'])
    df_test['is_promo']          = pd.to_numeric(df_test['is_promo'])
    df_test['c1']                = pd.to_numeric(df_test['c1'])
    #df_test['AC_updated']        = df_test['agg7']>0

    df_test["case_label"] = 0
    df_test.loc[df_test.item_rank.notnull(),"case_label"]=1
    #df_test.loc[df_test.item_rank.notnull() & df_test.c34.isnull(),"case_label"]=2
    df_test.loc[df_test.item_rank.isnull() & df_test.c34.notnull(),"case_label"]=3
    df_test.loc[df_test.item_rank.isnull() & df_test.c34.isnull(),"case_label"]=4
    return df_test


def testingmodel(df_test):
    exec_start_time = time.time()
    yesterday_string = (dt.date.today() - dt.timedelta(1)).strftime("%Y%m%d")
    modelinput_test = df_test.loc[df_test.case_label==1, Wide_x_[:-1]]
    if len(modelinput_test)>0:
        modelinput_test["q30"] = model_engine.predictq30(modelinput_test)
        df_test.loc[df_test.case_label==1,"q30"] = modelinput_test["q30"]
    #case 2 has been removed
    # #case 3 #need to be done later
    # #NOTE: 2016-07-29: Not enough data to train this on case 3/4, so combining these cases for now.
    # modelinput = df_test.loc[df_test.case_label==3, Case3_x_[:-1]]
    # if len(modelinput)>0:
    #     modelinput["q30"] = model_engine.predictq30(modelinput, responseColumn="log(q30)",modelname="3.x-Ensemble")
    #     df_test.loc[df_test.case_label==3,"q30"] =  modelinput["q30"]
    # df.loc[df_test.case_label==4,"q30"] = 0

    df_test.loc[df_test.case_label==4,"q30"] = 0
    df_test["etype"]                 = "JD"
    df_test["ref_date"]              = int(yesterday_string)
    df_test["prediction_confidence"] = df_test["case_label"]
    df_test["confidence_interval"]   = None
    df_test["update_date"]           = yesterday_string
    outputCSV = df_test.loc[:,["goodsid", "etype","ref_date","q30","prediction_confidence", "confidence_interval","update_date"]]
    outputCSV.to_csv("output/archive/%s.csv" % yesterday_string, index=True, header=False)
    outputCSV.to_csv("output/daily.csv", index=True, header=False)
    print "\nPrediction job complete!\nExecute time of Testing Model : %.1f seconds" % (time.time() - exec_start_time)




