
# coding: utf-8

# In[1]:

import pandas as pd
from numpy import isnan, isinf, random, add, exp, multiply
from matplotlib import pyplot as plt
import datetime as dt
import os
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
#get_ipython().magic(u'pylab inline')
import seaborn as sns
from SSv2_feature_spaces import Thin_x_,Poly_x_,Wide_x_, modelToSpaces

today_string = dt.date.today().strftime("%Y%m%d")


# In[2]:

def replace_nans_infs(x):
    x[isnan(x)] = 0
    x[isinf(x)] = 0
    return x


# In[3]:

def splitXy(training_data,responseColumn,predictorColumns="default"):
    try:
        y = training_data.loc[:,responseColumn]
        if predictorColumns == "default": predictorColumns = training_data.columns.drop(responseColumn)
        X = training_data.loc[:,predictorColumns]
        return X,y
        
    except AttributeError:
        raise AttributeError("Input training data is not a pandas DataFrame")
    except KeyError:
        raise KeyError("Column names for response or predictors not valid")
    except NameError:
        raise TypeError("generate_{MODEL} function takes at least 1 argument: training_data")


# In[4]:

def generate_MARS(training_data,
                  modelname="4.01-MARS", 
                  responseColumn="log(q30)",
                  predictorColumns="default", #default => all non-response columns in training data
                  max_degree=2,
                  minspan_alpha=0.5,
                  smooth=False,
                  trainingSplitRatio=0.8,
                  trainingSplitRandom=random.RandomState(),
                  persist=True,
                  returnTestSetResults=False,    #True ==> will return predictions, the actual, and the predictors
                  verbose=True  #setting this to True will still save a model, but not return any images/diagnostics
                  ):

    from pyearth import Earth
    
    model = Earth(max_degree=max_degree,
                  minspan_alpha=minspan_alpha, 
                  smooth=smooth)

    replace_nans_infs(training_data)
    X,y = splitXy(training_data, responseColumn, predictorColumns)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=trainingSplitRatio, random_state=trainingSplitRandom)
    model.fit(Xtrain,ytrain)

        
##Model evaluation:
        
    yhat  = model.predict(Xtest)
    R2    = r2_score(yhat,ytest)  #imported above via sklearn.metrics
  
    
##If model is successfully generated, output results##

    modelDir="models/%s/%s/" % (modelname,today_string)
    imageDir=modelDir + "0-images/"


    if persist==True:

        try: 
            os.listdir(modelDir)
        except:
            os.makedirs(modelDir)
            os.mkdir(imageDir)


        joblib.dump(model,modelDir + "model.pkl")
    

        plt.rcParams['figure.figsize']=[9,4]
        plt.subplot(121)
        plt.scatter(ytest, yhat, alpha=.1)
        plt.plot([0,10],[0,10])
        #plt.xlim(0,10); plt.ylim(0,10)
        plt.xlabel("actual");plt.ylabel("predicted")
        plt.title("Model = MARS(%i)\t\t $R^2$=%.2f" % (max_degree,R2))
        plt.subplot(122)
        sns.set(style="whitegrid")
        sns.residplot(ytest,yhat)#, lowess=True)
    
        plt.savefig(imageDir+"diagnostics.png")
        plt.close()

    if returnTestSetResults==True: 
        return yhat, ytest, Xtest
    
    if verbose==True:
        print "MARS(%i) model successfully generated! \t\t\t\t\t(Train: %i, Test: %i)\n\tModel file saved in:\t\t%s\n\tDiagnostics plots saved in:\t%s\n" % (max_degree,len(ytrain),len(ytest),modelDir,imageDir)
        #print "Model Trace:\n"
        #print model.trace()
        #print "Model Summary:\n"
        #print model.summary()
        #print "Evaluation:\n\n"
    


# In[5]:

def generate_PolyR(training_data,
                  modelname="4.0x-polyx", # --> it will be 4.0x-Polyx based on the degree x of polynomials 
                  responseColumn="log(q30)",
                  predictorColumns="default", #default => all non-response columns in training data
                  poly_degree=2,
                  trainingSplitRatio=0.8,
                  trainingSplitRandom=random.RandomState(),
                  persist=True,
                  returnTestSetResults=False,    #True ==> will return predictions, the actual, and the predictors
                  verbose=True  #setting this to True will still save a model, but not return any images/diagnostics
                  ):


    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    if modelname=="4.0x-polyx": modelname="4.0%s-poly%s" % (str(poly_degree),str(poly_degree))
    
    replace_nans_infs(training_data)
    X,y = splitXy(training_data, responseColumn, predictorColumns)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=trainingSplitRatio, random_state=trainingSplitRandom)
    
    poly = PolynomialFeatures(degree=poly_degree)
    Xtrain_ = poly.fit_transform(Xtrain)
    Xtest_  = poly.fit_transform(Xtest)
    
    model = LinearRegression()
    model.fit(Xtrain_,ytrain)
    
##Model evaluation:
        
    yhat  = model.predict(Xtest_)
    R2    = r2_score(yhat,ytest)  #imported above via sklearn.metrics
  
    
##If model is successfully generated, output results##

    modelDir="models/%s/%s/" % (modelname,today_string)
    imageDir=modelDir + "0-images/"


    if persist==True:

        try: 
            os.listdir(modelDir)
        except:
            os.makedirs(modelDir)
            os.mkdir(imageDir)


        joblib.dump(model,modelDir + "model.pkl")
    
    
        plt.rcParams['figure.figsize']=[9,4]
        plt.subplot(121)
        plt.scatter(ytest, yhat, alpha=.1)
        plt.plot([0,10],[0,10])
        #plt.xlim(0,10); plt.ylim(0,10)
        plt.xlabel("actual");plt.ylabel("predicted")
        plt.title("Model = Poly(%i)\t\t $R^2$=%.2f" % (poly_degree,R2))
        plt.subplot(122)
        sns.set(style="whitegrid")
        sns.residplot(ytest,yhat)#, lowess=True)
    
        plt.savefig(imageDir+"diagnostics.png")
        plt.close()

    
    if returnTestSetResults==True: 
        return yhat, ytest, Xtest    
    
    if verbose==True:
        print "Polynomial Regression(%i) model successfully generated!\t\t\t(Train: %i, Test: %i)\n\tModel file saved in:\t\t%s\n\tDiagnostics plots saved in:\t%s\n" % (poly_degree,len(ytrain),len(ytest),modelDir,imageDir)



# In[6]:

def generate_GBTR(training_data,
                 modelname="4.04-GBTR",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 n_trees = 100,
                 lossfctn = "lad",           #options: [‘ls’, ‘lad’, ‘huber’, ‘quantile’]
                 learningrate = 0.1,
                 trainingSplitRatio = 0.8,
                 trainingSplitRandom=random.RandomState(),
                 persist=True,
                 returnTestSetResults=False,    #True ==> will return predictions, the actual, and the predictors
                 verbose=True
                ):
    
    from sklearn.ensemble import GradientBoostingRegressor
    ###NOTE: replace this with xgboost###
    
    
    replace_nans_infs(training_data)
    X,y = splitXy(training_data, responseColumn, predictorColumns)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=trainingSplitRatio, random_state=trainingSplitRandom)
    
    model = GradientBoostingRegressor(n_estimators=n_trees, loss=lossfctn, learning_rate=learningrate)
    model.fit(Xtrain,ytrain)
    

        
##Model evaluation:
        
    yhat  = model.predict(Xtest)
    R2    = r2_score(yhat,ytest)  #imported above via sklearn.metrics

##If model is successfully generated, output results##

    modelDir="models/%s/%s/" % (modelname,today_string)
    imageDir=modelDir + "0-images/"

    if persist==True:

        try: 
            os.listdir(modelDir)
        except:
            os.makedirs(modelDir)
            os.mkdir(imageDir)


        joblib.dump(model,modelDir + "model.pkl")
    
        
        plt.rcParams['figure.figsize']=[13.5,12]
        plt.subplot(321)
        plt.scatter(ytest, yhat, alpha=.1)
        plt.plot([0,10],[0,10])
        #plt.xlim(0,10); plt.ylim(0,10)
        plt.xlabel("actual");plt.ylabel("predicted")
        plt.title("Model = GBTR(%i)\t\t $R^2$=%.2f" % (n_trees,R2))
        plt.subplot(322)
        sns.set(style="whitegrid")
        sns.residplot(ytest,yhat)#, lowess=True)
        plt.subplot(323)
        piethis = pd.DataFrame(data=model.feature_importances_,index=training_data.columns.drop(responseColumn), columns=["importance"])
        plt.pie(piethis.importance, labels=piethis.index,autopct="%.1f%%")#,colors=cm.jet_r(piethis.importance), autopct="%.1f%%")
        plt.subplot(325)
        plt.barh(range(len(piethis)),piethis.importance)
        plt.yticks(add(range(len(piethis)),0.5),piethis.index)
        plt.title("Feature importances")
        
        plt.savefig(imageDir+"diagnostics.png")
        plt.close()

    if returnTestSetResults==True: 
        return yhat, ytest, Xtest   
    
    if verbose==True:
        print "Gradient Boosting Regression model(%i trees) successfully generated!\t(Train: %i, Test: %i)\n\tModel file saved in:\t\t%s\n\tDiagnostics plots saved in:\t%s\n" % (n_trees,len(ytrain),len(ytest),modelDir,imageDir)
    

# In[7]:

def generate_RFR(training_data,
                 modelname="4.05-RFR",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 n_trees = 100,
                 trainingSplitRatio = 0.8,
                 trainingSplitRandom=random.RandomState(),
                 persist=True,
                 returnTestSetResults=False,    #True ==> will return predictions, the actual, and the predictors
                 verbose=True
                ):
    
    from sklearn.ensemble import RandomForestRegressor
    
    replace_nans_infs(training_data)
    X,y = splitXy(training_data, responseColumn, predictorColumns)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=trainingSplitRatio, random_state=trainingSplitRandom)
    
    model = RandomForestRegressor(n_estimators=n_trees)
    model.fit(Xtrain,ytrain)
    

        
##Model evaluation:
        
    yhat  = model.predict(Xtest)
    R2    = r2_score(yhat,ytest)  #imported above via sklearn.metrics

##If model is successfully generated, output results##

    modelDir="models/%s/%s/" % (modelname,today_string)
    imageDir=modelDir + "0-images/"

    if persist==True:

        try: 
            os.listdir(modelDir)
        except:
            os.makedirs(modelDir)
            os.mkdir(imageDir)


        joblib.dump(model,modelDir + "model.pkl")
    
        plt.rcParams['figure.figsize']=[13.5,12]
        plt.subplot(321)
        plt.scatter(ytest, yhat, alpha=.1)
        plt.plot([0,10],[0,10])
        #plt.xlim(0,10); plt.ylim(0,10)
        plt.xlabel("actual");plt.ylabel("predicted")
        plt.title("Model = RFR(%i)\t\t $R^2$=%.2f" % (n_trees,R2))
        plt.subplot(322)
        sns.set(style="whitegrid")
        sns.residplot(ytest,yhat)#, lowess=True)
        plt.subplot(323)
        piethis = pd.DataFrame(data=model.feature_importances_,index=training_data.columns.drop(responseColumn), columns=["importance"])
        plt.pie(piethis.importance, labels=piethis.index,autopct="%.1f%%")#,colors=cm.jet_r(piethis.importance), autopct="%.1f%%")
        plt.subplot(325)
        plt.barh(range(len(piethis)),piethis.importance)
        plt.yticks(add(range(len(piethis)),0.5),piethis.index)
        plt.title("Feature importances")
        
        plt.savefig(imageDir+"diagnostics.png")
        plt.close()

    if returnTestSetResults==True: 
        return yhat, ytest, Xtest

    if verbose==True:
        print "Random Forest Regression model(%i trees) successfully generated!\t(Train: %i, Test: %i)\n\tModel file saved in:\t\t%s\n\tDiagnostics plots saved in:\t%s\n" % (n_trees,len(ytrain),len(ytest),modelDir,imageDir)


# In[8]:

def generate_XGBR(training_data,
                 modelname="4.06-XGBR",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 n_trees = 100,
                 lossfctn = 'reg:linear',           #options: [‘ls’, ‘lad’, ‘huber’, ‘quantile’]
                 learningrate = 0.1,
                 trainingSplitRatio = 0.8,
                 trainingSplitRandom=random.RandomState(),
                 persist=True,
                 returnTestSetResults=False,    #True ==> will return predictions, the actual, and the predictors
                 verbose=True
                ):
    
    from xgboost import XGBRegressor
    ####http://xgboost.readthedocs.io/en/latest/python/python_api.html
    
    replace_nans_infs(training_data)
    X,y = splitXy(training_data, responseColumn, predictorColumns)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=trainingSplitRatio, random_state=trainingSplitRandom)
    
    model = XGBRegressor(n_estimators=n_trees, objective=lossfctn, learning_rate=learningrate)
    model.fit(Xtrain,ytrain)
    

        
##Model evaluation:
       
    yhat  = model.predict(Xtest)
    R2    = r2_score(yhat,ytest)  #imported above via sklearn.metrics

##If model is successfully generated, output results##


    modelDir="models/%s/%s/" % (modelname,today_string)
    imageDir=modelDir + "0-images/"

    if persist==True:

        try: 
            os.listdir(modelDir)
        except:
            os.makedirs(modelDir)
            os.mkdir(imageDir)


        joblib.dump(model,modelDir + "model.pkl")
    
        plt.rcParams['figure.figsize']=[9,4]
        plt.subplot(121)
        plt.scatter(ytest, yhat, alpha=.1)
        plt.plot([0,10],[0,10])
        #plt.xlim(0,10); plt.ylim(0,10)
        plt.xlabel("actual");plt.ylabel("predicted")
        plt.title("Model = GBTR(%i)\t\t $R^2$=%.2f" % (n_trees,R2))
        plt.subplot(122)
        sns.set(style="whitegrid")
        sns.residplot(ytest,yhat)#, lowess=True)
        
        plt.savefig(imageDir+"diagnostics.png")
        plt.close()

    if returnTestSetResults==True: 
        return yhat, ytest, Xtest

    
    if verbose==True:
        print "XGBoost Regression model(%i trees) successfully generated!\t\t(Train: %i, Test: %i)\n\tModel file saved in:\t\t%s\n\tDiagnostics plots saved in:\t%s\n" % (n_trees,len(ytrain),len(ytest),modelDir,imageDir)

        
# In[9]:

def generate_Ensemble(df,
                 modelname="4.x-Ensemble",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 modelChoices=None,   #if not None, is a list of indeces referring to models of choice.  the ensemble will use only these
                 trainingSplitRatio = 0.8,
                 trainingSplitRandom=random.RandomState(),
                 persist=True,
                 returnTestSetResults=False,    #True ==> will return predictions, the actual, and the predictors
                 verbose=True
                ):
    
    ###Model libraries###
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor#, AdaBoostRegressor
    from pyearth import Earth
    from xgboost import XGBRegressor
    #####################
    
    
    ######## First, check that this is from case 3 or case 4
    if modelname.startswith("4"): caseString="4"
    elif modelname.startswith("3"): caseString="3"
    else: 
            raise AttributeError("invalid modelname parameter:  must be a string starting with 3 or 4")
    
    ######## Then, read all most recent 3.xx or 4.xx models (which are not the Ensemble itself)
    ######## These will be read from the models directory
    models = sorted([model for model in os.listdir("models") if model.startswith(caseString)])[:-1]

    mostRecentModelDate = {}
    
    for val in models:
        mostRecentModelDate[val] = sorted(os.listdir("models/%s/" % val))[-1]

    yhat = {}
    poly_degree = 0

    
    ######## Allow for model selection.  
    if modelChoices is not None:
        models = [models[i] for i in modelChoices]
        
    ########Then, loop over the models, using them to generate predictions for each input row.  These predictions will
    ########be stored as a column in the yhat table, corresponding to the model which predicted them.
       
    for k, v in mostRecentModelDate.items():
        if k not in models: continue
        model  = joblib.load("models/%s/%s/model.pkl" % (k,v))
        columnset = modelToSpaces[k]
        training_data = df.loc[:,columnset]#df.loc[df.item_rank.notnull(), columnset]
        response = columnset[-1]      # RESPONSE column assumed to be at end
        replace_nans_infs(training_data)
        X,y = splitXy(training_data, response, "default")
   
        if k.find("poly")>0:
            if k.endswith("2"):poly_degree=2
            elif k.endswith("3"):poly_degree=3
            else: raise NameError("unexpected directory name for polynomial model: expecting modelname in the format '4.xx-polyd', where d is some small int")
        
            poly = PolynomialFeatures(degree=poly_degree)
            X = poly.fit_transform(X)
        
        if len(yhat)==0: yhat["y"]=y
        yhat[k] = model.predict(X)
    
    
    #########Train a linear regression on the yhat object, to "average out" each model's prediction
    trainingStacker = pd.DataFrame(data=yhat)

    X,y = splitXy(trainingStacker, "y", "default")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=trainingSplitRatio)#, random_state=42)

    stacker = LinearRegression()
    stacker.fit(Xtrain,ytrain)
    yhatBar = stacker.predict(Xtest)
    R2 = r2_score(yhatBar,ytest)
    
##If model is successfully generated, output results##

    modelDir="models/%s/%s/" % (modelname,today_string)
    imageDir=modelDir + "0-images/"

    if persist==True:

        try: 
            os.listdir(modelDir)
        except:
            os.makedirs(modelDir)
            os.mkdir(imageDir)

        joblib.dump(stacker,modelDir + "model.pkl")    
        
     
        plt.rcParams['figure.figsize']=[13.5,4]
        plt.subplot(131)
        plt.scatter(ytest, yhatBar, alpha=.2)
        plt.plot([0,10],[0,10])
    #   plt.xlim(0,50000); plt.ylim(0,50000)
        plt.xlabel("actual (log)");plt.ylabel("predicted (log)")
        plt.title("Model = Stacker(%i models)\t\t $R^2$=%.2f" % (len(models),R2))
        plt.subplot(132)
        sns.set(style="whitegrid")
        sns.residplot(ytest,yhatBar)#, lowess=True)
        plt.subplot(133)
        plt.scatter(exp(ytest), exp(yhatBar), alpha=.2)
        maxval = max(max(exp(ytest)), max(exp(yhatBar)))
        plt.plot([0,maxval],[0,maxval])
        plt.xlim(0, maxval); plt.ylim(0, maxval)
        plt.title("Predictions scaled back to original")
        plt.xlabel("actual");plt.ylabel("predicted")

        plt.savefig(imageDir+"diagnostics.png")
        plt.close()

        
    if verbose==True:
        print "Stacker (%i models) successfully generated!\t\t\t\t(Train: %i, Test: %i)\n\tModel file saved in:\t\t%s\n\tDiagnostics plots saved in:\t%s\n" % (len(models),len(ytrain),len(ytest),modelDir,imageDir)
        print "Individual model weights:"
        for k,v in zip(Xtrain.columns,stacker.coef_):
            print "\t%s\t%.2f" % (k,v) 
            
            
    if returnTestSetResults==True: 
        return yhatBar, ytest, Xtest

        



    
def predictq30(df,
               responseColumn="log(q30)",
               predictorColumns="default",
               modelChoices=None,   #if not None, is a list of indeces referring to models of choice.  the ensemble will use only these               verbose=True
               modelname="4.x-Ensemble"
               #ensembleMethod="4.x-Ensemble",
               ):
    
    ###Model libraries###
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor#, AdaBoostRegressor
    from pyearth import Earth
    from xgboost import XGBRegressor
    #####################
    
    
     ######## First, check that this is from case 3 or case 4
    if modelname.startswith("4"): caseString="4"
    elif modelname.startswith("3"): caseString="3"
    else: 
            raise AttributeError("invalid modelname parameter:  must be a string starting with 3 or 4")
  
    ######## Then, read all most recent 3.xx or 4.xx models (which are not the Ensemble itself)
    ######## These will be read from the models directory
    models = sorted([model for model in os.listdir("models") if model.startswith(caseString) and not model.startswith(caseString+".y")])

    mostRecentModelDate = {}
    
    for val in models:
        mostRecentModelDate[val] = sorted(os.listdir("models/%s/" % val))[-1]         #CHANGE: 2016-07-29

    yhat = {}
    poly_degree = 0

    ######## Allow for model selection.  
    if modelChoices is not None:
        models = [models[i] for i in modelChoices]
    
    
    ########Then, loop over the models, using them to generate predictions for each input row.  These predictions will
    ########be stored as a column in the yhat table, corresponding to the model which predicted them.
    #####
    #####   NOTE: This code is almost identical to the generate_Ensemble code, except that we load ALL models, and
    #####         when iterating over the dictionary, we remove the last tuple of ("4.x-Ensemble":"yyymmdd").  We
    #####         also remove the response column from all incoming feature spaces loaded from SSv2_feature_spaces
    
    for k, v in sorted(mostRecentModelDate.items())[:-1]:
        if k not in models: continue
        model  = joblib.load("models/%s/%s/model.pkl" % (k,v))
        columnset = modelToSpaces[k][:-1]
        X = df.loc[:,columnset]
#        response = columnset[-1]      # RESPONSE column assumed to be at end
        replace_nans_infs(X)
#        X,y = splitXy(training_data, response, "default")
   
        if k.find("poly")>0:
            if k.endswith("2"):poly_degree=2
            elif k.endswith("3"):poly_degree=3
            else: raise NameError("unexpected directory name for polynomial model: expecting modelname in the format '4.xx-polyd', where d is some small int")
        
            poly = PolynomialFeatures(degree=poly_degree)
            X = poly.fit_transform(X)
        
        yhat[k] = model.predict(X)
    
    stackerName,stackerMostRecentDate = sorted(mostRecentModelDate.items())[-1]
    stackerModel = joblib.load("models/%s/%s/model.pkl" % (stackerName,stackerMostRecentDate))
    
    stackerInput = pd.DataFrame(data=yhat)
    yhatBar = stackerModel.predict(stackerInput)
    
    if responseColumn=="log(q30)": 
        
        #### Case 3 model has a very high likelihood of over-predicting, so we'll scale the predictions back,
        #### so as not to miss too high. 
        ####
        #### Use a normalizing function to keep the predictions exponentially distributed.
        
        if caseString=="3":
            ceiling = max(yhatBar)
            if ceiling > 10:
                normalizingConstant = 10./max(yhatBar)
                yhatBar = multiply(yhatBar, normalizingConstant)
    
        return exp(yhatBar).astype(int)
    else:
        return yhatBar
    
    
    
# ###Testing area:

# In[36]:

#df = pd.read_csv("data/MV_Simtxn_WithSales.csv")
##df = pd.read_csv("data/training/MV_Simtxn_WithSales.csv")
##df = pd.read_csv("data/training/ensemble/MV_Simtxn_WithSales.csv")


##df['relative_item_rank']= divide(df['item_rank'],df['agg1'])
#df['brand_share']       = divide(df['agg5'],df['agg2'])
#df['log(c34)']          = log(df['c34'])
#df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1'])) #log(df['relative_item_rank'])
#df['log(q30)']          = log(df['SUM'])
#df['is_promo']          = pd.to_numeric(df['is_promo'])

###
### split the training set into 2 subsets-- one for training all models, one for training the ensemble
###
###
#
#modelinput = df.loc[df["item_rank"].notnull(), Wide_x_]
#modelinput, ensembleinput = train_test_split(modelinput,train_size = 0.73)
##ensembleinput = df.loc[df["item_rank"].notnull(), Wide_x_]

##generate_MARS(modelinput,trainingSplitRatio=0.82)
##generate_PolyR(modelinput, poly_degree=2, trainingSplitRatio=0.83)
##generate_RFR(modelinput, n_trees=118, trainingSplitRatio=0.83)
##generate_GBTR(modelinput,lossfctn="ls",n_trees=110, trainingSplitRatio=0.81)
##generate_XGBR(modelinput,n_trees=110, trainingSplitRatio=0.83)
##generate_Ensemble(ensembleinput)

