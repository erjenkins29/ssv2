
# coding: utf-8

import os 
import pandas as pd
from numpy import isnan, isinf, random, add, exp, multiply, argwhere,log

from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.linear_model import Lasso,LinearRegression
from preprocess import preprocess,comment_log,replace_nans_infs


def month_math(month,num):
    '''
    A function that makes month-related calculation easier.
    Input parameters:
    month: a string or an int that represents a specific month, format YYYYmm.
    num: an int. how many months to be added. for subtraction, write a negative integer.
    Output:
    a string in the format of YYYYmm.

    '''
    from datetime import datetime
    parsed_month= datetime.strptime(str(month),'%Y%m')
    from dateutil import relativedelta
    new_month = parsed_month + relativedelta.relativedelta(months=num)
    return new_month.strftime('%Y%m')        


# In[7]:

def getlastbutone():
    '''
    A month calculation function that relies on the 'month_math' function.
    Get the month before the last month of the current date.
    No input parameter needed.
    Output:
    a string in the format of YYYYmm.
    Eg: if the function is run on 2017/05/05, the output will be '201703'. 
    '''
    from datetime import datetime
    lastbutone = month_math(datetime.now().strftime('%Y%m'),-2)
    return lastbutone


# In[8]:

def getlastmonth():
    '''
    A month calculation function that relies on the 'month_math' function.
    Get the last month of the current date.
    No input parameter needed.
    Output:
    a string in the format of YYYYmm.
    Eg: if the function is run on 2017/05/05, the output will be '201704'. '''

    from datetime import datetime
    lastmonth = month_math(datetime.now().strftime('%Y%m'),-1)
    return lastmonth


# In[9]:

def getmonthlist(startmonth = '201605',endmonth='default',monthnum=None):
    '''
    Produce a list of strings.
    The strings represent months of a given time range that is defined by the parameters.
    Input parameters:
    startmonth : a string in the format of YYYYmm. It is the beginning of the time range.
                 if not given, the default value is '201605'.
                 if given, the value should not be earlier than '201605'
    endmonth:    a string in the format of YYYYmm. It is the end of the time range.
                 if not given, the default value is the month before last month of the current date.
                 if given, the value should not be later than the month before last month of the current date.
    monthnum:    a positive integer. telling how many months should be added on the starting month.
    Output: a list of strings. Each string is in the format YYYYmm
    
    '''
    lastbutone = getlastbutone()
    if monthnum!=None and (monthnum<0 or type(monthnum)!=int): raise Exception("Monthnum should be a positive integer or None.")
    if endmonth != 'default' and monthnum!=None: raise Exception("Can't define endmonth and monthnum at the same time.Choose one.")
    
    if endmonth !='default' and int(endmonth)<int(startmonth): raise Exception("Endmonth should not be earlier than startmonth.")
    if startmonth !='201605' and int(startmonth)<201605: raise Exception("Startmonth can't be earlier than 201605.")
    if endmonth == 'default':
        if monthnum!=None: endmonth = month_math(startmonth,monthnum)
        else: endmonth = lastbutone
    if endmonth > lastbutone: raise Exception("Endmonth can't be later than %s."%lastbutone)
    
    from datetime import datetime
    parsed_start = datetime.strptime(str(startmonth),'%Y%m')
    parsed_end = datetime.strptime(str(endmonth),'%Y%m')
    if monthnum == None:
        from dateutil import relativedelta
        monthnum = relativedelta.relativedelta(parsed_end,parsed_start).months
    datelist = []
    for i in range(0,monthnum+1):
        datelist.append(month_math(startmonth,i))
    return datelist


# In[272]:
def month_check(m, submodel=False):
    '''A function that checks whether a month string follows certain limitations(length, range and type etc.)
    Input parameters: 
    m: the month string to be checked.
    submodel: True/False. if True, the month string is a submodel's month. The corresponding upper limit is '201605' and the lower limit is currentmonth-2. 
              if False, if True, the month string is a stacker's month.  The corresponding upper limit is '201606' and the lower limit is currentmonth-1.
    Output: No value returns. Either error or pass.
 
    '''
    status = 0
    if type(m)!=list: m = [m]
    if submodel == True: upper= '201605';lower = getlastbutone()
    else:  upper= '201606';lower = getlastmonth()
    for i in m:
        if len(str(i)) != 6 or str(i)[0:2] != '20':
            print i
            status =1; print "Month format should be YYYYMM."
        if int(str(i)[-2:]) < 0 or int(str(i)[-2:]) > 13: status =1;print "Month should be between 01 and 12."
        if str(i)< upper: status =1; print "No month should be earlier than %s."%upper
        if str(i)> lower: status =1; print "No month should be later than %s."%lower
        if status != 0: import sys; sys.exit("Month format Error.")


def find_or_generate(max_look_back=2):
    ''' Find or generate a bigstack model.
    If it cannot find a bigstack model in the current month, this function will look back to find the most recent one.
    Input parameter:
    max_look_back: how many months at most to look back. the default value is 2.
    '''
    from datetime import datetime
    current_month = datetime.now().strftime('%Y%m')
    look_back_list = [month_math(current_month,-i) for i in range(1,max_look_back+1)]
    for m in look_back_list:
        for c in [3,4]:
            if os.path.isdir('models/%s.y-BigStack/bigstacks/%s/%s.y-BigStack'%(c,m,c)) and os.path.isfile('models/%s.y-BigStack/bigstacks/%s/%s.y-BigStack/model.pkl'%(c,m,c)):
                print "%s %s.y-BigStack found."%(m,c)
                month = m
            #    return m
            else:
                print "%s %s.y-BigStack not found. Try to train it."%(m,c)
                nlist = [j for j in os.listdir('data') if j.startswith(str(m) + '_JD_sales_plus_compass_joined')]
                print nlist
                if len(nlist) == 0: 
                    print "No data to train. Will try an earliear month instead."
                    pass
                else: 
                    generate_Bigstack(bigstackmonth=m,overwrite=True,modelname="%s.y-BigStack"%c); 
                    print "%s %s.y-BigStack trained."%(m ,c)
                    month = m
            # return m
    try: return month
    except: raise Exception('No model and data in the past %s month(s).'%max_look_back)
            
            
# In[263]:

def read_training_data(training_data='default'):
    ''' A function that reads training_data and does data cleaning by using the function 'preprocess'.
    Input parameters:
    training_data: can be a pandas dataframe or a path of csv/excel file.
                   if no value is given, it will try to find the last month's training data in the 'data' sub-folder.
    Output: a pandas dataframe
    '''
    print training_data
    if isinstance(training_data,pd.core.frame.DataFrame): df = training_data
    elif training_data=='default': df = find_month_data(getlastmonth())
    elif isinstance(training_data,str) and training_data.endswith('.csv'):df = pd.read_csv(training_data)
    elif isinstance(training_data,str) and (training_data.endswith('.xlsx') or training_data.endswith('.xls')):df = pd.read_excel(training_data)
    else: raise Exception('Cant parse provided training data. Please confirm.')
    df = preprocess(df,training=True)
    return df


# In[18]:

def find_month_data(amonth):
    '''A function that looks for the training data of a given month in the data subfolder.
    if more than one files are found, it will use the first one (sorting by alphabet).
    Input parameter: a string represents a month in the format YYYYmm
    Output: a pandas dataframe
    '''
    import os
    nlist = [j for j in os.listdir('data') if j.startswith(str(amonth) + '_JD_sales_plus_compass_joined')]
    if len(nlist) == 1:
        if nlist[0].endswith('.csv'): df0 = pd.read_csv('data/%s' % nlist[0])
        else: df0 = pd.read_excel('data/%s' % nlist[0])
        print "Read:" + " " + nlist[0]
    elif len(nlist) == 0:
        print "Can't find %s data. If it is in the folder, please make sure the file is named correctly." % amonth
    else:
        if min(nlist).endswith('.csv'): df0 = pd.read_csv('data/%s' % min(nlist))
        else: df0 = pd.read_excel('data/%s' % min(nlist))
        print "Read:" + " " + min(nlist)
    return df0


# In[259]:

def model_saving(model, modelname,month ='default', submodel= True,overwrite=True,submodel_list='default'):
    '''
    Input parameters:
    model: the model to be saved
    modelname: a string. the name of the model
    month: a string. the month which the model above belongs to
    submodel: True/False. submodels will be saved in different folders
    overwrite: True/False.
               if True, the previous model, if any, will be moved to 'bak' subfolder
               if False, the previous model, if any, will not be touched. and the newly saved model will be in 'custom' subfolder
    submodel_list: a list describing which submodels are included in an ensemble model
    Output:
    The model will be saved into a specific subfolder.
    The path of the subfolder and the path of images will be returned.
     '''
    print modelname
    if submodel==True: subfolder = 'month_ensembles'
    else: subfolder = 'bigstacks'
    if month=='default': month_string = getlastmonth()
    else:month_string = month
    destination = "models/%s.y-BigStack/%s/%s/%s/"%(modelname[0],subfolder,month_string,modelname)
    if os.path.isdir(destination) and overwrite == True: 
        import datetime; currentdate = datetime.datetime.now().strftime("%Y%m%d%H%M")
        backup = "models/%s.y-BigStack/%s/%s/bak/%s/%s/"%(modelname[0],subfolder,month_string,modelname,currentdate)
        import shutil
        if not os.path.isdir(backup):os.makedirs(backup)
        shutil.move(destination,backup)
    elif overwrite == False:
        import datetime; currentdate = datetime.datetime.now().strftime("%Y%m%d%H%M")
        custom = "models/%s.y-BigStack/%s/%s/custom/%s/%s/"%(modelname[0],subfolder,month_string,modelname,currentdate)
        destination = custom
    imageDir= destination + "0-images/"
    if not os.path.isdir(destination): os.makedirs(destination);os.mkdir(imageDir)
    print "This model will be saved in: " +destination
    joblib.dump(model,destination + "model.pkl")
    if submodel_list != 'default':
        f = open(destination+"submodel_list.txt","w")
        submodels =  ','.join([str(x) for x in submodel_list])
        f.write(submodels)
        f.close()
    return destination,imageDir


# In[262]:

def generate_single_model(modelname,month='default' , training_data= 'default', overwrite = True):
    '''
    generate a single model. A single model is a model like 'poly2 model for 201607'.
    Input parameters:
    modelname: only 12 names are allowed. see the codes below for the 12 names.
    month: the month which the model belongs to
    training_data: a pandas dataframe or a path of csv/excel file
    overwrite: True/False. overwrite the prvious model(if any) or not
    Output:
    returns nothing. the models are saved in specific subfolders.
    '''
    if modelname not in modelToSpaces.keys():raise Exception("Haven't seen this model name before. Please confirm.")
    if month!='default': month_string= month
    else: month_string = getlastmonth()
    
    if isinstance(training_data,str) and training_data=='default': df= find_month_data(month_string);df=preprocess(df,training=True)
    else:  df = read_training_data(training_data)
    if modelname.startswith("4"): modelinput = df.loc[df['case_label']==1, Wide_x_]
    else: modelinput = df.loc[df.cx.notnull(), Case3_x_]
    print modelinput.shape
                                               
    treecountlowlvl, treecount = 105, 225
    if modelname == "3.01-MARS": generate_MARS(modelinput,month_string,overwrite = overwrite,modelname="3.01-MARS",predictorColumns=Case3_x_[:-1], trainingSplitRatio=0.89, verbose=False)
    if modelname == "3.02-poly2": generate_PolyR(modelinput,month_string,overwrite = overwrite,modelname="3.02-poly2",predictorColumns=Case3_x_[:-1], poly_degree=2, trainingSplitRatio=0.89, verbose=False)
    if modelname == "3.03-poly3": generate_PolyR(modelinput,month_string,overwrite = overwrite,modelname="3.03-poly3", predictorColumns=Case3_x_[:-1], poly_degree=3, trainingSplitRatio=0.89, verbose=False)
    if modelname == "3.04-GBTR":  generate_GBTR(modelinput,month_string,overwrite = overwrite,modelname="3.04-GBTR",predictorColumns=Case3_x_[:-1],lossfctn="ls",n_trees=117, trainingSplitRatio=0.89, verbose=False)
    if modelname == "3.05-RFR":  generate_RFR(modelinput,month_string,overwrite = overwrite,modelname="3.05-RFR", predictorColumns=Case3_x_[:-1],n_trees=treecountlowlvl, trainingSplitRatio=0.89, verbose=False)
    if modelname == "3.06-XGBR": generate_XGBR(modelinput,month_string,overwrite = overwrite,modelname="3.06-XGBR",predictorColumns=Case3_x_[:-1],n_trees=treecount, trainingSplitRatio=0.89, verbose=False)
    if modelname == "4.01-MARS": generate_MARS(modelinput,month_string, overwrite = overwrite,predictorColumns=Thin_x_[:-1], trainingSplitRatio=0.89, verbose=False)
    if modelname == "4.02-poly2":  generate_PolyR(modelinput,month_string, overwrite = overwrite,predictorColumns=Poly_x_[:-1], poly_degree=2, trainingSplitRatio=0.89, verbose=False)
    if modelname == "4.03-poly3":  generate_PolyR(modelinput,month_string, overwrite = overwrite,predictorColumns=Poly_x_[:-1], poly_degree=3, trainingSplitRatio=0.89, verbose=False)
    if modelname == "4.04-GBTR":  generate_GBTR(modelinput,month_string,overwrite = overwrite,lossfctn="ls",n_trees=117, trainingSplitRatio=0.89, verbose=False)
    if modelname == "4.05-RFR": generate_RFR(modelinput,month_string,overwrite = overwrite, n_trees=treecountlowlvl, trainingSplitRatio=0.89, verbose=False)
    if modelname == "4.06-XGBR": generate_XGBR(modelinput,month_string,overwrite = overwrite,n_trees=treecount, trainingSplitRatio=0.89, verbose=False)
 


# In[161]:

def one_month_model(month,case='default',training_data= 'default', overwrite=True):
    '''Generate a month-ensemble(small-stack). A month-ensemble is constituted by several 'single model's of that month. 
    Input parameters:
    month: the month
    case: 4 or 3 or default. default means '3 and 4'
    training_data:a pandas dataframe or a path of csv/excel file
    overwrite: True/False. overwrite the prvious model(if any) or not
    Output:
    returns nothing. the models are saved in specific subfolders.
    '''
    if isinstance(training_data,str) and training_data== 'default': df=find_month_data(month);df = preprocess(df,training=True)
    else: df = read_training_data(training_data)
    
    month_string = month
    treecountlowlvl, treecount = 105, 225
    if case == "4" or case=='default': 
        modelinput = df.loc[df['case_label']==1, Wide_x_]
        modelinput, ensembleinput = train_test_split(modelinput,train_size = 0.68)
        generate_MARS(modelinput,month_string,overwrite=overwrite, predictorColumns=Thin_x_[:-1], trainingSplitRatio=0.89, verbose=False)
        generate_PolyR(modelinput,month_string,overwrite=overwrite, predictorColumns=Poly_x_[:-1], poly_degree=2, trainingSplitRatio=0.89, verbose=False)
        generate_PolyR(modelinput,month_string,overwrite=overwrite, predictorColumns=Poly_x_[:-1], poly_degree=3, trainingSplitRatio=0.89, verbose=False)
        generate_GBTR(modelinput,month_string,overwrite=overwrite,lossfctn="ls",n_trees=117, trainingSplitRatio=0.89, verbose=False)
        generate_RFR(modelinput,month_string,overwrite=overwrite, n_trees=treecountlowlvl, trainingSplitRatio=0.89, verbose=False)
        generate_XGBR(modelinput,month_string,overwrite=overwrite,n_trees=treecount, trainingSplitRatio=0.89, verbose=False)
        yhat,ytest,Xtest = generate_Ensemble(ensembleinput,month_string,overwrite=overwrite,modelChoices = [3,5],returnTestSetResults=True,trainingSplitRatio=0.78, verbose=False)
    if case == "3" or case=='default': 
        modelinput = df.loc[df.cx.notnull(), Case3_x_]
        #if len(modelinput)==0: f=open('no_3.txt','w');f.close();return
        modelinput, ensembleinput = train_test_split(modelinput,train_size = 0.68)
        generate_MARS(modelinput,month_string,modelname="3.01-MARS",overwrite=overwrite,predictorColumns=Case3_x_[:-1], trainingSplitRatio=0.89, verbose=False)
        generate_PolyR(modelinput,month_string,modelname="3.02-poly2",overwrite=overwrite,predictorColumns=Case3_x_[:-1], poly_degree=2, trainingSplitRatio=0.89, verbose=False)
        generate_PolyR(modelinput,month_string,modelname="3.03-poly3", overwrite=overwrite,predictorColumns=Case3_x_[:-1], poly_degree=3, trainingSplitRatio=0.89, verbose=False)
        generate_GBTR(modelinput,month_string,modelname="3.04-GBTR",overwrite=overwrite,predictorColumns=Case3_x_[:-1],lossfctn="ls",n_trees=117, trainingSplitRatio=0.89, verbose=False)
        generate_RFR(modelinput,month_string,modelname="3.05-RFR", overwrite=overwrite,predictorColumns=Case3_x_[:-1],n_trees=treecountlowlvl, trainingSplitRatio=0.89, verbose=False)
        generate_XGBR(modelinput,month_string,modelname="3.06-XGBR",overwrite=overwrite,predictorColumns=Case3_x_[:-1],n_trees=treecount, trainingSplitRatio=0.89, verbose=False)
        yhat,ytest,Xtest = generate_Ensemble(ensembleinput,month_string,modelname="3.x-Ensemble",overwrite=overwrite,modelChoices = [3,5],returnTestSetResults=True,trainingSplitRatio=0.78, verbose=False)
    if case not in ['3','4','default']: import sys; sys.exit("Case is '3' or '4' or 'default'.")            


# In[261]:

def generate_submodels(months,mode='overwrite',case='default',ensemble_components='default'):
    '''
    Generate all the submodels needed for a bigstack model, in bulk.
    Input parameters:
    months: a string or a list of strings. Each string should be in the format of YYYYmm.
    mode: either 'overwrite' or 'complement'. If overwrite, generate all the models for the given months and overwrite the existing ones.
          If complement, only generate the models that are missing.
    case: 3 or 4 or default. default means '3 and 4'.
    ensemble_components: the components of the month-ensembles, by default, it is [3,5]. This parameter should be changed in the future, since it is possible that the month-ensembles are constituted by different components.
    Output:
    returns nothing. Generated models are saved in specific subfolders.
    '''
    #mode:overwrite,complement
    if type(months)!=list: months = [months]
    month_check(months, submodel=True)
    if case=='default': case_list = ['3','4']
    else: case_list=[case]
    if ensemble_components=='default':ensemble_components=[3,5]
    if mode =='overwrite':
        for i in months:
            if case=='default' or case=='4': one_month_model(case= "4",month=i,training_data= 'default')
            if case=='default' or case=='3': one_month_model(case= "3",month=i,training_data= 'default')
            print "%s Trained and Saved"%i
    elif mode =='complement':
        for i in months:
            for c in case_list:
                model_list = [x for x in modelToSpaces.keys() if x.startswith(c)]
                model_list2 = []
                for model in model_list:
                    modelDir="models/%s.y-BigStack/month_ensembles/%s/%s/" % (c,i,model)
                    print modelDir
                    if os.path.isdir(modelDir) and os.path.isfile(modelDir+"model.pkl"):pass
                    else: model_list2.append(model)
                if len(model_list2)>0:
                    for mm in model_list2: generate_single_model(mm,month=i,training_data= 'default');print('Model %s for %s has been generated.'%(mm,i))
                    df = find_month_data(i);df = preprocess(df,training=True)
                    if c=='4': modelinput = df.loc[df['case_label']==1, Wide_x_]
                    if c=='3': modelinput = df.loc[df.cx.notnull(), Case3_x_]
                    modelinput, ensembleinput = train_test_split(modelinput,train_size = 0.68)
                    generate_Ensemble(ensembleinput,i,modelname="%s.x-Ensemble"%c,overwrite=True,modelChoices = ensemble_components,returnTestSetResults=True,trainingSplitRatio=0.78, verbose=False)
    else: raise Exception("Mode should be either 'overwrite' or 'complement'.")


# In[205]:

def read_submodel_list(path):
    '''
    Each month-ensemble(small-stack) is constituted by several models.
    The components may vary.
    There is a txt file in the subfolder of each month-ensemble that records the components of that month-ensemble.
    
    This function reads that txt file and returns the components as a list.
    Import parameter:
    path: the path of that txt file
    Output: a list
    '''
    f = open(path)
    submodels = f.readline() 
    submodel_list = submodels.split(',')
    f.close()
    submodel_list = [int(x) for x in submodel_list]
    return submodel_list


# In[297]:

def generate_Bigstack(bigstackmonth= 'default',
                      modelname="4.y-BigStack",overwrite=False,
                      training_data= 'default',
                      submodel_list = 'default', smallmonth=None,
                      persist=True,
                      verbose=False,returnTestSetResults=False):  
    """ Recommend:
     For formal model generation: generate_Bigstack(overwrite='True') and keep other parameters as default.
     For testing:                 generate_Bigstack(overwrite='False') and keep other parameters as default.

     It is not suggested to use the other parameters, if you insist, here is the introduction of the other parameters:
     bigstackmonth: The month where this bigstack's training data are from. By default it is the previous month. Should be in the vaild format.
     submodel_list: The list of submodel months. By default it is all the months before the previous month and after 201605. Each month should be in the valid format.
     smallmonth: An integer. The month amount that users want to keep in smallstack. If smallmonth=2, only the top 2 months from Lasso will be used to train an OLS.
     modelname: The name of bigstack. Can only be '4.y-BigStack' or '3.y-BigStack'.
     training_data: The training data to be used. If default, the function will try to find 'lastmonth(e.g.201610)_JD_sales_plus_compass_joined' in the 'data' folder. Otherwise can be a file path or dataframe.
     overwrite: By default, it is False. False means the model is going to be saved in the 'custom' subfolder, without overwriting any previous model. Otherwise it will be saved in the '3/4.y-Bigstack' subfolder and override the previous model if any."""
   
    
    if submodel_list=='default'and bigstackmonth == 'default':
    ## For daily model generation script. Get the last month, and the month before
    ## that.  NOTE: if month is OCT, and there is already a bigstack model, script
    ## will not generate a new model for this month.
    
        lastbutone = getlastbutone()
        submodel_list = getmonthlist(startmonth = '201605',endmonth=lastbutone)
        bigstackmonth = getlastmonth()
    elif submodel_list=='default'and bigstackmonth != 'default': submodel_list = getmonthlist(startmonth = '201605',endmonth=month_math(bigstackmonth,-1))
    elif submodel_list!='default' and bigstackmonth == 'default': bigstackmonth= month_math(max(submodel_list),1)

    ## checking that month is valid, according to doc/valid_months.txt
    month_check(submodel_list,submodel=True)
    month_check(bigstackmonth)
    
    ## check that smallmonth parameter is valid
    if smallmonth is not None and int(smallmonth)>len(submodel_list): raise Exception('smallmonth should be a number less than total month number.')
    
    
    ## gets training data from the month specified by "bigstackmonth", or the path
    ## specified by param "training_data" (optionally, training_data can be a df, but 
    ## use is only encouraged for testing purposes)
    
    if bigstackmonth !='default' and training_data=='default': lastmonth = bigstackmonth;df = find_month_data(bigstackmonth)
    else:  df = read_training_data(training_data)
    df = preprocess(df,training=True)
    if modelname.startswith("4"): df = df.loc[df['case_label']==1, Wide_x_]
    else: df = df.loc[df.cx.notnull(), Case3_x_]   

        
    case_string = modelname[0]            ## e.g. "4.y.Bigstack"
    
    ## Important step: 
    ## 1st, check that a submodel in "submodel_list" exists
    ## 2nd, if any missing, train that submodel
    
    generate_submodels(months = submodel_list, mode='complement',case = case_string)
    
    ## here, submodel_list is the final set of Ensembles to be used by Bigstack
    ## submodel_list.txt is a list of models used by (3|4).x-Ensembles, for example
    ##     4.01-MARS
    ## Because it is possible that (3/4).x.Ensemble models have different sets
    ## of models being used, they are specified in submodel_list.txt
    
    for i in submodel_list:
        list_path="models/%s/month_ensembles/%s/%s.x-Ensemble/submodel_list.txt" % (modelname,i,case_string)
        models = read_submodel_list(list_path)
        df['model%s'%i]= predictq30(df, modelmonth=i,modelname="%s.x-Ensemble"%case_string,modelChoices=models,responseColumn="other")
    
    traindf, testdf = train_test_split(df,train_size = 0.9)
    bigstack_x = [x for x in traindf.columns if x.startswith('model')]
    traindf[bigstack_x] = replace_nans_infs(traindf[bigstack_x])
    
    ## NOTE: Lasso forces no intercept and hard-coded alpha == 1
    bigstacklr = Lasso(fit_intercept=False,warm_start=True,positive=True,alpha=1)
    bigstacklr.fit(traindf[bigstack_x],traindf['log(q30)'])
    print "Lasso coefficient of Bigstack:", bigstacklr.coef_

    testdf['bigstack_result']=bigstacklr.predict(testdf[bigstack_x])
    bs_r2 = r2_score(testdf['bigstack_result'],testdf['log(q30)'])
    
    ## get the top performing months from the Lasso
    ## NOTE: months chosen for smallstack correspond to hard-coded value below (0.1)
    
    if smallmonth is not None: smallx =bigstacklr.coef_.argsort()[-smallmonth:][::-1]
    else: smallx = [x[0] for x in argwhere(bigstacklr.coef_>0.1)];print smallx
    smallmonth = [bigstack_x[x] for x in smallx]
    print "Months to be used:",smallmonth
    print "Monthly models' performance:",smallx

    ## fitting the OLS Regression on remaining months from Lasso
    bigstacklr2 = LinearRegression(fit_intercept=False)
    bigstacklr2.fit(traindf[smallmonth],traindf['log(q30)'])
    testdf['smallstack_result']=bigstacklr2.predict(testdf[smallmonth])
    ss_r2 = r2_score(testdf['smallstack_result'],testdf['log(q30)'])
    ytest = testdf['log(q30)']
    yhatBar = testdf['smallstack_result']
    Xtest = traindf[smallmonth]
    print "Bigstack R^2: %.2f,\tSmallstack R^2: %.2f"%(bs_r2,ss_r2)

    if persist==True:
        month_list = [x.replace('model','') for x in smallmonth]
        destination,imageDir = model_saving(bigstacklr2,modelname,month=bigstackmonth,submodel_list=month_list,submodel=False,overwrite=overwrite)        
   
        plt.rcParams['figure.figsize']=[13.5,4]
        plt.subplot(131)
        plt.scatter(ytest, yhatBar, alpha=.2)
        plt.plot([0,10],[0,10])
    #   plt.xlim(0,50000); plt.ylim(0,50000)
        plt.xlabel("actual (log)");plt.ylabel("predicted (log)")
        plt.title("Model = Stacker(%i models)\t\t $R^2$=%.2f" % (len(smallmonth),ss_r2))
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
            


# In[224]:

def bigstack_predictq30(df, modelmonth,
               responseColumn="log(q30)",
               predictorColumns="default",
               modelChoices=None,   #if not None, is a list of indeces referring to models of choice.  the ensemble will use only these               verbose=True
               modelname="4.y-BigStack"
               #ensembleMethod="4.x-Ensemble",
               ):
    
    ######## First, check that this is from case 3 or case 4
    if modelname.startswith("4"): caseString="4"
    elif modelname.startswith("3"): caseString="3"
    else: raise AttributeError("invalid modelname parameter:  must be a string starting with 3 or 4")
  
    ######## Then, read all most recent 3.xx or 4.xx models (which are not the Ensemble itself)
    ######## These will be read from the models directory
    month = modelmonth
    f = open("models/%s/bigstacks/%s/%s/submodel_list.txt"%(modelname,month,modelname),"r")
    submodels = f.readline() 
    submodel_list = submodels.split(',')
    f.close()
    
    #df = preprocess(df)
    for m in submodel_list:
        #print m
        list_path="models/%s/month_ensembles/%s/%s.x-Ensemble/submodel_list.txt" % (modelname,m,caseString)
        models = read_submodel_list(list_path)
        df['model%s'%m]= predictq30(df, modelmonth=m,modelname="%s.x-Ensemble"%caseString,modelChoices=models,responseColumn="other")
    
    bigstack_x = ['model'+str(x) for x in submodel_list]
    #print bigstack_x
    model  = joblib.load("models/%s/bigstacks/%s/%s/model.pkl" % (modelname,month,modelname))
    X = df.loc[:,bigstack_x]
    replace_nans_infs(X)
    yhatBar = model.predict(X)
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
    


# In[176]:

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


def generate_MARS(training_data,month,
                  modelname="4.01-MARS", 
                  responseColumn="log(q30)",
                  predictorColumns="default", #default => all non-response columns in training data
                  max_degree=2,
                  minspan_alpha=0.5,
                  overwrite = True,
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
    if persist==True:

        destination,imageDir = model_saving(model,modelname,month = month, overwrite = overwrite)
    
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
    

def generate_PolyR(training_data,month,
                  modelname="4.0x-polyx", # --> it will be 4.0x-Polyx based on the degree x of polynomials 
                  responseColumn="log(q30)",
                  predictorColumns="default", #default => all non-response columns in training data
                  poly_degree=2,
                  trainingSplitRatio=0.8,
                  overwrite = True,
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

    if persist==True:

        destination,imageDir = model_saving(model,modelname,month = month, overwrite = overwrite)
    
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



def generate_GBTR(training_data,month,
                 modelname="4.04-GBTR",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 n_trees = 100,
                 lossfctn = "lad",           #options: [‘ls’, ‘lad’, ‘huber’, ‘quantile’]
                 learningrate = 0.1,
                 trainingSplitRatio = 0.8,
                 overwrite = True,
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

    if persist==True:

        destination,imageDir = model_saving(model,modelname,month = month, overwrite = overwrite)
        
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
    


def generate_RFR(training_data,month,
                 modelname="4.05-RFR",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 n_trees = 100,
                 trainingSplitRatio = 0.8,
                 overwrite = True,
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

    if persist==True:

        destination,imageDir = model_saving(model,modelname,month = month, overwrite = overwrite)
 
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


def generate_XGBR(training_data,month,
                 modelname="4.06-XGBR",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 n_trees = 100,
                 lossfctn = 'reg:linear',           #options: [‘ls’, ‘lad’, ‘huber’, ‘quantile’]
                 learningrate = 0.1,
                 trainingSplitRatio = 0.8,
                 trainingSplitRandom=random.RandomState(),
                 persist=True,
                 overwrite = True,
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

    if persist==True:

        destination,imageDir = model_saving(model,modelname,month = month, overwrite = overwrite)
    
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

        

def generate_Ensemble(df,month,
                 modelname="4.x-Ensemble",
                 responseColumn = "log(q30)",
                 predictorColumns="default",
                 modelChoices=None,   #if not None, is a list of indeces referring to models of choice.  the ensemble will use only these
                 trainingSplitRatio = 0.8,
                 trainingSplitRandom=random.RandomState(),
                 persist=True,
                 overwrite = True,
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
    models = sorted([model for model in os.listdir("models/%s.y-BigStack/month_ensembles/%s"%(caseString,month)) if model.startswith(caseString) and model[3].isdigit()])

    #mostRecentModelDate = {}
    
    #for val in models:
    #    mostRecentModelDate[val] = sorted(os.listdir("models/%s/" % val))[-1]

    yhat = {}
    poly_degree = 0
    ######## Allow for model selection.  
    if modelChoices is not None:
        models = [models[i] for i in modelChoices]
        
    ########Then, loop over the models, using them to generate predictions for each input row.  These predictions will
    ########be stored as a column in the yhat table, corresponding to the model which predicted them.
       
    for k in models:
    #    if k not in models: continue
        model  = joblib.load("models/%s.y-BigStack/month_ensembles/%s/%s/model.pkl" % (caseString,month,k))
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

    if persist==True:
        if modelChoices is None:  modelChoices=[0,1,2,3,4,5]
        destination,imageDir = model_saving(stacker,modelname,month = month, overwrite = overwrite,submodel_list=modelChoices)

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

        

def predictq30(df, modelmonth,
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
    month = modelmonth
    models = sorted([model for model in os.listdir("models/%s.y-BigStack/month_ensembles/%s"%(caseString,month)) if model.startswith(caseString) and model[3].isdigit()])

    #mostRecentModelDate = {}
    
    #for val in models:
    #    mostRecentModelDate[val] = sorted(os.listdir("models/%s/" % val))[-1]         #CHANGE: 2016-07-29

    yhat = {}
    poly_degree = 0
    month = modelmonth

    ######## Allow for model selection.  
    if modelChoices is not None:
        models = [models[i] for i in modelChoices]
    
    
    ########Then, loop over the models, using them to generate predictions for each input row.  These predictions will
    ########be stored as a column in the yhat table, corresponding to the model which predicted them.
    #####
    #####   NOTE: This code is almost identical to the generate_Ensemble code, except that we load ALL models, and
    #####         when iterating over the dictionary, we remove the last tuple of ("4.x-Ensemble":"yyymmdd").  We
    #####         also remove the response column from all incoming feature spaces loaded from SSv2_feature_spaces
    for k in models:
        model  = joblib.load("models/%s.y-BigStack/month_ensembles/%s/%s/model.pkl" % (caseString,month,k))
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
    
    stackerName = max([x for x in os.listdir("models/%s.y-BigStack/month_ensembles/%s"%(caseString,month)) if x.endswith('x-Ensemble')])
    stackerModel = joblib.load("models/%s.y-BigStack/month_ensembles/%s/%s/model.pkl" % (caseString,month,stackerName))
    
    stackerInput = pd.DataFrame(data=yhat)
    yhatBar = stackerModel.predict(stackerInput)
    stackerInput['result'] = stackerModel.predict(stackerInput)
    
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

# In[42]:

Case3_x_= [#'currentprice', 
          #'log(allcomments)', 
          #'c1',
          #'c34',
          #'agg1',
          #'agg2', 
          #'agg3', 
          #'agg5',       #REMOVED: used in numerator to calculate brand_share
          'is_promo', 
          #'brand_share', #df['brand_share']       = divide(df['agg5'],df['agg2'])
          'log(cx)',    #df['log(c34)']          = log(df['c34'])
#REMOVE   'log1p(RIR)',  #df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1'])) #log(df['relative_item_rank'])
          'log(q30)'] 

Thin_x_= ['log(cx)',    #df['log(c34)']          = log(df['c34'])
          'log1p(RIR)',  #df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1']))
          'log(q30)']    #RESPONSE: df['log(q30)']= log(df['SUM'])
Poly_x_= Thin_x_         #[] Not sure if we'll need to define these
Wide_x_= ['currentprice', 
          'allcomments', 
          'c1', 
          'agg1',
          'agg2', 
          'agg3', 
          #'agg5',       #REMOVED: used in numerator to calculate brand_share
          'is_promo', 
          'brand_share', #df['brand_share']       = divide(df['agg5'],df['agg2'])
          'log(cx)',    #df['log(c34)']          = log(df['c34'])
          'log1p(RIR)',  #df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1'])) #log(df['relative_item_rank'])
          'log(q30)']    #RESPONSE: df['log(q30)']= log(df['SUM'])

Case3_x_= [#'currentprice', 
          #'log(allcomments)', 
          #'c1',
          #'c34',
          #'agg1',
          #'agg2', 
          #'agg3', 
          #'agg5',       #REMOVED: used in numerator to calculate brand_share
          'is_promo', 
          #'brand_share', #df['brand_share']       = divide(df['agg5'],df['agg2'])
          'log(cx)',    #df['log(c34)']          = log(df['c34'])
#REMOVE   'log1p(RIR)',  #df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1'])) #log(df['relative_item_rank'])
          'log(q30)']    #RESPONSE: df['log(q30)']= log(df['SUM'])
          #'q30']
    
modelToSpaces = {"3.01-MARS": Case3_x_,
                 "3.02-poly2":Case3_x_,
                 "3.03-poly3":Case3_x_,
                 "3.04-GBTR": Case3_x_,
                 "3.05-RFR":  Case3_x_,
                 "3.06-XGBR": Case3_x_,
                 "4.01-MARS": Thin_x_,
                 "4.02-poly2":Poly_x_,
                 "4.03-poly3":Poly_x_,
                 "4.04-GBTR": Wide_x_,
                 "4.05-RFR":  Wide_x_,
                 "4.06-XGBR": Wide_x_}




