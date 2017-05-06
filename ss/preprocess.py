
from numpy import isnan, isinf, log,divide
import pandas as pd


def comment_log(df):
    '''
    This function produces log-transformed comment columns(c columns).
    It receives a pandas dataframe as an input, looks for comment columns and returns log-transformed values as new columns of the original dataframe.
    It also produces a 'cx' column which is the comment column that is cloest to c30.
    For example, if there are multiple comment columns like c34, c7, c1 in a dataframe. c34 will be the 'cx', since it is closest to c30 among these.
    
    Input:
    A pandas dataframe that contains comment columns, The comment columns should have been named as 'c%d', like 'c26'.
    Output:
    The same dataframe with log-transformed columns appended. This dataframe also contains 'cx'.

    '''
    comment_columns = [str(x) for x in df.columns if str(x).startswith('c') and len(x)>1 and x[1:].isdigit() and str(x)!='c1' ]
    print comment_columns
    comment_days = [int(x[1:]) for x in comment_columns]
    closet_day = min(comment_days, key=lambda x:abs(x-30))
    for c in comment_columns:
        df['log(%s)'%c] = log(df[c])
    df['log(cx)']=df['log(c%s)'%closet_day]
    df['cx'] = df['c%s'%closet_day]
    return df

def replace_nans_infs(x):
    x[isnan(x)] = 0
    x[isinf(x)] = 0
    return x

def preprocess(df,training = False):
    '''
    This function does multiple preprocess operations that are usually needed in this project.
    It drops non-numeric columns, produces log-tranformed columns, gives case labels, calculates relative values and replace infinite/nan values.
    Input parameters:
    df: a pandas dataframe that to be transformed.
    training: True/False. If True, log(q30)(the target column) will be calculated.
    Output:
    The dataframe that has gone through the propress steps. 
    '''
    df.columns = [str(i) for i in df.columns]
    for i in df.columns:
        try: df[i]=pd.to_numeric(df[i])
        except: df.drop(i,axis=1,inplace=True); print "%s not numeric. Dropped."%i
    if training == True:
        if 'sum' in df.columns: df.rename(columns={'sum':'SUM'},inplace=True)
        if 'SUM' not in df.columns: raise Exception("Target column 'Sum' not found.")
        df = df[(df['SUM']>1)&(df['is_in_stock']!=0.0)];df['log(q30)']=log(df['SUM'])

    df = comment_log(df)
    df["case_label"] = 0
    df.loc[df.item_rank.notnull(),"case_label"]=1
    #df.loc[df.item_rank.notnull() & df.c34.isnull(),"case_label"]=2
    df.loc[df.item_rank.isnull() & df.cx.notnull(),"case_label"]=3
    df.loc[df.item_rank.isnull() & df.cx.isnull(),"case_label"]=4
    df['is_promo']          = pd.to_numeric(df['is_promo'])
    df['c1']                = pd.to_numeric(df['c1']) 
    df['egoodsid']                = pd.to_numeric(df['egoodsid']) 
    df['relative_item_rank']= divide(df['item_rank'],df['agg1'])
    df['brand_share']       = divide(df['agg5'],df['agg2'])
    df['log1p(RIR)']        = log(divide(df['item_rank'],df['agg1'])) #log(df['relative_item_rank'])
    
    df['log(allcomments)']  = log(df['allcomments'])
    
    #numeric_columns = df.columns.drop('goodsid')
    for i in df.columns:
        replace_nans_infs(df[i])
    return df
