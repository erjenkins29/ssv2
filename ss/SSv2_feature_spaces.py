#author: Evan Jenkins
#create date: July 26, 2016
#
#Feature spaces, with a relational table linking models to their respective feature spaces.  This is for the modeling types I've planned for in version 2.0-- so this doesn't update dynamically

#NOTE: it is assumed that the response variable is always at the end of the feature space lists.  This is important because in other processes requiring only the predictors, the last value of the list (i.e. the RESPONSE) is removed

Thin_x_= ['log(c34)',    #df['log(c34)']          = log(df['c34'])
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
          'log(c34)',    #df['log(c34)']          = log(df['c34'])
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
          'log(c34)',    #df['log(c34)']          = log(df['c34'])
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

#Models = sorted(modelToSpaces.keys())