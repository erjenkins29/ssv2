def lag(lag_day=7,
        until_day='yesterday'):
    
    ### you can put until_day as m days ago, where m must <=n
    ### one greater than 1 integer number is expected of until_day.
    ### example: m=3, means you generate df is data of 3 days ago, df_lag is data of "lag_day" days ago.
    
    import datetime as dt
    import time
    import pandas as pd
    exec_start_time=time.time()
    yesterday_string      = (dt.date.today() - dt.timedelta(1)).strftime("%Y%m%d")
    today_string          = dt.date.today().strftime("%Y%m%d")
    n_date_string = (dt.date.today() - dt.timedelta(lag_day)).strftime("%Y%m%d")

    if until_day=='yesterday': until_day = 1
    if until_day > lag_day:
        raise IOError ("until_day: %d must be no greater than lag_day: %d!" %(until_day, lag_day))
    if until_day < 1:
        raise IOError ("until_day: %d and lag_day: %d must be no less than 1!" %(until_day, lag_day))
    if type(until_day)!=int or type(lag_day)!=int:
        raise IOError ("until_day and lag_day must be integer number!")
    

    yesterday_string      = (dt.date.today() - dt.timedelta(until_day)).strftime("%Y%m%d")
    
    
    from jaydebeapi import _DEFAULT_CONVERTERS, _java_to_py
    import jaydebeapi
    _DEFAULT_CONVERTERS.update({'BIGINT':_java_to_py('longValue')})

    conn = jaydebeapi.connect('com.ingres.jdbc.IngresDriver', 
                              ['jdbc:ingres://192.168.6.199:vw7/compass_v1', 'mintel', 'Passwd'], 
                              "lib/iijdbc.jar")
    curs = conn.cursor()
    curs.execute('''
                 SELECT 
                 TOP 5000
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
                 ''' % (yesterday_string))
    rows = curs.fetchall()
    columnNames = [curs.description[i][0] for i in range(len(curs.description))]
    df = pd.DataFrame(data=rows, columns=columnNames)

    curs = conn.cursor()
    curs.execute('''
                 select 
                 Top 5000
                 a.egoodsid,
                 b.c30,
                 a.goodsid,
                 a.currentprice, 
                 a.allcomments,
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
    df_lag = pd.DataFrame(data=rows, columns=columnNames)
    return df, df_lag
