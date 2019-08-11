"""
Loading data into dataframe, cleaning and preprocessing
and returning dftrain(training dataset) and dftest(testing dataset)
"""
import numpy as np
import pandas as pd

def load():

    dftrain_dir="/home/bigdatatech14/bigdatatech14/data/dftrain_net_all_o/all"
    dftest_dir="/home/bigdatatech14/bigdatatech14/data/dftest_net_all_o/000000_0"

    dftrain_names=["user_id", "risk_flag", "age", "occupation_id", "city_id", "county_id", "online_time", "real_name_flag", "user_credit_id", "call_mark", "tele_type", "tele_fac", "smart_system", "fist_use_date", "gprs_volume", "arpu", "sp_fee", "comm_flag", "call_counts", "vpmn_call_counts", "toll_counts", "wj_call_counts", "out_call_counts", "callfw_counts", "qqw_call_counts", "bd_call_counts", "roam_counts", "call_duration_m", "bill_duration_m", "vpmn_call_duration_m", "wj_call_duration_m", "out_call_duration_m", "callfw_duration_m", "bd_call_duration_m", "roam_duration_m", "toll_duration_m", "qqw_call_duration_m", "stop_days06", "stop_cnt06", "num_of_comm", "mo_count", "mt_count", "r0", "r1", "r4", "r5", "r6", "r8", "app_ztbz_counts", "app_jtdh_counts", "app_bjsh_counts", "app_sysx_counts", "app_jrlc_counts", "app_bgsw_counts", "app_xtaq_counts", "app_txsj_counts", "app_sygj_counts", "app_jyxx_counts", "app_etqz_counts", "app_yxyl_counts", "app_gwyh_counts", "app_ydjk_counts", "app_qtzx_counts", "app_hljy_counts", "app_yyst_counts", "app_xwyd_counts", "app_lyjd_counts"]

    dftest_names=["user_id", "age", "occupation_id", "city_id", "county_id", "online_time", "real_name_flag", "user_credit_id", "call_mark", "tele_type", "tele_fac", "smart_system", "fist_use_date", "gprs_volume", "arpu", "sp_fee", "comm_flag", "call_counts", "vpmn_call_counts", "toll_counts", "wj_call_counts", "out_call_counts", "callfw_counts", "qqw_call_counts", "bd_call_counts", "roam_counts", "call_duration_m", "bill_duration_m", "vpmn_call_duration_m", "wj_call_duration_m", "out_call_duration_m", "callfw_duration_m", "bd_call_duration_m", "roam_duration_m", "toll_duration_m", "qqw_call_duration_m", "stop_days06", "stop_cnt06", "num_of_comm", "mo_count", "mt_count", "r0", "r1", "r4", "r5", "r6", "r8", "app_ztbz_counts", "app_jtdh_counts", "app_bjsh_counts", "app_sysx_counts", "app_jrlc_counts", "app_bgsw_counts", "app_xtaq_counts", "app_txsj_counts", "app_sygj_counts", "app_jyxx_counts", "app_etqz_counts", "app_yxyl_counts", "app_gwyh_counts", "app_ydjk_counts", "app_qtzx_counts", "app_hljy_counts", "app_yyst_counts", "app_xwyd_counts", "app_lyjd_counts"]

    dftrain=pd.read_csv(dftrain_dir,names=dftrain_names,encoding='utf-8',na_values=['\\N'],engine='python')
    dftest=pd.read_csv(dftest_dir,names=dftest_names,encoding='utf-8',na_values=['\\N'],engine='python')

    feature=[
    ## "user_id",
     "risk_flag",
     "age",
    # "occupation_id",
    # "city_id",
    # "county_id",
     "online_time",
    # "real_name_flag",
     "user_credit_id",
    # "call_mark",
    #"tele_type",
    #"tele_fac",
    #"smart_system",
     "fist_use_date",
     "gprs_volume",
     "arpu",
    # "sp_fee",
    # "comm_flag",
    # "call_counts",
    # "vpmn_call_counts",
     "toll_counts",
     "wj_call_counts",
     "out_call_counts",
    # "callfw_counts",
     "qqw_call_counts",
     "bd_call_counts",
     "roam_counts",
    # "call_duration_m",
     "bill_duration_m",
    # "vpmn_call_duration_m",
     "wj_call_duration_m",
     "out_call_duration_m",
    # "callfw_duration_m",
     "bd_call_duration_m",
     "roam_duration_m",
     "toll_duration_m",
     "qqw_call_duration_m",
     "stop_days06",
     "stop_cnt06",
     "num_of_comm",
     "mo_count",
     "mt_count",
     "r0",
    # "r1",
    # "r4",
    # "r5",
    # "r6",
    # "r8",
    "app_ztbz_counts",
     "app_jtdh_counts",
     "app_bjsh_counts",
     "app_sysx_counts",
     "app_jrlc_counts",
     "app_bgsw_counts",
     "app_xtaq_counts",
     "app_txsj_counts",
     "app_sygj_counts",
     "app_jyxx_counts",
     "app_etqz_counts",
     "app_yxyl_counts",
     "app_gwyh_counts",
     "app_ydjk_counts",
     "app_qtzx_counts",
     "app_hljy_counts",
     "app_yyst_counts",
     "app_xwyd_counts",
     "app_lyjd_counts",
    ]

    dftrain=dftrain[feature]
    dftest=dftest[feature[1:]]

    dftrain["r0"]=dftrain["r0"].fillna(0)
    #dftrain["r1"]=dftrain["r1"].fillna(0)
    #dftrain["r4"]=dftrain["r4"].fillna(0)
    #dftrain["r5"]=dftrain["r5"].fillna(0)
    #dftrain["r6"]=dftrain["r6"].fillna(0)
    #dftrain["r8"]=dftrain["r8"].fillna(0)

    dftrain["fist_use_date"]=dftrain["fist_use_date"].fillna(9)
    dftrain["mo_count"]=dftrain["mo_count"].fillna(1)
    dftrain["mt_count"]=dftrain["mt_count"].fillna(1)

    dftest["r0"]=dftest["r0"].fillna(0)
    #dftest["r1"]=dftest["r1"].fillna(0)
    #dftest["r4"]=dftest["r4"].fillna(0)
    #dftest["r5"]=dftest["r5"].fillna(0)
    #dftest["r6"]=dftest["r6"].fillna(0)
    #dftest["r8"]=dftest["r8"].fillna(0)

    dftest["fist_use_date"]=dftest["fist_use_date"].fillna(9)
    dftest["mo_count"]=dftest["mo_count"].fillna(1)
    dftest["mt_count"]=dftest["mt_count"].fillna(1)

    dftrain["app_ztbz_counts"]=dftrain["app_ztbz_counts"].fillna(0)
    dftrain["app_jtdh_counts"]=dftrain["app_jtdh_counts"].fillna(0)
    dftrain["app_bjsh_counts"]=dftrain["app_bjsh_counts"].fillna(0)
    dftrain["app_sysx_counts"]=dftrain["app_sysx_counts"].fillna(0)
    dftrain["app_jrlc_counts"]=dftrain["app_jrlc_counts"].fillna(0)
    dftrain["app_bgsw_counts"]=dftrain["app_bgsw_counts"].fillna(0)
    dftrain["app_xtaq_counts"]=dftrain["app_xtaq_counts"].fillna(0)
    dftrain["app_txsj_counts"]=dftrain["app_txsj_counts"].fillna(0)
    dftrain["app_sygj_counts"]=dftrain["app_sygj_counts"].fillna(0)
    dftrain["app_jyxx_counts"]=dftrain["app_jyxx_counts"].fillna(0)
    dftrain["app_etqz_counts"]=dftrain["app_etqz_counts"].fillna(0)
    dftrain["app_yxyl_counts"]=dftrain["app_yxyl_counts"].fillna(0)
    dftrain["app_gwyh_counts"]=dftrain["app_gwyh_counts"].fillna(0)
    dftrain["app_ydjk_counts"]=dftrain["app_ydjk_counts"].fillna(0)
    dftrain["app_qtzx_counts"]=dftrain["app_qtzx_counts"].fillna(0)
    dftrain["app_hljy_counts"]=dftrain["app_hljy_counts"].fillna(0)
    dftrain["app_yyst_counts"]=dftrain["app_yyst_counts"].fillna(0)
    dftrain["app_xwyd_counts"]=dftrain["app_xwyd_counts"].fillna(0)
    dftrain["app_lyjd_counts"]=dftrain["app_lyjd_counts"].fillna(0)

    dftest["app_ztbz_counts"]=dftest["app_ztbz_counts"].fillna(0)
    dftest["app_jtdh_counts"]=dftest["app_jtdh_counts"].fillna(0)
    dftest["app_bjsh_counts"]=dftest["app_bjsh_counts"].fillna(0)
    dftest["app_sysx_counts"]=dftest["app_sysx_counts"].fillna(0)
    dftest["app_jrlc_counts"]=dftest["app_jrlc_counts"].fillna(0)
    dftest["app_bgsw_counts"]=dftest["app_bgsw_counts"].fillna(0)
    dftest["app_xtaq_counts"]=dftest["app_xtaq_counts"].fillna(0)
    dftest["app_txsj_counts"]=dftest["app_txsj_counts"].fillna(0)
    dftest["app_sygj_counts"]=dftest["app_sygj_counts"].fillna(0)
    dftest["app_jyxx_counts"]=dftest["app_jyxx_counts"].fillna(0)
    dftest["app_etqz_counts"]=dftest["app_etqz_counts"].fillna(0)
    dftest["app_yxyl_counts"]=dftest["app_yxyl_counts"].fillna(0)
    dftest["app_gwyh_counts"]=dftest["app_gwyh_counts"].fillna(0)
    dftest["app_ydjk_counts"]=dftest["app_ydjk_counts"].fillna(0)
    dftest["app_qtzx_counts"]=dftest["app_qtzx_counts"].fillna(0)
    dftest["app_hljy_counts"]=dftest["app_hljy_counts"].fillna(0)
    dftest["app_yyst_counts"]=dftest["app_yyst_counts"].fillna(0)
    dftest["app_xwyd_counts"]=dftest["app_xwyd_counts"].fillna(0)
    dftest["app_lyjd_counts"]=dftest["app_lyjd_counts"].fillna(0)

    #dftrain["user_id"]=dftrain["user_id"].astype("str")
    #dftest["user_id"]=dftest["user_id"].astype("str")

    dftrain.loc[dftrain.age<0,"age"]=2
    dftest.loc[dftest.age<0,"age"]=2

    ## transform variables with continuous value into discrete
    dftrain["fist_use_date_d"]=pd.cut(dftrain.fist_use_date,bins=[-1,6,12,18,147],labels=[0,1,2,3])
    dftrain["mo_count_d"]=pd.cut(dftrain.mo_count,bins=[-1,19,44,90,7200],labels=[0,1,2,3])
    dftrain["mt_count_d"]=pd.cut(dftrain.mt_count,bins=[-1,23,51,102,4500],labels=[0,1,2,3])
    dftrain["r0_d"]=pd.cut(dftrain.r0,bins=[-1,37,85,174,11000],labels=[0,1,2,3])
    #dftrain["r1_d"]=pd.cut(dftrain.r1,bins=[-1,3,9,36,3300],labels=[0,1,2,3])
    #dftrain["r4_d"]=pd.cut(dftrain.r4,bins=[-1,9,32,86,7300],labels=[0,1,2,3])
    #dftrain["r5_d"]=pd.cut(dftrain.r5,bins=[-1,4,13,35,530],labels=[0,1,2,3])
    #dftrain["r6_d"]=pd.cut(dftrain.r6,bins=[-1,3,5,10,3000],labels=[0,1,2,3])
    #dftrain["r8_d"]=pd.cut(dftrain.r8,bins=[-1,2,5,10,200],labels=[0,1,2,3])

    dftest["fist_use_date_d"]=pd.cut(dftest.fist_use_date,bins=[-1,6,12,18,147],labels=[0,1,2,3])
    dftest["mo_count_d"]=pd.cut(dftest.mo_count,bins=[-1,19,44,90,7200],labels=[0,1,2,3])
    dftest["mt_count_d"]=pd.cut(dftest.mt_count,bins=[-1,23,51,102,5000],labels=[0,1,2,3])
    dftest["r0_d"]=pd.cut(dftest.r0,bins=[-1,37,85,174,11000],labels=[0,1,2,3])
    #dftest["r1_d"]=pd.cut(dftest.r1,bins=[-1,3,9,36,3300],labels=[0,1,2,3])
    #dftest["r4_d"]=pd.cut(dftest.r4,bins=[-1,9,32,86,7300],labels=[0,1,2,3])
    #dftest["r5_d"]=pd.cut(dftest.r5,bins=[-1,4,13,35,530],labels=[0,1,2,3])
    #dftest["r6_d"]=pd.cut(dftest.r6,bins=[-1,3,5,10,3000],labels=[0,1,2,3])
    #dftest["r8_d"]=pd.cut(dftest.r8,bins=[-1,2,3,5,200],labels=[0,1,2,3])

    dftrain["fist_use_date_d"]=dftrain["fist_use_date_d"].astype("int")
    dftrain["mo_count_d"]=dftrain["mo_count_d"].astype("int")
    dftrain["mt_count_d"]=dftrain["mt_count_d"].astype("int")
    dftrain["r0_d"]=dftrain["r0_d"].astype("int")
    dftest["fist_use_date_d"]=dftest["fist_use_date_d"].astype("int")
    dftest["mo_count_d"]=dftest["mo_count_d"].astype("int")
    dftest["mt_count_d"]=dftest["mt_count_d"].astype("int")
    dftest["r0_d"]=dftest["r0_d"].astype("int")

    dftrain=dftrain.drop(["fist_use_date","mo_count", "mt_count", "r0"],1)
    dftest=dftest.drop(["fist_use_date","mo_count", "mt_count", "r0"],1)

    return dftrain, dftest
