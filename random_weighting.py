ret = [f"{asset}_Ret" for asset in assets]
weight = [f"{asset}_weight" for asset in assets]
bm_weight = [f"{asset}_BM_weight" for asset in assets]

exret = [f"{asset}_Ret" for asset in excl]

reit_on_ret = list()
reit_off_ret = list()
reit_on_risk= list()
reit_off_risk = list()
rr_on = list()
rr_off = list()


def calc_rr(df):
    
    for i in range(100):
        # ランダムにウェイトを保有する
        for v in weight:
            df[v] = pd.Series( np.random.random( len(df) ), index=df.index )

        df["divider"] = df[weight].sum(axis=1)

        df_ex = df.drop(["REIT", "REIT_Ret", "REIT_weight"], axis=1)

        for v in only:
            df_ex[v] = pd.Series( np.random.random( len(df_ex) ), index=df.index )

        df_ex["divider"] = df_ex[only].sum(axis=1)

        for v in weight:
            df[v] = df[v]/df["divider"]

        for v in only:
            df_ex[v] = df_ex[v]/df_ex["divider"]
            
        # リターンとリスク計算
        df_tmp = df.copy()
        df_extmp = df_ex.copy()
        tmp = np.matrix(df_tmp[ret])@np.matrix(df_tmp[weight].T)
        extmp = np.matrix(df_extmp[exret])@np.matrix(df_extmp[only].T)
        df_tmp["return"] = tmp.diagonal().T
        df_extmp["return"] = extmp.diagonal().T

        ret_ = df_tmp["return"].mean()*12
        reit_on_ret.append(ret_)
        exret_ = df_extmp["return"].mean()*12
        reit_off_ret.append(exret_)
        risk = df_tmp["return"].std()*math.sqrt(12)
        reit_on_risk.append(risk)
        exrisk = df_extmp["return"].std()*math.sqrt(12)
        reit_off_risk.append(exrisk)
