効率的フロンティア
pip install PyPortfolioOpt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
def optimize(df):
    Mu = df.mean().values
    Covs = df.cov().values
​
    ef = EfficientFrontier(Mu,Covs)
    
    trets = np.arange(round(np.amin(Mu),3), round(np.amax(Mu),3),0.001)
    tvols = []
    res_ret = []
    res_risk = []
    weight_df = pd.DataFrame(columns=df.columns,index=trets)
​
    for tr in trets:
        try:
            w = ef.efficient_return(target_return=tr)
            w = ef.clean_weights()
            pref = ef.portfolio_performance()
            res_ret += [pref[0]]
            res_risk += [pref[1]]
            weight_df.loc[tr,:] = list(w.values())
​
        except:
            print(tr,'solve error')
​
    flont_line = pd.DataFrame({'risk':res_risk,'return':res_ret,})
    flont_line["ReturnRisk"] = flont_line["return"]/flont_line["risk"]
    
    w_ret_df = pd.concat([weight_df.reset_index(drop=True), flont_line.reset_index(drop=True)], axis=1)
    return w_ret_df
EfficientFrontier_df = optimize(df_tmp[ret])
sharp_max = EfficientFrontier_df.iloc[EfficientFrontier_df["ReturnRisk"].idxmax(),:][ret].T
for asset in assets:
    df_tmp[f"{asset}_weight"] = sharp_max[f"{asset}_Ret"]
for asset in assets:
    df_tmp[f"{asset}_BM_weight"] = df_tmp[f"{asset}_weight"].mean()
tmp = np.matrix(df_tmp[ret])@np.matrix(df_tmp[weight].T)
df_tmp["return"] = tmp.diagonal().T
tmp = np.matrix(df_tmp[ret])@np.matrix(df_tmp[bm_weight].T)
df_tmp["BM_return"] = tmp.diagonal().T
plt.plot(df_tmp.index, df_tmp["return"].cumsum(),label="port")
plt.plot(df_tmp.index, df_tmp["BM_return"].cumsum(),label="BM")
plt.legend()
<matplotlib.legend.Legend at 0x7fc39c9ef460>

