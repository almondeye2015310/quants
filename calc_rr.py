for i in range(len(training)):
    exec(f"pre_{i} = pd.DataFrame(training[i])")

out_data = pd.DataFrame()
for i in range(len(training)):
    exec(f"out_data = pd.concat([out_data, pre_{i}], axis=1)")

out_data.columns = ['World Equity_weight', 'Emerging Equity_weight', 'US Domestic Equity_weight', 'US Treasury_weight',
       'US Corporate Bond_weight', 'US Long Treasury_weight', 'US High Yield Bond_weight', 'REIT_weight','Comodity_weight'
         ]
out_data.index = df.iloc[31:,:].index

df_train = pd.concat([df.iloc[31:,:], out_data], axis=1)

ret = [f"{asset}_Ret" for asset in assets]
weight = [f"{asset}_weight" for asset in assets]
bm_weight = [f"{asset}_BM_weight" for asset in assets]

df_train[df_train.iloc[:,df_train.columns.str.endswith('weight')] < 0] = 0
df_train["sum"] = df_train.iloc[:,df_train.columns.str.endswith('weight')].sum(axis=1)


for i in out_data.columns:
    df_train[i] = df_train[i] / df_train["sum"]

for asset in assets:
    df_train[f"{asset}_BM_weight"] = df_train[f"{asset}_weight"].mean()
train = np.matrix(df_train[ret])@np.matrix(df_train[weight].T)
df_train["return"] = train.diagonal().T
tmp = np.matrix(df_train[ret])@np.matrix(df_train[bm_weight].T)
df_train["BM_return"] = tmp.diagonal().T

ret_ = df_train["return"].mean()*12
risk = df_train["return"].std()*math.sqrt(12)
print("自分")
print(f"return:{ret_}")
print(f"risk:{risk}")
print(f"return/risk:{ret_/risk}")

ret_ = df_train["BM_return"].mean()*12
risk = df_train["BM_return"].std()*math.sqrt(12)
print("基準")
print(f"return:{ret_}")
print(f"risk:{risk}")
print(f"return/risk:{ret_/risk}")

plt.plot(df_train.index, df_train["return"].cumsum(),label="port")
plt.plot(df_train.index, df_train["BM_return"].cumsum(),label="BM")
plt.legend()
plt.show()
df_train[weight].plot.bar(stacked=True, width=0.9, figsize=(30, 16), fontsize=24, rot=90)
plt.xlabel('date', size=26)
plt.ylim([0,1])
plt.yticks([0.1 * i for i in range(11)])
plt.ylabel('weight', size=26)
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=24)
plt.show()
