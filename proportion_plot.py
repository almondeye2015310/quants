df_tmp[weight].plot.bar(stacked=True, width=0.9, figsize=(30, 16), fontsize=24, rot=90)
plt.xlabel('date', size=26)
plt.ylim([0,1])
plt.yticks([0.1 * i for i in range(11)])
plt.ylabel('weight', size=26)
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=24)
