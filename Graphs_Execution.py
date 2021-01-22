import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


f_1 = open('execution_results_scene1.json', )
data_1 = json.load(f_1)
df_1 = pd.DataFrame(data_1)


f_2 = open('execution_results_scene2.json', )
data_2 = json.load(f_2)
df_2 = pd.DataFrame(data_2)

f_3 = open('execution_results_scene3.json', )
data_3 = json.load(f_3)
df_3 = pd.DataFrame(data_3)

# Plot Food
food_1=pd.DataFrame()
food_1['Scene'] = ['Food']*5
food_1['Time: seconds'] = [df_1['0_count_food_scene1'].loc[['time']][0],
                   df_1['1_count_food_scene1'].loc[['time']][0],
                   df_1['2_count_food_scene1'].loc[['time']][0],
                   df_1['3_count_food_scene1'].loc[['time']][0],
                   df_1['4_count_food_scene1'].loc[['time']][0],
                   ]

food_2=pd.DataFrame()
food_2['Scene'] = ['Food & Walls']*5
food_2['Time: seconds'] = [df_2['0_count_food_scene2'].loc[['time']][0],
                   df_2['1_count_food_scene2'].loc[['time']][0],
                   df_2['2_count_food_scene2'].loc[['time']][0],
                   df_2['3_count_food_scene2'].loc[['time']][0],
                   df_2['4_count_food_scene2'].loc[['time']][0],
                   ]

food_3=pd.DataFrame()
food_3['Scene'] = ['Food & Blocks']*5
food_3['Time: seconds'] = [df_3['0_count_food_scene3'].loc[['time']][0],
                   df_3['1_count_food_scene3'].loc[['time']][0],
                   df_3['2_count_food_scene3'].loc[['time']][0],
                   df_3['3_count_food_scene3'].loc[['time']][0],
                   df_3['4_count_food_scene3'].loc[['time']][0],
                   ]



food_all=pd.concat([food_1, food_2, food_3])
sns.boxplot(x='Scene', y='Time: seconds', data=food_all, palette='bright', showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"})
plt.savefig(f'./result/TIME')
plt.show()

#Food


time_1=pd.DataFrame()
time_1['Scene'] = ['Food']*5
time_1['Food blocks'] = [df_1['0_timer_func_scene1'].loc[['food']][0],
                   df_1['1_timer_func_scene1'].loc[['food']][0],
                   df_1['2_timer_func_scene1'].loc[['food']][0],
                   df_1['3_timer_func_scene1'].loc[['food']][0],
                   df_1['4_timer_func_scene1'].loc[['food']][0],
                   ]

time_2=pd.DataFrame()
time_2['Scene'] = ['Food & Walls']*5
time_2['Food blocks'] = [df_2['0_timer_func_scene2'].loc[['food']][0],
                   df_2['1_timer_func_scene2'].loc[['food']][0],
                   df_2['2_timer_func_scene2'].loc[['food']][0],
                   df_2['3_timer_func_scene2'].loc[['food']][0],
                   df_2['4_timer_func_scene2'].loc[['food']][0],
                   ]
print(time_2)
time_3=pd.DataFrame()
time_3['Scene'] = ['Food & Blocks']*5
time_3['Food blocks'] = [df_3['0_timer_func_scene3'].loc[['food']][0],
                   df_3['1_timer_func_scene3'].loc[['food']][0],
                   df_3['2_timer_func_scene3'].loc[['food']][0],
                   df_3['3_timer_func_scene3'].loc[['food']][0],
                   df_3['4_timer_func_scene3'].loc[['food']][0],
                   ]



time_all=pd.concat([time_1, time_2, time_3])
sns.boxplot(x='Scene', y='Food blocks', data=time_all, palette='bright', showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"})
plt.savefig(f'./result/FOOD')
plt.show()
