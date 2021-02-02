'''回測實證'''

import numpy as np #載入套件
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timezone
import time
start = time.time()

data_liq = pd.read_excel('data.xlsx') #讀入電子股資料
list1 = data_liq['證券代碼'].unique()

'''篩流動性，在運用策略從中選股'''
data_liq1= {} #求出2015-2020電子類股公司的平均市值
for i in list1:
    a = data_liq['市值(百萬元)'][data_liq['證券代碼']== i].mean()
    data_liq1[i] = a

data_liq2 = pd.DataFrame.from_dict(data_liq1, orient='index',columns=['mv'])

score1 = data_liq2.sort_values(by = ['mv'],ascending=False)
score1 = score1.iloc[150:] #只取前150家市值最大公司，故先列出剩下的再去除
list_select_drop = score1.index

data_liq_final = data_liq.set_index('證券代碼' , inplace=False)
data_liq_final = data_liq_final.drop(list_select_drop,axis=0) #去掉流動性差的股票

'''只取出2020年資料'''
data_liq_final["年月日"] = pd.to_datetime(data_liq_final["年月日"])
data_liq_final = data_liq_final[data_liq_final["年月日"].dt.year==2020] #只取出2020年資料


df_benchmark = pd.read_excel('benchmark.xlsx') #讀入大盤資料
df_benchmark["年月"] = pd.to_datetime(df_benchmark["年月"])
df_benchmark = df_benchmark[df_benchmark["年月"].dt.year==2020]



'''策略一: ROE>=7.095 & P/E>=9.095'''

ROE = data_liq_final['ROE－綜合損益']
ROE = ROE.dropna(axis=0, how='any') #去除空值

NN = []
for n in range(150):
    NN.append(n*3)
print(NN)

ROE_F = pd.DataFrame() #算出2020年各公司平均ROE
for i in NN:
    ROE1 = ROE[i:i+3].mean() #算平均一季的報酬率
    dict_ROE = {'證券代碼':ROE.index[i],'ROE':[ROE1]} #變成字典再轉成df
    ROE2 = pd.DataFrame(dict_ROE)
    ROE_F = ROE_F.append(ROE2)


ROE_F = ROE_F[ROE_F['ROE']>7.095] #篩選出ROE大於7.095的個股

PE = data_liq_final['本益比-TSE']

NN2 = []
for n2 in range(150):
    NN2.append(n2*12)
print(NN2)


PE_F = pd.DataFrame() #算出2020年各公司平均PE
for j in NN2:
    PE1 = PE[j:j+12].mean() #算平均一季的報酬率
    dict_PE = {'證券代碼':PE.index[j],'P/E':[PE1]} #變成字典再轉成df
    PE2 = pd.DataFrame(dict_PE)
    PE_F = PE_F.append(PE2)

PE_F = PE_F[PE_F['P/E']>9.095] #篩選出P/E大於9.095的個股

#將上述條件篩選出的股票取交集，取出42家公司
ROE_list = set(ROE_F['證券代碼']) #資料類型轉為set
PE_list = set(PE_F['證券代碼'])
asset = (ROE_list & PE_list)
res = [asset1 for asset1 in asset] #set轉成list
data_liq_final_s1 = data_liq_final.loc[res,['年月日','報酬率％_月']] #只留篩選出的股票


len(data_liq_final_s1.index.unique()) #檢查是否只剩下42檔股票


#製作樞紐分析表
strategy1 = pd.pivot_table(data_liq_final_s1,values='報酬率％_月',index=['年月日'],columns=data_liq_final_s1.index)

'''策略二: P/E>18.702'''

PE_F2 = PE_F[PE_F['P/E']>18.702] #篩選出P/E>=9.322 & P/E<=18.702的個股，共95檔股票
PE2_list = PE_F2['證券代碼']

data_liq_final_s2 = data_liq_final.loc[PE2_list,['年月日','報酬率％_月']] 
len(data_liq_final_s2.index.unique())


#製作樞紐分析表
strategy2 = pd.pivot_table(data_liq_final_s2,values='報酬率％_月',index=['年月日'],columns=data_liq_final_s2.index)


'''回測2020年選股策略績效，與大盤做比較'''

bench = df_benchmark['報酬率％_月'].values.tolist()

s1_return = []  #比重一樣的加權平均各策略報酬率，再進行回測比較
s2_return = []
for v in range(12):
    s1 = strategy1.iloc[v,:].mean() 
    s2 = strategy2.iloc[v,:].mean()
    s1_return.append(s1)
    s2_return.append(s2)


s1_returnc = [0,0,0,0,0,0,0,0,0,0,0,0]
s2_returnc = [0,0,0,0,0,0,0,0,0,0,0,0]
benchc = [0,0,0,0,0,0,0,0,0,0,0,0]
for w in range(12):
    s1_returnc[w] = s1_return[w-1] + s1_return[w]
    s2_returnc[w] = s2_return[w-1] + s2_return[w]
    benchc[w] = bench[w-1] + bench[w]
    
fig, ax = plt.subplots() #回測績效圖
ax.plot(strategy1.index,s1_returnc,label='strategy1',linestyle=':')
ax.plot(strategy1.index,s2_returnc,label='strategy2',linestyle='--')
ax.plot(strategy1.index,benchc,label='benchmark')
fig.set_figwidth(15)
fig.tight_layout()
plt.xticks(size='small',rotation=45,fontsize=8)
plt.legend()
plt.show

end = time.time()
print(end-start)

