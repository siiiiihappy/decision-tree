'''資料清理'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()

data_liq = pd.read_excel('data.xlsx') #載入2015-2020電子類股資料
data_liq.columns

list = data_liq['證券代碼'].unique() #不重複證券代碼

'''篩選流動性最佳的150家公司'''
data_liq1= {}
for i in list:
    a = data_liq['市值(百萬元)'][data_liq['證券代碼']== i].mean()
    data_liq1[i] = a #求出2015-2020電子類股公司的平均市值

data_liq2 = pd.DataFrame.from_dict(data_liq1, orient='index',columns=['mv'])


score1 = data_liq2.sort_values(by = ['mv'],ascending=False)
score1 = score1.iloc[150:] #只取前150家市值最大公司，故先列出剩下的再去除
list_select_drop = score1.index
#list_select_drop = [str(x) for x in list_select_drop]
list_select_drop


data_liq = pd.DataFrame(data_liq)
data_liq_final = data_liq.set_index('證券代碼' , inplace=False)


data_liq_final = data_liq_final.drop(list_select_drop,axis=0) #去掉流動性差的股票

#data_liq_final.to_csv('data_liq_final.csv', encoding='utf_8_sig') #檢查錯誤


data_liq_final = data_liq_final.drop(['證期會代碼','市值(百萬元)','收盤價(元)_月'],axis=1)#去掉不需要的行資料


'''將資料皆改為季資料'''
NN = []
for n in range(3527): 
    NN.append(n*3)


clear_all_F = pd.DataFrame()
for m in NN:
    clear = data_liq_final[['報酬率％_月']][m:m+3].mean() #算平均一季的報酬率
    clear2 = data_liq_final[['本益比-TSE']][m:m+3].mean() 
    clear3 = {'年月日':data_liq_final.iloc[[m+2],0], #變成字典再轉成df
              '報酬率％':clear.values,
              '本益比':clear2.values}
    clear_all = pd.DataFrame(clear3)
    clear_all_F = clear_all_F.append(clear_all)


'''與原本就為季資料的項目合併'''
origin_F = pd.DataFrame()
for m in NN:
    origin = data_liq_final.iloc[[m+2],[0,3,4]]
    origin_F = origin_F.append(origin)


origin_F = origin_F.reset_index()
clear_all_F = clear_all_F.reset_index()

combine_final = pd.merge(origin_F,clear_all_F,on=['證券代碼','年月日'])

#combine_final.to_csv('combine_final.csv', encoding='utf_8_sig')


'''去除無效樣本'''
combine_final = combine_final.dropna(axis=0, how='any') #去除空值列
combine_final = combine_final[combine_final['本益比']>0]  #去除本益比小於等於0的row


'''合併大盤報酬和電子股資料'''
benchmark = pd.read_excel('benchmark.xlsx',ascending = False) #讀大盤報酬率

NN_2 = []
for n in range(24): #將資料都改成季資料(平均)
    NN_2.append(n*3)
print(NN_2)


benchmark_F = pd.DataFrame()
for r in NN_2:
    clear4 = benchmark[['報酬率％_月']][r:r+3].mean() #算平均一季的報酬率
    clear5 = {'年月日':benchmark.iloc[[r+2],1], #變成字典再轉成df
              '報酬率％_bench':clear4.values}
    clear_alll = pd.DataFrame(clear5)
    benchmark_F = benchmark_F.append(clear_alll)


#合併大盤報酬和電子股資料
Data = pd.merge(combine_final,benchmark_F,on=['年月日'])


'''給予資料標籤，報酬率大於大盤為1，小於大盤為0'''
Data['target'] = np.where(Data['報酬率％']>Data['報酬率％_bench'],1,0)

Data.to_csv('Data.csv',encoding='utf_8_sig') #存取csv檔

end = time.time()
print(end - start)
