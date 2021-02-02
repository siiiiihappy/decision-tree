'''進行決策樹分析'''

import numpy as np #載入套件
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime, date, time, timezone
import graphviz #視覺化套件
import time
start = time.time()

'''載入Final project檔案的資料'''
import Final_project #輸入資料
Data = pd.read_csv('Data.csv')

'''只取2015-2019的資料做訓練樣本及測試樣本'''
Data['年月日'] = pd.to_datetime(Data['年月日'])
Data = Data[Data["年月日"]<pd.to_datetime('20200101')]


'''將ROE為負值或0的值刪除，為無效樣本'''
Data2 = Data[Data['ROE－綜合損益']>0]


'''設定因變數與自變數'''
feature = ['研究發展費用率','ROE－綜合損益','本益比'] #取出接下來要訓練和測試的因子
X2 = Data2[feature]
Y2 = Data2['target']

'''分為訓練及測試資料，訓練決策樹模型'''
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2,test_size=0.2, random_state=50)


from sklearn.model_selection import GridSearchCV  #使用GridSearchCV 網格搜尋對決策樹進行參數調整 entropy法最佳深度也為3
tree_params = {'max_depth': range(3, 14)}
locally_best_tree = GridSearchCV(tree.DecisionTreeClassifier(criterion='gini',random_state=50),
                                 tree_params, cv=5)

locally_best_tree.fit(X2_train, Y2_train)
locally_best_tree.best_params_ #最佳深度為3



clf_a = tree.DecisionTreeClassifier(criterion='gini',max_depth=3).fit(X2_train, Y2_train) #決策樹模型訓練 應用gini方法

Y2_Predict = clf_a.predict(X2_test)
accuracy2 = metrics.accuracy_score(Y2_test, Y2_Predict) #看決策樹績效
print(accuracy2)

clf_a.classes_ #確認class順序

'''應用entropy法進行訓練，準確率近似gini法'''
clf_a2 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3).fit(X2_train, Y2_train) #entropy
Y22_Predict = clf_a2.predict(X2_test)
accuracy22 = metrics.accuracy_score(Y2_test, Y22_Predict) #看決策樹績效
print(accuracy22)

'''繪製及輸出決策樹策略圖'''
featurename = ['R&D','ROE','P/E']
plt.figure(figsize=(13,13))
tree.plot_tree(clf_a, max_depth=3 ,feature_names =featurename, class_names= ['lower','higher'] , fontsize = 10)
plt.show()

dot_data = tree.export_graphviz(clf_a,feature_names =featurename,class_names= ['lower','higher'], out_file=None)
graph = graphviz.Source(dot_data)
graph.render('all')



'''只選擇ROE做決策樹分析'''
feature2 = ['ROE－綜合損益']
X3 = Data2[feature2]

X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y2, test_size=0.2, random_state=50)
clf3 = tree.DecisionTreeClassifier(criterion='gini',max_depth=3) #模型訓練，gini法
clf3 = clf3.fit(X3_train, Y3_train)

Y3_Predict = clf3.predict(X3_test)
accuracy3 = metrics.accuracy_score(Y3_test, Y3_Predict) #看決策樹績效
print(accuracy3)

#clf3.score(X3_test,Y3_test)


featurename3= ['ROE']
plt.figure(figsize=(13,13))
tree.plot_tree(clf3, max_depth=3 ,feature_names =featurename3, class_names= ['lower','higher'] , fontsize = 10)
plt.show()

dot_data2 = tree.export_graphviz(clf3,feature_names =featurename3,class_names= ['lower','higher'], out_file=None)
graph = graphviz.Source(dot_data)
graph.render('ROE')


'''只選擇P/E做分類'''
feature4 = ['本益比']
X4 = Data2[feature4]

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y2,test_size=0.2, random_state=50)
clf4 = tree.DecisionTreeClassifier(max_depth=3) #模型訓練
clf4 = clf4.fit(X4_train, Y4_train)

Y4_Predict = clf4.predict(X4_test)
accuracy4 = metrics.accuracy_score(Y4_test, Y4_Predict) #看決策樹績效
print(accuracy4)

featurename4= ['P/E']
plt.figure(figsize=(13,13))
tree.plot_tree(clf4, max_depth=3 ,feature_names =featurename4, class_names= ['lower','higher'] , fontsize = 10)
plt.show()


dot_data3 = tree.export_graphviz(clf4,feature_names =featurename4,class_names= ['lower','higher'], out_file=None)
graph = graphviz.Source(dot_data3)
graph.render('PE')


'''只選擇R&D做分類'''
feature5 = ['研究發展費用率']
X5 = Data2[feature5]


X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, Y2,test_size=0.2, random_state=50)
clf5 = tree.DecisionTreeClassifier(max_depth=3) #模型訓練
clf5 = clf5.fit(X5_train, Y5_train)

Y5_Predict = clf5.predict(X5_test)
accuracy5 = metrics.accuracy_score(Y5_test, Y5_Predict) #看決策樹績效
print(accuracy5)

featurename5 = ['R&D']
plt.figure(figsize=(13,13))
tree.plot_tree(clf5, max_depth=3 ,feature_names =featurename5, class_names= ['lower','higher'] , fontsize = 10)
plt.show()

dot_data4 = tree.export_graphviz(clf5,feature_names =featurename5,class_names= ['lower','higher'], out_file=None)
graph = graphviz.Source(dot_data4)
graph.render('R&D')

end = time.time()
print(end-start)
