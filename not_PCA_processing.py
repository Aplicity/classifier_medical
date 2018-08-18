import time
from sklearn.model_selection import train_test_split,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

start=time.clock()  # 显示程序开始运行时的时间

with open('sample.txt') as df:     #打开文件
    dataMat=[]; labels=[]
    for line in df.readlines():     #逐行读取文件
        lineArr=line.strip().split(',')        #文件以'，'分隔每一行的因素
        if '?' not in lineArr:      #数据中有部分缺失的数据用？替代了，在训练模型的时候应该无视这些样本
            dataMat.append(lineArr[:-2])    #每一行中倒数第二个元素之前的都是输入变量
            labels.append(lineArr[-2])      #倒数第二个为分类标签，而倒数第一个为'，'


X=np.array(dataMat)
y=np.array(labels)

model_knn=KNeighborsClassifier(n_neighbors=10)      #建立knn模型，k取值为10
model_svm=SVC(kernel='rbf')                         #建立SVM模型，核函数为rbf函数
model_dt =DecisionTreeClassifier(max_depth=20)      #建立决策树模型，树最大深度为20

score_knn=[]
score_svm=[]
score_dt=[]

kf=KFold(n_splits=10)   # 十折交叉样本划分
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_knn.fit(X_train,y_train)   #拟合knn
    model_svm.fit(X_train,y_train)   #拟合SVM
    model_dt.fit(X_train,y_train)    #拟合决策树

    y_knn = model_knn.predict(X_test)   # 用knn预测集的分类结果
    y_svm = model_svm.predict(X_test)   # 用SVM预测集的分类结果
    y_dt = model_dt.predict(X_test)     # 用决策树预测集的分类结果

    score_knn.append(model_knn.score(X_test,y_test))    # 把每一折分类准确率添加到空列中
    score_svm.append(model_svm.score(X_test,y_test))    # 因为是十折，因此训练后空列中有10个数据，分别为每一折的分类准确率
    score_dt.append(model_dt.score(X_test,y_test))

mean_score_knn=float(sum(score_knn))/len(score_knn)     # 计算每一个模型的平均准确率
mean_score_svm=float(sum(score_svm))/len(score_svm)
mean_score_dt=float(sum(score_dt))/len(score_dt)


print('score_knn:%f' %mean_score_knn)       #输出准确率
print('score_svm:%f' %mean_score_svm)
print('score_dt:%f' %mean_score_dt)


end=time.clock()        #程序结束时的时间

print('耗时:%f' %(end-start)) #输出程序的运行时间