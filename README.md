# classifier_medical
生物数据二分类

数据来源http://archive.ics.uci.edu/ml/datasets/p53+Mutants 

K9.data为原数据，大小为1G多，因此手动抽取了前面几个样本作为代码测试数据，另存为文件sample.txt

数据全为数值型，最后一个数值为分类标签，其余均为特征。部分缺少数据用'?'替代，在建模前需要删除。

模型准确率用10折交叉验证，比较KNN、SVM、决策树三个模型的准确率。

not_PCA_processing.py直接输入所有特征进行模型拟合，classifier_aftetPCA.py 先把特征使用PCA(主成分分析法)进行降维，再将降维后的数据作为输入特征，最后才进行模型拟合。
