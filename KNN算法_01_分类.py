
"""
KNN算法介绍(K Nearest Neighbors),K近邻算法
原理：
    基于 欧式距离(或者是其他距离计算方法)计算测试集和每个训练集之间的距离，然后根据距离升序排列，找到最近的k个样本。
    基于K个样本投票，票数多的就作为最终预测结果——>分类问题
    基于k个样本计算平均值，作为最终预测结果——>回归问题
实现思路：
    1. 分类问题：
         适用于：有特征，有标签，且标签是不连续的（离散的）
    2. 回归问题：
         适用于：有特征，有标签，且标签是连续的，
KNN算法：分类问题实现思路如下：
    1. 计算测试集和训练集之间的距离
    2. 按照距离升序排列
    3. 找到最近的k个样本
    4. 根据k个样本的标签，进行投票，票数多的就作为最终预测结果
代码实现思路：
    1.导包
    2.收集数据集（测试集和训练集）
    3.创建(KNN,分类模型)模型对象.
    4.模型训练
    5.模型预测
"""
from sklearn.neighbors import KNeighborsClassifier

# 准备数据集(测试集和训练集)
x_train=[[1,2],[2,3],[3,4],[4,5]]   #训练集特征,因为特征可以有多列所以是二维列表
y_train=[0,0,1,1]                   #训练集标签
x_test=[[1,1]]                      #测试集特征

# 创建（KNN,分类模型）模型对象.
KNN_model_class=KNeighborsClassifier(n_neighbors=2)
# 模型训练
KNN_model_class.fit(x_train,y_train)

# 模型预测
y_pred=KNN_model_class.predict(x_test)
print(f'预测结果为：{y_pred}')

