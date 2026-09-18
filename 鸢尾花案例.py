"""
案例：通过KNN算法实现鸢尾花分类

回顾：
    1.加载书局
    2.数据的预处理
    3.特征工程（提取，预处理...）
    4.模型训练
    5.模型评估
    6.模型预测
"""
from sklearn.datasets import load_iris   #加载鸢尾花测试集

def my_load_iris():
    #加载鸢尾花数据集
    iris_data=load_iris()    #字典形态
    #查看数据集
    print(f"数据集：{iris_data}")
    print(f"数据值对应的键：{iris_data.keys()}")
    #查看数据集键所对应的值
    print(f"具体的数值：{iris_data.data}")




if __name__=="__main__":
    my_load_iris()