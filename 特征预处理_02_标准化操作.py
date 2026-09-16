"""
案例：特征预处理-标准化操作

回顾：
利用专业的背景知识和技巧处理数据，用于提升模型的性能
步骤：
   1.特征提取
   2.特征预处理
   3.特征降维
   4.特征选择
   5.特征组合


预处理特征之 标准化介绍：
    目的：
       防止因为量纲（单位）问题，导致特征列的方差值相差较大，影响模型的最终结果
       通过公式把各项的值映射到均值为0,标准差为1的正态分布序列.
       公式：
           x'=(当前值-该列均值)/（该列标准差）σ
           应用场景:
              适用于大数据集的处理
"""
from sklearn.preprocessing import StandardScaler

x_train=[
    [90,2,10,40],
    [60,4,15,45],
    [75,3,13,46]
]
# 创建标准化对象
transfer=StandardScaler()
#对原始数据进行标准化操作
x_train_new=transfer.fit_transform(x_train)
#打印处理之后的数据集
print(f"标准化之后的数据集：\n{x_train_new}")

#打印数据集的均值和方差
print(f"数据集的均值：\n{transfer.mean_}")
print(f"数据集的方差：\n{transfer.var_}")
print(f"数据集的方差：\n{transfer.scale_}")

