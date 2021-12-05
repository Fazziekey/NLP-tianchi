# Developer：Fazzie
# Time: 2021/12/516:01
# File name: xgboost.py
# Development environment: Anaconda Python

import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

model_path = "./xgboost/"

train_df = pd.read_csv('./train_set.csv/train_set.csv', sep='\t', nrows=100000)
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, smooth_idf=True)
label = train_df['label']
train_test = tfidf.fit_transform(train_df['text'])

train_data = train_test[:90000]
train_label = label[:900000]
dtrain = xgb.DMatrix(train_data, label=train_label, nthread=-1)


test_data = train_test[90000:100000]
test_label = label[90000:100000]
dtest = xgb.DMatrix(test_data, label=test_label, nthread=-1)

params = {
    'booster': 'gbtree',           # gbliner 或 gbtree树模型或者线性模型
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 14,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 10,               # 构建树的深度，越大越容易过拟合
    'lambda': 3,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 5,         # 最小叶子节点样本权重和
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}
print("start train xgboost classifier")
xgbc = xgb.train(params, dtrain)  # 初始化xgboost分类器，原生接口默认启用全部线程
xgbc.save_model(model_path+'xgboost.model')  # 保存模型

# 喂给分类器训练numpy形式的训练特征向量和标签向量


print("xgboost train over")
pre_train = xgbc.predict(xgb.DMatrix(train_data, nthread=-1))   # 训练数据的预测概率矩阵，启用全部线程
pre_test = xgbc.predict(xgb.DMatrix(test_data, nthread=-1))     # 测试数据的预测概率矩阵，启用全部线程

print("train data f1score:")
print(f1_score(train_label, pre_train, average='macro'))

print("test data f1score:")
print(f1_score(test_label, pre_test, average='macro'))
