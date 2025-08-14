import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 新增：用于显示进度条
from tqdm import tqdm
# 为pandas操作添加进度条支持
tqdm.pandas()





# 文本预处理相关库
import re  # 正则表达式库，用于文本清洗（如去除特殊字符、数字等）
import nltk  # 自然语言工具包，提供文本处理的各种功能
from nltk.corpus import stopwords  # NLTK的停用词库，包含英语中常见的无实际意义词汇（如"the"、"and"）
from nltk.stem import WordNetLemmatizer  # 词形还原工具，将词语还原为基本形式（如将"running"还原为"run"）

# 特征提取相关库
from sklearn.feature_extraction.text import TfidfVectorizer  # 将文本转换为TF-IDF特征向量
# TF-IDF通过计算词频和逆文档频率，衡量词语在文本中的重要性，是文本分类/回归的常用特征表示方法

# 模型训练与评估相关库
from sklearn.model_selection import train_test_split  # 将数据集划分为训练集和验证集
from sklearn.model_selection import cross_val_score  # 实现交叉验证，评估模型的稳定性
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归器，用于预测作文分数（回归任务）
from sklearn.metrics import mean_squared_error  # 计算均方误差，评估回归模型的预测误差
from sklearn.metrics import cohen_kappa_score  # 计算Cohen's Kappa系数，特别是加权Kappa，是本竞赛的官方评价指标
from sklearn.pipeline import Pipeline  # 构建机器学习流水线，将特征提取和模型训练整合为一个流程
# 使用Pipeline可以避免数据泄露，同时简化代码流程

import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')

# 1. 数据加载与初步探索
def load_and_explore_data():
    # 加载数据
    train_df = pd.read_csv('D:/python/projects/automated_essay_scoring 2.0/data/train.csv')
    test_df = pd.read_csv('D:/python/projects/automated_essay_scoring 2.0/data/test.csv')

    # # 查看数据基本信息
    # print("训练集信息：")
    # print(train_df.info())
    # print("\n测试集信息：")
    # print(test_df.info())
    #
    # # 查看前几行数据
    # print("\n训练集前5行数据：")
    # print(train_df.head())
    # print("\n测试集前5行数据：")
    # print(test_df.head())
    #
    # # 检查缺失值
    # print("\n训练集缺失值情况：")
    # print(train_df.isnull().sum())
    # print("\n测试集缺失值情况：")
    # print(test_df.isnull().sum())

    # 新增：使用tqdm显示进度
    print("\n计算作文长度...")
    train_df['essay_length'] = train_df['full_text'].progress_apply(len)

    # 查看评分分布
    print("\n评分分布统计:")
    print(train_df['score'].describe())

    # 绘制评分分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['score'], kde=True)
    plt.title('Essay Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.show()

    # 查看作文长度分布
    train_df['essay_length'] = train_df['full_text'].apply(len)
    print("\n作文长度统计:")
    print(train_df['essay_length'].describe())

    # 绘制作文长度与评分的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='essay_length', y='score', data=train_df)
    plt.title('Essay Length vs Score')
    plt.xlabel('Essay Length (characters)')
    plt.ylabel('Score')
    plt.show()




    return train_df, test_df

# 2. 文本预处理

def preprocess_text(text):
    """
    对文本进行预处理
    步骤包括：小写转换、去除特殊字符、去除停用词、词形还原
    """
    # 转换为小写
    text = text.lower()

    # 去除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 分词（简单的空格分割）
    words = text.split()

    # 去除停用词（如"the", "and"等无实际意义的词）
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 词形还原（将词语还原为其基本形式）
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # 重新组合为字符串
    return ' '.join(words)


# 4. 特征工程与模型训练
def build_and_train_model(train_df):
    """构建和训练模型"""
    # 对训练集文本进行预处理
    print("开始文本预处理...")
    train_df['processed_text'] = train_df['full_text'].apply(preprocess_text)

    # 划分特征和目标变量
    X = train_df['processed_text']
    y = train_df['score']

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # # 创建一个包含特征提取和模型的管道
    # # TF-IDF: 将文本转换为数值特征，衡量词语在文档中的重要性
    # # 随机森林回归器: 用于预测作文分数
    # pipeline = Pipeline([
    #     ('tfidf', TfidfVectorizer(
    #         max_features=5000,  # 只考虑最重要的5000个词
    #         ngram_range=(1, 2)  # 考虑单个词和词组
    #     )),
    #     ('model', RandomForestRegressor(
    #         n_estimators=100,  # 100棵决策树
    #         random_state=42
    #     ))
    # ])

    # 新增：自定义随机森林训练进度监控
    class ProgressRandomForest(RandomForestRegressor):
        def fit(self, X, y, **kwargs):
            # 初始化进度条
            self.progress_bar = tqdm(total=self.n_estimators, desc="训练随机森林")

            # 重写每棵树训练完成后的回调
            def callback(tree, index):
                self.progress_bar.update(1)  # 更新进度条

            # 将回调函数注入到训练过程
            self._callback = callback
            super().fit(X, y, **kwargs)
            self.progress_bar.close()  # 训练完成后关闭进度条
            return self

    # 创建包含进度显示的模型管道
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('model', ProgressRandomForest(  # 使用自定义的带进度的随机森林
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # 并行计算加速
        ))
    ])
    # 训练模型
    print("开始训练模型...")
    pipeline.fit(X_train, y_train)

    # 在验证集上评估模型
    print("评估模型性能...")
    y_pred = pipeline.predict(X_val)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"验证集RMSE: {rmse:.4f}")

    # 计算Quadratic Weighted Kappa（竞赛官方评价指标）
    # 需要将预测值四舍五入为整数，因为评分是整数
    kappa = cohen_kappa_score(
        y_val,
        np.round(y_pred).astype(int),
        weights='quadratic'
    )
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")

    # 交叉验证
    print("进行交叉验证...")
    # cv_scores = cross_val_score(
    #     pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
    # )
    # cv_rmse = np.sqrt(-cv_scores)
    # print(f"交叉验证RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    #

    # 新增：为交叉验证添加进度显示
    cv_scores = []
    for i in tqdm(range(5), desc="交叉验证"):
        score = cross_val_score(
            pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
        )
        cv_scores.extend(score)

    cv_rmse = np.sqrt(-np.array(cv_scores))
    print(f"交叉验证RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

    return pipeline



# 5. 生成提交文件
def generate_submission(pipeline, test_df):
    """生成提交文件"""
    # print("对测试集文本进行预处理")
    # test_df['processed_text'] = test_df['full_text'].apply(preprocess_text)
    #
    # # 预测测试集分数
    # test_predictions = pipeline.predict(test_df['processed_text'])
    #
    # # 四舍五入为整数（因为评分是整数）
    # test_predictions = np.round(test_predictions).astype(int)

    print("预测测试集分数...")
    # 新增：显示预测进度
    test_predictions = []
    for text in tqdm(test_df['processed_text'], desc="预测进度"):
        pred = pipeline.predict([text])
        test_predictions.append(pred[0])

    test_predictions = np.round(test_predictions).astype(int)


    # 创建提交文件
    submission = pd.DataFrame({
        'essay_id': test_df['essay_id'],
        'score': test_predictions
    })

    # 保存为CSV文件
    submission.to_csv('submission.csv', index=False)
    print("提交文件已生成: submission.csv")

    return submission


# 主函数
def main():
    """主函数，执行整个流程"""
    # 加载和探索数据
    train_df, test_df = load_and_explore_data()

    # 构建和训练模型
    model = build_and_train_model(train_df)

    # 生成提交文件
    submission = generate_submission(model, test_df)

    # 显示提交文件前几行
    print("\n提交文件前5行:")
    print(submission.head())


# 程序入口
if __name__ == "__main__":
    main()
