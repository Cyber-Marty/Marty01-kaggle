# 使用XGBoost优化的自动化作文评分方案
# 适用于Kaggle竞赛: Learning Agency Lab - Automated Essay Scoring 2.0

# 1. 导入必要的库（新增XGBoost相关库）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

tqdm.pandas()

# 文本预处理相关库
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 特征提取和模型库（替换为XGBoost）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from sklearn.pipeline import Pipeline
import xgboost as xgb  # 导入XGBoost库

import warnings

warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')


# 2. 数据加载与探索（保持不变）
def load_and_explore_data():
    train_df = pd.read_csv('D:/python/projects/automated_essay_scoring 2.0/data/train.csv')
    test_df = pd.read_csv('D:/python/projects/automated_essay_scoring 2.0/data/test.csv')

    print("训练集形状:", train_df.shape)
    print("测试集形状:", test_df.shape)
    print("\n训练集前5行:")
    print(train_df.head())

    print("\n评分分布统计:")
    print(train_df['score'].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['score'], kde=True)
    plt.title('Essay Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.show()

    print("\n计算作文长度...")
    train_df['essay_length'] = train_df['full_text'].progress_apply(len)

    print("\n作文长度统计:")
    print(train_df['essay_length'].describe())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='essay_length', y='score', data=train_df)
    plt.title('Essay Length vs Score')
    plt.xlabel('Essay Length (characters)')
    plt.ylabel('Score')
    plt.show()

    return train_df, test_df


# 3. 文本预处理（保持不变）
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


# 4. 模型训练（核心修改：使用XGBoost）
def build_and_train_model(train_df):
    print("开始文本预处理...")
    train_df['processed_text'] = train_df['full_text'].progress_apply(preprocess_text)

    X = train_df['processed_text']
    y = train_df['score']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 关键修改：用XGBoost构建模型管道
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000,  # 可适当增加到8000，XGBoost能更好处理高维特征
            ngram_range=(1, 2)
        )),
        ('model', xgb.XGBRegressor(
            # 核心参数
            objective='reg:squarederror',  # 回归任务，平方误差损失
            n_estimators=100,  # 树的数量（比随机森林少，因为每棵树更复杂）
            max_depth=5,  # 控制树深度，防止过拟合（XGBoost默认6）
            learning_rate=0.1,  # 学习率（收缩因子），控制每棵树的贡献

            # 正则化参数（防止过拟合）
            subsample=0.8,  # 每棵树使用80%的样本
            colsample_bytree=0.8,  # 每棵树使用80%的特征
            reg_alpha=0.1,  # L1正则化
            reg_lambda=1,  # L2正则化

            # 计算优化
            n_jobs=-1,  # 并行计算（XGBoost的并行更高效）
            random_state=42,
            verbosity=0  # 静默模式，不输出额外信息
        ))
    ])

    print("开始训练模型...")
    # XGBoost内置进度条，直接训练即可
    pipeline.fit(
        X_train, y_train,
        # 可选：添加早停机制（根据验证集性能自动停止训练）
        model__early_stopping_rounds=10,
        model__eval_set=[(X_val, y_val)],
        model__verbose=False  # 关闭每轮训练的日志输出
    )

    # 模型评估（保持不变）
    print("评估模型性能...")
    y_pred = pipeline.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"验证集RMSE: {rmse:.4f}")

    kappa = cohen_kappa_score(
        y_val,
        np.round(y_pred).astype(int),
        weights='quadratic'
    )
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")

    # 交叉验证（时间会显著缩短）
    print("进行交叉验证...")
    cv_scores = []
    # 减少外层循环次数，XGBoost的稳定性更好
    for i in tqdm(range(3), desc="交叉验证"):
        score = cross_val_score(
            pipeline, X, y, cv=5, scoring='neg_mean_squared_error'
        )
        cv_scores.extend(score)

    cv_rmse = np.sqrt(-np.array(cv_scores))
    print(f"交叉验证RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

    return pipeline


# 5. 生成提交文件（保持不变）
def generate_submission(pipeline, test_df):
    """生成提交文件，确保正确处理测试集文本"""
    print("对测试集文本进行预处理...")

    # 关键修复：确保processed_text列被正确创建
    # 添加异常处理，验证预处理是否成功
    try:
        # 对测试集文本进行预处理
        test_df['processed_text'] = test_df['full_text'].progress_apply(preprocess_text)

        # 验证列是否成功创建
        if 'processed_text' not in test_df.columns:
            raise ValueError("预处理失败，未创建processed_text列")

        # 验证预处理结果不为空
        if test_df['processed_text'].isnull().any():
            print("警告：部分文本预处理结果为空")
            # 填充空值，避免后续错误
            test_df['processed_text'] = test_df['processed_text'].fillna('')

    except Exception as e:
        print(f"预处理过程出错: {str(e)}")
        # 尝试备选方案：简单处理文本
        test_df['processed_text'] = test_df['full_text'].str.lower().fillna('')

    print("预测测试集分数...")
    try:
        # 直接批量预测，比逐行循环更高效
        test_predictions = pipeline.predict(test_df['processed_text'])
        test_predictions = np.round(test_predictions).astype(int)
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        # 出错时生成默认预测（避免程序完全崩溃）
        test_predictions = [test_df['score'].median()] * len(test_df)

    submission = pd.DataFrame({
        'essay_id': test_df['essay_id'],
        'score': test_predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("提交文件已生成: submission.csv")

    return submission


# 主函数
def main():
    train_df, test_df = load_and_explore_data()
    model = build_and_train_model(train_df)
    submission = generate_submission(model, test_df)

    print("\n提交文件前5行:")
    print(submission.head())


if __name__ == "__main__":
    main()