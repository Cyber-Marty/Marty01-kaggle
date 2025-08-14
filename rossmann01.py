import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# sns.set(font_scale=1.2)
# sns.set_style("whitegrid")

# 1. 加载数据
def load_and_explore_data():
    # 加载训练集和测试集
    train = pd.read_csv('D:/python/projects/rossmann-store-sales/data/train.csv')
    test = pd.read_csv('D:/python/projects/rossmann-store-sales/data/test.csv')
    store = pd.read_csv('D:/python/projects/rossmann-store-sales/data/store.csv')


    # 探索数据集
    # 主要关注的信息就是数据的形状、缺失值情况、以及数据类型


    # print(train.info())
    # 直接train.info()即可打印到控制台，因为train.info()本身就有打印功能
    # train.info()
    # print('=========================================')
    # test.info()
    # print('=========================================')
    # store.info()

    # print("测试集形状：", test.shape)
    # print("训练集形状：", train.shape)
    # print("店铺信息形状：", store.shape)
    #
    # print("\n训练集前五行数据：")
    # print(train.head())
    #
    # # 查看数据类型和缺失值
    # print("\n训练集信息:")
    # print(train.info())
    #
    # print("\n店铺信息前5行:")
    # print(store.head())
    #
    # # 检查缺失值
    # print("\n训练集缺失值统计:")
    # print(train.isnull().sum())
    #
    # print("\n店铺信息缺失值统计:")
    # print(store.isnull().sum())


    return train, test, store

#  load_and_explore_data()   #测试的时候直接调用，用来查看数据集信息

# 2. 数据预处理
def preprocess_data(train, test, store):

    # 对数据文件查看发现，有同一个Store编号在训练集和测试集以及店铺信息表格中都存在
    # 并且店铺信息中表明Store之间是存在等级区别的，对收入是有影响的
    # 考虑合并店铺信息到训练集和测试集


    train = pd.merge(train, store, how='left', on='Store')
    test = pd.merge(test, store, how='left', on='Store')

    # 关键修复：将混合类型的列转换为字符串（根据DtypeWarning，这里是第7列，通常是StateHoliday）
    # 查看列名确认：print(train.columns[7])，假设输出是'StateHoliday'
    for df in [train, test]:
        df['StateHoliday'] = df['StateHoliday'].astype(str)  # 统一转为字符串类型

    # 处理日期特征
    # 数据集中Date列数据格式是‘object’,是pandas默认的字符串或者混合类型对象的格式，一般需要优化
    # 而且object类型在内存效率和数据分析上都不方便操作，决定优化
    # 结合实际意义，将日期原本的字符串转换为datetime格式
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    # 并提取年、月、日等特征，用来考虑这些时间特征在季节和周期性上的作用
    for df in [train, test]:

        # 年、月、日的默认类型是整数
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # 提取星期几
        df['DayOfWeek'] = df['Date'].dt.dayofweek

        # 提取一年中的第几周，即季度特征
        # 周数（WeekOfYear）的默认类型可能特殊，避免兼容性问题而转成标准整数类型int64
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

        # 处理缺失值

        # 对于数值型特征，使用中位数填充
        numeric_cols = ['CompetitionDistance', 'CompetitionOpenSinceMonth',
                        'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                        'Promo2SinceYear']

        # median表示中位数，mode表示众数，inplace = True表示直接在原数据上修改
        # for col in numeric_cols:
        #     train[col].fillna(train[col].median(), inplace=True)
        #     test[col].fillna(test[col].median(), inplace=True)

        # 修改后
        for col in numeric_cols:
            train[col] = train[col].fillna(train[col].median())  # 去掉inplace，改为直接赋值
            test[col] = test[col].fillna(test[col].median())

        # 对于分类特征，使用众数填充
        categorical_cols = ['PromoInterval']
        # for col in categorical_cols:
        #     train[col].fillna(train[col].mode()[0], inplace=True)
        #     test[col].fillna(test[col].mode()[0], inplace=True)

        # 修改后
        # 类别型特征的填充也同样修改
        for col in categorical_cols:
            train[col] = train[col].fillna(train[col].mode()[0])  # 去掉inplace
            test[col] = test[col].fillna(test[col].mode()[0])

        # # 检查是否还有缺失值
        # print("\n处理后训练集缺失值统计:")
        # print(train.isnull().sum())

    return train,test

# 3. 特征工程
def feature_engineering(train, test):

    # 数据集给了竞争队手存在的时间，因而考虑创建新特征，即竞争对手开业时间对销售额的影响
    for df in [train, test]:
        df['CompetitionOpenMonths'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                        (df['Month'] - df['CompetitionOpenSinceMonth'])
        df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].clip(lower=0)  # 确保不为负数

        # 对Promo2的处理
        df['Promo2Weeks'] = (df['Year'] - df['Promo2SinceYear']) * 52 + \
                            (df['WeekOfYear'] - df['Promo2SinceWeek'])
        df['Promo2Weeks'] = df['Promo2Weeks'].clip(lower=0)  # 确保不为负数

        # 创建一个特征表示当前月份是否在促销期间
        # 先将PromoInterval转换为数字月份
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

        # 分割PromoInterval并转换为数字
        df['PromoMonths'] = df['PromoInterval'].apply \
            (lambda x: x.split(',') if pd.notnull(x) else [])
        df['PromoMonths'] = df['PromoMonths'].apply \
            (lambda x: [month_map[m] for m in x])

        # 检查当前月份是否在促销期间

        # 模型无法直接理解促销周期列表，将其转换为 0/1 的二元特征后，
        # 能让模型清晰识别 "当前月份是否有促销"，从而学习促销活动对销售额的影响规律。
        df['IsPromoMonth'] = df.apply \
            (lambda x: 1 if x['Month'] in x['PromoMonths'] else 0, axis=1)

        # 移除不需要的列
        # 'PromoInterval' 列包含了促销周期信息，但在特征工程中已经转换为二元特征 'IsPromoMonth'，
        # 'PromoMonths' 中间过渡特征
        # 'Date' 列是日期信息，已经被拆分为多个特征（Year, Month, Day, DayOfWeek, WeekOfYear），
        # 因此不再需要原始的日期列。
        # 减少数据维度，降低模型计算成本，同时避免冗余特征对模型的干扰（如多重共线性问题）。
        df.drop(['PromoInterval', 'PromoMonths', 'Date'], axis=1, inplace=True)

    # 对分类特征进行编码
    categorical_features = ['StoreType', 'Assortment', 'StateHoliday']
    for feature in categorical_features:
        le = LabelEncoder()
        # 训练集拟合编码器后再同时处理训练集和测试集，确保相同类别在两个数据集中被编码为相同数字
        le.fit(train[feature].unique())
        train[feature] = le.fit_transform(train[feature])
        test[feature] = le.transform(test[feature])

    return train, test

# 4. 探索性数据分析
def exploratory_data_analysis(train):
    # 可视化销售额分布
    print("\n===== 探索性数据分析 =====")

    # 设置画布
    plt.figure(figsize =(18,12))

    # 1. 销售额分布
    plt.subplot(2,3,1)

    # histplot()函数用于绘制直方图
    # kde=True：同时绘制核密度曲线（一条平滑的曲线
    sns.histplot(train['Sales'], kde=True)
    plt.title('销售额分布')

    # 2. 销售额与促销的关系

    # boxplot()函数用于绘制箱线图
    # x轴为分类变量Promo（促销活动），y轴为数值变量
    plt.subplot(2,3,2)
    sns.boxplot(x = 'Promo', y = 'Sales', data = train)
    plt.title('促销与销售额关系')

    # 3. 销售额与店铺类型的关系
    plt.subplot(2, 3, 3)
    sns.boxplot(x='StoreType', y='Sales', data=train)
    plt.title('店铺类型对销售额的影响')

    # 4. 销售额与星期几的关系
    plt.subplot(2, 3, 4)
    sns.boxplot(x='DayOfWeek', y='Sales', data=train)
    plt.title('星期几对销售额的影响')

    # 5. 销售额与月份的关系
    plt.subplot(2, 3, 5)
    sns.boxplot(x='Month', y='Sales', data=train)
    plt.title('月份对销售额的影响')

    # 6. 竞争对手距离与销售额的关系
    plt.subplot(2, 3, 6)
    # 为了可视化更清晰，我们对距离取对数

    # scatterplot()函数用于绘制散点图
    # alpha=0.2：设置透明度，避免点过于密集

    sns.scatterplot(x=np.log1p(train['CompetitionDistance']), y=train['Sales'], alpha=0.2)
    plt.title('竞争对手距离与销售额的关系 (对数尺度)')

    plt.tight_layout()
    plt.savefig('sales_analysis.png')
    plt.close()

    # 相关性分析
    plt.figure(figsize=(12, 10))
    # 选取数值型特征计算相关性
    # 从训练集中筛选出所有数值类型的特征列（如销售额、促销标识、竞争对手距离等），
    # 排除文本型或类别型特征（因为相关性分析只适用于数值变量）。


    numeric_features = train.select_dtypes(include=[np.number]).columns

    # 使用 pandas 的corr()方法计算所有数值型特征之间的皮尔逊相关系数（取值范围为 - 1 到 1）：
    # 接近 1 表示强正相关（一个特征增大，另一个也倾向增大）
    # 接近 - 1 表示强负相关（一个特征增大，另一个倾向减小）
    # 接近 0 表示几乎无相关
    corr_matrix = train[numeric_features].corr()

    # seaborn 库绘制热图的函数，参数含义：
    # corr_matrix：要可视化的相关性矩阵数据
    # annot=True：在热图单元格中显示具体的相关系数值
    # cmap='coolwarm'：使用冷暖色调渐变（红色表示正相关，蓝色表示负相关，颜色深浅反映相关强度）
    # fmt='.2f'：相关系数保留两位小数显示
    # linewidths=0.5：单元格之间添加细线分隔，使热图更清晰
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('特征相关性矩阵')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    print("探索性数据分析完成，图表已保存为PNG文件")


# 5. 模型训练与评估
def train_and_evaluate_model(train):

    """训练模型并评估性能"""
    # 筛选出营业的店铺（Open=1）且销售额大于0的记录
    # 因为当店铺关门时，销售额为0，这不是我们要预测的情况
    train = train[(train['Open'] == 1) & (train['Sales'] > 0)]

    # 定义特征和目标变量
    # 排除不需要的特征

    # Sales（销售额）是目标变量，不能作为特征


    # Customers：信息泄漏风险决定其必须在建模阶段排除

    # Customers（顾客数量）与Sales高度相关（顾客越多销售额通常越高），但在实际业务中：
    # 预测销售额时，“未来的顾客数量” 是未知的（无法提前获取）；
    # 若将Customers作为特征，模型会过度依赖这个 “未来信息”，导致在真实场景中预测失效（
    # 这是典型的 “数据泄漏”）。
    # 但在 EDA 阶段，Customers有分析价值（如分析 “客单价”= 销售额 / 顾客数），因此需要保留到建模前

    exclude_features = ['Sales', 'Customers', 'Open', 'Store']
    features = [col for col in train.columns if col not in exclude_features]
    X = train[features]
    y = train['Sales']

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")

    # 使用随机森林回归模型
    # 随机森林是一种集成学习方法，对非线性关系和特征交互有很好的捕捉能力
    model = RandomForestRegressor(
        n_estimators=100,  # 树的数量
        max_depth=10,  # 树的最大深度，防止过拟合
        random_state=42,
        n_jobs=-1  # 使用所有可用的CPU核心
    )

    # 训练模型
    print("\n开始训练模型...")
    model.fit(X_train, y_train)

    # 在验证集上进行预测
    y_pred = model.predict(X_val)

    # 评估模型性能
    # 使用RMSE（均方根误差）作为评估指标，这是Kaggle竞赛中使用的指标
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"验证集RMSE: {rmse:.2f}")

    # 分析特征重要性
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n特征重要性（前10名）:")
    print(feature_importance.head(10))

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('特征重要性（前10名）')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return model, features


# 6. 生成预测结果
def generate_predictions(model, test, features):
    """使用训练好的模型对测试集进行预测并生成提交文件"""
    # 只对营业的店铺进行预测
    test_open = test[test['Open'] == 1].copy()
    # 对关门的店铺，销售额为0

    # 进行预测
    if not test_open.empty:
        test_open['Sales'] = model.predict(test_open[features])

    # 生成提交结果
    submission = test[['Id']].copy()
    # 合并预测结果，关门的店铺销售额为0
    submission = submission.merge(
        test_open[['Id', 'Sales']], on='Id', how='left'
    ).fillna(0)

    # 保存结果
    submission.to_csv('rossmann_submission01.csv', index=False)
    print("\n预测结果已保存为 rossmann_submission01.csv")
    print("提交文件前5行:")
    print(submission.head())


# 主函数
def main():
    # 加载和探索数据
    train, test, store = load_and_explore_data()

    # 数据预处理
    train, test = preprocess_data(train, test, store)

    # 特征工程
    train, test = feature_engineering(train, test)

    # 探索性数据分析
    exploratory_data_analysis(train)

    # 模型训练与评估
    model, features = train_and_evaluate_model(train)

    # 生成预测结果
    generate_predictions(model, test, features)

    print("\n===== 项目完成 =====")


if __name__ == "__main__":
    main()
