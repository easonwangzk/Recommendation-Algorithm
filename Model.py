#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: easonwang
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from nltk.tokenize import word_tokenize
import nltk
import logging
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 设置Gensim的日志记录功能
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from kerastuner.tuners import RandomSearch

# 数据加载到DataFrame中
movie_df = pd.read_csv('/Users/easonwang/Desktop/毕业论文/Movie.csv')
ratings_df = pd.read_csv('/Users/easonwang/Desktop/毕业论文/User.csv')

# 分词和预处理文本
def tokenize_text(text):
    if pd.isna(text):
        return []
    return word_tokenize(text.lower())

# 训练Word2Vec模型
def train_word2vec(text_lists, vector_size=50, window=5, min_count=1):
    model = Word2Vec(sentences=text_lists, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

# 将文本转换为Word2Vec向量
def text_to_vector(model, text_list):
    vectors = [model.wv[text] for text in text_list if text in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

# 完整的数据预处理函数
def preprocess_movie_data(movie_df):
    # 将发行日期转换为日期时间格式
    movie_df['release_date'] = pd.to_datetime(movie_df['release_date'], errors='coerce')

    # 处理所有文本字段
    text_fields = ['original_title', 'overview', 'production_companies','production_countries','spoken_languages','tagline', 'title', 'genres','adult', 'original_language', 'status', 'video']
    models = {}
    for field in text_fields:
        movie_df[field] = movie_df[field].apply(lambda x: tokenize_text(str(x)))  # 转换为字符串并分词
        print(f"Training Word2Vec model for {field}...")
        models[field] = train_word2vec(movie_df[field].tolist())

    # 将文本转换为向量
    for field in text_fields:
        movie_df[field + '_vector'] = movie_df[field].apply(lambda x: text_to_vector(models[field], x))

    # 归一化数值数据
    scaler = MinMaxScaler()
    numeric_features = ['revenue', 'runtime', 'vote_average', 'vote_count']
    movie_df[numeric_features] = scaler.fit_transform(movie_df[numeric_features])

    # 删除原始的文本字段
    for field in text_fields:
        if field in movie_df:
            movie_df.drop(field, axis=1, inplace=True)
    
    return movie_df

# 展开向量到多个列
def expand_vector_to_columns(df, vector_column_names):
    for vector_column in vector_column_names:
        # 尝试获取一个样本向量以确定长度
        sample_vector = df[vector_column].dropna().iloc[0]
        vector_length = len(sample_vector)

        # 为每个向量元素创建一个新列
        for i in range(vector_length):
            # 创建新列名
            column_name = f'{vector_column}_{i}'
            # 分配新列
            df[column_name] = df[vector_column].apply(lambda x: x[i] if len(x) > i else None)
        
        df.drop(vector_column, axis=1, inplace=True)
        
#深度学习embedding模型
def build_model(num_users, num_movies, num_features, embedding_size=30):
    # 输入层
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')
    additional_features_input = Input(shape=(num_features,), name='additional_features_input')

    # 用户和电影嵌入
    user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_user')(user_embedding)
    movie_embedding = Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)
    movie_vec = Flatten(name='flatten_movie')(movie_embedding)

    # 连接所有特征
    concat = Concatenate()([user_vec, movie_vec, additional_features_input])

    # 更深层次的网络结构
    dense = Dense(128, activation='relu')(concat)
    dense = Dense(64, activation='relu')(dense)
    output = Dense(1, activation='linear')(dense)

    model = Model(inputs=[user_input, movie_input, additional_features_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

#Wide&Deep模型
def build_model2():
    # Wide部分
    wide_inputs = tf.keras.layers.Input(shape=(len(wide_features),))
    wide = tf.keras.layers.Dense(1, activation=None)(wide_inputs)

    # Deep部分
    deep_inputs = tf.keras.layers.Input(shape=(600,))
    deep = tf.keras.layers.Dense(128, activation='relu')(deep_inputs)
    #deep = BatchNormalization()(deep)  # 添加批归一化层
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(1, activation=None)(deep)

    # 结合Wide和Deep
    both = tf.keras.layers.concatenate([wide, deep])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(both)
    
    model = tf.keras.models.Model(inputs=[wide_inputs, deep_inputs], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

#寻找最优超参数
def build_hypermodel(hp):
    num_users = 671  # 用户数
    num_movies = 2828  # 电影数
    num_features = 607  # 数值特征数

    # 超参数调整
    embedding_size = hp.Int('embedding_size', min_value=10, max_value=100, step=5)
    dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=320, step=32)
    dense_units_2 = hp.Int('dense_units_2', min_value=32, max_value=256, step=16)
    learning_rate = hp.Float('learning_rate', min_value=0.00001, max_value=0.1, sampling='LOG')

    # 模型架构
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    features_input = Input(shape=(num_features,))

    user_embedding = Embedding(num_users, embedding_size)(user_input)
    user_vec = Flatten()(user_embedding)

    movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
    movie_vec = Flatten()(movie_embedding)

    concat = Concatenate()([user_vec, movie_vec, features_input])
    dense_1 = Dense(dense_units_1, activation='relu')(concat)
    dense_2 = Dense(dense_units_2, activation='relu')(dense_1)
    output = Dense(1, activation='linear')(dense_2)

    model = Model([user_input, movie_input, features_input], output)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model


#构建最优超参数下的模型
def build_best_model(hp):
    num_users = 671  # 用户数
    num_movies = 2828  # 电影数
    num_features = 607  # 特征数

    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')
    features_input = Input(shape=(num_features,), name='additional_features_input')

    user_embedding = Embedding(num_users, hp.get('embedding_size'))(user_input)
    user_vec = Flatten()(user_embedding)

    movie_embedding = Embedding(num_movies, hp.get('embedding_size'))(movie_input)
    movie_vec = Flatten()(movie_embedding)

    concat = Concatenate()([user_vec, movie_vec, features_input])
    dense = Dense(hp.get('dense_units_1'), activation='relu')(concat)
    dense = Dense(hp.get('dense_units_2'), activation='relu')(dense)
    output = Dense(1, activation='linear')(dense)

    model = Model(inputs=[user_input, movie_input, features_input], outputs=output)
    model.compile(optimizer=Adam(hp.get('learning_rate')), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_hypermodel2(hp):
    dense_units_1 = hp.Int('dense_units_1', min_value=64, max_value=256, step=64)
    dense_units_2 = hp.Int('dense_units_2', min_value=32, max_value=128, step=32)
    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling='LOG')
    wide_inputs = Input(shape=(len(wide_features),), name='wide_inputs')
    wide = Dense(1, activation=None)(wide_inputs)
    deep_inputs = Input(shape=(600,), name='deep_inputs')
    deep = Dense(dense_units_1, activation='relu')(deep_inputs)
    deep = Dense(dense_units_2, activation='relu')(deep)
    deep = Dense(1, activation=None)(deep)
    both = Concatenate()([wide, deep])
    output = Dense(1, activation='sigmoid')(both)  
    model = Model(inputs=[wide_inputs, deep_inputs], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def build_best_model2(best_hps):
    wide_inputs = Input(shape=(len(wide_features),), name='wide_inputs')
    wide = Dense(1, activation=None)(wide_inputs)

    deep_inputs = Input(shape=(600,), name='deep_inputs')
    deep = Dense(best_hps.get('dense_units_1'), activation='relu')(deep_inputs)
    deep = Dense(best_hps.get('dense_units_2'), activation='relu')(deep)
    deep = Dense(1, activation=None)(deep)

    both = Concatenate()([wide, deep])
    output = Dense(1, activation='sigmoid')(both)
    
    model = Model(inputs=[wide_inputs, deep_inputs], outputs=output)
    model.compile(optimizer=Adam(best_hps.get('learning_rate')), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

#训练过程可视化
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    metrics_keys = history.history.keys()
    print(f"Available metrics: {metrics_keys}")
    # 绘制评估指标曲线（MSE 和 MAE）
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Metrics over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()


'''
数据预处理
'''
# 合并电影数据和用户评分数据
# 检查用于合并的键的数据类型
print("Data type of 'movieId' in ratings_df:", ratings_df['movieId'].dtype)
print("Data type of 'movieId' in movie_df:", movie_df['movieId'].dtype)

# 如果数据类型不匹配，转换它们以匹配
ratings_df['movieId'] = ratings_df['movieId'].astype(int)

invalid_movie_ids = movie_df[movie_df['movieId'].apply(lambda x: not str(x).isdigit())]
print("Invalid movieId entries:")
print(invalid_movie_ids)

# 移除非法行
movie_df = movie_df[movie_df['movieId'].apply(lambda x: str(x).isdigit())]
movie_df['movieId'] = movie_df['movieId'].astype(int)

# 安全转换日期列，处理无效日期
def safe_convert_dates(movie_df, date_column='release_date'):
    movie_df[date_column] = pd.to_datetime(movie_df[date_column], errors='coerce')
    return movie_df

# 示例处理过程
movie_df = safe_convert_dates(movie_df, 'release_date')

# 检查转换后有多少无效日期被转换为NaT
num_invalid_dates = movie_df['release_date'].isna().sum()
print(f"Number of invalid dates converted to NaT: {num_invalid_dates}")

# 删除包含NaT的行
movie_df = movie_df.dropna(subset=['release_date'])


'''
Embedding训练
'''
# 处理电影数据
processed_movies = preprocess_movie_data(movie_df)
# 合并电影数据和用户评分数据
final_df = ratings_df.merge(processed_movies, on="movieId")
# 确保日期列是 datetime 类型
final_df['release_date'] = pd.to_datetime(final_df['release_date'], errors='coerce')
# 转换日期为时间戳 (单位: 秒)
final_df['release_date_timestamp'] = final_df['release_date'].astype('int64') // 10**9


'''
模型一
'''
# 编码用户ID和电影ID
encoder_user = LabelEncoder()
encoder_movie = LabelEncoder()

final_df['user'] = encoder_user.fit_transform(final_df['userId'].values)
final_df['movie'] = encoder_movie.fit_transform(final_df['movieId'].values)

# 数值特征归一化
numerical_features = ['budget', 'popularity', 'release_date_timestamp', 'revenue', 'runtime', 'vote_average', 'vote_count']
scaler = MinMaxScaler()
final_df[numerical_features] = scaler.fit_transform(final_df[numerical_features])

deep_features = [col for col in final_df.columns if 'vector' in col]
expand_vector_to_columns(final_df, deep_features)
# 更新 numerical_features 列表，包含所有展开后的向量列
numerical_features.extend([col for col in final_df.columns if 'vector_' in col])


# 组合所有特征
feature_columns = ['user', 'movie'] + numerical_features
X = final_df[feature_columns]
y = final_df['rating']

# 用平均值填充 NaN
for column in X.columns:
    X[column].fillna(X[column].mean(), inplace=True)
    
num_users = final_df['user'].nunique()
num_movies = final_df['movie'].nunique()
model = build_model(num_users, num_movies, len(numerical_features))


# 将用户和电影ID提取为单独的特征
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 检查NaN和无限值
print("NaN in training data:", X_train.isnull().any().any())
print("Inf in training data:", np.isinf(X_train).any().any())


#拟合模型
history = model.fit([X_train['user'], X_train['movie'], X_train[numerical_features]], y_train,
                    validation_split=0.1, epochs=10, batch_size=32)
#可视化
plot_history(history)

results1 = model.evaluate([X_test['user'], X_test['movie'], X_test[numerical_features]], y_test)
print("Results for Model 1:")
print(f"MSE: {results1[1]}, MAE: {results1[2]}")

#搜索最优参数
tuner = kt.RandomSearch(
    build_hypermodel,
    objective='val_mean_squared_error',
    max_trials=10,  # 超参数组合的数量
    executions_per_trial=1,  # 每个超参数组合运行的次数
    directory='my_dir',
    project_name='hyperparameter_tuning'
)

# 开始搜索
tuner.search([X_train['user'], X_train['movie'], X_train[numerical_features]], y_train,
             validation_split=0.1, epochs=10, batch_size=32)

# 获取最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best embedding size: {best_hps.get('embedding_size')}")
print(f"Best dense units 1: {best_hps.get('dense_units_1')}")
print(f"Best dense units 2: {best_hps.get('dense_units_2')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = build_best_model(best_hps)
history = best_model.fit([X_train['user'], X_train['movie'], X_train[numerical_features]],
                    y_train, epochs=10,  # 增加训练周期以获得更好的结果
                    batch_size=32,
                    validation_split=0.1)  # 保持验证集的比例
best_results = best_model.evaluate([X_test['user'], X_test['movie'], X_test[numerical_features]], y_test)
print(f"Test Loss: {best_results[0]}, Test MSE: {best_results[1]}")


# 获取调优结果
tuner.results_summary()
# 提取调优过程中所有试验的历史数据
trials = tuner.oracle.get_best_trials(num_trials=10)
# 可视化每个试验的验证损失
val_losses = [trial.metrics.get_best_value('val_mean_squared_error') for trial in trials]


plt.figure(figsize=(10, 6))
plt.plot(val_losses, 'bo-', label='Validation Loss')
plt.xlabel('Trial')
plt.ylabel('Validation Loss')
plt.title('Hyperparameter Tuning Results')
plt.legend()
plt.show()


# 定义K折交叉验证参数
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# 准备记录交叉验证结果
fold_no = 1
loss_per_fold = []
acc_per_fold = []

# KFold交叉验证循环
for train, test in kfold.split(X):
    # 创建模型
    model = build_best_model(best_hps)
    
    # 选择'用户'和'电影'作为输入
    train_users, test_users = X['user'].iloc[train], X['user'].iloc[test]
    train_movies, test_movies = X['movie'].iloc[train], X['movie'].iloc[test]
    train_features, test_features = X[numerical_features].iloc[train], X[numerical_features].iloc[test]
    
    # 训练模型
    history = model.fit([train_users, train_movies, train_features], y.iloc[train],
                        batch_size=32,
                        epochs=10,
                        verbose=1,
                        validation_data=([test_users, test_movies, test_features], y.iloc[test]))
    
    # 记录性能
    scores = model.evaluate([test_users, test_movies, test_features], y.iloc[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
    loss_per_fold.append(scores[0])
    acc_per_fold.append(scores[1])
    
    # 增加折数
    fold_no = fold_no + 1

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')



'''
模型二
'''
final_df = ratings_df.merge(processed_movies, on="movieId")
# 确保日期列是 datetime 类型
final_df['release_date'] = pd.to_datetime(final_df['release_date'], errors='coerce')
# 转换日期为时间戳 (单位: 秒)
final_df['release_date_timestamp'] = final_df['release_date'].astype('int64') // 10**9

# 示例：选择特征
#wide_features = ['movieId', 'userId','budget','popularity','release_date_timestamp','revenue','runtime','vote_average','vote_count']  # 加上其他类别特征
wide_features = ['movieId', 'userId','popularity','revenue','vote_average']  # 加上其他类别特征
deep_features = [col for col in final_df.columns if 'vector' in col]  # 假设包括所有的嵌入向量特征


X_wide = final_df[wide_features]
X_deep = final_df[deep_features]
y = final_df['rating']

# 应用这个函数
expand_vector_to_columns(X_deep, deep_features)


X_wide = X_wide.astype('float32')
X_deep = X_deep.astype('float32')
y = y.astype('float32')

# 检查并处理 NaN 或无限值
for column in X_wide.columns:
    X_wide[column].fillna(X_wide[column].mean(), inplace=True)
    
for column in X_deep.columns:
    X_deep[column].fillna(X_deep[column].mean(), inplace=True)
# 示例代码：检查NaN和无限值
print("NaN in X_wide data:", X_wide.isnull().any().any())
print("Inf in X_wide data:", np.isinf(X_wide).any().any())
print("NaN in X_deep data:", X_deep.isnull().any().any())
print("Inf in X_deep data:", np.isinf(X_deep).any().any())
# 划分数据集
X_wide_train, X_wide_test, X_deep_train, X_deep_test, y_train, y_test = train_test_split(X_wide, X_deep, y, test_size=0.2, random_state=42)

model = build_model2()



history = model.fit([X_wide_train, X_deep_train], y_train, epochs=10, batch_size=32, validation_split=0.2)


#寻找最优超参数
tuner = RandomSearch(
    build_hypermodel2,
    objective='val_mean_squared_error',
    max_trials=10,  
    executions_per_trial=1,  
    directory='model_tuning',
    project_name='wide_and_deep_tuning'
)

tuner.search([X_wide_train, X_deep_train], y_train,
             epochs=10,
             validation_data=([X_wide_test, X_deep_test], y_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best dense units 1: {best_hps.get('dense_units_1')}")
print(f"Best dense units 2: {best_hps.get('dense_units_2')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")
model = build_best_model2(best_hps)

#进行K折验证
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_no = 1
loss_per_fold = []
mse_per_fold = []
for train_index, test_index in kf.split(X_wide):
    X_wide_train, X_wide_test = X_wide.iloc[train_index], X_wide.iloc[test_index]
    X_deep_train, X_deep_test = X_deep.iloc[train_index], X_deep.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = build_best_model2(best_hps)

    history = model.fit([X_wide_train, X_deep_train], y_train,
                        epochs=5,
                        batch_size=32,
                        verbose=1,
                        validation_data=([X_wide_test, X_deep_test], y_test)) 

    scores = model.evaluate([X_wide_test, X_deep_test], y_test, verbose=0)
    print(f'Test Score for fold {fold_no}: Loss = {scores[0]}, MSE = {scores[1]}')
    loss_per_fold.append(scores[0])
    mse_per_fold.append(scores[1])
    fold_no += 1


print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(num_folds):
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - MSE: {mse_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> MSE: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


y_pred = model.predict([X_wide_test, X_deep_test])
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Results for Model 2:")
print(f"MSE: {mse}, MAE: {mae}")








