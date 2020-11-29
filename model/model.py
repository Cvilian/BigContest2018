# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 01:23:37 2019

@author: Yangjiwon
"""

import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 가장 좋았던 모델 : LSTM_D2
path_dir = '../preprocess'
dataset_name = sys.argv[1]

# 전처리된 데이터를 불러들이는 함수
# A : acc_id를 저장하는 행렬
# x : feature data를 저장하는 행렬
# D : Label을 저장하는 행렬(학습 속도 개선을 위해 수치 조정을 일부 수행)
def get_data() :
    print("Read whole data..")
    x = []; A = []; D = []
    with open(path_dir+'/'+dataset_name+"_preprocess.csv", 'r', encoding='utf-8') as f :
        rdr = csv.reader(f)
        for line in rdr :
            a = (float(line[1]) - 1)
            b = float(line[2]) + random.random()/10000
            D.append([a+b, b])
            A.append(int(line[0]))
            x.append([float(i) for i in line[3:]])
    x = np.array(x, dtype=np.float64)
    D = np.array(D, dtype=np.float64)
    return x, A, D

# Loss 함수
# max_s : 예측이 100% 맞아 떨어졌을 때 가질 수 있는 score (score_function의 두 parameter를 모두 actual_label로 두었을 때)
# pre_s : 예측한 생존기간/지출금액을 토대로 구한 score
# 두 점수의 차이의 절대값에다가 상수 'B'를 나눠 loss를 계산
def custom_loss():

    def loss(y_true, y_pred):
        max_s = score_N(y_true, y_true)
        pre_s = score_N(y_true, y_pred)
        B = tf.constant(200, dtype=tf.float64)
        return tf.abs(max_s - pre_s)/B
    return loss

# 기존 score_function을 학습에 용이하게 (loss가 nan, 또는 0에 수렴하지 않도록) 변형한 버전
# y_true : actual_label
# y_pred : predict_label
def score_N(y_true, y_pred):
    S = tf.constant(30, dtype=tf.float64)
    alpha = tf.constant(0.01, dtype=tf.float64)
    L = tf.constant(0.1, dtype=tf.float64)
    sigma = tf.constant(15, dtype=tf.float64)
    day_true = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), tf.constant([0])))
    day_pred = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_pred), tf.constant([0])))
    cost_true = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), tf.constant([1])))
    cost_pred = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_pred), tf.constant([1])))
    
    day_true = tf.dtypes.cast(day_true, dtype=tf.float64)
    day_pred = tf.dtypes.cast(day_pred, dtype=tf.float64)
    cost_true = tf.dtypes.cast(cost_true, dtype=tf.float64)
    cost_pred = tf.dtypes.cast(cost_pred, dtype=tf.float64)
    
    day_true = day_true - cost_true
    day_pred = day_pred - cost_pred
    
    cost = alpha*S*cost_pred + 1e-6
    optimal_cost = alpha*S*cost_true + 1e-6

    gamma = tf.where(condition = tf.less(cost/optimal_cost, L),
                     x = tf.zeros_like(cost, dtype=tf.float64),
                     y = tf.where(condition = tf.greater_equal(cost/optimal_cost, tf.constant(1.0, dtype=tf.float64)),
                                  x = tf.ones_like(cost, dtype=tf.float64),
                                  y = cost/((1-L)*optimal_cost) - L/(1-L)))

    T_k = S * tf.exp(-tf.square(day_pred - day_true)/(2*tf.square(sigma)))

    add_rev = T_k * cost_true
    profit = gamma * add_rev - cost
    score = tf.reduce_sum(profit)
    return score

# 주최측에서 공개한 score_function을 tensor 버전으로 구현한 것
# y_true : actual_label
# y_pred : predict_label
def score_O(y_true, y_pred):
    S = tf.constant(30, dtype=tf.float64)
    alpha = tf.constant(0.01, dtype=tf.float64)
    L = tf.constant(0.1, dtype=tf.float64)
    sigma = tf.constant(15, dtype=tf.float64)
    day_true = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), tf.constant([0])))
    day_pred = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_pred), tf.constant([0])))
    cost_true = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), tf.constant([1])))
    cost_pred = tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_pred), tf.constant([1])))
    
    day_true = tf.dtypes.cast(day_true, dtype=tf.float64)
    day_pred = tf.dtypes.cast(day_pred, dtype=tf.float64)
    cost_true = tf.dtypes.cast(cost_true, dtype=tf.float64)
    cost_pred = tf.dtypes.cast(cost_pred, dtype=tf.float64)
    
    day_true = day_true - cost_true
    day_pred = day_pred - cost_pred
       
    cost = tf.where(condition = tf.less(day_pred, tf.constant(63.0, dtype=tf.float64)),
                    x = alpha*S*cost_pred,
                    y = tf.zeros_like(day_pred, dtype=tf.float64))
    optimal_cost = tf.where(condition = tf.less(day_pred, tf.constant(63.0, dtype=tf.float64)),
                            x = alpha*S*cost_true,
                            y = tf.zeros_like(day_pred, dtype=tf.float64))

    gamma = tf.where(condition = tf.equal(optimal_cost, tf.constant(0.0, dtype=tf.float64)),
                     x = tf.zeros_like(cost, dtype=tf.float64),
                     y = tf.where(condition = tf.less(cost/optimal_cost, L),
                                  x = tf.zeros_like(cost, dtype=tf.float64),
                                  y = tf.where(condition = tf.greater_equal(cost/optimal_cost, tf.constant(1.0, dtype=tf.float64)),
                                               x = tf.ones_like(cost, dtype=tf.float64),
                                               y = cost/((1-L)*optimal_cost) - L/(1-L))))

    T_k = tf.where(condition = tf.greater_equal(day_pred, tf.constant(63.0, dtype=tf.float64)),
                   x = tf.zeros_like(day_pred, dtype=tf.float64),
                   y = tf.where(condition = tf.greater_equal(day_true, tf.constant(63.0, dtype=tf.float64)),
                                x = tf.zeros_like(day_pred, dtype=tf.float64),
                                y = S * tf.exp(-tf.square(day_pred - day_true)/(2*tf.square(sigma)))))

    add_rev = T_k * cost_true
    profit = gamma * add_rev - cost
    score = tf.reduce_sum(profit)
    return score

# 모델 생성 (Recurrent Neural Network 계열의 LSTM)
# max_seq_len : 주어진 활동 데이터의 기간(28)
# attributes : 각 날짜별 feature의 개수
# loss는 위에서 언급한 custom_loss()를 사용하고, metric은 기존 score_function을 사용
def create_LSTMmodel(max_seq_len, attributes) :
    model = models.Sequential([
            layers.LSTM(128, input_shape=(max_seq_len, attributes)),
            layers.Dense(2, activation='relu')
    ])
    model.summary()
    model.compile(optimizer='Adam',
                  loss=custom_loss(),
                  metrics = [score_O])
    return model

if __name__ == '__main__' :
    x_d, A_d, D_d = get_data()
    x_d = np.reshape(x_d, (-1, 28*184))
    
    # 데이터를 섞어줌
    ridx = list(range(len(A_d)))
    random.shuffle(ridx)

    tx_d = np.reshape([x_d[i] for i in ridx], (-1, 28, 184))
    tA_d = np.reshape([A_d[i] for i in ridx], (-1, ))
    tD_d = np.reshape([D_d[i] for i in ridx], (-1, 2))

    rnn_T = create_LSTMmodel(28, 184)
    rnn_T.fit(tx_d, tD_d, epochs=200, batch_size=200)
    rnn_T.save("model-RNNv.h5")