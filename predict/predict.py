# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:48:45 2019

@author: Yangjiwon
"""

import csv
import sys
import math
import numpy as np
from tensorflow.keras import models
sys.path.append('../model/')
from model import custom_loss, score_N, score_O

preprocess_dir = '../preprocess/'
model_dir = '../model/'
dataset_name = sys.argv[1]

# 전처리한 data 불러오는 함수
# attributes : feature의 개수
def get_data(attributes) :
    print("Read whole data..")
    # x : feature 데이터
    # A : acc_id 저장
    x = []; A = []
    with open(preprocess_dir+'/'+dataset_name+"_preprocess.csv", 'r', encoding='utf-8') as f :
        rdr = csv.reader(f)
        for line in rdr :
            A.append(int(line[0]))
            x.append([float(i) for i in line[3:]])
    x = np.array(x, dtype=np.float64)
    return x, A

# 결과를 예측하는 함수
# model : 학습한 모델
# x_d : feature 데이터
# A_d : acc_id 데이터
# fname : 저장할 파일 이름
def predict_model(model, x_d, A_d, fname):
    x_d = np.reshape(x_d, (-1, 28, 184))
    y_pred = model.predict(x_d, batch_size=200)
    with open(fname, 'w', encoding='utf-8', newline='') as f :
        wr = csv.writer(f)
        wr.writerow(['acc_id', 'survival_time', 'amount_spent'])
        for i in range(len(A_d)) :
            wr.writerow([A_d[i], math.floor(max(0.0, min(y_pred[i][0] - y_pred[i][1], 64-1e-6))) + 1, max(0.0, y_pred[i][1])])
    
if __name__ == '__main__' :
    x_d, A_d = get_data(28*184)
    rnn_T = models.load_model(model_dir+"model-RNNv.h5", compile=False)
    rnn_T.compile(optimizer='Adam',
                  loss=custom_loss(),
                  metrics = [score_O])
    predict_model(rnn_T, x_d, A_d, dataset_name+'_predict.csv')