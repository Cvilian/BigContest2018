# -*- coding: utf-8 -*
"""
Created on Mon Dec  2 16:06:35 2019

@author: Yangjiwon
"""

import sys
import csv
import numpy as np
import pandas as pd

D = 28      # 활동 데이터의 날짜 범위 
FS = []     # 주어진 Feature 항목         

path_dir = '../raw/'
dataset_name = sys.argv[1]

class User:
    def __init__(self, acc_id):
        # acc_id : 계정 아이디
        # features : 유저 별 해당 날짜에 나타난 features
        # survival_time : 생존시간(Training Data만)
        # amount_spent : 결제금액(Training Data만)
        self.acc_id = acc_id
        self.features = {}
        self.survival_time = 0
        self.amount_spent = 0.0
        
    # Day 'i' 에 해당 유저가 소유한 캐릭터의 특정 feature 값 저장
    # feature_name : 추출한 feature 항목
    # day : feature가 나타난 날짜
    # val : feature 값
    def append_feature(self, feature_name, day, val):
        if feature_name not in self.features :
            # 해당 유저에 대한 feature 값이 최초로 등장했을 경우
            self.features[feature_name] = [[] for i in range(D)]
        self.features[feature_name][day].append(val)

    # 각 날짜마다 유저의 캐릭터 별 feature 통계 정보를 집계 : 최소값/최대값/평균/표준편차
    # Output은 날짜별 feature 정보를 통합한 list : D x #features x 4 (metrics : min/max/mean/std) = 28 x 46 x 4    
    def cal_stat(self) :
        S = [[] for i in range(D)]
        for fs in FS :
            if fs not in self.features :
                # 해당 유저에 대한 feature 'fs' 기록이 없을 경우 그 feature에 대해 최소값/최대값/평균/표준편차을 0.0으로 설정
                for i in range(D):
                    S[i] = S[i] + [0.0]*4
                continue
            for i in range(D) :
                tp = [0.0]*4
                if self.features[fs][i] :
                    # 해당 날짜의 feature별 min/max/mean/std
                    tp[0] = np.min(self.features[fs][i])
                    tp[1] = np.max(self.features[fs][i])
                    tp[2] = np.mean(self.features[fs][i])
                    tp[3] = np.std(self.features[fs][i])
                S[i] = S[i] + tp
        res = [S[i][j] for i in range(D) for j in range(4*len(FS))]
        return res
    
    # Training 시 예측 값과 실제 값 비교를 위한 labeling
    # sut : train_label.csv 에 주어진 생존시간
    # ams : train_label.csv 에 주어진 결제금액
    def labeling(self, sut, ams) :
        self.survival_time = sut
        self.amount_spent = ams
        
        
# 생존 기간 및 지출을 예측할 유저 탐색
# data : 데이터 파일
def get_Users(data) :
    print("Initialize users..")
    Users = {}
    df = pd.read_csv(data, sep=",", dtype='unicode')
    acc = df['acc_id'].unique()
    for a in acc :
        Users[int(a)] = User(int(a))
    return Users

# 파일 별로 feature 읽어들이고 저장
# data : 데이터 파일
# Users : 유저 정보
# elements : feature에 해당하는 열 번호
def read_features(data, Users, elements) :
    global FS
    print("Read : ", data)
    df = pd.read_csv(data, sep=",", dtype='unicode')
    hds = [df.columns[e] for e in elements]
    
    print("Extract : ", data)
    FS = FS + hds
    fv = df[['acc_id', 'day'] + hds]
    
    # 유저마다 feature 추가
    for index, row in fv.iterrows():
        acc_id = int(row['acc_id'])
        day = int(row['day']) - 1
        if acc_id not in Users :
            continue
        for e in hds :
            Users[acc_id].append_feature(e, day, float(row[e]))

# combat 파일로부터 유저마다 그날 접속한 직업의 비중 0~7 을 따로 Ncombat에 저장
# data : 데이터 파일
def combat_modify(data):
    print("Modify : ", data)
    df = pd.read_csv(data, sep=",", dtype='unicode')
    with open(path_dir+dataset_name+"_Ncombat.csv", 'w', encoding='utf-8', newline='') as f :
        wr = csv.writer(f)
        hds = [i for i in df.columns[:4]]+[i for i in df.columns[5:]]
        wr.writerow(hds+["class"+str(i) for i in range(8)])
        for index, row in df.iterrows():
            c = int(row['class'])
            wr.writerow([row[e] for e in hds] + [int(i == c) for i in range(8)])

# trade 파일로부터 유저마다 그날 교환한 Adena, 그리고 이외의 물건의 갯수 및 가격을 따로 Ntrade에 저장 (교환한 기록/받은 기록 분리)
# data : 데이터 파일
def trade_modify(data):
    print("Modify : ", data)
    df = pd.read_csv(data, sep=",", dtype='unicode')
    df = df.fillna(0.0)
    with open(path_dir+dataset_name+"_Ntrade.csv", 'w', encoding='utf-8', newline='') as f :
        wr = csv.writer(f)
        wr.writerow(['day', 'acc_id', 'char_id', 'server', 'Sitem_amount', 'Sitem_price',
                     'Bitem_amount', 'Bitem_price', 'Sadena_amount', 'Badena_amount'])
        for index, row in df.iterrows():
            it = str(row['item_type'])
            if 'adena' == it:
                # 아데나의 경우 교환한 아데나 교환량만 저장
                wr.writerow([row['day'], row['source_acc_id'], row['source_char_id'], row['server'],
                             0.0, 0.0, 0.0, 0.0, row['item_amount'], 0.0])
                wr.writerow([row['day'], row['target_acc_id'], row['target_char_id'], row['server'],
                             0.0, 0.0, 0.0, 0.0, 0.0, row['item_amount']])
            else :
                # 그 외 물건인 경우 교환한 아이템 갯수 및 가격 저장
                wr.writerow([row['day'], row['source_acc_id'], row['source_char_id'], row['server'],
                             row['item_amount'], row['item_price'], 0.0, 0.0, 0.0, 0.0])
                wr.writerow([row['day'], row['target_acc_id'], row['target_char_id'], row['server'],
                             0.0, 0.0, row['item_amount'], row['item_price'], 0.0, 0.0])

# Training data를 위한 라벨링
# data : 데이터 파일
# Users : 유저 정보
def labeling(data, Users):
    print("Labeling..")
    df = pd.read_csv(data, sep=",", dtype='unicode')
    for index, row in df.iterrows():
        Users[int(row['acc_id'])].labeling(int(row['survival_time']), float(row['amount_spent']))

# preprocessed 된 데이터를 저장
# Users : 유저 정보
def make_data(Users) :
    print("Read labeling..")
    if dataset_name == 'train' :
        labeling(path_dir+dataset_name+"_label.csv", Users)
    print("Preprocessing..")
    with open(dataset_name+"_preprocess.csv", 'w', encoding='utf-8', newline='') as f : 
        wr = csv.writer(f)
        for acc_id, u in Users.items():
            S = u.cal_stat()
            wr.writerow([acc_id, u.survival_time, u.amount_spent]+S)

if __name__ == '__main__' :
    combat_modify(path_dir+dataset_name+"_combat.csv")
    trade_modify(path_dir+dataset_name+"_trade.csv")
    Users = get_Users(path_dir+dataset_name+'_activity.csv')
    read_features(data=path_dir+dataset_name+'_activity.csv',
                  Users=Users, 
                  elements=list(range(4,17)))
    read_features(data=path_dir+dataset_name+'_Ncombat.csv',
                  Users=Users, 
                  elements=list(range(4,20)))
    read_features(data=path_dir+dataset_name+'_payment.csv',
                  Users=Users, 
                  elements=list(range(2,3)))
    read_features(data=path_dir+dataset_name+'_pledge.csv',
                  Users=Users, 
                  elements=list(range(5,15)))
    read_features(data=path_dir+dataset_name+'_Ntrade.csv',
                  Users=Users, 
                  elements=list(range(4,10)))
    make_data(Users)