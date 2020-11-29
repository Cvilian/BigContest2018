#시스템 구성==============================================================
*본 프로그램이 동작할 수 있는 환경 및 필요한 라이브러리는 아래와 같습니다.
└=== Python 3.7
└=== Tensorflow-gpu 1.14.0
└=== Numpy 1.16.3
└=== Scikit-learn
└=== Pandas

*소스 코드는 3가지이며 각각이 수행하는 일은 다음과 같습니다.
└=== preprocess.py : raw 폴더 내에 있는 원본 데이터를 전처리하는 코드
└=== model.py : 전처리된 학습용 데이터를 불러와 분류 학습을 수행하고 완성된 모델을 저장하는 코드 
└=== predict.py : model 폴더에 저장된 학습된 모델 parameter와 테스트용 데이터를 불러와 생존기간 및 예상 지출 금액을 예측하는 코드

#데이터 준비==============================================================
*raw 폴더 내에는 다음과 같은 데이터 파일들이 들어가 있어야 합니다.
└===[name]_activity.csv
└===[name]_combat.csv
└===[name]_payment.csv
└===[name]_pledge.csv
└===[name]_trade.csv

#전처리 과정==============================================================
*전처리를 통해 학습에 필요한 feature들을 선별하고 생존기간 및 예상 지출금액을 예측하고자 하는 유저들의 데이터를 가공합니다.
*해당 유저들의 목록은 [name]_activity 파일에서 참조하며 모든 데이터 셋에서 공통적으로 캐릭터 아이디/서버 등의 정보는 활용하지 않습니다.
*각각의 데이터 파일로부터 추출하는 특성 정보(총 46개)는 아래와 같습니다.
└===[name]_activity.csv : playtime/npc_kill/solo_exp/party_exp/quest_exp/rich_monster/death/revive/exp_recovery/
							fishing/private_shop/game_money_change/enchant_count
└===[name]_combat.csv :  level/pledge_cnt/random_attacker_cnt/random_defender_cnt/temp_cnt/same_pledge_cnt/
							etc_cnt/num_opponent/캐릭터 직업 : 군주(1 : yes/0: no)/캐릭터 직업 : 군주(1 : yes/0: no)/캐릭터 직업 : 기사(1 : yes/0: no)/
							캐릭터 직업 : 요정(1 : yes/0: no)/캐릭터 직업 : 마법사(1 : yes/0: no)/캐릭터 직업 : 다크엘프(1 : yes/0: no)/
							캐릭터 직업 : 용기사(1 : yes/0: no)/캐릭터 직업 : 환술사(1 : yes/0: no)/캐릭터 직업 : 전사(1 : yes/0: no)
└===[name]_payment.csv : 결제 금액
└===[name]_pledge.csv : play_char_cnt/combat_char_cnt/pledge_combat_cnt/random_attacker_cnt/random_defender_cnt/
							same_pledge_cnt/temp_cnt/etc_cnt/combat_play_time/non_combat_play_time
└===[name]_trade.csv : 준 adena 거래량/받은 adena 거래량/준 아이템(adena 이외의 모든)의 수량/받은 아이템(adena 이외의 모든)의 수량/
						   준 아이템(adena 이외의 모든)의 거래가격/받은 아이템(adena 이외의 모든)의 거래가격

* 날짜별로 특정 유저가 접속한 캐릭터들의 각 특성값들을 집계하여 최소/최대/평균/표준편차(전체 46 x 4 = 184)와 같은 통계치들을 얻어내었고 이들을
   한 유저가 28일간 활동한 총체적인 로그로 간주하였습니다.
*데이터 셋 별로 각 특성값의 분포가 다르기에 추가적인 Normalization을 수행하지 않았습니다.
* 전처리하는 코드는 아래 명령어를 통해 실행시킬 수 있습니다.
└===$ python preprocess.py [name]

* 학습용 데이터의 이름은 반드시 'train'으로 시작되어야 하며, 레이블 데이터(ex train_label.csv)가 같은 폴더 내에 존재해야만 합니다.

#학습 과정===============================================================
* 분류모델은 Recurrent Neural Network 계열의 LSTM (Long short-term memory)를 활용하였습니다.
* 이 모델은 로그가 기록된 날짜만큼인 28개의 cell로 구성되어 각 날짜별로 사전에 전처리한 특성값들을 input vector로 받아들입니다.
└=== Input size : 유저 수 x 28 x 184

*Loss 함수는 주최측에서 주어진 score_function을 참고하여 만들었습니다. 분류 모델은 2가지를 예측하게 되는데요,
└=== A : 예상 생존 시간 + 예상 지출 금액 - 1
└=== B : 예상 지출 금액
* 이 2가지 output에서 예상 생존 시간은 (A - B +1) 로 구하며 예상 지출 금액은 B로 구할 수 있습니다.
* 굳이 A를 '예상 생존 시간'로 정의하지 않은 이유는 생존 시간의 단위가 '일'이라 discrete한 range를 가지기 때문입니다
(학습의 효율성을 위해 예상해야할 값을 discrete domain에서 continous domain으로 옮기는 방안으로 예상 지출 금액을 더한 값을 모델이 예측하도록 설계하였습니다.)
* A 에서 '1'을 뺸 것은 예상 생존 시간이 1~64의 범위를 가지기에 0~63으로 평행이동시켜주기 위함입니다. (물론 최종 답안을 낼 시에는 +1을 합니다)

*Loss 함수의 정의는 
└=== | (max_s - pre_s)/C | 
└=== max_s : 예측이 100% 맞아 떨어졌을 때(예상 생존기간 및 예상 지출 금액을 완전히 동일한 값으로 예측할시/score_function에서의 두 parameter를 모두 actual_label로 두었을 때)의 score
└=== pre_s : 예측한 생존기간/지출금액과 실제 생존시간/지출금액을 토대로 구한 score
└=== C : 상수

*max_s 및 pre_s를 구하는 데 쓰인 score 함수는  score_function.py에 나와있는 함수와 비슷하나, score가 0또는 nan 값을 가지는 경우을 방지하기 위해
tensor버전으로 구현하는 도중에 일부 수식을 수정하였습니다. 
└=== LSTM 마지막 레이어에 relu activation을 추가해 예측값이 음이 아닌 실수가 되도록 설계
└=== cost나 optimal_cost 값이 '0'이 될 경우 이후 score 값이 0이 되어버려 학습 진행이 더뎌지므로 이 둘에 상수 10^-6 을 더하고 day_pred 또는 day_true를 64와 비교하는 부분을 삭제
└=== T_k가 '0'이 되어 pre_s나 max_s가 음수가 되는 상황을 방지하기 위해 day_pred 및 day_true를 비교하는 부분을 삭제

*분류 모델 성능 평가 지표는 기존의 score_function 함수를 그대로 사용합니다. 물론 이 지표는 실제 예측시에 어느 정도에 최종 score를 획득할 수 있을지 검증하기 위해 쓰였으며 학습에는 아무런 영향을 끼치지 않습니다.
* 실제 데이터에서 예상 지출 금액이 '0'인 케이스가 많아 학습이 비효율 적일 수 있으므로 0~10^-4 사이의 난수를 더하여 noise를 발생시킵니다.

*전처리한 데이터를 총 200 epoch로 학습을 시키고 모델 정보를 "model-RNNv.h5" 이름의 파일로 model 폴더 안에 저장합니다.
* 학습 코드는 다음과 같이 실행시킬 수 있습니다.
└===$ python model.py [name]

#최종 답안 예측==============================================================
*전처리한 데이터와 학습된 모델 파일을 불러와 최종 답안을 예측합니다 (답안을 도출하는 식은 #학습 과정의 2번째 문단 참고).
* 예측된 결과물을 [name]_predict.csv 와 같은 형식으로 저장합니다.
* 예측 코드는 다음과 같이 실행시킬 수 있습니다.
└===$ python predict.py [name]




