import pandas as pd

df = pd.read_csv('congestion_data.csv')
df2 = pd.read_csv('nohan_congest.csv')

new_df = df[df["congestionCar1"] != 0]
new_df2 = df2[df2["congestionCar1"] != 0]

new_df.reset_index(drop=True, inplace=True)
new_df2.reset_index(drop=True, inplace=True)

selected_columns = [
    'updnLine',
    'hh',
    'mm',
    'congestionCar1',
    'congestionCar2',
    'congestionCar3',
    'congestionCar4',
    'congestionCar5',
    'congestionCar6',
    'congestionCar7',
    'congestionCar8',
    'congestionCar9',
    'congestionCar10',
]

new_df_selected = new_df[selected_columns]
new_df_selected2 = new_df2[selected_columns]

df_line_1 = new_df_selected[new_df_selected['updnLine'] == 1]
df_line_0 = new_df_selected[new_df_selected['updnLine'] == 0]

df2_line_1 = new_df_selected2[new_df_selected2['updnLine'] == 1]
df2_line_0 = new_df_selected2[new_df_selected2['updnLine'] == 0]

# 수요일, 18~20시
# print(df_line_1)
# 전체 요일, 18시
# print(df2_line_1)

hh_mm_data = df_line_1[(df_line_1['hh'] == 18) & (df_line_1['mm'] == 30)].copy()
hh_mm_data.reset_index(drop=True, inplace=True)
hh_mm_data2 = df2_line_1[(df2_line_1['hh'] == 18) & (df2_line_1['mm'] == 30)].copy()
hh_mm_data2.reset_index(drop=True, inplace=True)

df1_nohan = hh_mm_data[['congestionCar1', 'congestionCar2', 'congestionCar3', 'congestionCar4', 'congestionCar5',
                         'congestionCar6', 'congestionCar7', 'congestionCar8', 'congestionCar9', 'congestionCar10']]
df2_nohan = hh_mm_data2[['congestionCar1', 'congestionCar2', 'congestionCar3', 'congestionCar4', 'congestionCar5',
                         'congestionCar6', 'congestionCar7', 'congestionCar8', 'congestionCar9', 'congestionCar10']]

base_list = []
for i in range(df1_nohan.shape[0]):
    first_club = list(df1_nohan.iloc[i])
    base_list.append(first_club)
print('처음 남은 진짜 데이터 : ', base_list)

# ------------- 실제 데이터를 상관계수를 위해 하나로 합치기 ---------------
df_nohan0 = pd.DataFrame({
    'congestionCar1': [df1_nohan['congestionCar1'].mean()] ,
    'congestionCar2': [df1_nohan['congestionCar2'].mean()] ,
    'congestionCar3': [df1_nohan['congestionCar3'].mean()] ,
    'congestionCar4': [df1_nohan['congestionCar4'].mean()] ,
    'congestionCar5': [df1_nohan['congestionCar5'].mean()] ,
    'congestionCar6': [df1_nohan['congestionCar6'].mean()] ,
    'congestionCar7': [df1_nohan['congestionCar7'].mean()] ,
    'congestionCar8': [df1_nohan['congestionCar8'].mean()] ,
    'congestionCar9': [df1_nohan['congestionCar9'].mean()],
    'congestionCar10':[df1_nohan['congestionCar10'].mean()]
})
# ------------- 실제 데이터를 상관계수를 위해 하나로 합치기 ---------------

df_nohan0_list = list(df_nohan0.iloc[0])
print('진짜 리스트', df_nohan0_list)


df2_nohan_list = []

for i in range(df2_nohan.shape[0]):
    abc = df2_nohan.iloc[i]
    inner_list = list(abc)
    df2_nohan_list.append(inner_list)

print('후보 리스트', df2_nohan_list)

# ----------------- 차원축소 및 유클리드 거리 -----------------------
import numpy as np

standard_data = np.array(df_nohan0_list)
candidates = np.array(df2_nohan_list)

# 유클리드 거리 함수
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Standard Scaler를 사용하여 데이터를 표준화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
all_data = np.vstack((standard_data, candidates))
all_data_standardized = scaler.fit_transform(all_data)

# PCA를 사용하여 차원을 축소
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
all_data_pca = pca.fit_transform(all_data_standardized)

# 차원 축소된 기준 및 후보 데이터 분할
standard_data_pca = all_data_pca[0]  # 차원 축소된 기준 데이터
candidates_pca = all_data_pca[1:]    # 차원 축소된 후보 데이터

# 차원 축소된 데이터에 대한 거리 계산
distances = [euclidean_distance(standard_data_pca, candidate) for candidate in candidates_pca]
min_idx = distances.index(min(distances))

filtered_candidates = [candidates[index] for index, distance in enumerate(distances) if distance <= 4]
print('유클리드 거리 :',distances)

print("유클리드 거리가 4 이하인 후보 데이터:")
filter_lis = []
for candidate in filtered_candidates:
    print(candidate)

    abdc = list(candidate)
    filter_lis.append(abdc)
print('최종 : ',filter_lis)
# ----------------- 유클리드 거리 -----------------------

filter_list = pd.DataFrame(filter_lis, columns=['congestionCar1', 'congestionCar2', 'congestionCar3', 'congestionCar4', 'congestionCar5',
                         'congestionCar6', 'congestionCar7', 'congestionCar8', 'congestionCar9', 'congestionCar10'])

print(filter_list)
# #--------------------------- df1_nohan과 df2_nohan을 합치기 -------------------------
merged_df = pd.concat([df1_nohan, filter_list], axis=0)
merged_df.reset_index(drop=True, inplace=True)
print('진짜 최종 데이터 : ', merged_df)
# #--------------------------- df1_nohan과 df2_nohan을 합치기 -------------------------

# # -------------- 7일 간격으로 '2023-04-26'부터 12개 날짜 생성 -------------------------
dates = pd.date_range(start='2023-05-24', periods=8, freq='7D')
# # -------------- 7일 간격으로 '2023-04-26'부터 12개 날짜 생성 -------------------------

# #-------------- 'date' 컬럼을 추가하여 데이터프레임에 저장 ------------------------
merged_df['date'] = dates
print(merged_df)
# #-------------- 'date' 컬럼을 추가하여 데이터프레임에 저장 ------------------------
from fbprophet import Prophet

# 결과를 저장할 빈 데이터프레임 생성
predictions = pd.DataFrame()

# 각 컬럼에 대해 반복하면서 예측을 수행
for car in range(1, 11):
    # 혼잡도 데이터를 데이터프레임에 저장
    car_df = merged_df[['date', f'congestionCar{car}']]
    car_df = car_df.rename(columns={'date': 'ds', f'congestionCar{car}': 'y'})

    # Prophet 예측 모델 생성
    model = Prophet()
    model.fit(car_df)

    # 미래 날짜 생성
    future_dates = model.make_future_dataframe(periods=1, freq='7D')

    # 예측 계산
    forecast = model.predict(future_dates)

    # 예측 결과를 저장
    predictions[f'congestionCar{car}'] = forecast['yhat']

# '2023-07-19' 날짜의 예측 결과 출력
pred_0719_1830 = predictions.iloc[-1]
print(pred_0719_1830)

import os

# 이 경로를 코틀린 프로젝트가 있는 경로로 설정.
target_directory = "테스트 중"

# 필요하다면, "data" 폴더가 없으면 만들어 줍니다.
os.makedirs(target_directory, exist_ok=True)

with open(os.path.join(target_directory, "pred_0719_1830.txt"), "w") as file:
    file.write(pred_0719_1830.to_string())





