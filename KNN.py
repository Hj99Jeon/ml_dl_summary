import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# fish_data = [[l, w] for l, w in zip(length, weight)]
fish_data = np.column_stack((fish_length, fish_weight)) #2열로 합침.
# fish_target = [1]*35+[0]*14
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# input_arr = np.array(fish_data)
# target_arr = np.array(fish_target)

# np.random.seed(42) #일정하게 랜덤
# index = np.arange(49) #랜덤하게 할 인덱스
# np.random.shuffle(index)
#
# train_input = input_arr[index[:35]]
# train_target = target_arr[index[:35]]
#
# test_input = input_arr[index[35:]]
# test_target = target_arr[index[35:]]

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

mean = np.mean(train_input, axis=0) #평균 계산
std = np.std(train_input, axis=0) #편차 계산
train_scaled = (train_input-mean)/std #표준 점수
test_scaled = (test_input-mean)/std #표준 점수
new = ([25, 150] - mean)/std

kn = KNeighborsClassifier() #n_neighbor로 참고할 주변부 갯수를 정할 수 있음. default=5
kn.fit(train_scaled, train_target) #모델 학습
kn.score(test_scaled, test_target) #모델 검사 0~1
print(kn.predict([new]))

distances, indexes = kn.kneighbors([new]) #해당 점에서 가장 가까운 것의 거리, 인덱스를 반환

plt.scatter(train_scaled[:,0], train_scaled[:,1]) #산점도
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()