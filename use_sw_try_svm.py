# coding:utf-8

import os
import time
import numpy as np
import cv2
from selectivesearch import selective_search as ss
from sklearn import cluster, preprocessing, mixture

import matplotlib.pyplot as plt

# 今回はmin_window_sizeを定数にし、正方形の枠のみ生成する
def sliding_windows(min_window_size, img_size, win_step, step):
	regions = []
	first_win_size =  min(img_size[0], img_size[1])
	for ws in range(first_win_size, min_window_size, win_step * -1):
		regions.extend(sliding_window((ws, ws), img_size, step))
	return regions


def sliding_window(window_size, img_size, step):
	regions = []
	for y in range(0, img_size[0], step):
		for x in range(0, img_size[1], step):
			regions.append({"rect": (x, y, window_size[0], window_size[1])})
	return regions


def meanshift(data):
	x = data
	sc = preprocessing.StandardScaler()
	sc.fit(x)
	x_norm = sc.transform(x)
	ms = cluster.MeanShift(seeds=x_norm)
	ms.fit(x_norm)
	for i in x_norm:
		plt.plot(i[0], i[1], ".")
	for i in ms.cluster_centers_:
		plt.plot(i[0], i[1], "o")
	plt.show()
	return ms.cluster_centers_	

start_time = time.time()

# HOGをとるためのパラメータ
win_size = (60, 60)
block_size = (16, 16)
block_stride = (4, 4)
cell_size = (4, 4)
bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

# 検出したい画像を選択
target_path = "/mnt/c/Users/user/Documents/dataset/dataset_favorite/"
target = os.listdir(target_path)[0]
print(target)

# 作成したSVMをロード
svm = cv2.ml.SVM_load("train.xml")

# HOG用とselectivesearch用に2種類用意
img_bgr = cv2.imread(target_path + target)
img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2GRAY)

img_load_time = time.time()
print(f"img_load:{img_load_time - start_time}")

# selectivesearchを実行
# img_lblには範囲で区切られている画像データが入っている
# regionsはselectivesearchで得られるrectの情報とか
#img_lbl, regions = ss(img_bgr, scale=500, sigma=0.9, min_size=100)
# 今回はSliding windowを用いる

min_window_size = 200
img_size = img_bgr.shape
win_step = 40
step = 40

regions = sliding_windows(min_window_size, img_size, win_step, step)

ss_time = time.time()
print(f"ss:{ss_time - img_load_time}")

hog_list = []
hog_list_rect = []
almosit_regions = len(regions)

# HOGの結果をリストにまとめる
for index, i in enumerate(regions):
	x, y, w, h = i["rect"]
	# rectの範囲が小さいものを排除
	if h < win_size[1] or w < win_size[0]:
		continue
	print(f"hog: {index + 1}/{almosit_regions}")
	img_temp = cv2.resize(img[y:y+h, x:x+w].copy(), win_size)
	hog_list.append(hog.compute(img_temp))
	hog_list_rect.append((x, y, w, h))

# SVMにHOGを渡し、検出する
result = svm.predict(np.array(hog_list))[1].ravel()

hog_and_svm_time = time.time()
print(f"hog_and_svm:{hog_and_svm_time - ss_time}")

print(result)

# 0、つまりは検出できたもののrectを保管
recog_face_rect = [hog_list_rect[index] for index, i in enumerate(result) if i == 0.]

# 赤色
color = (0, 0, 255)

# 元画像の検出できた先にrectを描画
if recog_face_rect:
	centroid_list = []
	for i in recog_face_rect:
		x, y, w, h = i
		centroid_list.append([y + h / 2, x + w / 2])
		#cv2.rectangle(img_bgr, (x,y), (x+w, y+h), color, thickness=8)
	print(meanshift(centroid_list))
	exit()
cv2.imwrite(f"./ppp.jpg", img_bgr)

end_time = time.time()
print(f"end:{end_time - hog_and_svm_time}")

print(f"total:{end_time - start_time}")
