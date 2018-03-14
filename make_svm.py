# coding:utf-8

import os
import numpy as np
import cv2

# HOGをとるためのパラメータ
win_size = (60, 60)
block_size = (16, 16)
block_stride = (4, 4)
cell_size = (4, 4)
bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

# HOGを計算し、データとラベルのリストを返す関数
def calc_hog():
	train_and_labels = []
	for file in os.listdir("./dataset/pos/"):
		img = cv2.imread(f"./dataset/pos/{file}", 0)
		img = cv2.resize(img, win_size)
		train_and_labels.append((hog.compute(img), 0))

	for file in os.listdir("./dataset/neg/"):
		img = cv2.imread(f"./dataset/neg/{file}", 0)
		img = cv2.resize(img, win_size)
		train_and_labels.append((hog.compute(img), 1))
	
	# 順番をランダムにする処理
	# もっといい方法がありそう
	np.random.shuffle(train_and_labels)
	train = []
	label = []
	for train_and_label in train_and_labels:
		train.append(train_and_label[0])
		label.append(train_and_label[1])

	return np.array(train), np.array(label)

if __name__ == "__main__":
	train, label = calc_hog()

	# SVMを新規作成
	svm = cv2.ml.SVM_create()
	# カーネル関数設定
	svm.setKernel(cv2.ml.SVM_LINEAR)
	# パラメータCを決定
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setC(0.5)
	# 訓練
	svm.train(train, cv2.ml.ROW_SAMPLE, label)
	svm.save('train.xml')
