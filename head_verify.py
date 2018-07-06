#============================================================#
############# SHUJI HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#

import cv2
import numpy as np
from sklearn.externals import joblib
from fea_extract import fea_hog
import os
import sys


svm_persist_file = "./model/head_v21.svm"       # with hog feature from v15
pca_persist_file = "./model/head_v21.pca"

if getattr(sys, 'frozen', False):
    # we are running in a bundle
    bundle_dir = sys._MEIPASS
else:
    # we are running in a normal Python environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))


svc = joblib.load(os.path.join(bundle_dir,svm_persist_file))
pca = joblib.load(os.path.join(bundle_dir,pca_persist_file))

# cnn_model = load_model("./model/model.h5")

# def pre_cnn(img):
# 	x = img_to_array(img)
# 	x_max = np.max(x)
# 	x_min = np.min(x)
# 	x= (x-x_min)/(x_max-x_min+1.0e-16)*2-1
# 	x = np.expand_dims(x, 0)
# 	# x=np.stack([x],axis=2)
# 	return x

def head_verify(img_src, mask):
    img_src = cv2.GaussianBlur(img_src, (3, 3), 1.5)
    
    _img = img_src.copy()
    pos = np.argwhere(mask)

    rect_list = []
    ## 对每一个候选点截取roi区域
    for m in pos:
        dep = img_src[m[0],m[1]]
        if m[0]<20 or m[0]>220 or m[1]<20 or m[1]>300:
            continue

        cv2.circle(_img, (m[1], m[0]), 10, 255, -1)
        
        radius = (dep - 35).astype(np.uint8)
        if radius > 50:
            radius = 50

        if radius <= 25:
            continue

        ## 框出ROI感兴趣区域矩形框
        i = m[0]
        j = m[1]
        y0=max(0,i-radius)
        y1=min(i+radius,240)
        x0=max(0,j-radius)
        x1=min(j+radius,320)
        cv2.rectangle(_img, (x0, y0), (x1, y1), 255, 1)
        head_img = img_src[y0:y1, x0:x1].copy()
        ## 调整大小100*100
        head_img = cv2.resize(head_img, (100,100), interpolation=cv2.INTER_AREA)
        # 保存ROI图像作为数据集
        # if f_cnt % 2 == 0:
        # cv2.imwrite("./roi/%s.png"%(datetime.now().strftime("%Y%m%d%H%M%S%f")), head_img)

        fea = fea_hog(head_img, 6)
        fea = fea.flatten()
        test_x = pca.transform([fea])
        pre = svc.predict(test_x)
        if pre == "pos":
            ## SVM分类为人头的概率
            pro = svc.predict_proba(test_x)
            proba = max(pro[0])
            ## 分类置信率阈值
            if proba >= 0.5:
                rect_list.append([x0,y0,x1,y1,i,j,proba])
    ## NMS
    dele_idx = []
    num = len(rect_list)
    if num > 1:
        for i in range(num-1):
            for j in range(i+1, num):
                overlap = max(rect_list[j][0], rect_list[i][0]) - min(rect_list[j][2], rect_list[i][2])
                ## 如果有重叠
                if overlap < -30:
                    if rect_list[i][-1] > rect_list[j][-1]:
                        dele_idx.append(j)
                    else:
                        dele_idx.append(i)
    ## NMS后的人头结果降序排列并删除
    dele_idx = list(set(dele_idx))
    dele_idx.sort(reverse=True)

    for idx in dele_idx:
        del rect_list[idx]
    ## 当前帧的人头 mask
    A = np.zeros(240 * 320, dtype=np.bool)
    A = A.reshape(240, 320)
    for r in rect_list:
        cv2.circle(_img, (r[5], r[4]), 40, 255, 1)
        # cv2.rectangle(_img, (r[0],r[1]), (r[2], r[3]), 255, 1)
        A[int(r[4]), int(r[5])] = 1

    return _img, A