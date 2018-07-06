# -*- coding: utf-8 -*-
#============================================================#
############# SHUJI HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#


import time
import cv2
import numpy as np
from HeadTracker import HeadTracker
from head_proposal import head_proposal
from head_verify import head_verify
from datetime import datetime

f_cnt = 166#2000
#mouse callback function
def draw_circle(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        mask_mouse = np.zeros(240 * 320, dtype=np.bool)
        mask_mouse = mask_mouse.reshape(240, 320)
        mask_mouse[y,x]=1
        _img, A = head_verify(img_src, mask_mouse, img_fill)
        cv2.putText(_img, "%s"%track_img[y,x], (10, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
        # cv2.circle(img_fill, (x, y), radius, 255, 1)
        cv2.imshow("rec", _img)

## 初始化人头跟踪子类
HT = HeadTracker()

file_dep = open("E:/trainee/data/20180416.bin", 'rb')

img = np.frombuffer(file_dep.read(2 * 320 * 240*f_cnt ), dtype=np.uint16)
img = np.frombuffer(file_dep.read(2 * 320 * 240), dtype=np.uint16).reshape(240, 320)

#mask_new = np.frombuffer(file_mask.read(320*240*f_cnt), dtype=np.uint8)
t_roi = 0
t_cls = 0
t_trc = 0
t_tol = 0
## 主程序入口
while True:
# while f_cnt < 1800:
    img = np.frombuffer(file_dep.read(2 * 320 * 240), dtype=np.uint16).reshape(240, 320)

    img_u8 = (img/20).astype(np.uint8)
    cv2.imshow("img_u8",img_u8 )

    # print("frame_num = ", f_cnt)
    # cv2.imwrite("./src_u8/%s.png"%(f_cnt), img_u8)
    f_cnt = f_cnt + 1
    print("f_cnt = ", f_cnt)
    t0 = time.time()
    ## 人头提案点检测
    track_img, mask, img_fill = head_proposal(img)#, v=5)#v原来等于2
    cv2.imshow("track_img",track_img )
    cv2.imshow("mask",mask )
    #cv2.imshow("img_fill",img_fill )
    #cv2.imshow("img_u8", img_fill)
    img_src = track_img.copy()
    # cv2.imshow("src", img_src)
    t1 = time.time()
    pro_time = t1 - t0
    ## 人头ROI特征提取与分类
    _img, A = head_verify(img_src, mask)
    t2 = time.time()
    reg_time = t2 - t1
    # ## 鼠标事件
    # cv2.namedWindow('rec')
    # cv2.setMouseCallback('rec', draw_circle)
    ## 人头跟踪与轨迹绘制
    HT.update( A)
    for j in HT.get_valid_tracks():
        # print(j.omission_cnt,j.path)
        pathLen = len(j.path)
        for w in range(1, pathLen):
            cv2.line(_img, j.path[w - 1][::-1], j.path[w][::-1], (255, 0, 0), 2)

    track_time = time.time() - t2
    total_time = time.time() - t0

    _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(_img, "#OUT: %s" % HT.inn, (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
    cv2.putText(_img, "#IN: %s" % HT.out, (180, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
    t_roi = t_roi + int(pro_time*1000)
    t_cls = t_cls + int(reg_time*1000)
    t_trc = t_trc + int(track_time*1000)
    t_tol = t_tol + int(total_time*1000)
    cv2.putText(_img, "@SJTU 2018", (200, 230), cv2.FONT_HERSHEY_COMPLEX_SMALL+cv2.FONT_ITALIC, 0.7, (255,255,0), 1)
    cv2.imshow("rec", _img)

    cv2.waitKey(0)
print(t_roi/f_cnt, t_cls/f_cnt, t_trc/f_cnt, t_tol/f_cnt)

# file_sav.close()
# human_mask =np.array(human_mask,dtype=np.bool)
# print(human_mask.shape)
# np.savez_compressed("rec_mask.npz", mask=human_mask)
# np.savez("head_det.npz", img=pre_img, mask=human_mask)
