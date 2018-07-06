#============================================================#
############# HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#

import sys
import time
import cv2
import dmcam
import numpy as np
from HeadTracker import HeadTracker
from head_proposal import head_proposal
from head_verify import head_verify
from datetime import datetime


f_cnt = 0


## 初始化人头跟踪子类
HT = HeadTracker()

#file_dep = open("I:\TOF\HEAD\data/20180416.bin", 'rb')


dmcam.init(None)
dmcam.log_cfg(dmcam.LOG_LEVEL_INFO,
              dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)

devs = dmcam.dev_list()
if devs is None:
    print(" No device found")
    sys.exit(1)

dev = dmcam.dev_open(None)
dmcam.cap_set_frame_buffer(dev, None, 320 * 240 * 4 * 10)
dmcam.cap_set_callback_on_frame_ready(dev, None)
dmcam.cap_set_callback_on_error(dev, None)

wparams = {
    dmcam.PARAM_INTG_TIME: dmcam.param_val_u(),
    dmcam.PARAM_FRAME_FORMAT: dmcam.param_val_u(),
}
wparams[dmcam.PARAM_INTG_TIME].intg.intg_us = 1400
wparams[dmcam.PARAM_FRAME_FORMAT].frame_format.format = 1
amp_min_val = dmcam.filter_args_u()
amp_min_val.min_amp = 0

if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_AMP, amp_min_val, sys.getsizeof(amp_min_val)):
    print(" set amp to %d %% failed" % 0)

if not dmcam.param_batch_set(dev, wparams):
    print(" set parameter failed")
assert dev is not None
print(" Start capture ...")
dmcam.cap_start(dev)

#img = np.frombuffer(file_dep.read(2 * 320 * 240 * f_cnt), dtype=np.uint16)
f = bytearray(320 * 240 * 4 * 2)
## 主程序入口
while True:
    finfo = dmcam.frame_t()
    ret = dmcam.cap_get_frames(dev, 1, f, finfo)
    # print("get %d frames" % ret)
    if ret > 0:
        w = finfo.frame_info.width
        h = finfo.frame_info.height

        #print(" frame @ %d, %d, %dx%d" %
        #      (finfo.frame_info.frame_idx, finfo.frame_info.frame_size, w, h))

        dist_cnt, dist = dmcam.frame_get_distance(dev, w * h, f, finfo.frame_info)
        if dist_cnt == w * h:
            #timer = cv2.getTickCount()
            img = (dist * 1000).astype(np.uint16).reshape(240, 320)
            # print("frame_num = ", f_cnt)
            f_cnt = f_cnt + 1
            t0 = time.time()
            ## 人头提案点检测
            track_img, mask = head_proposal(img)
            img_src = track_img.copy()
            # t1 = time.time()
            #pro_time = t1 - t0
            ## 人头ROI特征提取与分类
            _img, A = head_verify(img_src, mask)
            #t2 = time.time()
            #reg_time = t2 - t1
            ## 人头跟踪与轨迹绘制
            HT.update(A)
            for j in HT.get_valid_tracks():
                #print(j.omission_cnt,j.path)
                pathLen = len(j.path)
                for w in range(1, pathLen):
                    cv2.line(_img, j.path[w - 1][::-1], j.path[w][::-1], (255, 0, 0), 2)
            _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
            #track_time = time.time() - t2
            total_time = time.time() - t0
            cv2.putText(_img, "OUT: %s" % HT.inn, (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL+ cv2.FONT_ITALIC, 1, (0,255,255), 2)
            cv2.putText(_img, "IN: %s" % HT.out, (180, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL+ cv2.FONT_ITALIC, 1, (0,255,255), 2)
            cv2.putText(_img, "@SHUJI 2018", (200, 230), cv2.FONT_HERSHEY_COMPLEX_SMALL+ cv2.FONT_ITALIC, 0.7, (255,255,0), 1)
            #cv2.putText(_img, "pro_t=%s" % (int(pro_time * 1000)), (10, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 255, 1)
            #cv2.putText(_img, "reg_t=%s"%(int(reg_time*1000)), (10,190), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 255, 1)
            #cv2.putText(_img, "trk_t=%s"%(int(track_time*1000)), (10,210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 255, 1)
            cv2.putText(_img, "%s ms"%(int(total_time*1000)), (10,230), cv2.FONT_HERSHEY_COMPLEX_SMALL + cv2.FONT_ITALIC, 0.7, (255,255,0), 1)
            cv2.imshow("rec", _img)

            cv2.waitKey(1)

# file_sav.close()
# human_mask =np.array(human_mask,dtype=np.bool)
# print(human_mask.shape)
# np.savez_compressed("rec_mask.npz", mask=human_mask)
# np.savez("head_det.npz", img=pre_img, mask=human_mask)