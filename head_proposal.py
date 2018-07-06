#============================================================#
############# SHUJI HEAD DETECTION Version 1.0 ###############
#============================================================#
#==========================2018-04-17========================#



from core import preprocess, mean_pooling
import cv2
import numpy as np

mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)#不设置阴影检测器

params = cv2.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.filterByCircularity = False
params.filterByArea = False
params.filterByCircularity = True
params.minCircularity = 0.01
params.filterByArea = True
params.minDistBetweenBlobs = 60
params.minThreshold = 10
params.maxThreshold = 100
params.thresholdStep = 1
params.minArea = 1000.0
params.maxArea = 8000.0
detector = cv2.SimpleBlobDetector_create(params)


def head_proposal(img):
        # start = time.time()
        img,img_fill = preprocess(img)
        # print('  preprocess costs %d ms' % ((time.time() - start) * 1000))
        # start = time.time()
        fgmask = mog.apply(img)
        dilated = cv2.dilate(fgmask, np.ones((3,3), dtype=np.uint8), iterations=2)
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fgmask = np.zeros_like(fgmask, dtype=np.bool)
        for c in contours:
            if cv2.contourArea(c)>1000:
                (x,y,w,h) = cv2.boundingRect(c)
                fgmask[y:y+h,x:x+w] = True
        img_ = np.where(fgmask, img, 0)
        # print('  foreground extract costs %d ms' % ((time.time() - start) * 1000))
        # start = time.time()
        _img = mean_pooling(img_, 10).astype(np.uint8)
        _img = cv2.resize(_img, (320, 240))

        keypoints = detector.detect(_img)
        trace_mask = np.zeros_like(img)
        for hp in keypoints:
            trace_mask[int(hp.pt[1]),int(hp.pt[0])] = True
        # print('  blob detect costs %d ms' % ((time.time() - start) * 1000))
        return img, trace_mask, img_fill



        

# if __name__ == "__main__":
#     path = r"20180408.bin"
#
#
#
#     img = np.fromfile(path, dtype=np.uint16)
#     img = img.astype(np.uint16).reshape(img.shape[0]//240//320, 240, 320)
#     for idx, _img in enumerate(img):
#         img = preprocess(_img)
#         cv2.imshow('img', img)
#         fgmask = mog.apply(img)
#         dilated = cv2.dilate(fgmask, np.ones((3,3), dtype=np.uint8), iterations=2)
#         image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         fgmask = np.zeros_like(fgmask, dtype=np.bool)
#         for c in contours:
#             if cv2.contourArea(c)>1600:
#                 (x,y,w,h) = cv2.boundingRect(c)
#                 fgmask[y:y+h,x:x+w] = True
#         img_ = np.where(fgmask, img, 0)
#         cv2.imshow('-bg', img_)
#         #img_ = morph_open(img_, 55)
#         local_maxima = find_local_maxima(np.where(fgmask, img_, 255), 25)
#         cv2.imshow('local_maxima', (local_maxima*255).astype(np.uint8))
#         local_maxima = cluster(local_maxima)
#         cv2.imshow('cluster', (local_maxima*255).astype(np.uint8))
#         local_maxima = shape_center(img, local_maxima)
#         cv2.imshow('shape_center', (local_maxima*255).astype(np.uint8))
#         print('idx:%d'% idx)
#         for i,j in np.argwhere(local_maxima == 1):
#             cv2.circle(img, (j, i), 3, 255, 1)
#         cv2.imshow('final', img)
#         cv2.waitKey(0)






