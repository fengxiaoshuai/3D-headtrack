import numpy as np
from collections import deque
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import cv2
import warnings

import time

# params
DIST_RESTRICT = 120
MID_FIELD_MARGIN = 110

def _one_to_one_mapping(dist_matrix):
    """
    :param dist_matrix:
    :return:
    returns an array of shape (n_true_heads,2), stores the minimal distances and corresponding index in each column
    """
    x, y = dist_matrix.shape
    ret = np.ones((x, 2), dtype=np.float) * np.inf
    mask = np.ones_like(dist_matrix)
    while np.max(mask) > 0:
        idx, idy = np.unravel_index(np.argmin(np.where(mask == 1, dist_matrix, np.inf), axis=None),
                                    dist_matrix.shape)
        mask[:, idy] = 0
        ret[idx, 0] = dist_matrix[idx, idy]
        ret[idx, 1] = int(idy)
        mask[idx, :] = 0
    return ret


class HeadTracker(object):
    def __init__(self, threshold=DIST_RESTRICT):
        self.track_threshold = threshold
        self.head_tracks = list()
        self.inn = 0
        self.out = 0


    def update(self, proposal_mask):
        if len(self.head_tracks) == 0:
            for i,j in np.argwhere(proposal_mask == 1):
                ht = self.HeadTrack((i,j))
                self.head_tracks.append(ht)
            return
        head_track_list = np.zeros(len(self.head_tracks), dtype=np.bool)
        if np.any(proposal_mask):
            dist_matrix = cdist(np.argwhere(proposal_mask == 1), np.array(list(map(lambda x: x.lastPoint(), self.head_tracks))))
            dist_matrix = _one_to_one_mapping(dist_matrix)
            for i in range(len(np.argwhere(proposal_mask == 1))):
                if dist_matrix[i, 0] < self.track_threshold:
                    head_track_list[int(dist_matrix[i, 1])] = 1
                    self.head_tracks[int(dist_matrix[i, 1])].update(tuple(np.argwhere(proposal_mask == 1)[i]))
                else:
                    point = tuple(np.argwhere(proposal_mask == 1)[i])
                    if MID_FIELD_MARGIN < point[0] < 240 - MID_FIELD_MARGIN:
                        warnings.warn('Trace start from midfield. Abondoned!')
                        continue
                    ht = self.HeadTrack(point)
                    self.head_tracks.append(ht)

        for i in range(len(head_track_list)-1, -1, -1):
            if head_track_list[i]==0:
                ret = self.head_tracks[i].update()
                if ret:
                    if self.head_tracks[i].is_valid():
                        if MID_FIELD_MARGIN < self.head_tracks[i].path[-1][0] < 240 - MID_FIELD_MARGIN:
                            warnings.warn('Trace vanished in midfield. Abondoned!')
                        else:
                            if (self.head_tracks[i].path[0][0] - MID_FIELD_MARGIN) * (self.head_tracks[i].path[-1][0] - MID_FIELD_MARGIN) > 0:
                                warnings.warn('Trace started and vanished in same side. Abondoned!')
                            else:
                                if self.head_tracks[i].path[0][0] - self.head_tracks[i].path[-1][0] > 0:
                                    self.inn += 1
                                else:
                                    self.out += 1
                    del self.head_tracks[i]


    def get_valid_tracks(self):
        return list(filter(lambda x: x.is_valid(), self.head_tracks))

    class HeadTrack(object):
        def __init__(self, point, intrusion=2, omission=10):
            self.path = deque([], maxlen=600)
            self.omission = omission
            self.intrusion = intrusion
            self.omission_cnt = 0
            self.intrusion_cnt = 0
            self.valid = False
            self.update(point)



        def lastPoint(self):
            """
            calculate the distance of a given point and the last point of an existing track, for evaluating
            which track the new point should belong to
            :param point: a tuple that contains the x and y coordinates of a new point
            :return:  a float that indicates the Euclid distance
            """
            if len(self.path) < 1:
                raise Exception('Track not initialized!')
            else:
                if self.path[-1] is not None:
                    last_point = self.path[-1]
                else:
                    raise RuntimeError('last point is None')
                return last_point


        def update(self, point=None):
            """
            update a track and return true if it dies
            :return: boolean True if the track should be destroyed
            """
            if point:
                self.path.append(point)
                if self.valid:
                    self.omission_cnt = 0
                else:
                     self.intrusion_cnt += 1
                    if self.intrusion_cnt >= self.intrusion:
                        self.valid =True
                return False
            else:
                self.omission_cnt += 1
                if self.omission_cnt >= self.omission:
                    return True
                if not self.valid:
                    self.intrusion_cnt = 0
                    return False

        def is_valid(self):
            return self.valid

        def __str__(self):
            return '.'.join(map(lambda x: '(%s,%s)' % x, self.path))


if __name__=='__main__':
    from head_proposal import head_proposal

    img = np.fromfile(r"20180412-2m.bin", dtype=np.uint16)
    img = img.astype(np.uint16).reshape(img.shape[0] // 240 // 320, 240, 320)
    HT = HeadTracker()
    for idx, i in enumerate(img):
        timer = cv2.getTickCount()
        #cv2.imshow('i',(i/20).astype(np.uint8))
        #cv2.waitKey(50)
        print(idx)
        #start = time.time()
        im, trace_mask = head_proposal(i)
        #print('head proposal costs %d ms' % ((time.time() - start) * 1000))


        #start = time.time()
        HT.update(im, trace_mask)
        #print('head tracker costs %d ms' % ((time.time() - start) * 1000))
        _img = im.copy()
        print(np.argwhere(trace_mask))
        frame = im.copy()
        '''
        if 0:

            bbox = cv2.selectROI(frame, False)
            print(bbox)
            ok = tracker.init(frame, bbox)
        if 0:
            ok, bbox = tracker.update(frame)

            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Display result
            cv2.imshow("Tracking", frame)
        '''
        for j in HT.get_valid_tracks():
            # print(j.path)
            pathLen = len(j.path)
            for w in range(1, pathLen):
                cv2.line(_img, j.path[w - 1][::-1], j.path[w][::-1], (255, 0, 0), 2)      

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(_img, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.putText(_img,"# In:%d"%HT.inn, (0, 25), cv2.FONT_HERSHEY_COMPLEX,1,255,1)
        cv2.putText(_img, "# Out:%d" % HT.out, (150, 25), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 1)
        cv2.imshow('img', _img)
        cv2.waitKey(0)
    cv2.waitKey(0)