import os
import time
import numpy as np
from utils import onLane, putText, FPS, hconcatResize, toThreeChannel
import cv2
from tracker import CentroidTracker
import collections


class Counter():
    def __init__(self, videoPath, scale=3, skipFrame=2, outputName='demo.mp4'):
        self.videoPath = videoPath
        self.cap = cv2.VideoCapture(self.videoPath)
        self.scale = scale
        self.skipFrame = skipFrame
        self.bgs = cv2.createBackgroundSubtractorMOG2()
        # self.bgs = cv2.bgsegm.createBackgroundSubtractorGSOC()

        # optical flow
        self.featureParams = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
        self.lkParams = dict(winSize=(15, 15), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = (0, 255, 0)
        self.direction = 1
        self.disMove = 2.5
        self.marginCen = 5
        self.hsize, self.wsize = 20, 20

        # tracking
        self.ct = CentroidTracker()

        self.idStats = {}
        self.countDown = 0
        self.countUp = 0
        self.countedIds = {}
        self.fps = FPS()
        self.iter = 0
        self.frameNum = 0
        self.stableNum = 2
        self.carCoords = collections.defaultdict(list)
        self.skipedId = []

        os.makedirs('./outputs/', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # self.writer = cv2.VideoWriter(outputName, fourcc, 30, (1280, 720), True)
        self.writer = cv2.VideoWriter(outputName, fourcc, 30, (640, 360), True)

    def getFirstFrame(self):
        ret, fisrtFrame = self.cap.read()
        self.height, self.width = fisrtFrame.shape[0] // self.scale, fisrtFrame.shape[1] // self.scale
        self.size = (self.width, self.height)
        firstFrame = cv2.resize(fisrtFrame, self.size)
        return firstFrame

    def getCap(self):
        return self.cap

    def isOpened(self):
        return self.cap.isOpened()

    def getFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        else:
            frame = cv2.resize(frame, self.size)
            self.origin = frame.copy()
            self.optImage = np.zeros((self.height, self.width))
            self.cenImage = np.zeros((self.height, self.width))
            self.frameNum += 1
            return frame

    def skip(self):
        return self.frameNum % self.skipFrame != 0

    def releaseCap(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def settingLaneDirs(self):
        self.laneDirs = {}
        for landId in range(len(self.lanes)):
            self.laneDirs[landId] = 0

    def settingLanes(self, points):
        self.lanes = {}
        self.detRois = {}
        for i in range(len(points) // 4):
            self.lanes[i] = [points[i * 4 + j] for j in range(4)]
        if len(self.lanes) == 0:
            if self.videoPath == "./daytime1.mp4":
                self.lanes = {
                    0: [[120, 126], [242, 135], [243, 232], [63, 229]],
                    1: [[294, 132], [412, 129], [464, 222], [304, 220]],
                }
            elif self.videoPath == "./nighttime1.mp4":
                self.lanes = {
                    0: [[43, 160], [279, 164], [287, 295], [1, 296]],
                    1: [[349, 137], [476, 138], [537, 283], [371, 282]]
                }
            elif self.videoPath == "./daytime2.mp4":
                self.lanes = {
                    0: [[33, 364], [146, 370], [148, 461], [1, 467]],
                    1: [[152, 466], [333, 460], [358, 543], [154, 549]]
                }
        for i in range(len(self.lanes)):
            xlist = [p[0] for p in self.lanes[i]]
            ylist = [p[1] for p in self.lanes[i]]
            xmin, ymin = min(xlist), min(ylist)
            xmax, ymax = max(xlist), max(ylist)
            self.detRois[i] = (xmin, ymin, xmax, ymax)
            self.lanes[i] = np.array(self.lanes[i])

        # initial count for each lane
        self.settingLaneDirs()

    def drawLanes(self, frame, visual=False):
        self.lanes_mask = np.zeros((self.height, self.width))
        for lane in self.lanes.values():
            cv2.fillPoly(self.lanes_mask, pts=[lane], color=255)

        # self.lanes_mask = cv2.fillPoly(self.lanes_mask, pts=[list(lanes.values())[0]], color=255)
        _, self.lanesMask = cv2.threshold(self.lanes_mask, 0, 255, cv2.THRESH_BINARY)

        frame = frame.astype(np.uint8)
        self.lanesMask = self.lanesMask.astype(np.uint8)
        self.extractedLaneFrame = cv2.bitwise_and(frame, frame, mask=self.lanesMask)
        if visual:
            cv2.imshow(self.drawLanes.__name__, self.extractedLaneFrame)

    def drawRois(self):
        for lane in self.lanes.values():
            lane = lane.reshape((-1, 1, 2))
            cv2.polylines(self.origin, [lane], True, (0, 255, 255))

    def bgSubtraction(self, visual=False):
        self.extractedLaneFrame = cv2.GaussianBlur(self.extractedLaneFrame, (3, 3), cv2.BORDER_DEFAULT)
        self.diffMask = self.bgs.apply(self.extractedLaneFrame)

        # smaller mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.diffMask = cv2.erode(self.diffMask, kernel, iterations=1)
        # fill inside mask
        kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        self.diffMask = cv2.morphologyEx(self.diffMask, cv2.MORPH_CLOSE, kernelClose)
        # convert mask to binary
        _, self.diffMask = cv2.threshold(self.diffMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # smaller mask
        kernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        self.diffMask = cv2.erode(self.diffMask, kernelErode, iterations=1)
        self.contours = cv2.findContours(self.diffMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        self.contours = list(filter(lambda x: cv2.contourArea(x) > 300, self.contours))
        if visual:
            cv2.imshow(self.bgSubtraction.__name__, self.diffMask)

    def opticalFlowInitial(self, frame):
        self.prevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # TODO: remove this?
        # self.prev = cv2.goodFeaturesToTrack(self.prevGray, mask=None, **self.featureParams)

    def opticalFlow(self):
        gray = cv2.cvtColor(self.extractedLaneFrame, cv2.COLOR_BGR2GRAY)
        prev = cv2.goodFeaturesToTrack(self.prevGray, mask=None, **self.featureParams)
        curr, status, error = cv2.calcOpticalFlowPyrLK(self.prevGray, gray, prev, None, **self.lkParams)

        # Selects good feature points for previous position
        self.goodPre = prev[status == 1].astype(int)
        self.goodCur = curr[status == 1].astype(int)

        # Updates previous frame
        self.prevGray = gray.copy()
        self.prev = self.goodCur.reshape(-1, 1, 2)

    def createCarDirect(self, visual=False):
        for (curr, prev) in zip(self.goodCur, self.goodPre):
            a, b = curr.ravel()
            c, d = prev.ravel()

            if self.direction:
                dis = b - d
                a = c
            else:
                dis = a - c
                b = d
            if visual:
                self.origin = cv2.line(self.origin, (a, b), (c, d), self.color, 2)
                self.origin = cv2.circle(self.origin, (a, b), 3, self.color, -1)

            # moving distance is bigger than self.disMove and lying on lane
            if dis > self.disMove and onLane(self.diffMask, self.lanes, c, d) is not None:
                self.optImage[d, c] = 100
                if visual:
                    cv2.arrowedLine(self.origin, (c, d), (a, b), [240, 30, 30], 2, tipLength=0.5)
            elif dis < - self.disMove and onLane(self.diffMask, self.lanes, c, d) is not None:
                self.optImage[d, c] = 200
                if visual:
                    cv2.arrowedLine(self.origin, (c, d), (a, b), [30, 30, 240], 2, tipLength=0.5)

    def createBBox(self, visual=False):
        self.bboxes = list(map(lambda x: cv2.boundingRect(x), self.contours))
        for x, y, w, h in self.bboxes:
            car = self.diffMask[y:y + h, x:x + w]

            laneId = onLane(self.diffMask, self.lanes, x + w // 2, y + h // 2)
            # center in mask
            if laneId is not None:
                # xmin, ymin, xmax, ymax = self.detRois[laneId]
                # # top-down
                # if self.direction and (xmax - xmin) < 2 * w:
                #     continue
                # # left-right
                # if not self.direction and (ymax - ymin) < 2 * h:
                #     continue
                coords = cv2.findNonZero(car)
                car_cx = []
                car_cy = []
                if coords is not None:
                    for coord in coords:
                        car_cx.append(coord[0][0])
                        car_cy.append(coord[0][1])
                    cx = x + int(np.mean(car_cx))
                    cy = y + int(np.mean(car_cy))
                    self.cenImage[cy - self.marginCen: cy + self.marginCen, cx - self.marginCen:cx + self.marginCen] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.cenImage = cv2.dilate(self.cenImage, kernel, iterations=15)
        self.cenImage = cv2.erode(self.cenImage, kernel, iterations=1)
        self.cenImage = self.cenImage.astype(np.uint8)
        contours = cv2.findContours(self.cenImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        self.bboxes = list(map(lambda x: cv2.boundingRect(x), contours))
        if visual:
            cv2.imshow(self.createBBox.__name__, self.cenImage)

    def tracking(self):
        # xywh -> xyxy
        self.xyxyBBoxes = [(x1, y1, x1 + w1, y1 + h1) for (x1, y1, w1, h1) in self.bboxes if (w1 > self.wsize and h1 > self.hsize)]
        self.objects = self.ct.update(self.xyxyBBoxes)

    def counting(self, visual=False):
        for xyxy, (objectID, centroid) in zip(self.xyxyBBoxes, self.objects.items()):
            # objCx, objCy = centroid[0], centroid[1]
            x1, y1, x2, y2 = xyxy
            objCx, objCy = (x1 + x2) // 2, (y1 + y2) // 2

            optTmp = self.optImage[y1: y2, x1: x2]
            optTmp = optTmp.astype(np.uint8)

            for det_roi in self.detRois.values():
                xmin, ymin, xmax, ymax = det_roi
                if self.direction:
                    min_value = ymin
                    max_value = ymax
                    obj_c = objCy
                else:
                    min_value = xmin
                    max_value = xmax
                    obj_c = objCx

                if obj_c >= min_value and obj_c <= max_value:
                    redOptCount = np.count_nonzero(optTmp == 200)
                    blueOptCount = np.count_nonzero(optTmp == 100)
                    laneId = onLane(self.diffMask, self.lanes, objCx, objCy)
                    if blueOptCount > redOptCount:
                        self.laneDirs[laneId] += 1
                    if blueOptCount < redOptCount:
                        self.laneDirs[laneId] -= 1

                    move1 = 0
                    move2 = 0
                    move3 = 0
                    thres1 = 0
                    thres2 = 0
                    if len(self.carCoords[objectID]) >= 2:
                        oldObjCx = self.carCoords[objectID][-1][0]
                        oldObjCy = self.carCoords[objectID][-1][1]
                        if self.direction:
                            # current to previous
                            move1 = objCy - oldObjCy
                            thres1 = (ymax - ymin) * 0.9

                            move2 = objCx - oldObjCx
                            thres2 = (xmax - xmin) * 0.9

                            # current to mean
                            move3 = objCy - np.mean([coord[1] for coord in self.carCoords[objectID]])
                        else:
                            # current to previous
                            move1 = objCx - oldObjCx
                            thres1 = (xmax - xmin)

                            move2 = objCy - oldObjCy
                            thres2 = (ymax - ymin)
                            # current to mean
                            move3 = objCx - np.mean([coord[0] for coord in self.carCoords[objectID]])

                    self.carCoords[objectID].append((objCx, objCy))
                    if abs(move1) > thres1 or abs(move2) > thres2:
                        del self.carCoords[objectID][-1]
                        self.skipedId.append(objectID)
                        continue

                    # not counted and center on lane
                    if objectID not in list(self.countedIds.values()) and laneId is not None:
                        if objectID not in self.idStats.keys():
                            self.idStats[objectID] = collections.deque(maxlen=self.stableNum)

                        if blueOptCount > redOptCount and self.laneDirs[laneId] > 0 and move3 > 0:
                            self.idStats[objectID].append(1)
                            if len(self.idStats[objectID]) == self.stableNum and self.idStats[objectID].count(1) == self.stableNum:
                                self.countDown += 1
                                self.countedIds[objectID] = objectID
                                cv2.circle(self.origin, center=(objCx, objCy), radius=5, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                                cv2.circle(self.origin, center=(objCx, objCy), radius=5, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        elif blueOptCount < redOptCount and self.laneDirs[laneId] < 0 and move3 < 0:
                            self.idStats[objectID].append(-1)
                            if len(self.idStats[objectID]) == self.stableNum and self.idStats[objectID].count(-1) == self.stableNum:
                                self.countUp += 1
                                self.countedIds[objectID] = objectID
                                cv2.circle(self.origin, center=(objCx, objCy), radius=5, color=(30, 30, 240), thickness=-1, lineType=cv2.LINE_4, shift=0)
                if visual and objectID not in self.countedIds.keys() and objectID not in self.skipedId:
                    # draw both the ID of the object and the centroid of the
                    cv2.rectangle(self.origin, (x1, y1), (x2, y2), self.color, 5)
                    text = "ID {}".format(objectID + 1)
                    cv2.putText(self.origin, text, (objCx - 10, objCy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
                    cv2.circle(self.origin, (objCx, objCy), 4, self.color, -1)
        # print(self.carCoords)

    def getFPS(self):
        fps = round(self.fps())
        return fps

    def showResult(self):
        # put text
        if self.direction:
            putText(self.origin, 'Down:' + str(self.countDown), (self.origin.shape[1] // 2, self.height - 10), (0, 255, 255))
            putText(self.origin, 'Up:' + str(self.countUp), (self.origin.shape[1] // 10, self.height - 10), (0, 255, 255))
        else:
            putText(self.origin, 'Right:' + str(self.countDown), (self.width // 2, self.height - 10), (0, 255, 255))
            putText(self.origin, 'Left' + str(self.countUp), (self.width // 10, self.height - 10), (0, 255, 255))
        putText(self.origin, 'FPS: ' + str(self.getFPS()), (10, 30), (0, 255, 255))

        # draw car area
        self.drawRois()

        # show all result
        self.concatVisual()

    def concatVisual(self):
        diffMask = toThreeChannel(self.diffMask, self.origin)
        cenImage = toThreeChannel(self.cenImage, self.origin)
        optImage = toThreeChannel(self.optImage, self.origin)
        result1 = hconcatResize([self.origin, diffMask])
        result2 = hconcatResize([optImage, cenImage])
        result = cv2.hconcat([result1, result2])
        cv2.imshow(self.concatVisual.__name__, result)

    def saveImage(self):
        cv2.imwrite("./outputs/frame_num_{}.png".format(str(self.iter).zfill(5)), self.origin)
        self.iter += 1

    def saveVideo(self):
        self.writer.write(self.origin)
