import cv2
import numpy as np
import collections
import time

points = []


def mousePoints(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def setPoints(frame):
    cv2.imshow("firstFrame", frame)
    cv2.setMouseCallback("firstFrame", mousePoints)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
    if len(points):
        print(points)
    assert len(points) % 4 == 0, "Please check your points input!"
    return points


def onLane(frame, lanes, cx, cy):
    height, width = frame.shape[:2]

    maskImages = []
    for lane in lanes.values():
        mask_image = np.zeros((height, width))
        cv2.fillPoly(mask_image, pts=[lane], color=255)
        maskImages.append(mask_image)

    lane_id = None
    for i, mask_image in enumerate(maskImages):
        if mask_image[cy, cx] == 255:
            lane_id = i
            break

    return lane_id


def putText(frame, text, org, color):
    cv2.putText(frame, text=str(text),
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=color,
                thickness=3, lineType=cv2.LINE_AA)


def hconcatResize(imgList, interpolation=cv2.INTER_CUBIC):
    hmin = min(img.shape[0] for img in imgList)
    imListResize = [cv2.resize(img, (int(img.shape[1] * hmin / img.shape[0]), hmin), interpolation=interpolation) for img in imgList]
    return cv2.hconcat(imListResize)


def toThreeChannel(image, dest):
    result = np.zeros_like(dest)
    result[:, :, 0] = image
    result[:, :, 1] = image
    result[:, :, 2] = image
    return result


class FPS:
    def __init__(self, avarageof=5):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0
