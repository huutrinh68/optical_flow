import cv2
from counter import Counter
from utils import setPoints


def main():
    videoPath = "./daytime2.mp4"
    # initial counter class
    counter = Counter(videoPath)

    # get first frame
    firstFrame = counter.getFirstFrame()

    # setting point on first frame
    points = setPoints(firstFrame)

    # create lane from clicked points
    counter.settingLanes(points)

    # initial optical flow
    counter.opticalFlowInitial(firstFrame)

    while(counter.isOpened()):
        # get frame
        frame = counter.getFrame()
        if frame is None:
            break

        # skip or not
        if counter.skip():
            continue

        # extract lane area
        counter.drawLanes(frame, visual=False)

        # extract diff mask on lane area
        counter.bgSubtraction(visual=False)

        # apply optical flow on extracted lane area
        counter.opticalFlow()

        # create car move direction
        counter.createCarDirect(visual=True)

        # create bbox from diffmask
        counter.createBBox(visual=False)

        # tracking
        counter.tracking()

        # counting
        counter.counting(visual=True)

        # visual
        counter.showResult()

        # save result
        counter.saveImage()
        counter.saveVideo()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
    counter.releaseCap()


if __name__ == '__main__':
    main()
