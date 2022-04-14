import cv2
from counter import Counter


def main():
    videoPath = "./daytime2.mp4"
    cap = cv2.VideoCapture(videoPath)
    scale = 3

    # initial counter class
    counter = Counter()

    while(cap.isOpened()):
        ret, frame = cap.read()
        height, width = frame.shape[0] // scale, frame.shape[1] // scale
        frame = cv2.resize(frame, (width, height))

        if frame is None:
            break

        counter.process(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
