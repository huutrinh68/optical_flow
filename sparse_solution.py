import numpy as np
import cv2
import math
from scipy.spatial import distance
from collections import OrderedDict


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] for img in img_list)
    # image resizing
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.hconcat(im_list_resize)


def one_channel_to_three_channel(image, dest):
    result = np.zeros_like(dest)
    result[:, :, 0] = image
    result[:, :, 1] = image
    result[:, :, 2] = image
    return result


def main():
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # The video feed is read in as a VideoCapture object
    # cap = cv2.VideoCapture("./20211215_102922.mp4")
    cap = cv2.VideoCapture("./20211211_194628.mp4")

    # Variable for color to draw optical flow track
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (first_frame.shape[1] // 3, first_frame.shape[0] // 3))
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # google summer code for background subtraction
    # bgs = cv2.bgsegm.createBackgroundSubtractorGSOC()
    # bgs = cv2.createBackgroundSubtractorKNN()
    # bgs = cv2.bgsegm.createBackgroundSubtractorLSBP()
    bgs = cv2.createBackgroundSubtractorMOG2()
    # bgs = cv2.bgsegm.createBackgroundSubtractorMOG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    min_dist = 110
    min_optim = 3.5

    width, height = first_frame.shape[:2]
    car_obj = OrderedDict()
    count_flag = OrderedDict()
    nextcar_id = 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("demo.mp4", fourcc, 30, (1920, 720), True)

    while(cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        if not ret:
            break
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        bin_image = np.zeros((width, height))
        cen_image = np.zeros((width, height))
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv2.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next[status == 1].astype(int)

        # ************* background subtraction ***************
        frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
        bg_mask = bgs.apply(frame)
        bg_mask_3channel = one_channel_to_three_channel(bg_mask, frame)

        # smaller mask
        bg_mask = cv2.erode(bg_mask, kernel, iterations=3)
        erode = one_channel_to_three_channel(bg_mask, frame)

        # fill inside mask
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_close)

        # convert mask to binary
        _, bg_mask = cv2.threshold(bg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # smaller mask
        bg_mask = cv2.erode(bg_mask, kernel_erode, iterations=1)

        # bg_mask = cv2.dilate(bg_mask, kernel, iterations=2)
        # dilate = one_channel_to_three_channel(bg_mask, frame)
        closing = one_channel_to_three_channel(bg_mask, frame)
        combi = hconcat_resize([bg_mask_3channel, erode, closing])
        # bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        # bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = list(filter(lambda x: cv2.contourArea(x) > 300, contours))
        bboxes = list(map(lambda x: cv2.boundingRect(x), contours))
        # ************* background subtraction ***************

        # Draws the optical flow tracks
        final_bboxes = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()

            dist = b - d

            dist = math.sqrt((b - d)**2 + (c - a)**2)
            if dist > min_optim:
                bin_image[d, c] = 255
                # Draws line between new and old position with green color and 2 thickness
                frame = cv2.line(frame, (a, b), (c, d), color, 2)
                # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                frame = cv2.circle(frame, (a, b), 3, color, -1)
                for x, y, w, h in bboxes:
                    # if x <= a and a <= x + w and y <= b and b <= y + h and w > 100:
                    if x <= a and a <= x + w and y <= b and b <= y + h:
                        final_bboxes.append((x, y, w, h))

        # create center of optical flow
        for x, y, w, h in final_bboxes:
            car = bin_image[y:y + h, x:x + w]
            coords = cv2.findNonZero(car)
            car_cx = []
            car_cy = []
            if coords is not None:
                for coord in coords:
                    car_cx.append(coord[0][0])
                    car_cy.append(coord[0][1])
                cx = x + int(np.mean(car_cx))
                cy = y + int(np.mean(car_cy))
                cen_image[cy, cx] = 255
        cen_image = cv2.dilate(cen_image, kernel, iterations=15)
        cen_image = cen_image.astype(np.uint8)
        contours = cv2.findContours(cen_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        final_bboxes = list(map(lambda x: cv2.boundingRect(x), contours))

        for x, y, w, h in final_bboxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cx, cy = int(x + w // 2), int(y + h // 2)
            centroid = np.array([[cx, cy]])

            if len(car_obj) == 0:
                car_obj[nextcar_id] = centroid
                count_flag[nextcar_id] = 0
                nextcar_id += 1
            else:
                objectCentroids = list(car_obj.values())
                D = distance.cdist(np.array(objectCentroids).reshape(-1, 2), centroid)
                D = D.reshape(-1)
                inx_d = np.argmin(D)

                if D[inx_d] <= min_dist:
                    car_obj[inx_d + 1] = centroid
                else:
                    car_obj[nextcar_id] = centroid
                    count_flag[nextcar_id] = 0
                    nextcar_id += 1
            cv2.putText(frame, text='carID: ' + str(nextcar_id), org=(cx, cy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        binary = one_channel_to_three_channel(bin_image, frame)
        center = one_channel_to_three_channel(cen_image, frame)
        result = hconcat_resize([frame, binary, center])
        result = cv2.vconcat([result, combi])
        cv2.imshow("result", result)
        writer.write(result)
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
