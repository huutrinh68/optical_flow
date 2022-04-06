import numpy as np
import cv2
import math
import time
from scipy.spatial import distance
from collections import OrderedDict

points = []


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


def mouse_points(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def set_points(frame):
    cv2.imshow("first_frame", frame)
    cv2.setMouseCallback("first_frame", mouse_points)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
    print(points)


def on_lane(frame, lanes, cx, cy):
    width, height = frame.shape[:2]

    mask_images = []
    for lane in lanes.values():
        mask_image = np.zeros((width, height))
        cv2.fillPoly(mask_image, pts=[lane], color=255)
        mask_images.append(mask_image)

    lane_id = None
    for i, mask_image in enumerate(mask_images):
        if mask_image[cy, cx] == 255:
            lane_id = i
            break
    return lane_id


def main():
    global points
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # The video feed is read in as a VideoCapture object
    video_path = "./20211215_102922.mp4"
    # video_path = "./20211211_194628.mp4"
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture("")

    # Variable for color to draw optical flow track
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (first_frame.shape[1] // 3, first_frame.shape[0] // 3))
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # httdelta1://docs.opencv2.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    width, height = first_frame.shape[:2]
    min_dist = 100
    w1_size = 50
    h1_size = 50
    car_area = [0, height]
    count_down = 0
    count_up = 0
    dis_move = 3.5

    car_obj = OrderedDict()
    car_lane = OrderedDict()
    count_flag = OrderedDict()
    nextcar_id = 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("demo.mp4", fourcc, 30, (1280, 720), True)

    # ************setting lane*************
    # set_points(first_frame)
    if video_path == "./20211215_102922.mp4":
        lanes = {
            0: [[197, 6], [272, 7], [214, 357], [39, 357]],
            1: [[289, 4], [339, 5], [514, 356], [350, 356]]
        }
    else:
        lanes = {
            0: [[153, 4], [215, 4], [87, 359], [2, 357]],
            1: [[232, 3], [292, 4], [267, 356], [154, 356]],
            2: [[338, 3], [391, 3], [515, 357], [378, 356]]
        }
    for i in range(len(lanes)):
        lanes[i] = np.array(lanes[i])
    # ************setting lane*************

    while(cap.isOpened()):
        start_time = time.perf_counter()
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        if not ret:
            break
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        bin_image = np.zeros((width, height))
        dir_image = np.zeros((width, height))
        cen_image = np.zeros((width, height))
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        # ****** setting lane******
        mask_image = np.zeros((width, height))
        visual_frame = frame.copy()
        for lane in lanes.values():
            cv2.fillPoly(visual_frame, pts=[lane], color=(255, 255, 255))
            cv2.fillPoly(mask_image, pts=[lane], color=255)
        cv2.imshow("visual_frame", visual_frame)
        cv2.imshow("mask_image", mask_image)
        # ****** setting lane******

        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # httdelta1://docs.opencv2.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next[status == 1].astype(int)

        delta1 = 50
        delta2 = 30
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            dis = b - d
            a = c
            frame = cv2.line(frame, (a, b), (c, d), color, 2)
            frame = cv2.circle(frame, (a, b), 3, color, -1)

            # moving distance is bigger than dis_move and lying on lane
            # if dis > dis_move and on_lane(frame, lanes, c, d):
            if dis > dis_move:
                bin_image[d - delta1:d + delta1, int(c) - delta1:int(c) + delta1] = 255
                dir_image[d - delta2:d + delta2, int(c) - delta2:int(c) + delta2] = 100
            elif dis < - dis_move:
                bin_image[d - delta1:d + delta1, int(c) - delta1:int(c) + delta1] = 255
                dir_image[d - delta2:d + delta2, int(c) - delta2:int(c) + delta2] = 200

        cv2.imshow("before opening", bin_image)
        # bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        # bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        opening_size = 65
        kernel_size = (opening_size, opening_size)
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel_opening, iterations=2)

        # bitwise
        bin_image = bin_image.astype(np.uint8)
        contours = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in contours:
            (x1, y1, w1, h1) = cv2.boundingRect(c)

            if w1 > w1_size and h1 > h1_size:
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color, 5)
                x2 = w1 / 2
                y2 = h1 / 2
                cx = int(x1 + x2)
                cy = int(y1 + y2)
                centroid = np.array([[cx, cy]])
                cen_image[cy - 10:cy + 10, cx - 10:cx + 10] = 255

                lane_id = on_lane(frame, lanes, cx, cy)

                if len(car_obj) == 0:
                    car_obj[nextcar_id] = centroid
                    count_flag[nextcar_id] = 0
                    car_lane[nextcar_id] = lane_id
                    nextcar_id += 1
                else:
                    objectCentroids = list(car_obj.values())
                    D = distance.cdist(np.array(objectCentroids).reshape(-1, 2), centroid)
                    D = D.reshape(-1)
                    inx_d = np.argmin(D)

                    if D[inx_d] <= min_dist and list(car_lane.items())[inx_d][1] == lane_id or lane_id is None:
                        car_obj[inx_d + 1] = centroid
                    else:
                        car_obj[nextcar_id] = centroid
                        count_flag[nextcar_id] = 0
                        car_lane[nextcar_id] = lane_id
                        nextcar_id += 1

        for carID, val in car_obj.items():
            obj_cx = val[0, 0]
            obj_cy = val[0, 1]

            if obj_cy >= car_area[0] and obj_cy <= car_area[1]:
                if count_flag[carID] == 0:
                    red_direc_count = np.count_nonzero(dir_image[obj_cy - 10:obj_cy + 10, obj_cx - 10:obj_cx + 10] == 200)
                    blue_direc_count = np.count_nonzero(dir_image[obj_cy - 10:obj_cy + 10, obj_cx - 10:obj_cx + 10] == 100)
                    if blue_direc_count > red_direc_count:
                        count_down += 1
                        cv2.circle(frame, center=(obj_cx, obj_cy), radius=40, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        cv2.arrowedLine(frame, (obj_cx, obj_cy), (obj_cx, obj_cy + 100), [240, 30, 30], 10, tipLength=0.5)
                    elif blue_direc_count <= red_direc_count:
                        count_up += 1
                        cv2.circle(frame, center=(obj_cx, obj_cy), radius=40, color=(30, 30, 240), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        cv2.arrowedLine(frame, (obj_cx, obj_cy), (obj_cx, obj_cy - 100), [30, 30, 240], 10, tipLength=0.5)
                    count_flag[carID] = 1

                    cv2.putText(frame, text='carID[' + str(carID) + ']', org=(obj_cx, obj_cy),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        cv2.putText(frame, text='down ' + str(count_down),
                    org=(frame.shape[1] * 2 // 3, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 255),
                    thickness=3, lineType=cv2.LINE_AA)

        cv2.putText(frame,
                    text='up ' + str(count_up),
                    org=(frame.shape[1] // 4, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        fps = round(1 / elapsed_time)

        cv2.putText(frame,
                    text='FPS:' + str(fps),
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(200, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA)
        # Opens a new window and displays the output frame
        binary = one_channel_to_three_channel(bin_image, frame)
        center = one_channel_to_three_channel(cen_image, frame)
        direct = one_channel_to_three_channel(dir_image, frame)
        result1 = hconcat_resize([frame, binary])
        result2 = hconcat_resize([direct, center])
        result = cv2.vconcat([result1, result2])
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
