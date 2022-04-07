import numpy as np
import time
import cv2
# import time
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
    height, width = frame.shape[:2]

    mask_images = []
    for lane in lanes.values():
        mask_image = np.zeros((height, width))
        cv2.fillPoly(mask_image, pts=[lane], color=255)
        mask_images.append(mask_image)

    lane_id = None
    for i, mask_image in enumerate(mask_images):
        if mask_image[cy, cx] == 255:
            lane_id = i
            break

    return lane_id


def background_subtraction(bgs, frame):
    # ************* background subtraction ***************
    frame = cv2.GaussianBlur(frame, (3, 3), cv2.BORDER_DEFAULT)
    binary_mask = bgs.apply(frame)
    # smaller mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.erode(binary_mask, kernel, iterations=3)
    # fill inside mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    # convert mask to binary
    _, binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # smaller mask
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    binary_mask = cv2.erode(binary_mask, kernel_erode, iterations=1)
    contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = list(filter(lambda x: cv2.contourArea(x) > 300, contours))
    # bboxes = list(map(lambda x: cv2.boundingRect(x), contours))
    # ************* background subtraction ***************
    return binary_mask, contours


def draw_lanes(frame, lanes):
    for lane in lanes.values():
        lane = lane.reshape((-1, 1, 2))
        cv2.polylines(frame, [lane], True, (0, 255, 255))
    return frame


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

    height, width = first_frame.shape[:2]
    min_dist = 100
    w1_size = 50
    h1_size = 50
    car_area = [int(height * 0.25), int(height * 1.0) - 5]
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
    set_points(first_frame)
    lanes = {}
    for i in range(len(points) // 4):
        lanes[i] = [points[i * 4 + j] for j in range(4)]
    if len(lanes) == 0:
        if video_path == "./20211215_102922.mp4":
            lanes = {
                0: [[197, 6], [272, 7], [214, 357], [39, 357]],
                1: [[289, 4], [339, 5], [514, 356], [350, 356]]
            }
        else:
            lanes = {
                0: [[45, 182], [318, 200], [331, 354], [2, 357]]
            }
    for i in range(len(lanes)):
        lanes[i] = np.array(lanes[i])
    # ************setting lane*************

    bgs = cv2.createBackgroundSubtractorMOG2()

    frame_num = 0
    while(cap.isOpened()):
        start_time = time.perf_counter()
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % 1 != 0:
            continue
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        bin_image = np.zeros((height, width))
        dir_image = np.zeros((height, width))
        cen_image = np.zeros((height, width))
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        # ****** setting lane******
        lanes_mask = np.zeros((height, width))
        for lane in lanes.values():
            cv2.fillPoly(lanes_mask, pts=[lane], color=255)
        # lanes_mask = cv2.fillPoly(lanes_mask, pts=[list(lanes.values())[0]], color=255)
        # _, lanes_mask = cv2.threshold(lanes_mask, 0, 255, cv2.THRESH_BINARY)
        # frame = frame.astype(np.uint8)
        # lanes_mask = lanes_mask.astype(np.uint8)
        # frame = cv2.bitwise_and(frame, frame, mask=lanes_mask)
        # ****** setting lane******

        # ****** background subtraction ********
        binary_mask, contours = background_subtraction(bgs, frame)
        binary_mask = binary_mask.astype(np.uint8)
        lanes_mask = lanes_mask.astype(np.uint8)
        bin_image = cv2.bitwise_and(binary_mask, lanes_mask)
        bboxes = list(map(lambda x: cv2.boundingRect(x), contours))
        # ****** background subtraction ********

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

        delta2 = 50
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            dis = b - d
            a = c
            frame = cv2.line(frame, (a, b), (c, d), color, 2)
            frame = cv2.circle(frame, (a, b), 3, color, -1)

            # moving distance is bigger than dis_move and lying on lane
            if dis > dis_move and on_lane(frame, lanes, c, d) is not None:
                dir_image[d - delta2:d + delta2, int(c) - delta2:int(c) + delta2] = 100
            elif dis < - dis_move and on_lane(frame, lanes, c, d) is not None:
                dir_image[d - delta2:d + delta2, int(c) - delta2:int(c) + delta2] = 200

        for x, y, w, h in bboxes:
            car = bin_image[y:y + h, x:x + w]
            car_dir = dir_image[y:y + h, x:x + w]
            car = car_dir.astype(np.uint8)
            car_dir = car_dir.astype(np.uint8)
            and_image = cv2.bitwise_and(car, car_dir)
            overlap = cv2.findNonZero(and_image)
            if overlap is not None:
                coords = cv2.findNonZero(car)
                car_cx = []
                car_cy = []
                if coords is not None:
                    for coord in coords:
                        car_cx.append(coord[0][0])
                        car_cy.append(coord[0][1])
                    cx = x + int(np.mean(car_cx))
                    cy = y + int(np.mean(car_cy))
                    cen_image[cy - 10: cy + 10, cx - 10:cx + 10] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cen_image = cv2.dilate(cen_image, kernel, iterations=15)
        cen_image = cen_image.astype(np.uint8)
        contours = cv2.findContours(cen_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        bboxes = list(map(lambda x: cv2.boundingRect(x), contours))

        for bbox in bboxes:
            x1, y1, w1, h1 = bbox

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
                    car_obj[nextcar_id] = [centroid, bbox]
                    count_flag[nextcar_id] = 0
                    car_lane[nextcar_id] = lane_id
                    nextcar_id += 1
                else:
                    objectCentroids = [obj[0] for obj in list(car_obj.values())]
                    D = distance.cdist(np.array(objectCentroids).reshape(-1, 2), centroid)
                    D = D.reshape(-1)
                    inx_d = np.argmin(D)

                    if D[inx_d] <= min_dist and list(car_lane.items())[inx_d][1] == lane_id or lane_id is None:
                        car_obj[inx_d + 1] = [centroid, bbox]
                    else:
                        car_obj[nextcar_id] = [centroid, bbox]
                        count_flag[nextcar_id] = 0
                        car_lane[nextcar_id] = lane_id
                        nextcar_id += 1

        # overlap bin_mask with dir_image
        dir_image = dir_image.astype(np.uint8)
        dir_image = cv2.bitwise_and(bin_image, dir_image)

        for carID, val in car_obj.items():
            obj_cx = val[0][0][0]
            obj_cy = val[0][0][1]
            x, y, w, h = val[1]
            tmp = dir_image[y: y + h, x: x + w]
            tmp = tmp.astype(np.uint8)

            if obj_cy >= car_area[0] and obj_cy <= car_area[1]:
                # radius = 10
                # red_direc_count = np.count_nonzero(dir_image[obj_cy - radius:obj_cy + radius, obj_cx - radius:obj_cx + radius] == 200)
                # blue_direc_count = np.count_nonzero(dir_image[obj_cy - radius:obj_cy + radius, obj_cx - radius:obj_cx + radius] == 100)
                red_direc_count = np.count_nonzero(tmp == 200)
                blue_direc_count = np.count_nonzero(tmp == 100)
                if count_flag[carID] == 0:
                    if blue_direc_count > red_direc_count:
                        count_down += 1
                        cv2.circle(frame, center=(obj_cx, obj_cy), radius=40, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        cv2.arrowedLine(frame, (obj_cx, obj_cy), (obj_cx, obj_cy + 100), [240, 30, 30], 10, tipLength=0.5)
                    elif blue_direc_count < red_direc_count:
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
        frame = draw_lanes(frame, lanes)
        cv2.line(frame, (0, car_area[0]), (width, car_area[0]), (255, 0, 0), thickness=1, lineType=cv2.LINE_8)
        cv2.line(frame, (0, car_area[1] - 5), (width, car_area[1] - 5), (255, 0, 0), thickness=1, lineType=cv2.LINE_8)
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
