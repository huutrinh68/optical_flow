import numpy as np
import time
import cv2
from tracker import CentroidTracker

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
    if len(points):
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
    # video_path = "./20211215_102922.mp4"
    video_path = "./20211211_194628.mp4"
    cap = cv2.VideoCapture(video_path)

    color = (0, 255, 0)
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (first_frame.shape[1] // 3, first_frame.shape[0] // 3))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    height, width = first_frame.shape[:2]
    # min size of car
    w1_size = 70
    h1_size = 70
    car_area = [int(height * 0.65), int(height * 1.0) - 5]
    count_down = 0
    count_up = 0
    # movement quantity
    dis_move = 3.5
    # use in dilation
    radius = 20

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
                0: [[345, 186], [469, 190], [538, 358], [353, 357]],
                1: [[45, 182], [318, 200], [331, 354], [2, 357]]
            }
    for i in range(len(lanes)):
        lanes[i] = np.array(lanes[i])
    # ************setting lane*************

    bgs = cv2.createBackgroundSubtractorMOG2()

    frame_num = 0
    ct = CentroidTracker()
    counted_ids = []
    while(cap.isOpened()):
        start_time = time.perf_counter()
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % 1 != 0:
            continue
        bin_image = np.zeros((height, width))
        dir_image = np.zeros((height, width))
        cen_image = np.zeros((height, width))
        frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        origin = frame.copy()
        # ****** setting lane******
        lanes_mask = np.zeros((height, width))
        for lane in lanes.values():
            cv2.fillPoly(lanes_mask, pts=[lane], color=255)

        # lanes_mask = cv2.fillPoly(lanes_mask, pts=[list(lanes.values())[0]], color=255)
        _, lanes_mask = cv2.threshold(lanes_mask, 0, 255, cv2.THRESH_BINARY)
        frame = frame.astype(np.uint8)
        lanes_mask = lanes_mask.astype(np.uint8)
        frame = cv2.bitwise_and(frame, frame, mask=lanes_mask)
        # ****** setting lane******

        # ****** background subtraction ********
        binary_mask, contours = background_subtraction(bgs, frame)
        binary_mask = binary_mask.astype(np.uint8)
        lanes_mask = lanes_mask.astype(np.uint8)
        bin_image = cv2.bitwise_and(binary_mask, lanes_mask)
        bboxes = list(map(lambda x: cv2.boundingRect(x), contours))
        # ****** background subtraction ********

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            onlane = on_lane(frame, lanes, x + w // 2, y + h // 2)
            # center in mask
            if overlap is not None and onlane is not None:
                coords = cv2.findNonZero(car)
                car_cx = []
                car_cy = []
                if coords is not None:
                    for coord in coords:
                        car_cx.append(coord[0][0])
                        car_cy.append(coord[0][1])
                    cx = x + int(np.mean(car_cx))
                    cy = y + int(np.mean(car_cy))
                    cen_image[cy - radius: cy + radius, cx - radius:cx + radius] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cen_image = cv2.dilate(cen_image, kernel, iterations=15)
        cen_image = cv2.erode(cen_image, kernel, iterations=5)
        cen_image = cen_image.astype(np.uint8)
        contours = cv2.findContours(cen_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        bboxes = list(map(lambda x: cv2.boundingRect(x), contours))

        # ******** Tracking **********
        # xywh -> xyxy
        xyxy_bboxes = [(x1, y1, x1 + w1, y1 + h1) for (x1, y1, w1, h1) in bboxes if (w1 > w1_size and h1 > h1_size)]
        objects = ct.update(xyxy_bboxes)
        # ******** Tracking **********

        # ******** Counting **********
        # overlap bin_mask with dir_image
        dir_image = dir_image.astype(np.uint8)
        dir_image = cv2.bitwise_and(bin_image, dir_image)
        for xyxy, (objectID, centroid) in zip(xyxy_bboxes, objects.items()):
            # draw both the ID of the object and the centroid of the
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.rectangle(origin, (x1, y1), (x2, y2), color, 5)
            text = "ID {}".format(objectID + 1)
            cv2.putText(origin, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(origin, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            obj_cx, obj_cy = centroid[0], centroid[1]

            tmp = dir_image[y: y + h, x: x + w]
            tmp = tmp.astype(np.uint8)

            if obj_cy >= car_area[0] and obj_cy <= car_area[1]:
                red_direc_count = np.count_nonzero(tmp == 200)
                blue_direc_count = np.count_nonzero(tmp == 100)
                if objectID not in counted_ids:
                    if blue_direc_count > red_direc_count:
                        count_down += 1
                        cv2.circle(origin, center=(obj_cx, obj_cy), radius=40, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        cv2.circle(origin, center=(obj_cx, obj_cy), radius=40, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        counted_ids.append(objectID)
                    elif blue_direc_count < red_direc_count:
                        count_up += 1
                        cv2.circle(frame, center=(obj_cx, obj_cy), radius=40, color=(30, 30, 240), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        cv2.circle(origin, center=(obj_cx, obj_cy), radius=40, color=(30, 30, 240), thickness=-1, lineType=cv2.LINE_4, shift=0)
                        counted_ids.append(objectID)
        print(counted_ids)
        # ******** Counting **********

        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        cv2.putText(origin, text='down ' + str(count_down),
                    org=(frame.shape[1] * 2 // 3, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 255),
                    thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(frame, text='down ' + str(count_down),
                    org=(frame.shape[1] * 2 // 3, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 255),
                    thickness=3, lineType=cv2.LINE_AA)

        cv2.putText(origin,
                    text='up ' + str(count_up),
                    org=(frame.shape[1] // 4, frame.shape[0] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA)
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

        cv2.putText(origin,
                    text='FPS:' + str(fps),
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(200, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA)
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
        origin = draw_lanes(origin, lanes)
        cv2.line(frame, (0, car_area[0]), (width, car_area[0]), (255, 0, 0), thickness=1, lineType=cv2.LINE_8)
        cv2.line(frame, (0, car_area[1] - 5), (width, car_area[1] - 5), (255, 0, 0), thickness=1, lineType=cv2.LINE_8)
        cv2.line(origin, (0, car_area[0]), (width, car_area[0]), (255, 0, 0), thickness=1, lineType=cv2.LINE_8)
        cv2.line(origin, (0, car_area[1] - 5), (width, car_area[1] - 5), (255, 0, 0), thickness=1, lineType=cv2.LINE_8)
        binary = one_channel_to_three_channel(bin_image, frame)
        center = one_channel_to_three_channel(cen_image, frame)
        direct = one_channel_to_three_channel(dir_image, frame)
        result1 = hconcat_resize([origin, binary])
        # result1 = hconcat_resize([frame, binary])
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
