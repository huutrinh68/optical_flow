######################################################
# import
######################################################
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from collections import OrderedDict

######################################################
# 設定すべきパラメータ
# lk_params: オプティカルフローのパラメータ
#      1. winSize: オプティカルフローの推定の計算に使う周辺領域サイズ 小さくするとノイズに敏感になり、大きな動きを見逃す可能性
#      2. maxLevel: ピラミッド数 (デフォルト0：2なら1/4画像まで使用) ピラミッドを使用すると、画像のさまざまな解像度でオプティカルフローを見つけることができる
#      3. criteria: 探索アルゴリズムの終了条件　cv2.TERM_CRITERIA_EPSは指定された精度(epsilon)に到達　cv2.TERM_CRITERIA_COUNTは指定された繰り返しの最大回数
# feature_params: 特徴点算出のパラメータ
#      1. maxCorners: 検出するコーナの数
#      2. qualityLevel: 最良値(最大固有値の割合?), double
#      3. minDistance: この距離内のコーナーを棄却, double
#      4. blockSize: 使用する近傍領域のサイズ, int
# BinarIMG: オプティカルフローを二値化する表示するマトリックス
#      1. ps: 一つのオプティカルフローの点をどれくらいのサイズで表示するか。値が小さいと、車の特定が難しくなるが車同士が接近している場合は有効。値が大きいと、車の特定が簡単になるが車がすれ違う場合に不便になる
# OTHERS:
#      1. min_opti: 最小のオプティカルフローの移動量
#      2. opening_size: オープニング処理のカーネルサイズ
#      3. w1_size: バウンディングボックスの大きさがw1_size以上で車と認識する
#      4. h1_size: バウンディングボックスの大きさがh1_size以上で車と認識する
#      5. min_dist: 同じIDの車と認識する最小距離
#      6. car_area: 車をカウントする範囲


######################################################
# プログラムで使用するパラメータ
######################################################
video_path = './20211215_102922.mp4'
win_size = (10, 10)
maxLevel = 2
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
maxCorners = 200
qualityLevel = 0.1
minDistance = 7
blockSize = 7
ps = 50
ps_dict = 30
min_opti = 4.5
opening_size = 65
w1_size = 110
h1_size = 110
min_dist = 100
car_area = [150, 300]


######################################################
# オプティカルフロー関数
# オプティカルフローを実装します。
# input:
#     old_gray: グレースケール化された一つ前のフレーム
#     frame_gray: グレースケール化された現在のフレーム
#     p0: 一つ前のフレーム野特徴点
# output:
#     good_new: 現在フレームの特徴
#     good_old: 一つ前のフレームの特徴
######################################################
def optical_flow(old_gray, frame_gray, p0):
    # Lucas-Kanade法のパラメータ
    # winSize: オプティカルフローの推定の計算に使う周辺領域サイズ 小さくするとノイズに敏感になり、大きな動きを見逃す可能性
    # maxLevel: ピラミッド数 (デフォルト0：2なら1/4画像まで使用) ピラミッドを使用すると、画像のさまざまな解像度でオプティカルフローを見つけることができる
    # criteria: 探索アルゴリズムの終了条件　cv2.TERM_CRITERIA_EPSは指定された精度(epsilon)に到達　cv2.TERM_CRITERIA_COUNTは指定された繰り返しの最大回数
    lk_params = dict(winSize=win_size,
                     maxLevel=maxLevel,
                     criteria=criteria)

    # p1: 検出した対応点, numpy.ndarray
    # st: 各点において，見つかれば1(True), 見つからなければ0(False), numpy.ndarray
    # err: 検出した点の誤差, numpy.ndarray
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        return good_new, good_old
    else:
        return None, None


######################################################
# ビデオ読み込み
######################################################
cap = cv2.VideoCapture(video_path)
# 車の進行方向　0: 横　1: 縦
DIREC = 1


######################################################
# Shi-Tomasiのコーナー検出パラメータ
# maxCorners: 検出するコーナの数
# qualityLevel: 最良値(最大固有値の割合?), double
# minDistance: この距離内のコーナーを棄却, double
# blockSize: 使用する近傍領域のサイズ, int
######################################################
feature_params = dict(maxCorners=maxCorners,
                      qualityLevel=qualityLevel,  # 0~1
                      minDistance=minDistance,  # 7
                      blockSize=blockSize)  # 7


######################################################
# 最初のフレームの処理
######################################################
ret, old_frame = cap.read()
print(old_frame.shape)
old_frame = cv2.resize(old_frame, (old_frame.shape[1] // 2, old_frame.shape[0] // 2))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


######################################################
# 輪郭と輪郭の交点(コーナー)を求める
######################################################
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


######################################################
# プログラムで使用する変数たちの初期化
######################################################
frame_count = 0
count_down = 0
count_up = 0
car_obj = OrderedDict()
count_flag = OrderedDict()
nextcarID = 1
width, height, channel = old_frame.shape


######################################################
# メインの処理
# 動画に対するオプティカルフロー処理と車カウント
######################################################
while(1):
    # 処理時間計測
    start_time = time.perf_counter()
    frame_count += 1

    # 現在のフレームを読み込み
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー
    good_new, good_old = optical_flow(old_gray, frame_gray, p0)
    # オプティカルフローに二値化表示や、中心点描画に使うマトリックス
    # direcMat: 車の方向を示すマトリックス
    binarIMG = np.zeros((width, height))
    centerIMG = np.zeros((width, height))
    direcMat = np.zeros((width, height))
    # 現在のフレーム上に動きがあるならば、以下の処理を実装する
    if good_new is not None:
        min_distance = 0.8

        # フレーム上の各特徴点に関して処理
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # 現在のフレーム上のある特徴点
            a, b = new.ravel()

            # 一つ前のフレーム上のある特徴点
            c, d = old.ravel()

            # 車の進行方向が縦方向ならば、以下の処理
            if DIREC == 1:
                dis = b - d
                a = c  # 間違った横方向の矢印を消す
            # 車の進行方向が横方向ならば、以下の処理
            else:
                dis = a - c
                b = d  # 間違った縦方向の矢印を消す

            # 動きの小さいオプティカルフロー以外を表示
            if dis > min_opti:  # 進行方向が縦ならば、下方向に移動する車。
                binarIMG[int(d) - ps:int(d) + ps, int(c) - ps:int(c) + ps] = 1
                direcMat[int(d) - ps_dict:int(d) + ps_dict, int(c) - ps_dict:int(c) + ps_dict] = 100
            elif dis < -min_opti:  # 進行方向が縦ならば、上方向に移動する車。
                binarIMG[int(d) - ps:int(d) + ps, int(c) - ps:int(c) + ps] = 1
                direcMat[int(d) - ps_dict:int(d) + ps_dict, int(c) - ps_dict:int(c) + ps_dict] = 200

    ######################################################
    # オープニング処理
    # 目的：オプティカルフローの特徴点を二値化した点を連結する。特徴点がどの車を示すのかまとめる。
    # kernel : MORPH_RECT 長方形
    #          MORPH_CROSS 十字
    #          MORPH_ELLIPSE 楕円
    ######################################################
    kernel_size = (opening_size, opening_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binarIMG = cv2.morphologyEx(binarIMG, cv2.MORPH_OPEN, kernel, iterations=2)

    ######################################################
    # ポレゴンの中心を求める
    # FindContorusを使って、オブジェクトの中心点を求める
    ######################################################
    binarIMG_8bit = binarIMG.astype(np.uint8)
    contours, _ = cv2.findContours(binarIMG_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # 輪郭点からバウンディングボックスを求める
        (x1, y1, w1, h1) = cv2.boundingRect(c)

        # バウンディングボックスの大きさが一定以上ならば、以下の処理を実装
        if w1 > w1_size and h1 > h1_size:
            # 車を囲む
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 5)

            # 車の中心点を計算
            x2 = w1 / 2
            y2 = h1 / 2
            cx = int(x1 + x2)
            cy = int(y1 + y2)
            centroid = np.array([[cx, cy]])
            print('cx:{}, cy:{}'.format(cx, cy))
            centerIMG[cy - 10:cy + 10, cx - 10:cx + 10] = 1

            ###########################
            # ID判定
            # 目的：車ごとにIDを流布し、既に数えた車とそうでない車を区別するために使用
            # 方法：
            #   1. 画面に映った車にIDを流布
            #   2. 新たに求められた中心点と、元から各IDに登録されている中心点との距離を算出
            #   3. ある閾値よりも距離が近い中心点は特定のIDの中心点として認識。もし、現在登録しているIDの中心点よりも距離が遠いならば、新たに画面上に車が入ってきたと考え新たなIDを発行
            ###########################
            if len(car_obj) == 0:
                car_obj[nextcarID] = centroid
                count_flag[nextcarID] = 0
                nextcarID += 1
            else:
                #############################
                # 求めたセンターから距離を計算
                # 近いならIDの中心点更新
                # 遠いならIDを発行して車を新規登録
                #############################
                # 距離行列
                objectCentroids = list(car_obj.values())
                # 新たな中心点との距離を計算
                D = distance.cdist(np.array(objectCentroids).reshape(-1, 2), centroid)
                D = D.reshape(-1)
                inx_d = np.argmin(D)

                # もし新たな中心点がIDと距離100より近いなら
                if D[inx_d] <= min_dist:
                    # 車の中心点を更新
                    car_obj[inx_d + 1] = centroid
                # 遠いなら
                else:
                    # 新しい車が入ってきた為、新たにIDを配布
                    car_obj[nextcarID] = centroid
                    count_flag[nextcarID] = 0
                    nextcarID += 1

    ######################################################
    # 車カウント処理
    ######################################################
    for carID, val in car_obj.items():
        print('ID:{} centroid:{}'.format(carID, val))
        print('count:{}'.format(count_flag[carID]))
        obj_cx = val[0, 0]
        obj_cy = val[0, 1]

        # もしカウントエリアに入ったならば、以下の処理を実装
        if obj_cy >= car_area[0] and obj_cy <= car_area[1]:
            # もしまだカウントされていないならば、以下の処理を実装
            if count_flag[carID] == 0:
                ###############################
                # カウントされる車の方向を判定する。
                # 方法:
                #    1. 該当するエリア周辺の進行方向情報(directMat)を参照し、各方向をカウント
                #    2. 各方向カウント数が多い方が、車の進行方向として車をカウント
                #    3. カウントされた車はカウント済みフラグを立てる
                ###############################
                red_direc_count = np.count_nonzero(direcMat[obj_cy - 10:obj_cy + 10, obj_cx - 10:obj_cx + 10] == 200)
                blue_direc_count = np.count_nonzero(direcMat[obj_cy - 10:obj_cy + 10, obj_cx - 10:obj_cx + 10] == 100)
                if blue_direc_count > red_direc_count:
                    count_down += 1
                    cv2.circle(frame, center=(obj_cx, obj_cy), radius=40, color=(240, 30, 30), thickness=-1, lineType=cv2.LINE_4, shift=0)
                    cv2.arrowedLine(frame, (obj_cx, obj_cy), (obj_cx, obj_cy + 100), [240, 30, 30], 10, tipLength=0.5)
                elif blue_direc_count <= red_direc_count:
                    count_up += 1
                    cv2.circle(frame, center=(obj_cx, obj_cy), radius=40, color=(30, 30, 240), thickness=-1, lineType=cv2.LINE_4, shift=0)
                    cv2.arrowedLine(frame, (obj_cx, obj_cy), (obj_cx, obj_cy - 100), [30, 30, 240], 10, tipLength=0.5)
                # カウント済み
                count_flag[carID] = 1

            cv2.putText(frame, text='carID[' + str(carID) + ']', org=(obj_cx, obj_cy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)

    cv2.putText(frame, text='down ' + str(count_down), org=(550, 500),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    cv2.putText(frame,
                text='up ' + str(count_up),
                org=(200, 500),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 255),
                thickness=3,
                lineType=cv2.LINE_AA)

    # 処理時間計測
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

    # 矢印を含めて映し出す。
    cv2.imshow('frame', frame)
    # cv2.imshow('binar frame', binarIMG)
    # cv2.imshow('center frame', centerIMG)

    # ESCキー押下で終了
    if cv2.waitKey(30) & 0xff == 27:
        break

    ######################################################
    # 次のフレーム、特徴点の準備
    ######################################################
    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


# 終了処理
cv2.destroyAllWindows()
cap.release()
