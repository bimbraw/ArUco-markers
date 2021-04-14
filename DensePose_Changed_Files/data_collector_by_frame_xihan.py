import numpy as np
from cv2 import aruco
from cv2 import cv2
import matplotlib.pyplot as plt
import math
import csv

camera_matrix = np.array(
    [[662.1790, 0.0, 322.3619], [0.0, 662.8344, 252.0131], [0.0, 0.0, 1.0]])

dist_coeff = np.array([0.0430651, -0.1456001, 0.0, 0.0])


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def getBodyPart(IUV, part_id):
    IUV_chest = np.zeros((IUV.shape[0], IUV.shape[1], IUV.shape[2]))
    torso_idx = np.where(IUV[:, :, 0] == part_id)
    IUV_chest[torso_idx] = IUV[torso_idx]

    return IUV_chest


def detectFiducial(frame):
    # marker detection
    fid_id = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # pose estimation
    try:
        loc_marker = corners[np.where(ids == fid_id)[0][0]]
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
            loc_marker, 0.048, camera_matrix, dist_coeff)
        marker_frame = aruco.drawAxis(
            frame, camera_matrix, dist_coeff, rvecs, tvecs, 0.15)

        rmat = cv2.Rodrigues(rvecs)[0]
        RotX = rotationMatrixToEulerAngles(rmat)[1]
        RotX_formatted = float("{0:.2f}".format(-RotX*180/3.14))     # 2 digits
    except:
        RotX_formatted = -1

    return marker_frame, RotX_formatted


def detectMarker(frame):
    # marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    marker_frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # marker position
    num_markers = 4
    pos = np.zeros((num_markers, 2))
    for i in range(1, num_markers+1):
        try:
            marker = corners[np.where(ids == i)[0][0]][0]
            pos[i-1, :] = [marker[:, 0].mean(), marker[:, 1].mean()]
        except:
            pos[i-1, :] = [-1, -1]      # if marker is not detected
        # print("id{} center:".format(i), pos[i-1, 0], pos[i-1, 1])

    return marker_frame, pos


def createMask(IUV, IUV_chest, frame):
    opacity = 0.3
    mask = np.zeros((IUV.shape[0], IUV.shape[1], IUV.shape[2]))
    mask[:, :, 1] = IUV_chest[:, :, 0]*100
    overlay = cv2.addWeighted(mask, opacity, frame, 1.-opacity, -10., dtype=1)
    return overlay


def getUV(IUV_chest, pos):
    UV = np.zeros((pos.shape[0], pos.shape[1]))
    for id in range(pos.shape[0]):
        if pos[id, 0] != -1 and pos[id, 1] != -1:
            row = int(pos[id, 1])
            col = int(pos[id, 0])
            UV[id, 0] = IUV_chest[row, col, 1]     # store U
            UV[id, 1] = IUV_chest[row, col, 2]     # store V
        else:
            UV[id, 0] = -1
            UV[id, 1] = -1
    return UV


def initVideoStream():
    cap = cv2.VideoCapture(0)
    focus = 0               # min: 0, max: 255, increment:5
    cap.set(28, focus)      # manually set focus
    return cap


def getVideoStream(cap):
    # patch_size = 480
    _, frame = cap.read()
    #frame = frame[:, 80:560]
    return frame


def main():
    part_id = 2
    save_dir = '/home/keshav/frame.png'
    infer_dir = '/home/keshav/frame_IUV.png'
    cap = initVideoStream()

    while(True):
        frame = getVideoStream(cap)
        key = cv2.waitKey(33)
        cv2.imwrite(save_dir, frame)
        frame, angleX = detectFiducial(frame)   # detect fiducial marker
        frame, pos = detectMarker(frame)        # detect overlay marker

        try:
            inferred = cv2.imread(infer_dir)
        except Exception as e:
            print('error: '+str(e))

        if inferred is not None:
            IUV_chest = getBodyPart(inferred, part_id)
            frame = createMask(inferred, IUV_chest, frame)
            UV = getUV(IUV_chest, pos)
        else:
            UV = -1*np.ones((4, 2))

        cv2.imshow('frame', frame)

        if key == ord('q'):   # quit
            print('exit')
            break
        elif key == ord('c'):   # capture
            # print 'data captured: (u,v) {}'.format(UV)
            # save v
            with open('./angle_v.csv', 'a') as file_out:
                writer = csv.writer(file_out)
                for i in range(len(UV)):
                    writer.writerow(
                        [i+1, angleX, UV[i, 1]])
            # save u
            with open('./angle_u.csv', 'a') as file_out:
                writer = csv.writer(file_out)
                for i in range(len(UV)):
                    writer.writerow(
                        [i+1, angleX, UV[i, 0]])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

