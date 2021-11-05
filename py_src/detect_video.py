"""
Author: Baoyun Peng
Date: 2021-09-11 17:21:37
Description: using mediapipe to detect the pose of input image list
"""
# coding: utf-8
import cv2
import mediapipe as mp
import numpy as np
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
head_idx_list = [10, 199, 234, 454, 1]
body_idx_list = [11, 12, 13, 14, 15, 16, 23, 24]

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    # refine_landmarks=True,
    min_detection_confidence=0.5,
)
pose = mp_pose.Pose(
    static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
)

def detect(video_path):
    cap = cv2.VideoCapture(video_path)

    detect_result = {}
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_info = { 'frame_width': width, 'frame_height': height, 'frame_counter': frame_counter, 'fps': fps }
    print(video_info)
    detect_result['video_info'] = video_info
    frame_idx = 0
    while(cap.isOpened() and frame_idx < frame_counter):
        # Capture frame-by-frame
        _, image = cap.read()
        if image is None:
            frame_idx += 1
            continue

        # image = cv2.imread('imgs/2.jpg')
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        img = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(cv2.flip(img, 1))
        if not results.pose_landmarks:
            detect_result[f'frame_{frame_idx}'] = 'failed'
            continue
        pose_landmarks = results.pose_landmarks.landmark
        height, width, _ = img.shape
        landmarks = []
        landmarks = [[pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z, pose_landmarks[i].visibility] for i in body_idx_list]

        c_ab_x = (pose_landmarks[23].x + pose_landmarks[24].x) / 2
        c_ab_y = (pose_landmarks[23].y + pose_landmarks[24].y) / 2
        c_ab_z = (pose_landmarks[23].z + pose_landmarks[24].z) / 2
        c_ab_v = (pose_landmarks[23].visibility + pose_landmarks[24].visibility) / 2
        
        c_sh_x = (pose_landmarks[11].x + pose_landmarks[12].x) / 2
        c_sh_y = (pose_landmarks[11].y + pose_landmarks[12].y) / 2
        c_sh_z = (pose_landmarks[11].z + pose_landmarks[12].z) / 2
        c_sh_v = (pose_landmarks[11].visibility + pose_landmarks[12].visibility) / 2

        ratio_list = [1.5, 3, 6]
        for _idx in range(3):
            ratio = ratio_list[_idx]
            _x = (c_sh_x + (ratio - 1) * c_ab_x) / ratio
            _y = (c_sh_y + (ratio - 1) * c_ab_y) / ratio
            _z = (c_sh_z + (ratio - 1) * c_ab_z) / ratio
            _v = (c_sh_v + (ratio - 1) * c_ab_v) / ratio
            landmarks.append([_x, _y, _z, _v])
        landmarks.append([c_sh_x, c_sh_y, c_sh_z, c_sh_v])

        results = face_mesh.process(cv2.flip(img, 1))
        if not results.multi_face_landmarks:
            detect_result[f'frame_{frame_idx}'] = 'failed'
            continue
        face_lms = [results.multi_face_landmarks[0].landmark[i] for i in head_idx_list]
        landmarks += [[lms.x, lms.y, lms.z, lms.visibility] for lms in face_lms]

        detect_result[f'frame_{frame_idx}'] = [item for sublist in landmarks for item in sublist]
        # # image = cv2.flip(image, 1)
        # for lms in landmarks:
        #     pos = ( int(width*lms[0]), int(height*lms[1]) )
        #     cv2.circle(image, pos, 5, (0, 255, 0), 3)
        # cv2.imshow("MediaPipe Pose", image)
        # if cv2.waitKey(10) & 0xFF == 27:
        #     exit()

        frame_idx += 1

    cap.release()
    return detect_result
