# -*- coding: utf-8 -*-

from ultralytics import YOLO
from ultralytics.solutions import object_counter
from ultralytics.utils.plotting import Annotator, colors
import cv2
from collections import defaultdict
import time
import numpy as np

# 初始化YOLO模型
model = YOLO("F:\\ultralytics\\runs\\segment\\train3\\weights\\best.pt")

cap = cv2.VideoCapture("F:\\0.8m_bar\\big_column_train.MP4")

assert cap.isOpened(), "读取视频文件出错"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# 获取模型的类别名称
class_names = model.names
# 创建从类别名称到索引的映射
class_to_index = {name: idx for idx, name in class_names.items()}

# 确保'T-R-10'和'Bend-R'在class_names中，并获取它们的索引
classes_to_count = []
for class_name in ["T-R-10", "Bend-R"]:
    if class_name in class_to_index:
        class_index = class_to_index[class_name]
        classes_to_count.append(class_index)
    else:
        raise ValueError(f"模型的类别名称中未找到'{class_name}'。")

# 视频写入器
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# 初始化对象计数器
counter = object_counter.ObjectCounter(
    view_img=True,
    reg_pts=[(0.2 * w, 0.2 * h), (0.8 * w, 0.2 * h)],
    classes_names=class_names,
    draw_tracks=True,
    line_thickness=2,
)

cumulative_count = defaultdict(int)  # 初始化累计计数


def calculate_vertical_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    distance = abs(y1_center - y2_center)
    return distance


distance_threshold = 60  # 距离阈值，需要根据实际情况调整

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # YOLO检测
    results = model(frame)
    detections = results[0].boxes  # 获取results中的检测结果
    boxes = []

    for detection in detections:
        class_id = detection.cls.item()
        if class_id in classes_to_count:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2, class_id])

    # 提取所有检测到的T-R-10模型钢筋
    rebars = [box for box in boxes if box[4] == class_to_index["T-R-10"]]

    if len(rebars) >= 2:
        for i in range(len(rebars) - 1):
            for j in range(i + 1, len(rebars)):
                distance = calculate_vertical_distance(rebars[i], rebars[j])
                color1 = (0, 255, 0)  # 第一个钢筋的颜色
                color2 = (0, 255, 0)  # 第二个钢筋的颜色

                if distance > distance_threshold:
                    color2 = (0, 0, 255)  # 如果距离超过阈值，第二个钢筋的边界框标红

                # 画出钢筋的边界框
                cv2.rectangle(
                    frame,
                    (rebars[i][0], rebars[i][1]),
                    (rebars[i][2], rebars[i][3]),
                    color1,
                    2,
                )
                cv2.rectangle(
                    frame,
                    (rebars[j][0], rebars[j][1]),
                    (rebars[j][2], rebars[j][3]),
                    color2,
                    2,
                )

                # 画出垂直线并显示距离
                cv2.line(
                    frame, (50, rebars[i][1]), (50, rebars[j][1]), (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"{distance:.2f} px",
                    (60, (rebars[i][1] + rebars[j][1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

    # 显示处理后的帧
    video_writer.write(frame)
    cv2.imshow("Frame", frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
