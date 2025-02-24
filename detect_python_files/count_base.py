# -*- coding: utf-8 -*-

from ultralytics import YOLO
from ultralytics.solutions import object_counter
from ultralytics.utils.plotting import Annotator, colors
import cv2
from collections import defaultdict
import time
import numpy as np

# 初始化YOLO模型
model = YOLO("F:\\ultralytics\\runs\\segment\\train3\\weights\\column.pt")

cap = cv2.VideoCapture("F:\\0.8m_bar\\big_column_train.MP4")

assert cap.isOpened(), "读取视频文件出错"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# 获取模型的类别名称
class_names = model.names

# 确保'T-R-10'和'Bend-R'在class_names中，并获取它们的索引
classes_to_count = []
for class_name in ["T-R-10", "Bend-R"]:
    if class_name in class_names.values():
        class_index = list(class_names.values()).index(class_name)
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


def calculate_segmentation_area(mask):
    """使用提供的掩码张量计算分割面积。"""
    mask_array = np.array(
        mask, dtype=object
    )  # 将列表转换为带有对象类型的numpy数组以处理序列
    return np.sum([np.sum(m > 0) for m in mask_array])  # 计算每个分割中所有正像素的总和


def calculate_box_area(box):
    """计算框的面积。"""
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_parallel_rebar_distance(centers):
    """计算最近的平行rebar之间的中心距离。"""
    if len(centers) < 2:
        return None  # 如果少于2个中心点，则无法计算距离

    centers.sort(key=lambda x: x[0])  # 根据x坐标排序
    min_distance = float("inf")

    for i in range(len(centers) - 1):
        for j in range(i + 1, len(centers)):
            if centers[j][0] - centers[i][0] > min_distance:
                break  # 由于已经排序，后续的距离只会更大
            distance = abs(centers[j][1] - centers[i][1])
            if distance < min_distance:
                min_distance = distance

    return min_distance


def draw_annotations(annotator, results, masks):
    total_count = defaultdict(int)
    qualified_count = defaultdict(int)
    centers = []

    for result, mask in zip(results, masks):
        box = result["box"]
        cls = result["class"]
        box_area = calculate_box_area(box)  # 框的面积

        segment_area = calculate_segmentation_area(mask)  # 分割面积

        ratio = segment_area / box_area
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        centers.append((center, box))

        total_count[class_names[cls]] += 1
        if ratio > 0.5:  # 仅当分割面积占比超过50%时计数
            qualified_count[class_names[cls]] += 1

        label = f"{class_names[cls]}: 1, Box Area: {box_area:.2f}, Segment Area: {segment_area:.2f}, Ratio: {ratio:.2f}"
        annotator.box_label(box, label, color=colors(cls, True))

        if isinstance(mask, list):
            mask = [
                np.array(m, dtype=np.int32) for m in mask
            ]  # 转换为具有正确类型的numpy数组

        cv2.polylines(
            annotator.im, mask, isClosed=True, color=colors(cls, True), thickness=2
        )

    # 绘制包含所有钢筋的边界框
    if centers:
        min_x = int(min(c[0][0] for c in centers))
        max_x = int(max(c[0][0] for c in centers))
        min_y = int(min(c[0][1] for c in centers))
        max_y = int(max(c[0][1] for c in centers))
        cv2.rectangle(annotator.im, (min_x, min_y), (max_x, max_y), (0, 255, 0), 5)

    min_distance = calculate_parallel_rebar_distance([c[0] for c in centers])
    print(f"Min Parallel Rebar Distance: {min_distance}")  # Debug print statement

    if min_distance is not None:
        cv2.putText(
            annotator.im,
            f"Min Parallel Rebar Distance: {min_distance:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    # 计算每对钢筋中心的距离并在每个框上显示
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i][0]) - np.array(centers[j][0]))
            box = centers[i][1]
            next_box = centers[j][1]
            mid_point_x = int((box[0] + box[2]) / 2)
            mid_point_y = int((box[1] + box[3]) / 2)
            next_mid_point_x = int((next_box[0] + next_box[2]) / 2)
            next_mid_point_y = int((next_box[1] + next_box[3]) / 2)
            cv2.putText(
                annotator.im,
                f"{dist:.2f}",
                (mid_point_x, mid_point_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotator.im,
                f"{dist:.2f}",
                (next_mid_point_x, next_mid_point_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    # 计算合格率
    for cls in classes_to_count:
        class_name = class_names[cls]
        if total_count[class_name] > 0:
            qualification_rate = qualified_count[class_name] / total_count[class_name]
            print(f"{class_name}的合格率: {qualification_rate:.2f}")

    # 更新累计计数
    for cls in classes_to_count:
        class_name = class_names[cls]
        cumulative_count[class_name] += total_count[class_name]

    # 计算框内分段的数量
    in_box_count = sum(1 for center in centers if min_x <= center[0][0] <= max_x)

    # 在视频上显示总计数和框内计数
    cv2.putText(
        annotator.im,
        f"Total Count: {sum(total_count.values())}, In Box Count: {in_box_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    return total_count, qualified_count


start_time = time.time()
while cap.isOpened():
    success, im0 = cap.read()
    if not success or (time.time() - start_time) > 300:  # 确保至少处理300秒
        print("视频帧为空或视频处理已成功完成。")
        break

    annotator = Annotator(im0, line_width=2)

    # 使用YOLO模型处理帧
    results = model.track(im0, persist=True, classes=classes_to_count)

    filtered_results = []
    masks = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls in classes_to_count:
                box_coords = box.xyxy[0].tolist()  # 确保坐标格式正确
                if result.masks is not None:
                    mask = result.masks.xy  # 获取掩码数据
                    filtered_results.append(
                        {
                            "box": box_coords,
                            "class": cls,
                        }
                    )
                    masks.append(mask)

    # 计数对象
    im0 = counter.start_counting(im0, results)

    # 自定义注释绘制，包含计数和面积
    draw_annotations(annotator, filtered_results, masks)

    im0 = annotator.result()
    video_writer.write(im0)
    cv2.imshow("Object Counting", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
