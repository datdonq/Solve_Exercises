import cv2
import os
import numpy as np
def sort_bbox(bounding_boxes):
    updated_boxes = []
    # Duyệt qua từng bounding box
    for i, box in enumerate(bounding_boxes):
        if i not in updated_boxes:
            y_values = [box[0][1]]
            num_y_values = 1
            need_update=[]
            need_update.append(i)
            # Duyệt qua các bounding box còn lại
            for j, other_box in enumerate(bounding_boxes):
                if i != j and j not in updated_boxes:
                    # Kiểm tra chênh lệch y giữa hai bounding box
                    diff_y = abs(other_box[0][1] - box[0][1])
                    if diff_y < 10:
                        need_update.append(j)
                        y_values.append(other_box[0][1])
                        num_y_values += 1
                        # updated_boxes.append(j)  # Đánh dấu bounding box đã được cập nhật
            # Tính trung bình cộng của các giá trị y tìm được
            if num_y_values > 1:
                avg_y = sum(y_values) / num_y_values
                # Cập nhật lại giá trị y của tất cả các bounding box
                for j, other_box in enumerate(bounding_boxes):
                    if j not in updated_boxes and j in need_update:
                        other_box[0][1] = avg_y
                        other_box[1][1] = avg_y
                        updated_boxes.append(j)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0][1], box[0][0]))
    return bounding_boxes

def crop_image_with_boxes(image, bounding_boxes):
    cut_images=[]
    for i, box in enumerate(bounding_boxes):
        pts = [[int(x), int(y)] for x, y in box]
        pts = np.array(pts)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = image[y:y+h, x:x+w].copy()
        cut_images.append(cropped)
    return cut_images
    