def calc_l_point(x_center, y_center):
    """
    Change 640 x 480 frame coordinate to 40x30 L point map coordinate

    Args:
    x_center: frame point x coordinate (0 to 639)
    y_center: frame point y coordinate (0 to 479)

    Returns:
    l_point_index : 40x30 L point map coordinate (0 to 1199)

    """

    # Mapping frame to L point map
    Lmap_x = int((x_center + 1) / 16)   # 1 to 40
    Lmap_y = int((y_center + 1) / 16)      # 1 to 30
    # print('Lmap X', Lmap_x, 'Lmap Y', Lmap_y)

    # Mapping L point map to L point index
    if Lmap_y == 0:
        if Lmap_x == 0:
            l_point_index = Lmap_x
        else:
            l_point_index = abs(Lmap_x - 1)
    elif Lmap_y > 0 :
        l_point_index = (int(Lmap_y-1)*40) + int(Lmap_x-1)

    return l_point_index


# if __name__ == "__main__":
#     x_center = 319
#     y_center = 239
#     l_point_index = calc_l_point(x_center, y_center)
#     print("L point index val:", l_point_index)


    # def multiple_yolo_detection(self, webcam_frame):
    #     # Resize and convert the frame for model input
    #     original_frame_height, original_frame_width = webcam_frame.shape[:2]
    #     webcam_frame = cv2.resize(webcam_frame, (640, 640))
    #     frame_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)

    #     # Model inference
    #     results = self.yolo_model([frame_rgb])
    #     labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    #     # Storing results
    #     self.yolo_list = []
    #     for label, coord in zip(labels, coordinates):
    #         if coord[4] >= 0.8:  # Confidence threshold
    #             x1, y1, x2, y2 = coord[0:4] * [original_frame_width, original_frame_height, original_frame_width, original_frame_height]
    #             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    #             class_name = self.class_names[int(label)]
    #             yolo_x = (x1 + x2) / 2
    #             yolo_y = (y1 + y2) / 2

    #             result = {
    #                 "class_name": class_name,
    #                 "score": coord[4],
    #                 "x_coordinate": yolo_x,
    #                 "y_coordinate": yolo_y
    #             }
    #             self.yolo_list.append(result)

    #             # Drawing bounding boxes (optional)
    #             cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(webcam_frame, f"{class_name}: {coord[4]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     print('YOLO LIST:', self.yolo_list)
    #     return self.yolo_list