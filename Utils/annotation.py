from Models.box_Info import BoxInfo
from Utils.img import show_image
from Utils.path import get_frame_img_path, get_players_annot_path
import cv2


def load_frame_players_annot(path: str):
    with open(path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_players_boxes = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        # let's create view from frame to boxes
        for player_ID, boxes_info in player_boxes.items():
            # let's keep the middle 9 frames only (enough for this task empirically)
            boxes_info = boxes_info[5:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_players_boxes:
                    frame_players_boxes[box_info.frame_ID] = []

                frame_players_boxes[box_info.frame_ID].append(box_info)

        return frame_players_boxes


def show_annotated_frame(video: int, frame_id: int, figsize=(80, 20)):
    frame_players_boxes = load_frame_players_annot(
        get_players_annot_path(video, frame_id))
    font = cv2.FONT_HERSHEY_SIMPLEX

    img_path = get_frame_img_path(video, frame_id, frame_id)
    image = cv2.imread(img_path)

    boxes_info = frame_players_boxes[frame_id]
    for box_info in boxes_info:
        x1, y1, x2, y2 = box_info.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, box_info.category, (x1, y1 - 10),
                    font, 0.5, (255, 0, 0), 1)

    show_image(image, '', figsize=figsize)
