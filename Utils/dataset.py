from Models.box_Info import BoxInfo
from Utils.visualization import add_box, show_clip, show_image
import app_config as cf
import cv2
import pickle
import os


def get_video_path(video: int):
    return f'{cf.get_config().dataset.videos_path}/{str(video)}'


def get_frame_img_path(video: int, clip: int, frame: int):
    return f'{cf.get_config().dataset.videos_path}/{str(video)}/{str(clip)}/{str(frame)}.jpg'


def get_players_box_annot_path(video: int, frame: int):
    return f'{cf.get_config().dataset.tracking_boxes_annotation_path}/{str(video)}/{str(frame)}/{str(frame)}.txt'


def get_video_annot_path(video: int):
    return f'{cf.get_config().dataset.videos_path}/{str(video)}/annotations.txt'


def load_frame_players_box_annot(path: str):
    with open(path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_players_boxes = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_players_boxes:
                    frame_players_boxes[box_info.frame_ID] = []

                frame_players_boxes[box_info.frame_ID].append(box_info)

        return frame_players_boxes


def show_players_with_box_frame(video: int, clip: int, frame_id: int, figsize=(80, 20)):
    _annot_path = get_players_box_annot_path(video, clip)
    frame_players_boxes = load_frame_players_box_annot(_annot_path)

    _img_path = get_frame_img_path(video, clip, frame_id)
    image = cv2.imread(_img_path)

    boxes_info = frame_players_boxes[frame_id]
    for box_info in boxes_info:
        add_box(image, box_info)

    show_image(image, f'Video {video} - Frame {frame_id}', figsize=figsize)


def show_players_with_box_clip(video: int, clip: int):
    images = []
    _annot_path = get_players_box_annot_path(video, clip)
    frame_players_boxes = load_frame_players_box_annot(_annot_path)

    for _frame_id, boxes_info in frame_players_boxes.items():
        _img_path = get_frame_img_path(video, clip, _frame_id)
        image = cv2.imread(_img_path)

        for box_info in boxes_info:
            add_box(image, box_info)
        images.append(image)

    show_clip(images)

# ToDo
# def show_annotated_video(video: int):


def load_video_annot(video: int):
    with open(get_video_annot_path(video), 'r') as file:
        clip_category_map = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_map[clip_dir] = items[1]

        return clip_category_map


def load_dataset_full_annotations():
    videos_root = cf.get_config().dataset.videos_path
    tracking_boxes_annotation_root = cf.get_config(
    ).dataset.tracking_boxes_annotation_path

    videos_dirs = os.listdir(cf.get_config().dataset.videos_path)
    videos_dirs.sort()

    videos_annot = {}

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        clip_category_map = load_video_annot(video_dir)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            assert clip_dir in clip_category_map

            annot_file = os.path.join(
                tracking_boxes_annotation_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_players_boxes = load_frame_players_box_annot(annot_file)

            clip_annot[clip_dir] = {
                'label': clip_category_map[clip_dir],
                'frame_players_boxes': frame_players_boxes
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version():
    annots = load_dataset_full_annotations()

    with open(cf.get_config().dataset.pkl_path, 'wb') as file:
        pickle.dump(annots, file)


def load_pkl_version():
    with open(cf.get_config().dataset.pkl_path, 'rb') as file:
        return pickle.load(file)
