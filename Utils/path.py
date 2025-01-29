import app_config as cf


def get_players_annot_path(video: int, frame: int):
    return f'{cf.get_config().dataset_path}/volleyball_tracking_annotation/{str(video)}/{str(frame)}/{str(frame)}.txt'


def get_frame_img_path(video: int, main_frame: int, frame: int):
    return f'{cf.get_config().dataset_path}/videos/{str(video)}/{str(main_frame)}/{str(frame)}.jpg'


def get_video_annot_path(video: int):
    return f'{cf.get_config().dataset_path}/videos{str(video)}/annotations.txt'
