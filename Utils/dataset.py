from Models.config_mixin import _ConfigMixin


def get_video_path(video: int):
    return f'{_ConfigMixin.get_cf(None).dataset.videos_dir}/{str(video)}'


def get_frame_img_path(video: int, clip: int, frame: int):
    return f'{_ConfigMixin.get_cf(None).dataset.videos_dir}/{str(video)}/{str(clip)}/{str(frame)}.jpg'


def get_players_box_annot_path(video: int, clip: int):
    return f'{_ConfigMixin.get_cf(None).dataset.tracking_boxes_annotation_dir}/{str(video)}/{str(clip)}/{str(clip)}.txt'


def get_video_annot_path(video: int):
    return f'{_ConfigMixin.get_cf(None).dataset.videos_dir}/{str(video)}/annotations.txt'
