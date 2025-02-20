import os
from Models.clip_annotations import ClipAnnotations
from Utils.dataset import get_players_box_annot_path, get_video_annot_path


class VideoAnnotations:
    def __init__(self, video: int):
        self.video = video
        self.__clips_annotations: dict[int, ClipAnnotations] = {}
        with open(get_video_annot_path(video), 'r') as file:
            for line in file:
                clip, category = line.strip().split(' ')[:2]
                clip = int(clip.replace('.jpg', ''))
                if os.path.exists(get_players_box_annot_path(self.video, clip)):
                    self.__clips_annotations[clip] = ClipAnnotations(
                        self.video, clip, category)

    def get_clip_annotations(self, clip: int) -> ClipAnnotations:
        return self.__clips_annotations[clip]

    def get_all_clips_annotations(self):
        return self.__clips_annotations.items()
