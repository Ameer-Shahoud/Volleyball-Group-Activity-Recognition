import os
from Models.clip_annotations import ClipAnnotations
from Utils.dataset import get_players_box_annot_path, get_video_annot_path


class VideoAnnotations:
    """
    Class to manage video-level annotations and organize them by clips.

    Attributes:
        video (int): Video ID.
        __clips_annotations (dict): Dictionary containing clip IDs and corresponding ClipAnnotations objects.
    """
    def __init__(self, video: int):
        """
        Initializes VideoAnnotations by loading clip annotations for the given video.

        Args:
            video (int): Video ID.
        """
        self.video = video
        self.__clips_annotations: dict[int, ClipAnnotations] = {}
        # Load video-level annotations from the corresponding annotation file
        with open(get_video_annot_path(video), 'r') as file:
            for line in file:
                clip, category = line.strip().split(' ')[:2]
                clip = int(clip.replace('.jpg', ''))
                # Check if player box annotation exists for the clip
                if os.path.exists(get_players_box_annot_path(self.video, clip)):
                    self.__clips_annotations[clip] = ClipAnnotations(
                        self.video, clip, category)

    def get_clip_annotations(self, clip: int) -> ClipAnnotations:
        """Retrieves ClipAnnotations object for a specific clip."""
        return self.__clips_annotations[clip]

    def get_all_clips_annotations(self):
        """Retrieves all clip annotations for the video."""
        return self.__clips_annotations.items()
