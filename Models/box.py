class BoxInfo:
    """
    Class to parse and store bounding box information for players.

    Attributes:
        player_ID (int): ID of the player.
        category (str): Action category (e.g., standing, spiking).
        box (tuple): Coordinates of the bounding box (x1, y1, x2, y2).
        frame_ID (int): Frame ID in the video.
        lost (int): Indicates if the object was lost in tracking.
        grouping (int): Grouping ID for the player.
        generated (int): Indicates if the box was auto-generated.
    """

    def __init__(self, line: str):
        """
        Initializes a BoxInfo object by parsing a line from the annotation file.

        Args:
            line (str): A line from the annotation file containing box data.
        """
        words: list[str] = line.split()
        self.category: str = words.pop()
        words: list[int] = [int(string) for string in words]
        self.player_ID: int = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated

    def __repr__(self):
        """
        Provides a string representation of the BoxInfo object for debugging.
        
        Returns:
            str: A formatted string containing player ID, category, and box coordinates.
        """
        return f"""player: {self.player_ID} \t class: {self.category} \t box: (x1: {self.box[0]}, y1: {self.box[1]}) - (x2: {self.box[2]}, y2: {self.box[3]})
        """
