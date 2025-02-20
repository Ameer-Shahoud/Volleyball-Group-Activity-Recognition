class BoxInfo:
    def __init__(self, line: str):
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
        return f"""player: {self.player_ID} \t class: {self.category} \t box: (x1: {self.box[0]}, y1: {self.box[1]}) - (x2: {self.box[2]}, y2: {self.box[3]})
        """
