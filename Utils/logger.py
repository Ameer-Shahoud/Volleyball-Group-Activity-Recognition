import logging
import logging.handlers
import os
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, log_name: str, writer: SummaryWriter = None):
        self._writer = writer

        Path(log_dir).mkdir(exist_ok=True)

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)

        log_path = os.path.join(log_dir, f"{log_name}.log")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(self._get_formatter())

        self.memory_handler = logging.handlers.MemoryHandler(10 * 1024)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(self.memory_handler)

    def _get_formatter(self):
        return logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M"
        )

    def info(self, *messages: list[any]):
        for message in messages:
            self.logger.info(message)

        print(*messages)

    def warning(self, *messages: list[any]):
        for message in messages:
            self.logger.warning(message)

        print(*messages)

    def error(self, *messages: list[any]):
        for message in messages:
            self.logger.error(message)

        print(*messages)

    def log_to_tensorboard(self):
        if self._writer:
            formatter = self._get_formatter()
            messages = list(
                map(lambda r: formatter.format(r), self.memory_handler.buffer)
            )
            self._writer.add_text(
                'Logs',
                ' '.join(map(lambda m: str(m) + '\n', messages)),
            )

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
