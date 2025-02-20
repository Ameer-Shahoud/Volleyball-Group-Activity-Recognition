import cv2
import matplotlib.pyplot as plt
from Models.box import BoxInfo
from Models.config_mixin import _ConfigMixin


def add_box(img, box_info: BoxInfo, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0)):
    x1, y1, x2, y2 = box_info.box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, box_info.category, (x1, y1 - 10),
                font, 0.5, color, 1)


def show_image(img, title: str, figsize=(80, 20)):
    if _ConfigMixin.get_cf(None).is_notebook:
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()
    else:
        _img = cv2.resize(img, (1000, 500))
        cv2.imshow(title, _img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def show_clip(images):
    for img in images:
        _img = cv2.resize(img, (1000, 500))
        cv2.imshow('Clip', _img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
