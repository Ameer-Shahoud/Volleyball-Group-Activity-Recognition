import cv2
import matplotlib.pyplot as plt


def show_image(img, title: str, figsize=(80, 20)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()
