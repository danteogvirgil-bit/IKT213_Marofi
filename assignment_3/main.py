import cv2
import numpy as np


#Finner edges med Sobel på det blurrye gråtone-bildet og lagrer resultatet
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    sobel = cv2.Sobel(blur, cv2.CV_32F, dx=1, dy=1, ksize=1)
    sobel = cv2.normalize(np.abs(sobel), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("sobel_edge_detection.png", sobel)


#Finner edges med Canny på det blurrye gråtone-bildet og lagrer resultatet
def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    edges = cv2.Canny(blur, threshold_1, threshold_2)
    cv2.imwrite("canny_edge_detection.png", edges)


# Finner alle steder templaten matcher (terskel 0.9) (markerer dem med røde firkanter)
def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("template_match.png", image)


# Zoomer inn eller ut (Image Pyramid) (pyrUp / pyrDown) og lagrer resultatet
def resize(image, scale_factor: int, up_or_down: str):
    rows, cols = image.shape[:2]

    if up_or_down == "up":
        result = cv2.pyrUp(image, dstsize=(cols * scale_factor, rows * scale_factor))
    elif up_or_down == "down":
        result = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))

    cv2.imwrite("resize_" + up_or_down + ".png", result)

#Laster bildet 
if __name__ == "__main__":
    lambo = cv2.imread("lambo.png")
    shapes = cv2.imread("shapes-1.png")
    shapes_template = cv2.imread("shapes_template.jpg", 0)
    #kjører begge funksjonene
    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, 50, 50)
    template_match(shapes, shapes_template)
    resize(lambo, 2, "up")
    resize(lambo, 2, "down")