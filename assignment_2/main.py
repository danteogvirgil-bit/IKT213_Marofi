import cv2
import numpy as np


#Lager en kant rundt bildet som speiler kantpikslene (BORDER_REFLECT)
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width,
                                      border_width, border_width,
                                      cv2.BORDER_REFLECT)
    cv2.imwrite("padding.png", padded_image)
    return padded_image


#Klipper ut området vi vil ha med slicing: først rader (y), så kolonner (x)
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("crop.png", cropped_image)
    return cropped_image


#Endrer størrelsen på bildet, resize tar (bredde, høyde)
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("resize.png", resized_image)
    return resized_image


#Kopierer bildet piksel for piksel over i det tomme bildet
def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    cv2.imwrite("copy.png", emptyPictureArray)
    return emptyPictureArray


#Gjør bildet om til gråtoner (imread gir BGR, så vi bruker BGR2GRAY)
def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale.png", gray)
    return gray


#Gjør bildet om fra BGR til HSV
def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png", hsvImage)
    return hsvImage


#Legger hue til alle fargeverdiene. int16 og % 256 gjør at verdier over 255 eller under 0 starter på nytt i stedet for å bli feil i uint8
def hue_shifted(image, emptyPictureArray, hue):
    shifted = (image.astype(np.int16) + hue) % 256
    emptyPictureArray[:] = shifted.astype(np.uint8)
    cv2.imwrite("hue_shifted.png", emptyPictureArray)
    return emptyPictureArray


#Visker ut bildet med Gaussian blur, kernel 15x15 og standard kant
def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite("smoothing.png", blurred)
    return blurred


#Roterer bildet 90 grader eller 180 grader
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("rotation_90.png", rotated)
        return rotated
    if rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("rotation_180.png", rotated)
        return rotated


#Leser inn bildet og kjører alle funksjonene med verdiene fra oppgaven
if __name__ == "__main__":
    image = cv2.imread("iris.png")
    height, width, channels = image.shape

    padding(image, 100)
    # 200 piksler fra venstre og topp, 130 fra høyre og bunn
    crop(image, 200, width - 130, 200, height - 130)
    resize(image, 200, 200)

    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, emptyPictureArray)

    grayscale(image)
    hsv(image)

    emptyHueArray = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted(image, emptyHueArray, 50)

    smoothing(image)
    rotation(image, 90)
    rotation(image, 180)