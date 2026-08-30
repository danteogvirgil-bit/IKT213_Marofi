import cv2
import os
import time


# Skriver ut bildeinformasjon
def print_image_information(image):
    # Henter høyde, bredde og kanaler
    height, width, channels = image.shape

    # Printer verdiene
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", image.size)
    print("data type:", image.dtype)


# Henter kamerainfo og lagrer til fil
def save_camera_information():
    # Åpner kamera 0 med MSMF-backend
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

    # Sjekker at kameraet er åpent
    if not camera.isOpened():
        print("Fant ikke kameraet")
        return

    # Leser fps, bredde og høyde
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Måler fps selv hvis driveren melder 0
    if fps == 0:
        start = time.time()
        for i in range(60):
            camera.read()
        fps = round(60 / (time.time() - start), 1)

    # Frigjør kameraet
    camera.release()

    # Lager solutions-mappen
    os.makedirs("solutions", exist_ok=True)

    # Skriver verdiene til fil
    with open("solutions/camera_outputs.txt", "w") as file:
        file.write("fps: " + str(fps) + "\n")
        file.write("height: " + str(height) + "\n")
        file.write("width: " + str(width) + "\n")

    print("Lagret kamerainfo")


# Kjører begge funksjonene
def main():
    # Leser bildet i farger
    image = cv2.imread("iris-1.jpg")

    # imread gir None hvis stien er feil
    if image is None:
        print("Fant ikke iris-1.jpg")
        return

    print_image_information(image)
    save_camera_information()


# Kjører main kun ved direkte kjøring
if __name__ == "__main__":
    main()