import cv2
import numpy as np


#Harris corner-detection
def harris_corner_detection(reference_image):
    # Finner hjørnene med Harris på gråtonebildet
    gray = np.float32(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    #Markerer hjørnene med røde punkterpå en kopi og lagrer bildet
    result = reference_image.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite("harris.png", result)


#Alignment ved bruk av SIFT og FLANN (finner en match mellom to bilder og vrir det ene over på det andre)
def align_images(image_to_align, reference_image, max_features, good_match_precent):
    # Finner SIFT-keypoints og descriptors i begge bildene
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_align, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    #Matcher med FLANN og beholder gode matcher med ratio test
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    #Finner homografien med RANSAC og vrir bildet over på referansen (HUSK!: Trenger flere enn max_features matcher)
    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = reference_image.shape[:2]
        aligned = cv2.warpPerspective(image_to_align, M, (w, h))
        cv2.imwrite("aligned.png", aligned)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None

    #Tegner matchene og lagrer
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                       matchesMask=matchesMask, flags=2)
    img3 = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, **draw_params)
    cv2.imwrite("matches.png", img3)


if __name__ == "__main__":
    reference = cv2.imread("reference_img.png")
    to_align = cv2.imread("align_this.jpg")

    harris_corner_detection(reference)
    align_images(to_align, reference, 10, 0.7)