import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('img.png', cv2.IMREAD_COLOR_BGR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blur, 30, 150)

contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    vertices = len(approx)

    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)


cv2.imshow('cubissimo', img)
cv2.waitKey(0)

