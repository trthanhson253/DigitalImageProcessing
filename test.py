import cv2
import numpy as np

i1 = cv2.imread('a.jpg')
i2 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)

template1 = cv2.imread('b.jpg',0)
template2 = cv2.imread('c.jpg',0)
template3 = cv2.imread('d.jpg',0)

w, h = template1.shape[::-1]
w1, h1 = template2.shape[::-1]
w2, h2 = template3.shape[::-1]

res = cv2.matchTemplate(i2,template1,cv2.TM_CCOEFF_NORMED)
res1 = cv2.matchTemplate(i2,template2,cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(i2,template3,cv2.TM_CCOEFF_NORMED)

threshold = 0.35
threshold1 = 0.55
threshold2 = 0.475

loc = np.where( res >= threshold)
loc1 = np.where( res1 >= threshold1)
loc2 = np.where( res2 >= threshold2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(i1, pt, (pt[0] + w, pt[1] + h), (0,255,125), 1) # mark by green rectangle

for pt1 in zip(*loc1[::-1]):
    cv2.rectangle(i1, pt1, (pt1[0] + w1, pt1[1] + h1), (0,0,255), 1) # mark by red rectangle

for pt2 in zip(*loc2[::-1]):
    cv2.rectangle(i1, pt2, (pt2[0] + w2, pt2[1] + h2), (255,255,255), 1)

cv2.imshow('Result',i1)

cv2.waitKey(0)
cv2.destroyAllWindows()

