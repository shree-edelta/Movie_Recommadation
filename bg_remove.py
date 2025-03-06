import cv2
import numpy as np

img = cv2.imread('image_bg.jpg',cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
print(mask)


foregrd  = cv2.bitwise_and(img, img, mask=mask)

# kernel = np.ones((5, 5), np.uint8)
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
img_bg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.imwrite('result.jpg', foregrd)

cv2.imshow("foreground",foregrd)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

