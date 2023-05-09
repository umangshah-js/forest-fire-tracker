import cv2
from kafka import KafkaConsumer
import numpy as np
def get_center(im, radius=5, save_path = None):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    borderSize = 16
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    gap = 9                                
    # Adjust borderSize and gap for circle detection
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)
    contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
    num = 0


    centers = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
        # mxloc[0]+x : Xcoor of centers mxloc[1]+y: Ycoor of centers
        if mx >radius:
            centers.append((mxloc[0]+x, mxloc[1]+y))
            cv2.circle(im, (mxloc[0]+x, mxloc[1]+y), 2, (255, 0), -1)

    # write text over image
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    # img_name = save_path.split('/')[-1].split('.')[0]
    # cv2.putText(im, img_name, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # cv2.imwrite(save_path, im)
    # cv2.imshow('a',im)
    return centers




# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer('image-preprocess-1',
                         group_id='image_preprocess_1',
                         bootstrap_servers=['localhost:9092'])
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    # print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
    #                                       message.offset, message.key,
    #                                       message.value))
    image_np = np.frombuffer(message.value, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    centers = get_center(img)
    print(message.key.decode(),len(centers))
    # break
