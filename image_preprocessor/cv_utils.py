import cv2
import numpy as np
import os
import sys
import math


def get_contour(im, radius=10, save_path=None):
    np.set_printoptions(threshold=sys.maxsize)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    borderSize = 16
    distborder = cv2.copyMakeBorder(
        dist,
        borderSize,
        borderSize,
        borderSize,
        borderSize,
        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED,
        0,
    )
    gap = 9
    # Adjust borderSize and gap for circle detection
    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1)
    )
    kernel2 = cv2.copyMakeBorder(
        kernel2, gap, gap, gap, gap, cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0
    )
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)
    contours, hierarchy = cv2.findContours(
        peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    peaks8u = cv2.convertScaleAbs(peaks)  # to use as mask
    arr = np.zeros((256, 256))

    num = 0
    count = 0

    centers = []

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, mx, _, mxloc = cv2.minMaxLoc(
            dist[y : y + h, x : x + w], peaks8u[y : y + h, x : x + w]
        )

        # mxloc[0]+x : Xcoor of center mxloc[1]+y: Ycoor of center
        if mx > radius:
            cv2.circle(
                im, (int(mxloc[0] + x), int(mxloc[1] + y)), int(mx), (255, 0, 0), 2
            )
            r = mx
            xc = mxloc[0] + x
            yc = mxloc[1] + y

            centers.append(np.array([xc, yc, r]))

            for u in range(int(max(xc - r, 0)), int(min(xc + r + 1, arr.shape[0]))):
                for v in range(int(max(yc - r, 0)), int(min(yc + r + 1, arr.shape[1]))):
                    if (u - xc) ** 2 + (v - yc) ** 2 <= r**2:
                        arr[v, u] = 1
                        # count+=1
            # num+=1
    # print(num)
    # print(count)

    # cv2.imshow('a',im)
    # cv2.imshow('b',convert_to_image(arr))
    centers = np.array(centers)
    # np.savez(save_path, arr=arr, centers=centers)
    return arr, centers


def get_color_centers(im, centers):
    ans = []

    for center in centers:
        dist = []
        states = {
            0: np.array([255, 255, 255]),  # nothing
            1: np.array([0, 255, 0]),  # "alive",
            2: np.array([0, 255, 255]),  # "heating",
            3: np.array([0, 0, 255]),  # "burning",
            4: np.array([0, 0, 0]),  # "dead",
        }

        pixel = im[math.ceil(center[1]), math.ceil(center[0]), :]
        dist.append([np.linalg.norm(pixel - states[0])])
        dist.append([np.linalg.norm(pixel - states[1])])
        dist.append([np.linalg.norm(pixel - states[2])])
        dist.append([np.linalg.norm(pixel - states[3])])
        dist.append([np.linalg.norm(pixel - states[4])])

        # print(pixel, dist)
        state = np.argmin(np.array(dist))
        threshold = 20

        if dist[state][0] > threshold:
            state = 0

        # if state==2:
        #     cv2.imshow('img', im)
        #     print(pixel, dist)
        #     pass

        # x,y,r = center[0], center[1], center[2]
        # xm,ym = np.meshgrid(
        #     np.arange(max(x-r,0),min(x+r+1,255)),
        #     np.arange(max(y-r,0),min(y+r+1,255))
        # )
        # ind = np.column_stack((xm.ravel(), ym.ravel()))
        # ind = np.ceil(ind).astype(int)

        # pixels = im[ind[:,0], ind[:,1], :]

        # states = {
        # 1: np.array([0, 255, 0]),  # "alive",
        # 2: np.array([0, 255, 255]),  # "heating",
        # 3: np.array([0, 0, 255]),  # "burning",
        # 4: np.array([0, 0, 0]),  # "dead",
        # }

        # dist = []
        # for i in range(len(states)):
        #     dist.append(np.linalg.norm(pixels-states[i], axis=1))

        # state = np.argmin(np.array(dist))
        ans.append(np.array([center[0], center[1], center[2], state]))

    return np.array(ans)
