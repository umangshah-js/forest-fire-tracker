import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import math

def get_center(im, radius=5, save_path=None):
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
    num = 0

    center = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, mx, _, mxloc = cv2.minMaxLoc(
            dist[y : y + h, x : x + w], peaks8u[y : y + h, x : x + w]
        )
        # mxloc[0]+x : Xcoor of center mxloc[1]+y: Ycoor of center
        if mx > radius:
            center.append((mxloc[0] + x, mxloc[1] + y))
            cv2.circle(im, (mxloc[0] + x, mxloc[1] + y), 2, (255, 0), -1)

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
    img_name = save_path.split("/")[-1].split(".")[0]
    cv2.putText(im, img_name, org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, im)
    # cv2.imshow('a',im)
    return center


def get_color_center(im, centers):
    states = []

    for center in centers:
        pixel = im[center[0], center[1], :]

        green = np.array([0, 255, 0])
        red = np.array([255, 0, 0])
        black = np.array([0, 0, 0])
        yellow = np.array([255, 255, 0])

        state = [
            0,  # "alive",
            2,  # "burning",
            3,  # "dead",
            1,  # "heating"
        ]

        dist = []
        dist.append([np.linalg.norm(pixel - green)])
        dist.append([np.linalg.norm(pixel - red)])
        dist.append([np.linalg.norm(pixel - black)])
        dist.append([np.linalg.norm(pixel - yellow)])

        states.append(state[np.argmin(np.array(dist))])

    return states


def get_color_contour(im, arr):
    # cv2.imshow('img', im)
    im[arr == 0] = np.array([255, 255, 255])

    states = {
        0: np.array([255, 255, 255]),  # nothing
        1: np.array([0, 255, 0]),  # "alive",
        2: np.array([0, 255, 255]),  # "heating",
        3: np.array([0, 0, 255]),  # "burning",
        4: np.array([0, 0, 0]),  # "dead",
    }

    color_img = np.zeros((im.shape[0], im.shape[1], len(states), 3), dtype=np.uint8)

    im = im[:, :, np.newaxis, :]
    im = np.repeat(im, len(states), axis=2)

    for i in range(len(states)):
        color_img[:, :, i, :] = states[i]

    dist = np.linalg.norm(im - color_img, axis=3)
    dist = np.argmin(dist, axis=2)

    return dist

def get_color_centers(im, centers):
    ans = []
    
    for center in centers:
        dist = []
        states = {
        1: np.array([0, 255, 0]),  # "alive",
        2: np.array([0, 255, 255]),  # "heating",
        3: np.array([0, 0, 255]),  # "burning",
        4: np.array([0, 0, 0]),  # "dead",
        }

        dist.append([np.linalg.norm(im[math.ceil(center[0]),math.ceil(center[1]),:] - states[1])])
        dist.append([np.linalg.norm(im[math.ceil(center[0]),math.ceil(center[1]),:] - states[2])])
        dist.append([np.linalg.norm(im[math.ceil(center[0]),math.ceil(center[1]),:] - states[3])])
        dist.append([np.linalg.norm(im[math.ceil(center[0]),math.ceil(center[1]),:] - states[4])])
        
        state = np.argmin(np.array(dist))+1

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
        ans.append(np.array([center[0],center[1],center[2],state]))

    return np.array(ans)

def get_world_img(timestamp, m, n, data_pth, x_dim, y_dim):
    # Create an array og m*x_dim, n*y_dim, 3
    world_img = np.zeros((m * x_dim, n * y_dim, 3), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            # img_path = os.path.join(f"{data_pth}",f"{i}_{j}",f"{timestamp}.png")
            img_path = os.path.join(f"{data_pth}", f"{i}_{j}.png")
            im = cv2.imread(img_path)
            world_img[
                (m - 1 - i) * x_dim : (m - i) * x_dim, j * y_dim : (j + 1) * y_dim, :
            ] = im

    cv2.imshow("a", world_img)
    return world_img

def get_world_arr(lst_arr, m,n, x_dim, y_dim):
    combine_arr = np.zeros((m * x_dim, n * y_dim), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            combine_arr[(m - 1 - i) * x_dim : (m - i) * x_dim, j * y_dim : (j + 1) * y_dim] = lst_arr[i][j]
    
    return combine_arr

def generate_marker(world_img, sim_centers, m, n, x_dim, y_dim):
    n_trees = len(sim_centers)

    for tree in range(n_trees):
        x = sim_centers[tree][2]
        y = sim_centers[tree][3]
        for j in range(len(sim_centers[tree][0])):
            cv2.circle(
                world_img,
                (
                    sim_centers[tree][0][j][0] + (m - x - 1) * x_dim,
                    sim_centers[tree][0][j][1] + y * y_dim,
                ),
                2,
                (255, 0, 0),
                -1,
            )
    return world_img


def get_contour(im, radius=5, save_path=None):
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
    np.savez(save_path, arr=arr, centers=centers)
    return arr, centers


def arr_to_image(arr):
    img = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    img[arr == 0] = np.array([255, 255, 255])
    img[arr == 1] = np.array([0, 255, 0])
    img[arr == 2] = np.array([0, 255, 255])
    img[arr == 3] = np.array([0, 0, 255])
    img[arr == 4] = np.array([0, 0, 0])

    return img