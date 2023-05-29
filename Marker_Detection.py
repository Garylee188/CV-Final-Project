import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_MarkerPoints(time_stamp_path, marker_thresh=150):
    #'./seq1/dataset/1681710717_541398178'
    rd_marker = np.loadtxt(f'{time_stamp_path}/detect_road_marker.csv', delimiter=',')
    marker_img = cv2.imread(f'{time_stamp_path}/raw_image.jpg', cv2.IMREAD_GRAYSCALE)

    marker_pts = []
    drw = []
    for box in rd_marker[:, :4]:  # [x1, y1, x2, y2]
        # mask = cv2.rectangle(marker_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 0, 2)
        crop_img = marker_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        ret, output = cv2.threshold(crop_img, marker_thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnts in contours:
            appr_pts = cv2.approxPolyDP(cnts, cv2.arcLength(cnts, True) * 0.01, True)
            appr_pts[:, :, 0] += int(box[0])
            appr_pts[:, :, 1] += int(box[1])

            for pts in appr_pts:
                marker_pts.append(cv2.KeyPoint(int(pts[0][0]), int(pts[0][1]), 1))
                drw.append((int(pts[0][0]), int(pts[0][1])))
        # plt.imshow(crop_img)
        # plt.show()
        # plt.close()
    # cv2.drawContours(marker_img, drw, -1, 0, 2)
    # cv2.imshow('final', marker_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return marker_pts, drw

# marker_pts = get_MarkerPoints('./seq1/dataset/1681710717_541398178')
# print(marker_pts)

