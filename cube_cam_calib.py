#  
# Check triangulation points from several images (several 2d to 3d) by usung a color rendered 3d cube
# TODO:
# Add points cross check on each plane
# Add support images where are two sides of a box
# Add the some order for each side of a box. Now the some for each box an image
# если у стороны найдено точек меньше чем требуется, то точки не добавлять, но не прерывать
# верх имеет минимум У, который больше минимума следующеей стороный на величину не меньшую чем четверть разницы минимума и максимума У
# если несколько сторон имеют минимум с разницой меньше чем четверть У стороны, то верх не видно и считать, что видны только левая и правая стороны и различать их по минимуму Х и сортировать только слева направо  
# если несколько максимумов У у верхей стороны, то значит на рисунке видны всего две стороны и тогда считать верх ровной стороной и сортировать только слева направо
# если видна больше чем одна сторона, то
# если верх ровная сторона, то нижнюю тоже считать ровной и сортировать только слева направо 
# если верх не ровная сторона, то левая сторона будет та у которой левый верх выше чем правый верх а у правой наоборот

import numpy as np
# import scipy
import cv2
from matplotlib import pyplot as plt
# import mpl_toolkits.mplot3d as plt3d
# import random


# find and order points between the begging and the end points
def points_on_line(line, pnts, limit, sort_by_y=0):
    on_line = []
    p2p1 = line[1] - line[0]
    norm_p2p1 = np.linalg.norm(p2p1)
    if norm_p2p1 != 0: 
        for k, p in enumerate(pnts):
            on_line.append([k, round(abs(np.cross(p2p1, p-line[0]) / norm_p2p1), 3), p[0], p[1]])
        # sort all points by distance to a line
        on_line.sort(key=lambda x: x[1])
        # get the first n
        on_line = on_line[:limit]
        #  sort points by y val
        if sort_by_y==0:
            on_line.sort(key=lambda x: x[3], reverse=True)
        else:
            on_line.sort(key=lambda x: x[2], reverse=True)
    return on_line

# find and order points on a side
def points_on_plane(corners, pnts, points_in_line, side_name):
    columns = np.zeros((points_in_line, points_in_line, 2), dtype=np.int)
    # check if enough info about corners: remove duplicates and check new array size if any more or equal zero
    corners_0 = list(set(corners))
    if(min(corners) >= 0 and len(corners_0) == 4):
        left_bottom_indx, left_top_indx, right_bottom_indx, right_top_indx = corners
        # print("for the {} side: left_bot:{} left_top:{} right_bot:{} right_top{}".format(side_name, pnts[left_bottom_indx], pnts[left_top_indx], pnts[right_bottom_indx], pnts[right_top_indx]))
        # find points between a left top point and left bottom point
        is_y_ord = 0
        if side_name == "right":
            line_last = points_on_line([pnts[left_bottom_indx], pnts[left_top_indx]], pnts, points_in_line, is_y_ord)
            line_0 = points_on_line([pnts[right_bottom_indx], pnts[right_top_indx]], pnts, points_in_line, is_y_ord)
            is_y_ord = 1
        else:
            line_0 = points_on_line([pnts[left_bottom_indx], pnts[left_top_indx]], pnts, points_in_line, is_y_ord)
            line_last = points_on_line([pnts[right_bottom_indx], pnts[right_top_indx]], pnts, points_in_line, is_y_ord)

        # if side_name == "right":
        #     is_y_ord += 1
        # print("line_last=", line_last)
        # find all points between the first and the last columns as rows
        for m in range(points_in_line):
            columns[0, m] = pnts[line_0[m][0]]
            columns[points_in_line - 1, m] = pnts[line_last[m][0]]
            line = points_on_line([columns[0, m], columns[points_in_line - 1, m]], pnts, points_in_line, is_y_ord)
            # print("i:{} len:{}".format(m, len(line)), "line: ".join('{}'.format(i) for i in line))
            for j in range(points_in_line):
                columns[j, m] = pnts[line[j][0]]
        # add the some for columns and then compare as a test!!!!!!!!!!!!!
    else:
        print("Corners error on a side:", side_name)
    # print(columns[0, 0], columns[0, 5], columns[5, 5])
    return columns

def color_box_to_2d_points(im, dots):
    number_of_dots = dots**2 # total number of dots on each cube side
    # Define filters colors
    colors = {
        'blue': ([110, 70, 70], [140, 255, 255]), # ok
        'cyan':([90, 60, 70], [100, 255, 255]),
        'yellow': ([150, 60, 70], [160, 255, 255]), # ok
        'pink': ([20, 60, 70], [30, 255, 255]), # ok
        'red': ([160, 60, 70], [180, 255, 255]),  # ok
        'green': ([30, 60, 70], [100, 255, 255]),  #ok
        }
    #  ordered corners
    pts_all = []
    #  unordered corners
    pts_2d = []
    pts_minmax_xy = []
    # create mask for image
    mask = np.zeros(image.shape, dtype=np.uint8)
    im = cv2.bilateralFilter(image, 9, 75, 75)
    im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)
    hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)   # HSV image
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # detect each side by a color filter
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(hsv_img, lower, upper)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)
        color_mask = cv2.merge([color_mask, color_mask, color_mask])
        thresh = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
        # find contours for each point on a side
        conts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(conts) > 3:
            print(color, len(conts))
        if len(conts) == number_of_dots:
            ps = []
            for cnt in conts:
                x, y, w, h = cv2.boundingRect(cnt)
                tmp = [x + w / 2, y + h / 2]
                ps.append(np.array(tmp, dtype=np.int))
            # print(len(pts))
            ps = np.array(ps)
            pts_2d.append(ps)
            # find min amd max coordinate values for a side
            pts_minmax_xy.append(np.array([np.amin(ps, axis=0), np.amax(ps, axis=0)]).reshape(1, 4).squeeze())
            # create a mask for color filters visual testing
            mask = cv2.bitwise_or(mask, color_mask)
        else:
            if len(conts) > 4:
                print("Wrong numbers of dots for a plane:", color, len(conts))
                return False, np.array(pts_all), mask
    pts_2d = np.array(pts_2d, dtype=np.int)
    if len(pts_minmax_xy) == 0:
        print("Not detected any dots on a cube")
        return False, np.array(pts_all), mask
    pts_minmax_xy = np.array(pts_minmax_xy)
    # detect the top, the left and the right side indexes in unordered array
    top_plane_indx = np.argmin(pts_minmax_xy[:, 1])
    left_plane_indx = np.argmin(pts_minmax_xy[:, 0])
    right_plane_indx = np.argmax(pts_minmax_xy[:, 0])
    # only for boxes where we can detect three sides
    if len(pts_minmax_xy) == 3:
        # print("can see tree cube sides")
        # print("top side has index=", top_plane_indx)
        # detect box top plane points
        top_pts = pts_2d[top_plane_indx]
        # left bottom has min x, left top has min y (y on a picture from top to down), right bottom has max y, right top has max x
        # top_corners =[left_bottom_indx, left_top_indx, right_bottom_indx, right_top_indx]
        top_corners = [np.argmin(top_pts[:, 0]), np.argmin(top_pts[:, 1]), np.argmax(top_pts[:, 1]), np.argmax(top_pts[:, 0])]
        # add all points on line between the first and the last elements
        pts_all.append(points_on_plane(top_corners, top_pts, number_of_line_dots, "top"))

        # order points on the left side!!!!!
        left_pts = pts_2d[left_plane_indx]
        # _corners = [left_bottom_indx, left_top_indx, right_bottom_indx, right_top_indx]
        # right bottom has max y
        left_corners = [-1, -1, np.argmax(left_pts[:, 1]), -1]
        # try detect as nearest to top side left bottom and top side right bottom points
        left_left_top_indx = [1000000, -1, -1, -1]
        left_right_top_indx = [1000000, -1, -1, -1]
        for i, var in enumerate(left_pts):
            # right top of the left side close to right bottom of the top side
            dist_right = abs(np.linalg.norm(top_pts[top_corners[2]] - var))
            # left top of the left side close to  left bottom of the top side
            dist_left = abs(np.linalg.norm(top_pts[top_corners[0]] - var))
            if dist_left < left_left_top_indx[0]:
                left_left_top_indx = [dist_left, i, var[0], var[1]]
            if dist_right < left_right_top_indx[0]:
                left_right_top_indx = [dist_right, i, var[0], var[1]]
        left_corners[1] = left_left_top_indx[1]
        left_corners[3] = left_right_top_indx[1]
        # left botton is the closest to opposite point of the right top by the center
        # find the cetre of side
        center = np.median(np.array([left_pts[left_left_top_indx[1]], left_pts[left_corners[2]]]).T, axis=1).astype(np.int)
        m_2 = left_pts[left_right_top_indx[1]] - center
        # find a point oposite to the top left corner
        left_left_bottom_test = center - m_2
        left_left_bottom_indx = [1000000, -1, -1, -1]
        for i, var in enumerate(left_pts):
            dist_left = abs(np.linalg.norm(left_left_bottom_test - var))
            if dist_left < left_left_bottom_indx[0]:
                left_left_bottom_indx = [dist_left, i, var[0], var[1]]
        left_corners[0] = left_left_bottom_indx[1]
        pts_all.append(points_on_plane(left_corners, left_pts, number_of_line_dots, "front"))
        
        # order points on the right side!!!
        right_pts = pts_2d[right_plane_indx]
        #  the left bottom has max y
        # _corners = [left_bottom_indx, left_top_indx, right_bottom_indx, right_top_indx]
        right_corners = [np.argmax(right_pts[:, 1]), -1, -1, -1]
        # try detect as nearest to top side right bottom and top side right top points
        right_left_top_indx = [1000000, -1, -1, -1]
        right_right_top_indx = [1000000, -1, -1, -1]
        for i, var in enumerate(right_pts):
            dist_left = abs(np.linalg.norm(top_pts[top_corners[2]] - var))
            dist_right = abs(np.linalg.norm(top_pts[top_corners[3]] - var))
            if dist_left < right_left_top_indx[0]:
                right_left_top_indx = [dist_left, i, var[0], var[1]]
            if dist_right < right_right_top_indx[0]:
                right_right_top_indx = [dist_right, i, var[0], var[1]]
        right_corners[1] = right_left_top_indx[1]
        right_corners[3] = right_right_top_indx[1]
        # left botton is the closest to opposite point of the left top by the center
        center = np.median(np.array([right_pts[right_right_top_indx[1]], right_pts[right_corners[0]]]).T, axis=1).astype(np.int)
        m_2 = center - right_pts[right_left_top_indx[1]]
        right_right_bottom_test = center + m_2
        right_right_bottom_indx = [1000000, -1, -1, -1]
        for i, var in enumerate(right_pts):
            dist_bot = abs(np.linalg.norm(right_right_bottom_test - var))
            if dist_bot < right_right_bottom_indx[0]:
                right_right_bottom_indx = [dist_bot, i, var[0], var[1]]
        right_corners[2] = right_right_bottom_indx[1]
        # print(right_corners, number_of_line_dots)
        pts_all.append(points_on_plane(right_corners, right_pts, number_of_line_dots, "right"))
        print("Done!")
    return True, np.array(pts_all), mask
    
if __name__ == '__main__':
    im_pathes = [
        'images/Cube_6x6_Camera_0.png',
        'images/Cube_6x6_Camera_1.png',
        # 'images/Cube_6x6_Camera_2.png',
        # 'images/Cube_6x6_Camera_3.png',
        ]
    cenral_pt_loc = []
    number_of_line_dots = 6 # total number of dots on each cube side line
    ready = []
    for im_path in im_pathes:
        pts_p = []
        image = cv2.imread(im_path)
        if not image is None: 
            ret, pts_2d_eval, im_mask = color_box_to_2d_points(image, number_of_line_dots)
            if ret:
                fig = plt.figure(figsize=(10, 10)) #
                fig.canvas.set_window_title(im_path)
                # Display detected points on image
                for p in range(len(pts_2d_eval)):
                    for l in range(len(pts_2d_eval[p])):
                        if np.any(pts_2d_eval[p, l]):
                            for i in range(len(pts_2d_eval[p][l])):
                                cv2.putText(image, str(l)+str(i), (pts_2d_eval[p, l, i][0]-20, pts_2d_eval[p, l, i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (199, 199, 199), 2)
                                cv2.circle(image, (pts_2d_eval[p, l, i][0], pts_2d_eval[p, l, i][1]), 4, (255, 255, 255), -1)
                                cv2.circle(image, (pts_2d_eval[p, l, i][0], pts_2d_eval[p, l, i][1]), 2, (0, 0, 0), -1)
                fp = fig.add_subplot()
                fp.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                print("Error detect some points!")
                fig = plt.figure() #figsize=(12, 4)
                fig.canvas.set_window_title(im_path)
                p_0 = fig.add_subplot(121)
                p_0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                p_1 = fig.add_subplot(122)
                p_1.imshow(cv2.cvtColor(im_mask, cv2.COLOR_BGR2RGB))
        else:
            print("Can not open file: ", im_path)
        np.save(im_path+'_2d.npy', pts_2d_eval)
    plt.show()
