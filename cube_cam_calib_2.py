import numpy as np
from scipy.linalg import rq
import cv2
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as plt3d

def generate_real_box_pts(delta_side, number_of_dots, step=1):
    #  real blender box coordinates
    #         delta_side = 0.6
    #         number_of_dots = 6
    # 
    #                                         top side
    #                                         left bottom     -5     0    6.2
    #                                         left top        -5     5    6.2
    #                                         right bottom     0     0    6.2
    #                                         roght top        0     5    6.2
    # left side                             # front side                            # right side
    # left bottom    -5.6   5   0.6         # left bottom    -5  -0.6  0.6          # left bottom     0.6   0   0.6
    # left top       -5.6   5   5.6         # left top       -5  -0.6  5.6          # left top        0.6   0   5.6
    # right bottom   -5.6   0   0.6         # right bottom    0  -0.6  0.6          # right bottom    0.6   5   0.6
    # roght top      -5.6   0   5.6         # roght top       0  -0.6  5.6          # roght top       0.6   5   5.6
    #                                         bottom side
    #                                         left bottom     -5     5    0
    #                                         left top        -5     0    0
    #                                         right bottom     0     5    0
    #                                         roght top        0     0    0
    #                                         back side
    #                                         left bottom     0     5.6    0.6
    #                                         left top        0     5.6    5.6
    #                                         right bottom    -5    5.6    0.6
    #                                         roght top       -5    5.6    5.6
    pts_3d = [[] for i in range(3)]
    for x in range(number_of_dots):
        for y in range(number_of_dots):
            # generate top side
            z = 5 * step + 2 * delta_side
            dx = -x * step
            dy = y * step
            pts_3d[0].append([dx, dy, z])
            # generate front side
            dz = y * step + delta_side
            dy = -delta_side
            dx = -x * step
            pts_3d[1].append([dx, dy, dz])
            # generate right side
            dx = delta_side
            dy = x * step
            dz = y * step + delta_side
            pts_3d[2].append([dx, dy, dz])
    return np.array(pts_3d, dtype=np.float32)
    

def calibrate(n2d, n3d):
    ''' This function computes camera projection matrix from 3D scene points
    and corresponding 2D image points with the Direct Linear Transformation (DLT).
    Usage:  P = calibrate(n2d, n3d)
    Input:  
    n2d: nx3 homogeneous coordinates image points
    n3d: nx4 homogeneous coordinates scene points
    Output: P: 3x4 camera projection matrix '''
    # Warning: The svd function from Numpy returns U, Sigma and V_transpose (not V, unlike Matlab)
    # xt = np.transpose(n2d)
    print("input shape=", n2d.shape, n3d.shape)
    sz = len(n2d)
    # Xt = np.transpose(n3d)
    n3d = np.hstack((n3d, np.ones((sz, 1))))
    zero4 = np.zeros(4)
    M = np.array((2 * len(n2d), 12))         #M=np.array((56,12))
    for i in range(0, sz):             # for i in range(0,28):
        A = np.hstack((zero4, -n3d[i], n2d[i][1] * n3d[i]))
        B = np.hstack((n3d[i], zero4, -n2d[i][0] * n3d[i]))
        A = np.reshape(A, (1, 12))
        B = np.reshape(B, (1, 12))
        if i == 0:
            M = np.vstack((A, B))
        else:
            M = np.vstack((M, A, B))
    u, s, vtranspose = np.linalg.svd(M)
    v = np.transpose(vtranspose)
    p = v[:, 11]
    P = p.reshape((3,4))
    return P

def P_to_KRt(P):
    ''' This function computes the decomposition of the projection matrix into intrinsic parameters, K, 
    and extrinsic parameters Q (the rotation matrix) and t (the translation vector)
    Usage: K, Q, t = P_to_KRt(P)
    Input: P: 3x4 projection matrix
    Outputs:
        K: 3x3 camera intrinsics
        Q: 3x3 rotation matrix (extrinsics)
        t: 3x1 translation vector(extrinsics) '''  
    M = P[0:3, 0:3]
    R, Q = rq(M)
    # R, Q = scipy.linalg.rq(M)
    K = R / float(R[2, 2])
    if K[0, 0] < 0:
        K[:, 0] = -1 * K[:, 0]
        Q[0, :] = -1 * Q[0, :]
    if K[1, 1] < 0:
        K[:, 1] = -1 * K[:, 1]
        Q[1, :] = -1 * Q[1, :]
    if np.linalg.det(Q) < 0:
        print('Warning: Determinant of the supposed rotation matrix is -1')
    P_3_3 = np.dot(K, Q)
    P_proper_scale = (P_3_3[0,0] * P) / float(P[0,0])
    t = np.dot(np.linalg.inv(K), P_proper_scale[:,3])
    return K, Q, t

if __name__ == '__main__':
    im_pathes = [
        'images/Cube_6x6_Camera_0.png',
        'images/Cube_6x6_Camera_1.png',
        # 'images/Cube_6x6_Camera_2.png',
        # 'images/Cube_6x6_Camera_3.png',
        ]
    # Get from Blender: Camera object->Properties->
    #  tv - Object -> Transform-> Location (degree)
    #  rv - Object -> Transform-> Rotation (degree)
    # f - Data-> Focal Lengh (mm)
    cam_params = {
        'Camera_0': {'tv':[7, -10, 18], 'rv':[48.7, 0, 32.1], 'f': 35},
        'Camera_1': {'tv':[13, -7, 13], 'rv':[60, 0, 54], 'f': 35},
        'Camera_2': {'tv':[8, -14.78, 11.09], 'rv':[65, 0, 34], 'f': 35},
        'Camera_3': {'tv':[5, -12, 12], 'rv':[60, 0, 25], 'f': 35}
    }
        # Create camera matrix from Blender properties data
    K_real = np.diag([cam_params['Camera_0']['f'], cam_params['Camera_0']['f'], 1]).astype(np.float)
    img_resolution = [960, 540]
    K_real[0, 2] = int(img_resolution[0]/2)
    K_real[1, 2] = int(img_resolution[1]/2)

        # 'K' = np.array([[ 3.38625072e+03,  1.27159260e+01,  8.52916937e+02],
        #     "       [ 1.85319235e+00,  3.42886718e+03,  6.27551184e+02],
        #     "       [-1.11022302e-15,  2.99760217e-15,  1.00000000e+00]])
    # print(K_real)
    # Blender rendered images is non distorted.
    # But if you want add this feature: plug the rendered layers to the undistort node and use the distort option
    number_of_line_dots = 6
    pts_3d_real = generate_real_box_pts(0.6, number_of_line_dots, 1)
    pts_3d_real_all = pts_3d_real.reshape(3*number_of_line_dots*number_of_line_dots, 3)
    shp = pts_3d_real_all.shape
    print("3d shape", shp)

    fig = plt.figure(figsize=(8, 8))
    fp3d = fig.add_subplot(projection='3d')
    dist_coeffs = np.zeros(5, dtype=np.float32)
    fp3d.text(0, -0.6, 0.6, "Center", size=10, zorder=1, color='k')
    fp3d.scatter([0], [-0.6], [0.6], c="k")
    colors = ['b', 'r', 'g', 'y', 'c']
    deltas = []
    for i, im_path in enumerate(im_pathes):
        pts_2d_eval = np.load(im_path+'_2d.npy').astype(np.float32)
        pts_2d_eval_all = pts_2d_eval.reshape(3*number_of_line_dots*number_of_line_dots, 2)
        # print("2d_all", pts_2d_eval_all.shape, pts_3d_real_all.shape)
        # pts_2d_eval_all = cv2.convertPointsToHomogeneous(pts_2d_eval_all).squeeze()
        # pts_3d_real_all = cv2.convertPointsToHomogeneous(pts_3d_real_all).squeeze()

        P = calibrate(pts_2d_eval_all, pts_3d_real_all)
        print("P=", P)
        ret_cv, K_cv, dist_cv, rvecs_cv, tvecs_cv = cv2.calibrateCamera(pts_3d_real_all, pts_2d_eval_all, (img_resolution[0],img_resolution[1]), None, None)
        K, R, pose = P_to_KRt(P)
        print("P_cv=", K_cv)
        print ('K = ', K)
        print ('K_real = ', K_real)
        print ('R = ', R)
        print ('t = ', pose)
        # print(dist_coeffs)
        # ret, R, t = cv2.solvePnP(pts_3d_real_all, pts_2d_eval_all, K_real, dist_coeffs)
        # rt = cv2.solvePnPRansac(pts_3d_real_all, pts_2d_eval_all, K_real, None)
        # # print(rt)
        # R = cv2.Rodrigues(rt[1])[0]
        # print('R=', R)
        # pose = np.hstack((R, rt[2]))
        # pose = -np.matrix(R).T * np.matrix(rt[2])
        # print("pose_0=", pose)
        # pose = np.array(pose).flatten()
        # print("pose=", pose)
        deltas.append(pose)
        # fp3d.text(pos[0], pos[2], pos[2], str(i)+"_pos", size=10, zorder=1, color='b')
        fp3d.scatter(pose[0], pose[2], pose[2], c=colors[i])
        # [K, R, t] = P_to_KRt(P)
        # print ('K = ', K)
        # print ('R = ', R)
        # print ('t = ', t)
    #     # draw the box center
    #     # fp3d.quiver([0], [0], [0], [0], [2], [0], length=5)

        for name, param in cam_params.items():
            # R_cur = cv2.Rodrigues(np.deg2rad(np.array(param['rv'])))
    #         # draw camera position
            if name in im_path:
                fp3d.text(param['tv'][0], param['tv'][1], param['tv'][2], str(name)+"_real", size=10, zorder=1, color=colors[i])
                fp3d.scatter(param['tv'][0], param['tv'][1], param['tv'][2], c=colors[i])
        
    #     # print(pts_3d_real)
    #     for side in range(len(pts_3d_real)):
    #         xyz = pts_3d_real[side].T
    #         fp3d.scatter(xyz[0], xyz[1], xyz[2], c=colors[side])    
    #         #  Display 3d view
    #         # fp3d = fig.add_subplot(122, projection='3d')
    #         # draw cameras
    #         # fp3d.plot([3.0, 3.0], [2.0, 0.0], [1, 3], c='b')
    #         # fp3d.scatter(cam_params['cam_0']['tv'][0], cam_params['cam_0']['tv'][1], cam_params['cam_0']['tv'][2], c="b")
    #         # fp3d.quiver(cam_params['cam_0']['tv'][0], cam_params['cam_0']['tv'][1], cam_params['cam_0']['tv'][2],
    #         #             cam_params['cam_0']['rv'][0], cam_params['cam_0']['rv'][1], cam_params['cam_0']['rv'][2]
    #         #             , length=0.1, normalize=True)
    print(np.linalg.norm(deltas[0]-deltas[1]))
    
    plt.show()