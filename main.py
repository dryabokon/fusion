import cv2
import numpy
import open3d as o3d
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import tools_draw_numpy
from CV import tools_pr_geom
import tools_render_CV
import tools_render_GL
import tools_GL3D
# ----------------------------------------------------------------------------------------------------------------------
import utils_parser
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
folder_in = "./data/"
pcd_file = folder_in + '000547.pcd'
im_file = folder_in + '000547.jpg'
# ----------------------------------------------------------------------------------------------------------------------
P = utils_parser.LParser(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
camera_calibration = {"rotation_x": -0.458161,"rotation_y": 0.525006,"rotation_z": -0.522040,"rotation_w": 0.491865,"f_x": 3.13509204e+03,"f_y": 3.11577686e+03,"c_x": 1.94519315e+03,"c_y": 1.07760672e+03,"x":0.050,"y":0.999,"z":-0.100}
# ----------------------------------------------------------------------------------------------------------------------
# im = cv2.undistort(cv2.imread(im_file), camera_matrix_3x3,distortion, None, None)
# im = cv2.undistort(cv2.imread(im_file), init_intrinsic, distortion, None, camera_matrix_3x3)
# ----------------------------------------------------------------------------------------------------------------------
def import_pcd(file_name):
    pcd = o3d.t.io.read_point_cloud(file_name)
    points = pcd.point.positions.numpy()
    intensities = pcd.point.intensity.numpy()
    return points, intensities.flatten()
# ----------------------------------------------------------------------------------------------------------------------
def construct_camera_matrix_3x3():
    camera_matrix_3x3 = numpy.eye(3)
    camera_matrix_3x3[0, 0] = camera_calibration['f_x']
    camera_matrix_3x3[1, 1] = camera_calibration['f_y']
    camera_matrix_3x3[0, 2] = camera_calibration['c_x']
    camera_matrix_3x3[1, 2] = camera_calibration['c_y']
    return camera_matrix_3x3
# ----------------------------------------------------------------------------------------------------------------------
def construct_RT():
    RT = numpy.eye(4)
    RT[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([camera_calibration['rotation_w'], camera_calibration['rotation_x'], camera_calibration['rotation_y'],camera_calibration['rotation_z']])
    RT[:3, 3] = [camera_calibration['x'], camera_calibration['y'], camera_calibration['z']]
    RT = numpy.linalg.inv(RT)
    return RT
# ----------------------------------------------------------------------------------------------------------------------
init_intrinsic=numpy.array( [[3087.031494140625, 0.0, 1939.39697265625],[0.0, 3087.031494140625, 1079.7120361328125],[0.0, 0.0, 1.0 ]])
distortion=numpy.array([1.2756537199020386, -45.257240295410156, -0.001470720861107111, 0.0016440246254205704, 200.98423767089844, 1.1310863494873047, -44.19110107421875, 197.37249755859375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((1,14))
# ----------------------------------------------------------------------------------------------------------------------
def generate_colors(intensity):
    colors = intensity-numpy.min(intensity)
    colors = (255*colors/numpy.max(colors)).astype(int)
    colors = tools_draw_numpy.get_colors(256,colormap = 'viridis')[colors]

    return colors
# ----------------------------------------------------------------------------------------------------------------------
def generate_ground_plane(distance,best_plane,z_shift=0.0):
    step = 10
    points_3d = numpy.array([(distance * numpy.sin(angle * numpy.pi / 180.0), distance * numpy.cos(angle * numpy.pi / 180.0), 0) for angle in range(0, 360, step)])
    points_3d = [tools_render_CV.line_plane_intersection(best_plane[:3], (0,0,-best_plane[3]/best_plane[2]-z_shift), (0,0,1), p) for p in points_3d]

    return points_3d
# ----------------------------------------------------------------------------------------------------------------------
def render():
    points_3d0, intensity = import_pcd(pcd_file)
    idx_good = intensity > 60
    intensity = intensity[idx_good]
    points_3d = points_3d0[idx_good]
    colors = generate_colors(intensity)
    P.pointcloud_df_to_obj(pd.DataFrame(points_3d), '000547.obj', image=None, max_V=8000, noise=0.05,auto_triangulate=False, color=(192, 192, 192))

    best_plane, best_inliers = P.fit_dominant_plane(points_3d0)
    P.pointcloud_df_to_obj(pd.DataFrame(generate_ground_plane(10,best_plane,z_shift=0.00)),'ground_plane_10.obj', noise=0,auto_triangulate=True,color=(192, 192, 0))
    P.pointcloud_df_to_obj(pd.DataFrame(generate_ground_plane(20,best_plane,z_shift=0.01)),'ground_plane_20.obj', noise=0,auto_triangulate=True,color=(192, 96, 0))
    P.pointcloud_df_to_obj(pd.DataFrame(generate_ground_plane(40,best_plane,z_shift=0.02)),'ground_plane_50.obj', noise=0,auto_triangulate=True,color=(192, 0, 0))

    camera_matrix_3x3 = construct_camera_matrix_3x3()
    tg_half_fovx = camera_matrix_3x3[0, 2] / camera_matrix_3x3[0, 0]
    RT_CV = construct_RT()
    RT_GL = tools_pr_geom.to_RT_GL(RT_CV)

    im = cv2.imread(im_file)
    H, W = im.shape[0], im.shape[1]
    R = tools_GL3D.render_GL3D(filename_obj=folder_out, do_normalize_model_file=False, W=W//4, H=H//4,is_visible=False, projection_type='P', textured=False)

    cv2.imwrite(folder_out + 'render_GL.png', R.get_image_perspective_M(RT_GL, tg_half_fovx, tg_half_fovx, do_debug=True, mat_view_to_1=True))
    cv2.imwrite(folder_out + 'render_CV.png', tools_draw_numpy.draw_points(im, tools_pr_geom.project_points_M(points_3d, RT_CV, camera_matrix_3x3=camera_matrix_3x3), color=colors, w=20, transperency=0.5))

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    render()