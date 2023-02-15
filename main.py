import numpy
import cv2
import tools_animation
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import utils_parser
import tools_GL3D
from CV import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
folder_in_lidar_raw = 'E:/users/Dm/GL_sample_rover_data/LiDAR/rosbag2_2023_01_18-16_56_50/'
folder_in_lidar_import = 'E:/users/Dm/GL_sample_rover_data/LiDAR/import/'
filename_in_video_normal = 'E:/users/Dm/GL_sample_rover_data/Normal_FoV/Normal_FOV.mp4'
filename_timestamps_video_normal = 'E:/users/Dm/GL_sample_rover_data/Normal_FoV/timestamps.json'
filename_in_video_wide = 'E:/users/Dm/GL_sample_rover_data/Wide_FoV/Wide_FoV.mp4'
filename_timestamps_video_wide = 'E:/users/Dm/GL_sample_rover_data/Wide_FoV/timestamps.json'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
cmap_names = ['hsv', 'viridis', 'jet', 'rainbow', 'gist_rainbow','gist_ncar', 'nipy_spectral']
# ----------------------------------------------------------------------------------------------------------------------
P = utils_parser.LParser(folder_out)
# ----------------------------------------------------------------------------------------------------------------------
def ex_render():
    filename_in_video, frame_ID = filename_in_video_normal,9561
    #filename_in_video, frame_ID = filename_in_video_wide, 11992

    image = P.get_camera_frame(filename_in_video, frame_ID = frame_ID)
    df_lidar = P.import_lidar_df_fast(folder_in_lidar_import, lidar_frame_id= 7156)
    image_render = P.pointcloud_df_to_render_image(df_lidar)
    image_render = numpy.concatenate([image, image_render], axis=1)

    cv2.imwrite(P.folder_out + 'image_render.png', image_render)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    P.video_BEV(folder_in_lidar_raw,folder_in_lidar_import,filename_in_video_normal,filename_timestamps_video_normal,start_time_sec = 6*60+22)
    #P.video_BEV(folder_in_lidar_raw,folder_in_lidar_import, filename_in_video_wide  ,filename_timestamps_video_wide  ,start_time_sec = 6*60+30)


