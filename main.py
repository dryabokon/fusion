import numpy
import pandas as pd
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import utils_parser
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
def ex_convert_lidar_camera_data(folder_in_lidar):
    P.export_lidar_frames(folder_in_lidar)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_render():
    df_camera_timestamps = pd.read_csv(P.folder_out + 'df_camera_timestamps_normal.csv')
    filename_in_video, cam_frame_id = filename_in_video_normal,9380
    #filename_in_video, frame_ID = filename_in_video_wide, 11992

    #lidar_frame_id = int(df_camera_timestamps[df_camera_timestamps['frame_id'] == cam_frame_id]['lidar_frame_id'])


    #image = P.get_camera_frame(filename_in_video, frame_ID = frame_ID)
    df_lidar = P.import_lidar_df_fast(folder_in_lidar_import, lidar_frame_id= 0)
    image_render = P.pointcloud_df_to_render_image(df_lidar)
    #image_render = numpy.concatenate([image, image_render], axis=1)
    cv2.imwrite(P.folder_out + 'image_render.png', image_render)

    return
# ----------------------------------------------------------------------------------------------------------------------
def detect_lines(folder_in_lidar_raw,filename_in_video,filename_timestamps_video):

    #df_lidar_timestamps = P.get_lidar_timestamps(folder_in_lidar_raw)
    #df_camera_timestamps = P.get_camera_timestamps(filename_timestamps_video)
    #df_camera_timestamps = P.fetch_lidar_frames(df_lidar_timestamps, df_camera_timestamps)
    #df_camera_timestamps.to_csv(P.folder_out + 'df_camera_timestamps.csv', index=False)
    df_camera_timestamps = pd.read_csv(P.folder_out + 'df_camera_timestamps_normal.csv')

    for cam_frame_id in [8358,8361,9361,9380,9542,9555]:
        lidar_frame_id = int(df_camera_timestamps[df_camera_timestamps['frame_id']==cam_frame_id]['lidar_frame_id'])

        df_lidar = P.import_lidar_df_fast(folder_in_lidar_import, lidar_frame_id= lidar_frame_id)
        image = P.get_camera_frame(filename_in_video, frame_ID = cam_frame_id)
        image_render = P.pointcloud_df_to_render_image(df_lidar)
        image_render = numpy.concatenate([image, image_render], axis=1)
        cv2.imwrite(P.folder_out + '%06d.png'%cam_frame_id, image_render)

    return
# ----------------------------------------------------------------------------------------------------------------------
def batch_rename_lidar_files():
    from shutil import copyfile

    folder_in  = 'E:/users/Dm/GL_sample_rover_data/LiDAR/import/'
    folder_out = 'E:/users/Dm/GL_sample_rover_data/LiDAR_Wide/'

    df_camera_timestamps = pd.read_csv('./data/output/df_camera_timestamps_wide.csv')
    for r in range(df_camera_timestamps.shape[0]):
        cam_frame_id = df_camera_timestamps.iloc[r,0]
        lidar_frame_id = df_camera_timestamps.iloc[r,4]
        lidar_filename_source = '%06d.pcd'%lidar_frame_id
        lidar_filename_target = '%06d_%06d.pcd' % (lidar_frame_id,cam_frame_id)
        copyfile(folder_in + lidar_filename_source, folder_out + lidar_filename_target)


    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #P.video_BEV(folder_in_lidar_raw,folder_in_lidar_import,filename_in_video_normal,filename_timestamps_video_normal,start_time_sec = 6*60+22)
    #P.video_BEV(folder_in_lidar_raw,folder_in_lidar_import, filename_in_video_wide  ,filename_timestamps_video_wide  ,start_time_sec = 6*60+30)

    #P.pointcloud_df_to_obj_v2(P.import_lidar_df_fast(folder_in_lidar_import, lidar_frame_id= 7156))

    #detect_lines(folder_in_lidar_raw,filename_in_video_normal,filename_timestamps_video_normal)

    #ex_convert_lidar_camera_data(folder_in_lidar_raw)
    #ex_render()
    batch_rename_lidar_files()