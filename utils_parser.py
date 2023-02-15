# https://github.com/daavoo/pyntcloud
# https://github.com/strawlab/python-pcl
# http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#DBSCAN-clustering
import itertools
import cv2
import pickle
import struct
import numpy
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pyransac3d as pyrsc
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import open3d as o3d
# ----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_IO
import tools_wavefront
from CV import tools_pr_geom
import tools_draw_numpy
import tools_image
import tools_time_convertor
import tools_GL3D
# ----------------------------------------------------------------------------------------------------------------------
class LParser(object):
    def __init__(self,folder_out=None):
        self.folder_out = folder_out
        self.R = self.init_render()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_render(self):
        W, H = 720, 720
        cam_fov_deg = 90
        eye = (0, 0, 0)
        target = (0, -1, 0)
        up = (-1, 0, 0)
        M_obj = tools_pr_geom.compose_RT_mat((0, 0, 0), (0, 0, 0), do_rodriges=False, do_flip=False, GL_style=True)
        self.R = tools_GL3D.render_GL3D(filename_obj=None, W=W, H=H, is_visible=False, do_normalize_model_file=False,
                                   textured=True, projection_type='P', cam_fov_deg=cam_fov_deg, scale=(1, 1, 1),
                                   eye=eye, target=target, up=up, M_obj=M_obj)
        return self.R
# ----------------------------------------------------------------------------------------------------------------------
    def str_to_int(self, strng):
        res = int(re.sub(r"[^0-9]", "", strng))
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def float_from_bytes(self,bytes0,big_endian=False):
        if big_endian:
            fmt = '>f'
        else:
            fmt = '<f'
        flt = struct.unpack(fmt, bytes0)[0]

        return flt
# ----------------------------------------------------------------------------------------------------------------------
    def export(self,df,filename):
        with open(filename, "wb") as f:
            pickle.dump(df.values, f)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def import_lidar_rawdata(self, rawdata, connection):
        msg = deserialize_cdr(rawdata, connection.msgtype)
        frame = []
        for r in range(int(msg.row_step / msg.point_step)):
            x = self.float_from_bytes(msg.data[msg.point_step * r + 0:msg.point_step * r + 4])
            y = self.float_from_bytes(msg.data[msg.point_step * r + 4:msg.point_step * r + 8])
            z = self.float_from_bytes(msg.data[msg.point_step * r + 8:msg.point_step * r + 12])
            i = self.float_from_bytes(msg.data[msg.point_step * r +12:msg.point_step * r + 16])
            frame.append([x, y, z, i])

        th_intensity = 28

        frame = numpy.array(frame)
        frame = frame[~numpy.all(frame == 0, axis=1)]
        frame = frame[frame[:,3]>th_intensity]
        df_frame = pd.DataFrame(frame)

        return df_frame
# ----------------------------------------------------------------------------------------------------------------------
    def get_lidar_df_slow(self,folder_in,lidar_frame_id):
        with Reader(folder_in) as reader:
            connections = [x for x in reader.connections if x.topic == '/livox/lidar']
            generator = reader.messages(connections=connections)
            next(itertools.islice(generator, lidar_frame_id, None))
            for connection, ts, rawdata in generator:
                df_lidar = self.import_lidar_rawdata(rawdata, connection)
                break
        return df_lidar
# ----------------------------------------------------------------------------------------------------------------------
    def import_lidar_df_fast(self,folder_in,lidar_frame_id):
        df_lidar = pd.read_csv(folder_in+'%06d.csv'%lidar_frame_id)
        return df_lidar
# ----------------------------------------------------------------------------------------------------------------------
    def get_video_frame_pos_mapping(self,filename_in_video):

        # fast unstable
        # vidcap = cv2.VideoCapture(filename_in_video)
        # total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = vidcap.get(cv2.CAP_PROP_FPS)
        # vidcap.set(cv2.CAP_PROP_POS_FRAMES,total_frames)
        # end_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        #
        # #check
        # fps2 = total_frames/end_time
        # pos_sec = numpy.linspace(0,end_time,total_frames)
        # df = pd.DataFrame({'frame_id': numpy.arange(0,total_frames)})
        # df['pos'] = tools_time_convertor.pretify_timedelta(pos_sec)
        # df.to_csv(self.folder_out + 'df_video_timestamps_fast.csv', index=False)

        # slow stable
        vidcap = cv2.VideoCapture(filename_in_video)
        frame_id,pos_sec = [],[]
        success = True
        count = 0
        while success:
            success, image = vidcap.read()
            frame_id.append(count)
            pos_sec.append(vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
            count += 1

        df = pd.DataFrame({'frame_id':frame_id})
        df['pos'] = tools_time_convertor.pretify_timedelta(pos_sec)
        df.to_csv(self.folder_out+'df_video_timestamps_slow.csv',index=False)


        return df
# ----------------------------------------------------------------------------------------------------------------------
    def get_camera_timestamps(self, filename_timestamps_video, interpolate=False):
        lines = tools_IO.get_lines(filename_timestamps_video)
        frame_ID, ts_sec, msec = [], [], []
        for i,line in enumerate(lines):
            split = line[0].split("timeStamp")
            #frame_ID.append(self.str_to_int(split[0].split(':')[1]))
            frame_ID.append(i)
            ts = self.str_to_int(split[1])
            ts = str(ts) +''.join(['0']*(17-len(str(ts))))

            ts_sec.append(ts[:10])
            msec.append(ts[10:])
        df = pd.DataFrame({'frame_id': frame_ID, 'ts_sec': ts_sec,'msec':msec})

        if interpolate:
            S1 = df['ts_sec'].astype(int)
            Y = 1.0*(S1.values - S1.min())
            Y= Y+numpy.array([int(s[:3]) / 1000 for s in df['msec'].astype('str').values])
            d0 = pd.DataFrame({'frame_id': df['frame_id'].values, 'Y': Y})
            dd = pd.DataFrame({'frame_id':numpy.arange(frame_ID[0],frame_ID[-1])})
            dd = tools_DF.fetch(dd,'frame_id',d0,'frame_id','Y')
            dd['Y'] = dd['Y'].interpolate()
            dd['ts_sec'] = S1.min()+dd['Y'].astype('int')
            dd['msec'] = [('%.7f'%(f%1))[2:] for f in dd['Y']]
            df = dd[['frame_id','ts_sec','msec']]

        df['frame_id'] = df['frame_id'].astype('int')
        df['ts_sec']= df['ts_sec'].astype('int')
        df['msec'] = df['msec'].astype('str')


        delta1 = df['ts_sec'].values - df['ts_sec'].values[0]
        delta2 = numpy.array([int(str(s)[:3]) / 1000 for s in df['msec'].values]) - int( str(df['msec'].values[0])[:3]) / 1000
        delta = delta1 + delta2
        df['time_delta'] = tools_time_convertor.pretify_timedelta(delta)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def get_lidar_timestamps(self, folder_in_lidar):

        with Reader(folder_in_lidar) as reader:
            connections = [x for x in reader.connections if x.topic == '/livox/lidar']

            timestamps = [timestamp for connection, timestamp, rawdata in reader.messages(connections=connections)]
            timestamps = [str(ts) + ''.join(['0'] * (17 - len(str(ts)))) for ts in timestamps]
            ts_sec = [str(ts)[:10] for ts in timestamps]
            msec   = [str(ts)[10:] for ts in timestamps]

            df_lidar_timestamps = pd.DataFrame({'lidar_frame_id':numpy.arange(0,len(timestamps)),'ts_sec':ts_sec,'msec':msec})
            df_lidar_timestamps.to_csv(self.folder_out+'df_lidar_timestamps.csv',index=False)

        return df_lidar_timestamps

# ----------------------------------------------------------------------------------------------------------------------
    def get_lidar_frame_id(self, df_lidar_timestamps, ts_sec, msec):

        s1 = 1.0*(df_lidar_timestamps['ts_sec'].values - ts_sec)
        s2 = (df_lidar_timestamps['msec'].values/1e9 - int(msec)/1e7)
        s1+= s2
        s1 = abs(s1)
        i = numpy.argmin(s1)
        if (s1[i]) > 1.0:
            return -1

        frame_id = df_lidar_timestamps['lidar_frame_id'].iloc[i]
        lidar_ts = df_lidar_timestamps['ts_sec'].iloc[i]
        lidar_msec = df_lidar_timestamps['msec'].iloc[i]

        return frame_id,lidar_ts,lidar_msec
# ----------------------------------------------------------------------------------------------------------------------
    def fetch_lidar_frames(self,df_lidar_timestamps, df_camera_timestamps):

        df_camera_timestamps['lidar_frame_id'] = -1
        df_camera_timestamps['lidar_ts'] = -1
        df_camera_timestamps['lidar_msec'] = -1

        for r in range(df_camera_timestamps.shape[0]):
            ts_sec = df_camera_timestamps['ts_sec'].iloc[r]
            msec = df_camera_timestamps['msec'].iloc[r]
            lidar_frame_id,lidar_ts,lidar_msec = self.get_lidar_frame_id(df_lidar_timestamps,ts_sec, msec)
            df_camera_timestamps.iloc[r,df_camera_timestamps.columns.get_loc('lidar_frame_id')] = lidar_frame_id
            df_camera_timestamps.iloc[r, df_camera_timestamps.columns.get_loc('lidar_ts')] = lidar_ts
            df_camera_timestamps.iloc[r, df_camera_timestamps.columns.get_loc('lidar_msec')] = lidar_msec

        return df_camera_timestamps
# ----------------------------------------------------------------------------------------------------------------------
    def export_lidar_frames(self,folder_in_lidar):
        with Reader(folder_in_lidar) as reader:
            connections = [x for x in reader.connections if x.topic == '/livox/lidar']
            i=0
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                df_frame = self.import_lidar_rawdata(rawdata, connection)
                df_frame.to_csv(self.folder_out+'/lidar/%06d.csv'%i,index=False)
                print(i)
                i+=1
        return
# ----------------------------------------------------------------------------------------------------------------------
    def random_orto_tripple(self,r=0.15):
        rvec = numpy.array([numpy.pi / 2, 0, 0])
        v1 = numpy.ones(3) * r
        v2 = tools_pr_geom.apply_rotation(rvec, v1)[0]
        v3 = numpy.cross(v1, v2)
        res = numpy.concatenate([v1, v2, v3]).reshape(((-1, 3)))
        return res
    # ----------------------------------------------------------------------------------------------------------------------
    def pointcloud_df_to_BEV_image(self, df):

        image = numpy.full((720,720,3),26,dtype=numpy.uint8)
        points = (df.iloc[:,:2].values)*10
        points[:,0]+=image.shape[0]//2
        points[:,1]=image.shape[1]//2-points[:,1]

        image = tools_draw_numpy.draw_points(image, points,color=tools_draw_numpy.color_white,w=1,transperency=0.95)


        image = tools_image.rotate_image(image, 90, reshape=True)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def pointcloud_df_to_render_image(self, df_lidar):

        tools_IO.remove_files(self.folder_out, '*.obj,*.mtl')
        #self.pointcloud_df_to_obj(df_lidar,'lidar.obj',max_V=9000,noise=0.05,auto_triangulate=False,color=(200,200,200))
        self.pointcloud_df_to_obj_v2_clusters(df_lidar)

        self.R.my_VBO.remove_total()
        self.R.init_object(self.folder_out)
        self.R.bind_VBO()
        image = self.R.get_image()

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def construct_noise(self,df):
        a = 0.05
        df['yaw'] = df.T.apply(lambda x: numpy.arctan(x[1] / (x[0] + 1e-8)))
        p1 = numpy.array((0, 0, +a / numpy.sqrt(3), 1))
        p2 = numpy.array((0, +a / 2, -a * numpy.sqrt(3) / 6.0, 1))
        p3 = numpy.array((0, -a / 2, -a * numpy.sqrt(3) / 6.0, 1))

        p1r = df.T.apply(lambda x: p1.dot(tools_pr_geom.compose_RT_mat((x['yaw'], 0 * x['yaw'], 0), (0, 0, 0), do_rodriges=True, do_flip=False,GL_style=True))).T
        p2r = df.T.apply(lambda x: p2.dot(tools_pr_geom.compose_RT_mat((x['yaw'], 0 * x['yaw'], 0), (0, 0, 0), do_rodriges=True, do_flip=False,GL_style=True))).T
        p3r = df.T.apply(lambda x: p3.dot(tools_pr_geom.compose_RT_mat((x['yaw'], 0 * x['yaw'], 0), (0, 0, 0), do_rodriges=True, do_flip=False,GL_style=True))).T
        df_noise = pd.concat([p1r, p2r, p3r], axis=0)
        df_noise['i'] = numpy.concatenate([numpy.arange(0, 3 * df.shape[0], 3), numpy.arange(1, 3 * df.shape[0], 3),numpy.arange(2, 3 * df.shape[0], 3)])
        df_noise = df_noise.sort_values(by='i')
        return df_noise
# ----------------------------------------------------------------------------------------------------------------------
    def pointcloud_df_to_obj(self, df, filename_out, image=None, max_V=None, noise=0.05, auto_triangulate=False, color=(128, 128, 128)):

        if max_V is not None and df.shape[0]>max_V:
            df = df.iloc[numpy.random.choice(df.shape[0], max_V, replace=False)].copy()

        if auto_triangulate:
            df3 = df
            idx_vertex3 = None
        else:
            df3 = pd.concat([df, df, df], axis=0)
            df3['i'] = numpy.concatenate([numpy.arange(0, 3 * df.shape[0], 3), numpy.arange(1, 3 * df.shape[0], 3),numpy.arange(2, 3 * df.shape[0], 3)])
            df3 = df3.sort_values(by='i')
            df3.drop(columns=['i'],inplace=True)
            idx_vertex3 = numpy.arange(0, df3.shape[0]).reshape((-1, 3))

            if noise>0:
                df_noise = pd.DataFrame(numpy.array([self.random_orto_tripple(r=noise)] * df.shape[0]).reshape((-1, 3)))
                df3.iloc[:,:3]+=df_noise.iloc[:,:3].values


        coord_texture3 = numpy.array([numpy.linalg.norm(v[:3]) for v in df3.values])
        coord_texture3-= numpy.min(coord_texture3)
        coord_texture3 = coord_texture3/numpy.max(coord_texture3+(1e-4))
        coord_texture3 = numpy.concatenate([coord_texture3.reshape((-1,1)),numpy.full((coord_texture3.shape[0],1),0.95)],axis=1)

        base_name=''.join(filename_out.split('.')[:-1])

        filename_obj = base_name+'.obj'
        filename_material = base_name+'.mtl'
        filename_texture = None
        if image is not None:
            filename_texture = base_name+'.jpg'
            self.create_texture(filename_texture, image,'jet')

        obj = tools_wavefront.ObjLoader()
        obj.export_material(self.folder_out + filename_material, color, filename_texture=filename_texture)
        obj.export_mesh(self.folder_out+filename_obj, df3.values, coord_texture=coord_texture3, idx_vertex=idx_vertex3, do_transform=False, filename_material=filename_material)

        return

# ----------------------------------------------------------------------------------------------------------------------
    def get_clusters(self,df,eps=0.5, min_points=100):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df.iloc[:, :3].values)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            labels = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
            #labels = numpy.array(labels)
            #max_label = numpy.max(labels)
            #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            #colors[labels < 0] = 0
            #pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            #o3d.visualization.draw_geometries([pcd])
        return numpy.array(labels)
# ----------------------------------------------------------------------------------------------------------------------
    def strighten_plane(self,df,eq=None):

        if eq is None:
            plane1 = pyrsc.Plane()
            eq, best_inliers = plane1.fit(df.iloc[:, :3].values,thresh=0.1, minPoints=10)
            df = df.iloc[best_inliers]

        df_res = self.plane_to_polygon(df,eq)
        return df_res
# ----------------------------------------------------------------------------------------------------------------------
    def get_ortho_planes(self, df,do_streight=True,remove_ground=True,augment_bottom=True):

        cuboid = pyrsc.Cuboid()
        best_eqs,best_inliers = cuboid.fit(df.iloc[:, :3].values, thresh=0.05, maxIteration=5000)
        df_best=df.iloc[best_inliers].copy()
        df_best['I'] = 1
        loss = abs(numpy.array([best_eqs[0].reshape((-1,4)).dot(df_best.T.values),
                                best_eqs[1].reshape((-1,4)).dot(df_best.T.values),
                                best_eqs[2].reshape((-1,4)).dot(df_best.T.values)])).reshape((3, -1)).T

        i = numpy.argmin(loss,axis=1)
        df_xx = df.iloc[[i for i in range(df.shape[0]) if i not in best_inliers]]
        df_p0 = df_best.iloc[i==0,:3]
        df_p1 = df_best.iloc[i==1,:3]
        df_p2 = df_best.iloc[i==2,:3]

        if do_streight:
            df_p0 = self.strighten_plane(df_p0, best_eqs[0])
            df_p1 = self.strighten_plane(df_p1, best_eqs[1])
            df_p2 = self.strighten_plane(df_p2, best_eqs[2])

        df_res_list = [df_p0,df_p1,df_p2]

        if remove_ground:
            loss = numpy.array([min(numpy.linalg.norm(best_eqs[i,:3]-(0,0,1)),numpy.linalg.norm(best_eqs[i,:3]-(0,0,-1))) for i in range(3)])
            df_res_list = [df for i,df in enumerate(df_res_list) if i!=numpy.argmin(loss)]

            if augment_bottom:
                df_best = pd.concat([df_res_list[0], df_res_list[1]], axis=0)
                best_eqs, best_inliers = cuboid.fit(df_best.iloc[:, :3].values)
                df_best['I'] = 1
                loss = abs(numpy.array([best_eqs[0].reshape((-1, 4)).dot(df_best.T.values),
                                        best_eqs[1].reshape((-1, 4)).dot(df_best.T.values),
                                        best_eqs[2].reshape((-1, 4)).dot(df_best.T.values)])).reshape((3, -1)).T
                df_best.drop(columns=['I'], inplace=True)
                df_res_list=df_res_list+[df_best[loss[:,2]<0.2]]




        return [df_xx]+df_res_list

# ----------------------------------------------------------------------------------------------------------------------
    def plane_to_polygon(self, df,eq):

        E = numpy.array((0,0,0*eq[3]))
        forward = numpy.array(eq[:3])
        side = tools_pr_geom.perpendicular_vector3d(forward)
        U = numpy.cross(forward,side)

        M = tools_pr_geom.ETU_to_mat_view(E, E+forward, U)
        df['I']=1
        X_plane = df.dot(numpy.linalg.pinv(M)).values

        if X_plane.shape[0]>4:
            hull = ConvexHull(X_plane[:, :2])
            cntrs_2d = X_plane[hull.vertices,:2]
            cntrs_3d = numpy.concatenate([cntrs_2d, numpy.full((cntrs_2d.shape[0],1),X_plane[:,2].mean()),numpy.ones(cntrs_2d.shape[0]).reshape((-1,1))], axis = 1)
            X2 = cntrs_3d.dot(M)
            df_res = pd.DataFrame(X2[:,:3])
        else:
            X_plane[:,2] = X_plane[:,2].mean()
            X2 = X_plane.dot(M)
            df_res = pd.DataFrame(X2[:,:3])

        return df_res
# ----------------------------------------------------------------------------------------------------------------------
    def pointcloud_df_to_obj_v2_clusters(self,df):

        labels = self.get_clusters(df,eps=0.5, min_points=100)

        df_cluster0=pd.DataFrame([])
        colors = tools_draw_numpy.get_colors(numpy.max(labels)+1,colormap = 'jet')
        for l in numpy.unique(labels):
            df_cluster = df.iloc[labels == l, :3]

            if l==-1:
                df_cluster0 = df_cluster.copy()
            else:
                max_V,noise,auto_triangulate,color = None,0,True,colors[l]

                df_plenes = self.get_ortho_planes(df_cluster,do_streight=True,remove_ground=True)
                df_cluster0 = pd.concat([df_cluster0,df_plenes[0]],axis=1,ignore_index=True)
                for p,df_plane in enumerate(df_plenes[1:]):
                    filename_out = 'cluster_%02d%c.obj' % (l, chr(ord('A')+p))
                    self.pointcloud_df_to_obj(df_plane, filename_out, image=None, max_V=max_V, noise=noise,auto_triangulate=auto_triangulate, color=color)

        max_V,noise,auto_triangulate,color = 8000,0.05,False,(192,192,192)
        self.pointcloud_df_to_obj(df_cluster0, 'cluster_none.obj', image=None, max_V=max_V, noise=noise,auto_triangulate=auto_triangulate, color=color)

        return

# ----------------------------------------------------------------------------------------------------------------------
    def get_camera_frame(self,filename_in_video,frame_ID):
        vidcap = cv2.VideoCapture(filename_in_video)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_ID)
        success, image = vidcap.read()
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def create_bar(self,H,W,cm_name):
        image = numpy.full((H, W, 3), 0, dtype=numpy.uint8)
        N = plt.get_cmap(cm_name).N
        colors = tools_draw_numpy.get_colors(N, colormap=cm_name, interpolate=False)
        for n in range(N - 1):
            j1 = int(n * W / (N - 1))
            j2 = int((n + 1) * W / (N - 1))
            image[:, j1:j2, :] = colors[n]
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def create_texture(self,filanema_out,image,cm_name=None):

        if cm_name is None:
            image_bar =numpy.full((int(image.shape[0]*0.1),image.shape[1],3),255,dtype=numpy.uint8)
        else:
            image_bar = self.create_bar(int(image.shape[0]*0.1),image.shape[1],cm_name)
        cv2.imwrite(self.folder_out+filanema_out,numpy.concatenate([image, image_bar], axis=0))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def video_BEV(self,folder_in_lidar_raw,folder_in_lidar_import,filename_in_video,filename_timestamps_video,start_time_sec=None,stop_time_sec=None):

        #df_lidar_timestamps = self.get_lidar_timestamps(folder_in_lidar_raw)
        df_lidar_timestamps = pd.read_csv(self.folder_out+'df_lidar_timestamps_wide.csv')

        df_camera_timestamps = self.get_camera_timestamps(filename_timestamps_video)
        df_camera_timestamps = self.fetch_lidar_frames(df_lidar_timestamps,df_camera_timestamps)
        df_camera_timestamps.to_csv(self.folder_out + 'df_camera_timestamps.csv', index=False)

        frame_ID = df_camera_timestamps['frame_id'].iloc[0]
        vidcap = cv2.VideoCapture(filename_in_video)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_ID)

        if start_time_sec is not None:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
            frame_ID = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))


        success = True
        #and ((stop_time_sec is not None) and vidcap.get(cv2.CAP_PROP_POS_MSEC)<stop_time_sec)
        while success and (frame_ID<df_camera_timestamps['frame_id'].iloc[-1]) :
            success, image = vidcap.read()
            DD = df_camera_timestamps[df_camera_timestamps['frame_id'] == frame_ID]
            lidar_frame_id = DD['lidar_frame_id'].iloc[0]

            df_lidar = self.import_lidar_df_fast(folder_in_lidar_import, lidar_frame_id)
            image_3D_render = self.pointcloud_df_to_BEV_image(df_lidar)
            #image_3D_render = self.pointcloud_df_to_render_image(df_lidar)
            image = numpy.concatenate([image, image_3D_render],axis=1)

            label1 = 'camra_frame_id: %d' % (frame_ID)
            label2 = 'lidar_frame_id: %d' % (lidar_frame_id)
            label3 = 'camra ts:%d:%s' % (DD['ts_sec'].iloc[0], DD['msec'].iloc[0])
            label4 = 'lidar ts:%d:%s' % (DD['lidar_ts'].iloc[0], DD['lidar_msec'].iloc[0])

            image = tools_draw_numpy.draw_text(image,label1,(1280+10,720-120), tools_draw_numpy.color_blue,font_size=24)
            image = tools_draw_numpy.draw_text(image,label2,(1280+10,720-90 ), tools_draw_numpy.color_blue,font_size=24)
            image = tools_draw_numpy.draw_text(image,label3,(1280+10,720-60 ), tools_draw_numpy.color_blue,font_size=24)
            image = tools_draw_numpy.draw_text(image,label4,(1280+10,720-30 ), tools_draw_numpy.color_blue,font_size=24)

            cv2.imwrite(self.folder_out+'%06d.jpg'%int(frame_ID),image)
            print(frame_ID)
            frame_ID+=1



        return
# ----------------------------------------------------------------------------------------------------------------------
