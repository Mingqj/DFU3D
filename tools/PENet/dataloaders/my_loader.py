from dataloaders import calibration_kitti
import numpy as np
from skimage import io
import cv2
from PIL import Image
import os
import copy
import torch
from dataloaders.spconv_utils import replace_feature, spconv
from torch import nn
import torch.nn.functional as F
import open3d as o3d

import torch
import numpy as np
from rectangle_fitting.rectangle_fitting import LShapeFitting
tv = None
try:
    import cumm.tensorview as tv
except:
    pass
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

voxel_generator = VoxelGeneratorWrapper(
        vsize_xyz=[200, 0.002, 0.002],
        coors_range_xyz=[-100,-5,-5,100,5,5],
        num_point_features=11,
        max_num_points_per_voxel=100,
        max_num_voxels=1000000,
    )

voxel_generator0 = VoxelGeneratorWrapper(
        vsize_xyz=[200, 0.002, 0.002],
        coors_range_xyz=[-100,-5,-5,100,5,5],
        num_point_features=12,
        max_num_points_per_voxel=100,
        max_num_voxels=1000000,
    )

def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

def load_depth_input(calib, image, points):
    image = copy.deepcopy(image)
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, image.shape, calib)
    points = points[fov_flag]

    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)

    val_inds = (pts_img[:, 0] >= 0) & (pts_img[:, 1] >= 0)
    val_inds = val_inds & (pts_img[:, 0] < image.shape[1]) & (pts_img[:, 1] < image.shape[0])

    pts_img = pts_img[val_inds].astype(np.int32)
    depth = pts_rect_depth[val_inds]

    new_im = np.zeros(shape=image.shape[0:2])
    new_im[pts_img[:, 1], pts_img[:, 0]] = depth
    depth = np.expand_dims(new_im, -1)
    rgb_png = np.array(image, dtype='uint8')

    return rgb_png, depth

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float32) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)

    return depth

def depth2points(depth, calib):
    depth[depth<0.1] = 0
    uv = depth.nonzero()
    depth_val = depth[depth>0]

    p_rect = calib.img_to_rect(uv[1], uv[0], depth_val)
    p_lidar = calib.rect_to_lidar(p_rect)

    return p_lidar

def depth2pointsrgb(depth, image, calib):
    depth[depth<0.1] = 0
    uv = depth.nonzero()
    depth_val = depth[depth>0]

    new_p = np.zeros(shape=(uv[0].shape[0], 6))

    p_rect = calib.img_to_rect(uv[1], uv[0], depth_val)
    p_lidar = calib.rect_to_lidar(p_rect)
    new_p[:, 0:3] = p_lidar
    new_p[:, 3:] = image[uv[0], uv[1]]

    return new_p

def to_sphere_coords(points):
    r = np.linalg.norm(points[:, 0:3], ord=2, axis=-1)
    theta = np.arccos(points[:, 2]/r)
    fan = np.arctan(points[:, 1]/points[:, 0])

    new_points = copy.deepcopy(points)
    new_points[:, 0] = r
    new_points[:, 1] = theta
    new_points[:, 2] = fan
    mask1 = new_points[:, 1]>1.5

    new_points=new_points[mask1]
    points = points[mask1]

    return new_points, points

def de_noise(points, vert_res = 0.05, hor_res = 0.05):
    new_points = copy.deepcopy(points)

    sp_coords, new_points = to_sphere_coords(new_points)

    voxel_dict = {}

    for i, point in enumerate(sp_coords):

        vert_coord = point[1]//vert_res
        hor_coord = point[2]//hor_res

        voxel_key = str(vert_coord)+'_'+str(hor_coord)

        if voxel_key in voxel_dict:

            voxel_dict[voxel_key]['sp'].append(point)
            voxel_dict[voxel_key]['pts'].append(new_points[i])
        else:
            voxel_dict[voxel_key] = {'sp': [point], 'pts': [new_points[i]]}

    sampled_list = []

    for voxel_key in voxel_dict:

        sp = voxel_dict[voxel_key]['pts']
        if len(sp)<=20:
            continue

        sampled_list+=sp

    return np.array(sampled_list)

def la_sampling(points, vert_res = 0.002, hor_res = 0.002):
    new_points = copy.deepcopy(points)

    sp_coords, new_points = to_sphere_coords(new_points)
    voxel_dict = {}

    for i, point in enumerate(sp_coords):

        vert_coord = point[1]//vert_res
        hor_coord = point[2]//hor_res

        voxel_key = str(vert_coord)+'_'+str(hor_coord)

        if voxel_key in voxel_dict:

            voxel_dict[voxel_key]['sp'].append(point)
            voxel_dict[voxel_key]['pts'].append(new_points[i])
        else:
            voxel_dict[voxel_key] = {'sp': [point], 'pts': [new_points[i]]}

    sampled_list = []

    for voxel_key in voxel_dict:

        sp = voxel_dict[voxel_key]['pts'] #N,10

        arg_min = np.argmin(np.array(sp)[:, 0])
        min_point = voxel_dict[voxel_key]['pts'][arg_min]
        sampled_list.append(min_point)

    return np.array(sampled_list)

def la_sampling2(points, vert_res=0.002, hor_res=0.002):
    new_points = copy.deepcopy(points)

    sp_coords, new_points = to_sphere_coords(new_points)

    cat_points = np.concatenate([sp_coords, new_points[:,0:3]],-1)
    voxels, coordinates, num_points = voxel_generator.generate(cat_points)
    finals = []
    for i,voxel in enumerate(voxels):
        pt_n = num_points[i]
        arg_min = np.argmin(np.array(voxel[:pt_n, 10]))
        finals.append(voxel[arg_min])
    finals = np.array(finals)
    return np.concatenate([finals[:, 8:11], finals[:, 3:8]],-1)

def la_sampling20(points, vert_res=0.002, hor_res=0.002):
    new_points = copy.deepcopy(points)

    sp_coords, new_points = to_sphere_coords(new_points)

    cat_points = np.concatenate([sp_coords, new_points[:,0:3]],-1)
    voxels, coordinates, num_points = voxel_generator0.generate(cat_points)
    finals = []
    for i,voxel in enumerate(voxels):
        pt_n = num_points[i]
        arg_min = np.argmin(np.array(voxel[:pt_n, 10]))
        finals.append(voxel[arg_min])
    finals = np.array(finals)
    return np.concatenate([finals[:, 9:12], finals[:, 3:9]],-1)

def voxel_sampling(point2, res_x=0.05, res_y=0.05, res_z = 0.05):

    min_x = -100
    min_y = -100
    min_z = -10

    voxels = {}

    for point in point2:
        x = point[0]
        y = point[1]
        z = point[2]

        x_coord = (x-min_x)//res_x
        y_coord = (y-min_y)//res_y
        z_coord = (z-min_z)//res_z

        key = str(x_coord)+'_'+str(y_coord)+'_'+str(z_coord)

        voxels[key] = point

    return np.array(list(voxels.values()))

def lidar_guied_voxel_sampling(point2, ref_points, res_x=0.2, res_y=0.2, res_z = 0.2):

    min_x = -100
    min_y = -100
    min_z = -10

    voxels = {}

    for point in ref_points:
        x = point[0]
        y = point[1]
        z = point[2]

        x_coord = (x-min_x)//res_x
        y_coord = (y-min_y)//res_y
        z_coord = (z-min_z)//res_z

        key = str(x_coord)+'_'+str(y_coord)+'_'+str(z_coord)

        voxels[key] = 1

    new_points = []
    for point in point2:
        x = point[0]
        y = point[1]
        z = point[2]

        x_coord = (x - min_x) // res_x
        y_coord = (y - min_y) // res_y
        z_coord = (z - min_z) // res_z

        key = str(x_coord) + '_' + str(y_coord) + '_' + str(z_coord)

        if key in voxels:
            new_points.append(point)

    return np.array(new_points)

def lidar_guied_dis_sampling(point2, ref_points, dis = 0.3, res_z = 0.3):
    point2[np.abs(point2[:, 0] > 100)] = 100
    point2[np.abs(point2[:, 1] > 100)] = 100
    new_points=[]
    for i, point in enumerate(ref_points):
        if i%1000==0:
            print(i)
        x = point[0]
        y = point[1]
        z = point[2]
        mask_x = np.abs(point2[:, 0] - x) < dis
        mask_y = np.abs(point2[:, 1] - y) < dis
        mask_z = np.abs(point2[:, 2] - z) < res_z

        mask = mask_x*mask_z*mask_y

        new_points.append(point2[mask])

        point2[mask]=10000

    return np.concatenate(new_points)

def range_sampling(points2, ref_points, calib, pix_dis_x = 1, pix_dis_y = 7, depth_dis = 0.3):
    pts_img2, pts_depth2 = calib.lidar_to_img(points2[:, 0:3])
    ref_img, ref_depth = calib.lidar_to_img(ref_points[:, 0:3])

    pts = np.concatenate([pts_img2, pts_depth2.reshape(pts_img2.shape[0], 1)], -1)
    ref = np.concatenate([ref_img, ref_depth.reshape(ref_img.shape[0], 1)], -1)

    new_points=[]

    for i, point in enumerate(ref):
        if i%1000==0:
            print(i)
        x = point[0]
        y = point[1]
        dis = point[2]
        mask_x = np.abs(pts[:, 0] - x) < pix_dis_x
        mask_y = np.abs(pts[:, 1] - y) < pix_dis_y
        mask_z = np.abs(pts[:, 2] - dis) < depth_dis

        mask = mask_x*mask_z*mask_y

        new_points.append(points2[mask])

        pts[mask]=100000

    return np.concatenate(new_points)

def range_sampling_torch(points2, ref_points, calib, pix_dis_x = 4, pix_dis_y = 7, depth_dis = 0.5):
    pts_img2, pts_depth2 = calib.lidar_to_img(points2[:, 0:3])
    ref_img, ref_depth = calib.lidar_to_img(ref_points[:, 0:3])

    pts = np.concatenate([pts_img2, pts_depth2.reshape(pts_img2.shape[0], 1)], -1)
    ref = np.concatenate([ref_img, ref_depth.reshape(ref_img.shape[0], 1)], -1)

    pts_t = torch.from_numpy(pts).cuda()

    mask_all = torch.zeros((points2.shape[0],)).bool().cuda()

    for i, point in enumerate(ref):

        x = point[0]
        y = point[1]
        dis = point[2]
        mask_x = torch.abs(pts_t[:, 0] - x) < pix_dis_x
        mask_y = torch.abs(pts_t[:, 1] - y) < pix_dis_y
        mask_z1 = (pts_t[:, 2] - dis) < depth_dis
        mask_z2 = (pts_t[:, 2] - dis) > 0
        mask_z = mask_z1*mask_z2

        mask = mask_x*mask_z*mask_y
        pts_t[mask] = 100000
        mask_all+=mask

    return points2[mask_all.cpu().numpy()]

def depth2pointsrgbp(depth, image, calib, lidar):
    depth[depth<0.01] = 0
    uv = depth.nonzero()
    depth_val = depth[depth>0]

    new_p = np.zeros(shape=(uv[0].shape[0], 8))

    p_rect = calib.img_to_rect(uv[1], uv[0], depth_val)
    p_lidar = calib.rect_to_lidar(p_rect)
    new_p[:, 0:3] = p_lidar
    new_p[:, 4:7] = image[uv[0], uv[1]]/3
    new_p = new_p[new_p[:, 2] < 1.]
    new_p = la_sampling2(new_p)
    new_p[:, -1] = 1

    new_lidar = np.zeros(shape=(lidar.shape[0], 8))
    new_lidar[:, 0:4] = lidar[:, 0:4]
    new_lidar[:, 3] *= 10
    new_lidar[:, -1] = 2

    #new_p = new_p[new_p[:, 2]<1.]
    #_, new_p = to_sphere_coords(new_p)
    #new_p = voxel_sampling(new_p)
    #new_p = range_sampling_torch(new_p, new_lidar, calib)

    all_points = np.concatenate([new_lidar, new_p], 0)

    return all_points

from sklearn.linear_model import RANSACRegressor
import scipy

# def estimate_plane(origin_ptc, max_hs=-1.5, it=1, ptc_range=((-30, 80), (-30, 80))): # kitti
def estimate_plane(origin_ptc, max_hs=8, it=1, ptc_range=((-54, 54), (-54, 54))): # nuscenes
    mask = (origin_ptc[:, 2] < max_hs) & \
        (origin_ptc[:, 0] > ptc_range[0][0]) & \
        (origin_ptc[:, 0] < ptc_range[0][1]) & \
        (origin_ptc[:, 1] > ptc_range[1][0]) & \
        (origin_ptc[:, 1] < ptc_range[1][1])
    for _ in range(it):
        ptc = origin_ptc[mask]
        reg = RANSACRegressor().fit(ptc[:, [0, 1]], ptc[:, 2])
        w = np.zeros(3)
        w[0] = reg.estimator_.coef_[0]
        w[1] = reg.estimator_.coef_[1]
        w[2] = -1.0
        h = reg.estimator_.intercept_
        norm = np.linalg.norm(w)
        w /= norm
        h = h / norm
        result = np.array((w[0], w[1], w[2], h))
        result *= -1
        mask = np.logical_not(above_plane(
            origin_ptc[:, :3], result, offset=0.2))
    return result

def above_plane(ptc, plane, offset=0.05, only_range=((-54, 54), (-54, 54))):
    mask = distance_to_plane(ptc, plane, directional=True) < offset
    if only_range is not None:
        range_mask = (ptc[:, 0] < only_range[0][1]) * (ptc[:, 0] > only_range[0][0]) * \
            (ptc[:, 1] < only_range[1][1]) * (ptc[:, 1] > only_range[1][0])
        mask *= range_mask
    return np.logical_not(mask)

def distance_to_plane(ptc, plane, directional=False):
    d = ptc @ plane[:3] + plane[3]
    if not directional:
        d = np.abs(d)
    d /= np.sqrt((plane[:3]**2).sum())
    return d


from PIL import Image

def BallQuery(point1, point2, C):
    distance = (point1[:, None, :] - point2[None, :, 0:3]).norm(dim=-1)
    min_dis, min_dis_idx = distance.min(dim=-1)
    roi_max_dim = (point2[min_dis_idx, 3:6] / 3).norm(dim=-1)
    point_mask = min_dis < roi_max_dim + C
    return distance, min_dis, min_dis_idx, point_mask

def remove_center(ptc, x_range=(-1.15, 1.75), y_range=(-0.65, 0.65)):
    mask = (ptc[:, 0] < x_range[1]) & (ptc[:, 0] >= x_range[0]) & (
        ptc[:, 1] < y_range[1]) & (ptc[:, 1] >= y_range[0])
    mask = np.logical_not(mask)
    return ptc[mask]

def depth2pointsrgbpm(depth, image, image_ori, calib, lidar, thing_classes, mask_image, classes, scores, boxes2D, lidar_depth, file_idx):
    root_path = "/opt/data/private/codeN1/mmdetection3d/data/nuscenes_kitti_format/train/label_train/"
    calib_path = "/opt/data/private/codeN1/mmdetection3d/data/nuscenes_kitti_format/train/calib/"


    depth[depth<0.001] = 0
    uv = depth.nonzero()
    depth_val = depth[depth>0]

    plane = estimate_plane(lidar[:, :3])
    plane_mask = above_plane(lidar[:, :3], plane)
    lidar_mask = lidar[plane_mask]
    print('ori lidar shape:', lidar.shape)
    print('above plane lidar shape:', lidar_mask.shape)

    # lidar2depthmap
    pts_img, pts_depth = calib.lidar_to_img(lidar_mask[:, :3])

    pts_img = np.round(pts_img).tolist()
    lidar_object_mask_list = []
    for j in range(mask_image.shape[0]):
        mask_one_object = mask_image[j]
        mask_one_object = mask_one_object.detach().cpu().numpy()
        mask_one_object1 = Image.fromarray(np.uint8(mask_one_object))
        mask_list = [mask_one_object1.getpixel(tuple(xy)) for xy in pts_img if 0 <= xy[0] < 1600 and 0 <= xy[1] < 900]
        min_length = min(len(mask_list), lidar_mask.shape[0])
        mask_list = mask_list[:min_length]
        lidar_mask = lidar_mask[:min_length]
        lidar_object_mask_list.append(lidar_mask[np.array(mask_list) > 0][:, :3])

    new_p = np.zeros(shape=(uv[0].shape[0], 8))

    # kitti image2lidar
    p_rect = calib.img_to_rect(uv[1], uv[0], depth_val)
    p_lidar = calib.rect_to_lidar(p_rect)

    new_p[:, 0:3] = p_lidar
    new_p[:, 4:7] = image[uv[0], uv[1]]/3
    new_p0 = new_p[new_p[:, 2] < 1.]
    new_p1 = la_sampling2(new_p0)
    new_p1[:, -1] = 1
    
    # select objects
    file = open(os.path.join(root_path, file_idx + '.txt'), 'w')
    forvis = []
    for i in range(mask_image.shape[0]):
        # image_mask to mask of pseudo points
        mask_one_object = mask_image[i] # [352, 1216]
        class_one_object = classes[i] # [class]
        one_object_class = thing_classes[class_one_object]
        score_one_object = scores[i] # [score]
        boxes2D_one_object = boxes2D[i] # [2D coordinate]
        new_p_mask = mask_one_object[uv[0], uv[1]][new_p[:, 2] < 1.].unsqueeze(1)
        new_p_withMask = np.concatenate((new_p0, new_p_mask.detach().cpu().numpy()), axis=1)
        new_p1_withMask = la_sampling20(new_p_withMask) # [vir_points_number, 9]
        one_object_pseudo_point = new_p1_withMask[new_p1_withMask[:, 8] > 0.0][:, :3]

        point_type = 'multi-modal'
        if point_type == 'point_cloud':
            one_object_points = lidar_object_mask_list[i]
        elif point_type == 'pseudo_points':
            one_object_points = one_object_pseudo_point
        elif point_type == 'multi-modal':
            one_object_point_cloud = lidar_object_mask_list[i]

        # filtering
        if point_type == 'pseudo_points': # pseudo point
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(one_object_points)
            uni_down_pcd = pcd.uniform_down_sample(every_k_points=4)
            cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=20, radius=3)
            one_object_points = np.array(cl.points)
        elif point_type == 'point_cloud': # point cloud 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(one_object_points)
            cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=3)
            one_object_points = np.array(cl.points)
        elif point_type == 'multi-modal':
            # point cloud
            if one_object_class == "Car" or one_object_class == 'Truck' or one_object_class == 'Construction_vehicle' or one_object_class == 'Bus' or one_object_class == 'Trailer':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(one_object_point_cloud)
                cl, ind = pcd.remove_radius_outlier(nb_points=1, radius=3)
                one_object_point_cloud = torch.from_numpy(np.array(cl.points))
                # pseudo point
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(one_object_pseudo_point)
                cl1, ind = pcd1.remove_radius_outlier(nb_points=1, radius=3)
                one_object_pseudo_point = torch.from_numpy(np.array(cl1.points))
            elif one_object_class == "Pedestrian" or one_object_class == "Bicycle" or one_object_class == "Motorcycle" or one_object_class == "Barrier" or one_object_class == "Traffic_cone":
                pcd = o3d.geometry.PointCloud()
                cl, ind = pcd.remove_radius_outlier(nb_points=1, radius=0.6)
                one_object_point_cloud = torch.from_numpy(np.array(cl.points))
                # pseudo point
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(one_object_pseudo_point)
                cl1, ind = pcd1.remove_radius_outlier(nb_points=1, radius=0.6)
                one_object_pseudo_point = torch.from_numpy(np.array(cl1.points))

            # fuse
            if one_object_point_cloud.shape[0] > 0 and one_object_pseudo_point.shape[0] > 0:
                _, _, _, pseudo_point_idx = BallQuery(one_object_pseudo_point, one_object_point_cloud, 0.1)
                one_object_pseudo_point = one_object_pseudo_point[pseudo_point_idx]
            one_object_points = torch.cat((one_object_point_cloud, one_object_pseudo_point), 0).numpy()

        forvis = GenerateAnns(one_object_class, one_object_points, class_one_object, score_one_object, boxes2D_one_object, calib_path, file_idx, file, forvis)
    file.close()


    new_lidar = np.zeros(shape=(lidar.shape[0], 8))
    new_lidar[:, 0:4] = lidar[:, 0:4]
    new_lidar[:, 3] *= 10
    new_lidar[:, -1] = 2
    all_points = np.concatenate([new_lidar, new_p1], 0)

    return all_points

def get_lowest_point_rect(ptc, xz_center, l, w, ry):
    ptc_xz = ptc[:, [0, 2]] - xz_center
    rot = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry), np.cos(ry)]
    ])
    ptc_xz = ptc_xz @ rot.T
    mask = (ptc_xz[:, 0] > -l/2) & \
        (ptc_xz[:, 0] < l/2) & \
        (ptc_xz[:, 1] > -w/2) & \
        (ptc_xz[:, 1] < w/2)
    ys = ptc[mask, 1]
    return ys.max()

def GenerateAnns(one_object_class, one_object_points, class_one_object, score_one_object, boxes2D_one_object, calib_path, file_idx, file, forvis):
    thing_classes = ['Car', 'Truck', 'Construction_vehicle', 'Bus', 'Trailer', 'Barrier', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic_cone'] # nuscenes
    if one_object_points.shape[0] != 0:
        # L-shape fitting
        lshapefitting = LShapeFitting()
        rects, idsets = lshapefitting.fitting(one_object_points[:, 0], one_object_points[:, 1])
        boxes2D_one_object_np = boxes2D_one_object.tensor.detach().numpy()[0]
        x_min, y_min, x_max, y_max = boxes2D_one_object_np[0], boxes2D_one_object_np[1], boxes2D_one_object_np[2], boxes2D_one_object_np[3]
        center_type = None
        if rects != None:
            for rect in rects:
                rect_c_x, rect_c_y = calc_rect_contour(rect.a, rect.b, rect.c)
                center_x = (rect_c_x[0] + rect_c_x[2]) / 2
                center_y = (rect_c_y[0] + rect_c_y[2]) / 2
                center_z = one_object_points[:, 2].max() / 2 - 1.5
                height = one_object_points[:, 2].max()
                l1 = np.sqrt((rect_c_x[0] - rect_c_x[3]) ** 2 + (rect_c_y[0] - rect_c_y[3]) ** 2)
                l2 = np.sqrt((rect_c_x[0] - rect_c_x[1]) ** 2 + (rect_c_y[0] - rect_c_y[1]) ** 2)
                if one_object_class == "Car" and (l1 / l2 > 5 or l2 / l1 > 5): continue
                if l1 >= l2:
                    length, width = l1, l2
                    rotation = np.arctan((rect_c_y[3] - rect_c_y[0]) / (rect_c_x[3] - rect_c_x[0] + 1e-8))
                    if center_type == 'prior_centerline': # prior get center of x, y
                        length, width = 3.9, 1.6
                        if l2 < 1.5 and l1 < 2 : length, width = 1.6, 3.9
                        w_x = (rect_c_x[0] + rect_c_x[1]) / 2
                        w_y = (rect_c_y[0] + rect_c_y[1]) / 2
                        center_x = w_x + length / 2 * np.sin(rotation)
                        center_y = w_y + length / 2 * np.cos(rotation)
                elif l1 < l2:
                    length, width = l2, l1
                    rotation = np.arctan((rect_c_y[1] - rect_c_y[0]) / (rect_c_x[1] - rect_c_x[0] + 1e-8))
                    if center_type == 'prior_centerline': # prior get center of x, y
                        length, width = 3.9, 1.6
                        if l1 < 1.5 and l2 < 2: length, width = 1.6, 3.9
                        w_x = (rect_c_x[0] + rect_c_x[3]) / 2
                        w_y = (rect_c_y[0] + rect_c_y[3]) / 2
                        center_x = w_x + width / 2 * np.sin(rotation)
                        center_y = w_y + width / 2 * np.cos(rotation)
                rotation = -rotation - np.pi / 2
                theta = np.arctan((-center_x / (center_y + 1e-8)))
                alpha = rotation - theta

                # lidar to camera # kitti
                calib_file = os.path.join(calib_path, file_idx + '.txt')
                calib = calibration_kitti.Calibration(calib_file)
                [[center_x, center_y, center_z]] = calib.lidar_to_rect(np.array([[center_x, center_y, center_z]]))
                file.write(thing_classes[ class_one_object] + ' ' + str(0) + ' ' + str(0) + ' ' + str(alpha) + ' ' + str(x_min) +  ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + str(height) + ' ' + str(width) + ' ' + str(length) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(center_z) + ' ' + str(rotation) + '\n' )

                # save for vir
                forvis.append([center_x, center_y, center_z, length, width, height, rotation])
    return forvis

def calc_rect_contour(a, b, c):
    rect_c_x = [None] * 5
    rect_c_y = [None] * 5
    rect_c_x[0], rect_c_y[0] = calc_cross_point(
        a[0:2], b[0:2], c[0:2])
    rect_c_x[1], rect_c_y[1] = calc_cross_point(
        a[1:3], b[1:3], c[1:3])
    rect_c_x[2], rect_c_y[2] = calc_cross_point(
        a[2:4], b[2:4], c[2:4])
    rect_c_x[3], rect_c_y[3] = calc_cross_point(
        [a[3], a[0]], [b[3], b[0]], [c[3], c[0]])
    return rect_c_x, rect_c_y

def calc_cross_point(a, b, c):
    x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
    y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
    return x, y

class MyLoader():
    def __init__(self, root_path=''):
        self.root_path = root_path
        self.file_list = self.include_all_files()

    def include_all_files(self):
        velo_path = os.path.join(self.root_path, 'velodyne')
        all_files = os.listdir(velo_path)
        all_files.sort()

        all_files = [x[0:6] for x in all_files]

        return all_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file_idx = self.file_list[item]
        file_image_path = os.path.join(self.root_path, 'image_2', file_idx+'.png')
        file_velo_path = os.path.join(self.root_path, 'velodyne', file_idx+'.bin')
        file_calib = os.path.join(self.root_path, 'calib', file_idx+'.txt')

        calib = calibration_kitti.Calibration(file_calib)
        points = np.fromfile(str(file_velo_path), dtype=np.float32).reshape(-1, 4)
        image = np.array(io.imread(file_image_path), dtype=np.int32)

        rgb, depth = load_depth_input(calib, image, points)

        return rgb, depth