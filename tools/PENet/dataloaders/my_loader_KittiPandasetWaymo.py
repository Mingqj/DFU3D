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

def estimate_plane(origin_ptc, max_hs=-1.5, it=1, ptc_range=((-30, 80), (-30, 80))):
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

def above_plane(ptc, plane, offset=0.05, only_range=((-30, 80), (-30, 80))):
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


import matplotlib.pyplot as plt
from PIL import Image

def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    # x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    # y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 100               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    print(x_max/dpi, y_max/dpi)
    # fig, ax = plt.subplots(figsize=(1216, 352), dpi=dpi)
    print(x_img.shape, y_img.shape)
    ax.scatter(x_img, y_img, s=1, c=pixel_values, linewidths=0, alpha=1)
    fig.savefig("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth.png", dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    # ax.set_axis_bgcolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        fig.show()

def plot_to_matrix(img, x, y, depth, points, file_idx):
    # x_lidar = points[:, 0]
    # y_lidar = points[:, 1]
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # depth = -d_lidar

    dpi= 100
    width= img.shape[1] / dpi
    height= img.shape[0] / dpi
    fig_convert, ax = plt.subplots()

    # ax.scatter(x, y, s=1, c=depth, linewidths=0, alpha=1, cmap="jet")
    # axes_convert = fig_convert.add_axes([0.16, 0.15, 0.75, 0.75])
    # axes_convert.cla()
    # axes_convert.plot(x, y)

    # fig_convert.canvas.draw()
    # fig_str = fig_convert.canvas.tostring_rgb()
    # fig_convert.savefig("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth.png", dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    # data = np.frombuffer(fig_str, dtype=np.uint8).reshape((int(height * dpi), -1, 3))

    ax.scatter(x, y, s=1, linewidths=0, alpha=1)
    fig_convert.savefig(os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/visual/sparse_depth_map/", file_idx + '.png'), dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    data = cv2.imread(os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/visual/sparse_depth_map/", file_idx + '.png'))
    return data

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

# import pandaset
# import pandaset as ps

def depth2pointsrgbpm(depth, image, image_ori, calib, lidar, thing_classes, mask_image, classes, scores, boxes2D, lidar_depth, file_idx):
    # print(depth.shape, lidar_depth.shape, image.shape, lidar.shape, mask_image.shape, classes)
    root_path = "/opt/data/private/codeN/VirConv/tools/PENet/visual/label_train/"
    calib_path = "/opt/data/private/codeN/VirConv/data/kitti/training/calib/" # kitti
    # calib_path = "/opt/data/private/codeN/OpenPCDet/data/pandaset_kitti/training/calib/" # pandaset

    # # completionformer
    # depth_2_path = "/opt/data/private/codeN/CompletionFormer/results/depth_npy/"
    # depth = np.load(os.path.join(depth_2_path, str(int(file_idx)) + '.npy'))[:352, :1216, :]
    # depth[depth<0.001] = 0
    # uv = depth.nonzero()
    # depth_val = depth[depth>0]

    # PENet
    depth[depth<0.001] = 0
    uv = depth.nonzero()
    depth_val = depth[depth>0]

    # # ground truth depth map
    # lidar_depth = np.expand_dims(lidar_depth, 2)
    # lidar_depth[lidar_depth < 0.1] = 0
    # uv = lidar_depth.nonzero()
    # depth_val = lidar_depth[lidar_depth > 0]
    # print(depth_val.shape)

    # generate above plane
    plane = estimate_plane(lidar[:, :3])
    plane_mask = above_plane(lidar[:, :3], plane)
    lidar_mask = lidar[plane_mask]

    # # lidar2depthmap
    # pts_img, pts_depth = calib.lidar_to_img(lidar_mask[:, :3])
    # img_sparse_depth = plot_to_matrix(image_ori, pts_img[:, 0], pts_img[:, 1], pts_depth, lidar_mask, file_idx)
    # img_sparse_depth = img_sparse_depth[:352, :1216]

    # # cv2.imwrite(os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/visual/sparse_depth_map/", file_idx + '.jpg') , img_sparse_depth)
    # sparse_depth_path = "/opt/data/private/codeN/VirConv/tools/PENet/visual/sparse_depth_map/"
    # # cm = plt.cm.get_cmap('jet')
    # plt.imshow(img_sparse_depth)
    # plt.savefig(os.path.join(sparse_depth_path, file_idx + '.jpg'))
    # plt.clf()
    # # # depth = img_sparse_depth
    # # img_sparse_depth[img_sparse_depth<0.1] = 0
    # # uv = img_sparse_depth.nonzero()
    # # depth_val = img_sparse_depth[img_sparse_depth>0]
    # # print(depth_val.shape, 222)

    # lidar index to img
    pts_img, pts_depth = calib.lidar_to_img(lidar_mask[:, :3]) # [:, 2], [:, 1] # kitti

    # dataset = pandaset.DataSet("/data/PandaSet")
    # seq002 = dataset["002"]
    # seq_idx = 1
    # camera_name = "front_camera"
    # seq002.load()
    # lidar = seq002.lidar
    # points3d_lidar_xyz = lidar.data[seq_idx].to_numpy()[:, :3]
    # choosen_camera = seq002.camera[camera_name]
    # projected_points2d, camera_points_3d, inner_indices = ps.geometry.projection(lidar_points=points3d_lidar_xyz, camera_data=choosen_camera[seq_idx], camera_pose=choosen_camera.poses[seq_idx], camera_intrinsics=choosen_camera.intrinsics, filter_outliers=True)

    num_pts_img = pts_img.shape[0]
    pts_img = np.round(pts_img).tolist()
    lidar_object_mask_list = []
    for j in range(mask_image.shape[0]):
        mask_one_object = mask_image[j]
        # mask_one_object = mask_one_object.unsqueeze(2).expand(mask_image[j].shape[0], mask_image[j].shape[1], 3).detach().cpu().numpy()
        mask_one_object = mask_one_object.detach().cpu().numpy()
        mask_one_object1 = Image.fromarray(np.uint8(mask_one_object))
        # mask_list = [mask_one_object1.getpixel(tuple(xy)) for xy in pts_img]
        mask_list = [mask_one_object1.getpixel(tuple(xy)) for xy in pts_img if 0 <= xy[0] < 1216 and 0 <= xy[1] < 352]
        lidar_mask = lidar_mask[:len(mask_list)]
        lidar_object_mask_list.append(lidar_mask[np.array(mask_list) > 0][:, :3])

    new_p = np.zeros(shape=(uv[0].shape[0], 8))
    # print('new_p', new_p.shape)

    # kitti image2lidar
    p_rect = calib.img_to_rect(uv[1], uv[0], depth_val)
    p_lidar = calib.rect_to_lidar(p_rect)

    new_p[:, 0:3] = p_lidar
    new_p[:, 4:7] = image[uv[0], uv[1]]/3
    new_p0 = new_p[new_p[:, 2] < 1.]
    new_p1 = la_sampling2(new_p0)
    new_p1[:, -1] = 1

    # # save depth map
    # pcd_depth = o3d.geometry.PointCloud()
    # pcd_depth.points = o3d.utility.Vector3dVector(new_p1[:, 0:3])
    # pcd_lidar = o3d.geometry.PointCloud()
    # pcd_lidar.points = o3d.utility.Vector3dVector(lidar[:, 0:3])
    # pcd_above_plane = o3d.geometry.PointCloud()
    # pcd_above_plane.points = o3d.utility.Vector3dVector(lidar_mask[:, 0:3])
    # o3d.io.write_point_cloud(os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/", file_idx + '.ply'), pcd_depth)
    # o3d.io.write_point_cloud(os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/", 'Lidar' + file_idx + '.ply'), pcd_lidar)
    # o3d.io.write_point_cloud(os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/", 'Lidar_above_plane' + file_idx + '.ply'), pcd_above_plane)
    
    # select objects
    # if mask_image.shape[0] < 50:
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
            # uni_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
            # cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.3)
            cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=20, radius=3)
            one_object_points = np.array(cl.points)
        elif point_type == 'point_cloud': # point cloud 
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(one_object_points)
            if one_object_points.shape[0] > 50:
                cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=2)
            elif one_object_points.shape[0] <= 50:
                cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=3)
            one_object_points = np.array(cl.points)
        elif point_type == 'multi-modal':
            # point cloud
            if one_object_class == "Car":
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(one_object_point_cloud)
                if one_object_point_cloud.shape[0] > 50:
                    cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=2)
                elif one_object_point_cloud.shape[0] <= 50:
                    cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=3)
                one_object_point_cloud = torch.from_numpy(np.array(cl.points))
                # pseudo point
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(one_object_pseudo_point)
                if one_object_pseudo_point.shape[0] > 500:
                    uni_down_pcd = pcd1.uniform_down_sample(every_k_points=8)
                    cl1, ind = uni_down_pcd.remove_radius_outlier(nb_points=20, radius=3)
                elif one_object_pseudo_point.shape[0] <= 500 and one_object_pseudo_point.shape[0] > 100:
                    uni_down_pcd = pcd1.uniform_down_sample(every_k_points=4)
                    cl1, ind = uni_down_pcd.remove_radius_outlier(nb_points=20, radius=3)
                elif one_object_pseudo_point.shape[0] <= 100:
                    cl1, ind = pcd1.remove_radius_outlier(nb_points=20, radius=3)
                one_object_pseudo_point = torch.from_numpy(np.array(cl1.points))
            elif one_object_class == "Pedestrian" or one_object_class == "Person ride a bike": # kitti
            # elif one_object_class == "Pedestrian" or one_object_class == "Bicycle": # pandaset
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(one_object_point_cloud)
                if one_object_point_cloud.shape[0] > 50:
                    uni_down_pcd = pcd.uniform_down_sample(every_k_points=2)
                    cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=10, radius=0.3)
                elif one_object_point_cloud.shape[0] <= 50:
                    uni_down_pcd = pcd.uniform_down_sample(every_k_points=2)
                    cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=10, radius=0.6)
                one_object_point_cloud = torch.from_numpy(np.array(cl.points))
                # pseudo point
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(one_object_pseudo_point)
                if one_object_pseudo_point.shape[0] > 500:
                    uni_down_pcd = pcd1.uniform_down_sample(every_k_points=16)
                    cl1, ind = uni_down_pcd.remove_radius_outlier(nb_points=10, radius=0.6)
                elif one_object_pseudo_point.shape[0] <= 500 and one_object_pseudo_point.shape[0] > 100:
                    uni_down_pcd = pcd1.uniform_down_sample(every_k_points=8)
                    cl1, ind = uni_down_pcd.remove_radius_outlier(nb_points=10, radius=0.6)
                elif one_object_pseudo_point.shape[0] <= 100:
                    uni_down_pcd = pcd1.uniform_down_sample(every_k_points=4)
                    cl1, ind = uni_down_pcd.remove_radius_outlier(nb_points=10, radius=0.6)
                one_object_pseudo_point = torch.from_numpy(np.array(cl1.points))
            # fuse
            if one_object_point_cloud.shape[0] > 0 and one_object_pseudo_point.shape[0] > 0:
                _, _, _, pseudo_point_idx = BallQuery(one_object_pseudo_point, one_object_point_cloud, 0.1)
                one_object_pseudo_point = one_object_pseudo_point[pseudo_point_idx]
            one_object_points = torch.cat((one_object_point_cloud, one_object_pseudo_point), 0).numpy()


        # forvis = GenerateAnns(one_object_class, one_object_points, class_one_object, score_one_object, boxes2D_one_object, calib_path, file_idx, file, forvis)

        # visual
        # o3d.io.write_point_cloud("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/" + file_idx + "_" + str(i) + "after.ply", cl)
        # o3d.io.write_point_cloud("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/" + file_idx + "_" + str(i) + "before.ply", pcd)\

        
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(one_object_pseudo_point.numpy())
        # o3d.io.write_point_cloud("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth_2025/" + file_idx + "_" + str(i) + "pseudo.ply", pcd1)
        # o3d.io.write_point_cloud("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth_2025/" + file_idx + "_" + str(i) + "point_raw.ply", cl)

        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(one_object_points)
        # o3d.io.write_point_cloud("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/" + file_idx + "_" + str(i) + "fusion.ply", pcd1)

        # one_remove_center = remove_center(one_object_points)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(one_remove_center)
        # o3d.io.write_point_cloud("/opt/data/private/codeN/VirConv/tools/PENet/visual/depth/" + file_idx + "_" + str(i) + "remove.ply", pcd2)

    file.close()
    # np.save(os.path.join(root_path, file_idx + '.npy'), forvis)

    new_lidar = np.zeros(shape=(lidar.shape[0], 8))
    new_lidar[:, 0:4] = lidar[:, 0:4]
    new_lidar[:, 3] *= 10
    new_lidar[:, -1] = 2

    #new_p = new_p[new_p[:, 2]<1.]
    #_, new_p = to_sphere_coords(new_p)
    #new_p = voxel_sampling(new_p)
    #new_p = range_sampling_torch(new_p, new_lidar, calib)

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
    thing_classes = ["Car", "Pedestrian", "Cyclist"] # kitti
    # thing_classes = ["Car", "Pedestrian", "Bicycle"] # pandaset
    if one_object_points.shape[0] != 0:
        # L-shape fitting
        lshapefitting = LShapeFitting()
        rects, idsets = lshapefitting.fitting(one_object_points[:, 0], one_object_points[:, 1])
        # print(one_object_class, one_object_points, class_one_object, score_one_object, boxes2D_one_object)
        boxes2D_one_object_np = boxes2D_one_object.tensor.detach().numpy()[0]
        x_min, y_min, x_max, y_max = boxes2D_one_object_np[0], boxes2D_one_object_np[1], boxes2D_one_object_np[2], boxes2D_one_object_np[3]
        # center_type = 'prior_centerline'
        # center_type = 'ori'
        # center_type = 'prior_linecorner'
        center_type = None
        # center_type = 'prior_center'
        if rects != None:
            for rect in rects:
                rect_c_x, rect_c_y = calc_rect_contour(rect.a, rect.b, rect.c)
                
                center_x = (rect_c_x[0] + rect_c_x[2]) / 2
                center_y = (rect_c_y[0] + rect_c_y[2]) / 2

                center_z = one_object_points[:, 2].max() / 2 - 1.5
                # center_z = one_object_points[:, 2].max() / 2 - 1.56 # 1.3 bad: 39.1, 0.18, 1.4bad, 1.56bad
                height = one_object_points[:, 2].max()
                if one_object_class == "Car":
                    if height < 1.0: height = 1.56
                elif one_object_class == "Pedestrian":
                    if height < 1.0: height = 1.73
                elif one_object_class == "Person ride a bike": # kitti
                # elif one_object_class == "Bicycle": # pandaset
                    if height < 1.0: height = 1.73
                # if height < 1.45: height = 1.45

                l1 = np.sqrt((rect_c_x[0] - rect_c_x[3]) ** 2 + (rect_c_y[0] - rect_c_y[3]) ** 2)
                l2 = np.sqrt((rect_c_x[0] - rect_c_x[1]) ** 2 + (rect_c_y[0] - rect_c_y[1]) ** 2)
                if one_object_class == "Car" and l1 / l2 > 5 or l2 / l1 > 5: continue
                if l1 >= l2:
                    length, width = l1, l2
                    if center_type == 'ori' and one_object_class == "Car": length, width = 3.9, 1.6
                    elif center_type == 'ori' and one_object_class == "Pedestrian": length, width = 0.8, 0.6
                    elif center_type == 'ori' and one_object_class == "Person ride a bike": length, width = 1.76, 0.6 # kitti
                    # elif center_type == 'ori' and one_object_class == "Bicycle": length, width = 1.76, 0.6 # pandaset

                    rotation = np.arctan((rect_c_y[3] - rect_c_y[0]) / (rect_c_x[3] - rect_c_x[0] + 1e-8))
                    if center_type == 'prior_centerline': # prior get center of x, y
                        length, width = 3.9, 1.6
                        if l2 < 1.5 and l1 < 2 : length, width = 1.6, 3.9
                        w_x = (rect_c_x[0] + rect_c_x[1]) / 2
                        w_y = (rect_c_y[0] + rect_c_y[1]) / 2
                        center_x = w_x + length / 2 * np.sin(rotation)
                        center_y = w_y + length / 2 * np.cos(rotation)
                    elif center_type == 'prior_linecorner':
                        if one_object_class == "Car" and l2 < 1.5 and l1 < 2:  
                            rotation = np.arctan((rect_c_y[1] - rect_c_y[0]) / (rect_c_x[1] - rect_c_x[0] + 1e-8))
                            if rotation < 0.7854:
                                center_x = (3.9 - length) / 2 * np.cos(rotation) + center_x
                                center_y = (3.9 - length) / 2 * np.sin(rotation) + center_y
                                length, width = 3.9, 1.6
                            elif rotation >= 0.7854:
                                center_x = (3.9 - length) / 2 * np.cos(rotation) + center_x
                                center_y = (3.9 - length) / 2 * np.sin(rotation) + center_y
                                length, width = 3.9, 1.6
                    elif center_type == 'prior_center':
                        if l2 < 1.5 and l1 < 2:  
                            rotation = np.arctan((rect_c_y[1] - rect_c_y[0]) / (rect_c_x[1] - rect_c_x[0] + 1e-8))
                            if rotation < 0.7854:
                                theta = np.arctan((1.6 - width) / (3.9 - length) + 1e-8)
                                delta = theta + rotation
                                TwoCenterDis = np.sqrt(((3.9 - length) / 2) ** 2 + ((1.6 - width) / 2) ** 2)
                                center_x = TwoCenterDis * np.cos(delta) + center_x
                                center_y = TwoCenterDis * np.sin(delta) + center_y
                                length, width = 3.9, 1.6
                            elif rotation >= 0.7854:
                                theta = np.arctan((1.6 - width) / (3.9 - length) + 1e-8)
                                delta = theta + rotation
                                TwoCenterDis = np.sqrt(((3.9 - length) / 2) ** 2 + ((1.6 - width) / 2) ** 2)
                                center_x = TwoCenterDis * np.cos(delta) + center_x
                                center_y = TwoCenterDis * np.sin(delta) + center_y
                                length, width = 3.9, 1.6
                    
                    if one_object_class == "Car": length, width = 3.9, 1.6
                    elif one_object_class == "Pedestrian": length, width = 0.8, 0.6
                    elif one_object_class == "Person ride a bike": length, width = 1.76, 0.6 # kitti
                    # elif one_object_class == "Bicycle": length, width = 1.76, 0.6 # pandaset

                elif l1 < l2:
                    length, width = l2, l1
                    if center_type == 'ori' and one_object_class == "Car": length, width = 3.9, 1.6
                    elif center_type == 'ori' and one_object_class == "Pedestrian": length, width = 0.8, 0.6
                    elif center_type == 'ori' and one_object_class == "Person ride a bike": length, width = 1.76, 0.6 # kitti
                    # elif center_type == 'ori' and one_object_class == "Bicycle": length, width = 1.76, 0.6 # pandaset

                    rotation = np.arctan((rect_c_y[1] - rect_c_y[0]) / (rect_c_x[1] - rect_c_x[0] + 1e-8))
                    if center_type == 'prior_centerline': # prior get center of x, y
                        length, width = 3.9, 1.6
                        if l1 < 1.5 and l2 < 2: length, width = 1.6, 3.9
                        w_x = (rect_c_x[0] + rect_c_x[3]) / 2
                        w_y = (rect_c_y[0] + rect_c_y[3]) / 2
                        center_x = w_x + width / 2 * np.sin(rotation)
                        center_y = w_y + width / 2 * np.cos(rotation)
                    elif center_type == 'prior_linecorner':
                        if one_object_class == "Car" and l1 < 1.5 and l2 < 2:  
                            rotation = np.arctan((rect_c_y[3] - rect_c_y[0]) / (rect_c_x[3] - rect_c_x[0] + 1e-8))
                            if rotation < 0.7854:
                                center_x = (3.9 - length) / 2 * np.cos(rotation) + center_x
                                center_y = (3.9 - length) / 2 * np.sin(rotation) + center_y
                                length, width = 3.9, 1.6
                            elif rotation >= 0.7854:
                                center_x = (3.9 - length) / 2 * np.cos(rotation) + center_x
                                center_y = (3.9 - length) / 2 * np.sin(rotation) + center_y
                                length, width = 3.9, 1.6
                    elif center_type == 'prior_center':
                        if l1 < 1.5 and l2 < 2:  
                            rotation = np.arctan((rect_c_y[3] - rect_c_y[0]) / (rect_c_x[3] - rect_c_x[0] + 1e-8))
                            if rotation < 0.7854:
                                theta = np.arctan((1.6 - width) / (3.9 - length) + 1e-8)
                                delta = theta + rotation
                                TwoCenterDis = np.sqrt(((3.9 - length) / 2) ** 2 + ((1.6 - width) / 2) ** 2)
                                center_x = TwoCenterDis * np.cos(delta) + center_x
                                center_y = TwoCenterDis * np.sin(delta) + center_y
                                length, width = 3.9, 1.6
                            elif rotation >= 0.7854:
                                theta = np.arctan((1.6 - width) / (3.9 - length) + 1e-8)
                                delta = theta + rotation
                                TwoCenterDis = np.sqrt(((3.9 - length) / 2) ** 2 + ((1.6 - width) / 2) ** 2)
                                center_x = TwoCenterDis * np.cos(delta) + center_x
                                center_y = TwoCenterDis * np.sin(delta) + center_y
                                length, width = 3.9, 1.6
                    
                    if one_object_class == "Car": length, width = 3.9, 1.6
                    elif one_object_class == "Pedestrian": length, width = 0.8, 0.6
                    elif one_object_class == "Person ride a bike": length, width = 1.76, 0.6 # kitti
                    # elif one_object_class == "Bicycle": length, width = 1.76, 0.6

                # if np.sqrt(center_x ** 2 + center_y ** 2) < 10 or np.sqrt(center_x ** 2 + center_y ** 2) > 80: continue
                if one_object_class == "Car" and np.sqrt(center_x ** 2 + center_y ** 2) < 10: continue
                # if y_max - y_min < 20: continue

                rotation = -rotation - np.pi / 2
                theta = np.arctan((-center_x / (center_y + 1e-8)))
                alpha = rotation - theta

                # bottom = get_lowest_point_rect(one_object_points, np.array([center_x, center_y]), length, width, rotation)
                # height = bottom - one_object_points[:, 1].min()

                # lidar to camera # kitti
                calib_file = os.path.join(calib_path, file_idx + '.txt')
                calib = calibration_kitti.Calibration(calib_file)
                [[center_x, center_y, center_z]] = calib.lidar_to_rect(np.array([[center_x, center_y, center_z]]))
                file.write(thing_classes[ class_one_object] + ' ' + str(0) + ' ' + str(0) + ' ' + str(alpha) + ' ' + str(x_min) +  ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + str(height) + ' ' + str(width) + ' ' + str(length) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(center_z) + ' ' + str(rotation) + '\n' )

                # save for vir
                # forvis.append([height, width, length, center_x, center_y, center_z, rotation])
                forvis.append([center_x, center_y, center_z, length, width, height, rotation])
        # elif rects == None:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(one_object_points)
        #     aabb = pcd.get_axis_aligned_bounding_box()
        #     [center_x, center_y, center_z] = aabb.get_center()
        #     [l1, l2, height] = aabb.get_extent()
        #     if l1 >= l2:
        #         length, width = l1, l2
        #     elif l1 < l2:
        #         length, width = l2, l1
        #     rotation = 1.57
        #     alpha = 1.57
            
        #     if (length / width > 1.0 and length / width < 7):
        #         file.write(one_objecy_class + ' ' + str(0) + ' ' + str(0) + ' ' + str(alpha) + ' ' + str(x_min) +  ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + str(height) + ' ' + str(width) + ' ' + str(length) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(center_z) + ' ' + str(rotation) + '\n' )

        #         # save for vir
        #         # forvis.append([height, width, length, center_x, center_y, center_z, rotation])
        #         forvis.append([center_x, center_y, center_z, length, width, height, rotation])
        
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
    # rect_c_x[4], rect_c_y[4] = rect_c_x[0], rect_c_y[0]
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
        image = image[:352, :1216]

        rgb, depth = load_depth_input(calib, image, points)

        return rgb, depth