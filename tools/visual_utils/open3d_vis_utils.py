"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
from scipy.spatial import Delaunay

# colormap = [
#     [1, 1, 1],
#     [0, 1, 0],
#     [0, 1, 1],
#     [1, 1, 0],
# ]

box_colormap = [
    # [1, 1, 1],
    # [0, 0.6, 1], # 橙
    # [0, 0, 1], # 红
    # [0.8, 0, 0.8], # 紫
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],

]

box_colormap_gt = [
    # [1, 1, 1],
    # [0, 0.6, 1], # 橙
    # [0, 0, 1], # 红
    # [0.8, 0, 0.8], # 紫
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],
    [1, 0, 0.8],

]

# colormap = np.array([[128, 130, 120], 
#                      [235, 0, 205], 
#                      [0, 215, 0], 
#                      [235, 155, 0]]) / 255.0 

colormap = np.random.randint(0, 255, (200, 3))/255

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
 
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
 
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
 
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
 
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
 
    return corners3d.numpy() if is_numpy else corners3d

def draw_point_in_box3d(vis, points, boxes3d, labels):
 
    corner3ds = boxes_to_corners_3d(boxes3d) # [N,8,3]
    pc_in_boxes_sum = np.zeros((1,4))
    # colors = []
    for i in range(corner3ds.shape[0]):
        flag = in_hull(points[:,0:3], corner3ds[i])
        pc_in_boxes = points[flag]
        pc_in_boxes_sum = np.vstack((pc_in_boxes_sum, pc_in_boxes))
        # for j in range(pc_in_boxes_sum.shape[0]):
            # if labels[i] == 1:
            #     colors.append([0., 0.6, 1.0])
            # elif labels[i] == 2:
            #     colors.append([0., 0., 1.])
            # elif labels[i] == 3:
            #     colors.append([0.8, 0., 0.8])
 
    points_in_boxes = pc_in_boxes_sum
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points_in_boxes[:,:3])
    vis.add_geometry(pts)
    
    colors = [[1., 0., 0.] for _ in range(points_in_boxes.shape[0])]
    # print(len(colors))
    pts.colors = open3d.utility.Vector3dVector(np.asarray(colors))
    return vis
 
def in_hull(p, hull):
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)
    return flag
 

def gradient_point_cloud_color_map(points):
    # 根据距离生成色彩
    colors = np.zeros([points.shape[0], 3])
    # 使用x,y计算到中心点的距离
    dist = np.sqrt(np.square(points[:,0]) + np.square(points[:,1]))
    
    dist_max = np.max(dist)
    print(f"dist_max: {dist_max}")
    # 调整渐变半径
    dist = dist / 70	# 我这里的半径是51.2m，
    # dist = dist / 2
    
    # RGB
    min = [127,0,255]   # 紫色
    max = [255,255,0]   # 黄色
    
    # 最近处为紫色
    colors[:,0] = 127
    colors[:,2] = 255
    
    # 减R(127 -> 0),加G(0->255),再减B(255->0)，再加R(0 -> 255)
    # 127+255+255+255
    all_color_value = 127+255+255+255
    dist_color = dist * all_color_value
    
    # 减R (127 -> 0)
    clr_1 = 127
    dy_r = 127-dist_color
    tmp = np.zeros([colors[dist_color<clr_1].shape[0], 3])
    tmp[:, 0] = dy_r[dist_color<clr_1]
    tmp[:, 1] = 0
    tmp[:, 2] = 255
    colors[dist_color<clr_1] = tmp
    
    # 加G (0->255)
    clr_2 = 127+255
    dy_g = dist_color-clr_1
    tmp = np.zeros([colors[(dist_color>=clr_1) & (dist_color<clr_2)].shape[0], 3])
    tmp[:, 0] = 0
    tmp[:, 1] = dy_g[(dist_color>=clr_1) & (dist_color<clr_2)]
    tmp[:, 2] = 255
    colors[(dist_color>=clr_1) & (dist_color<clr_2)] = tmp
    
    # 减B (255->0)
    clr_3 = 127+255+255
    dy_b = dist_color-clr_2
    tmp = np.zeros([colors[(dist_color>=clr_2) & (dist_color<clr_3)].shape[0], 3])
    tmp[:, 0] = 0
    tmp[:, 1] = 255
    tmp[:, 2] = dy_b[(dist_color>=clr_2) & (dist_color<clr_3)]
    colors[(dist_color>=clr_2) & (dist_color<clr_3)] = tmp
    
    # 加R(0 -> 255)
    clr_4 = 127+255+255+255
    dy_r = dist_color-clr_3
    tmp = np.zeros([colors[(dist_color>=clr_3) & (dist_color<clr_4)].shape[0], 3])
    tmp[:, 0] = dy_r[(dist_color>=clr_3) & (dist_color<clr_4)]
    tmp[:, 1] = 255
    tmp[:, 2] = 0
    colors[(dist_color>=clr_3) & (dist_color<clr_4)] = tmp
    
    '''
    '''
    # 外围都为黄色
    tmp = np.zeros([colors[dist_color>clr_4].shape[0], 3])
    tmp[:, 0] = 255
    tmp[:, 1] = 255
    tmp[:, 2] = 0
    colors[dist_color>clr_4] = tmp
    
    points = np.concatenate((points[:,:3], colors),axis=1)

    return points


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, images=None, num=None, view_type=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    print(ref_scores, ref_scores.shape)
        
    vis = open3d.visualization.Visualizer()

    vis.create_window(window_name="pcd", width=1920, height=1061)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = 255 * np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    # # intensity 
    # points_intensity = points[:, 3]  
    # point_colors = [colormap[int(points_intensity[i]) % colormap.shape[0]] for i in range(points_intensity.shape[0])]

    # pointcatcolor = gradient_point_cloud_color_map(points)
    # point_colors = pointcatcolor[:, 3:6].tolist()

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)


    # if gt_boxes is not None:
    #     vis = draw_box(vis, gt_boxes, (0, 0, 1))
    vis, line_set_list = draw_box_gt(vis, gt_boxes, (1, 1, 1), ref_labels, ref_scores)

    if ref_boxes is not None:
        vis, line_set_list = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

        vis = draw_point_in_box3d(vis, points, ref_boxes, ref_labels)

    # camera = open3d.camera.PinholeCameraParameters()
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(camera,True)

    # ctl = vis.get_view_control()
    # # ctl.set_zoom(0.8)
    # ctl.rotate(0, 90, 0)

    # line_set_list.append(pts)
    # open3d.visualization.draw_geometries(line_set_list, window_name="world frame")

    ctl = vis.get_view_control()

    # 只有这个是BEV行驶视角
    if view_type == 'BEV':
        ctl.set_up([0.01, 0.0, 0.0])
        ctl.set_zoom(0.6)

    # 行驶正视角
    if view_type == 'Front':
        ctl.set_front([-0.05, 0.0, 0.02])
        ctl.set_up([0.01, 0.0, 0.0]) 
        # ctl.set_lookat([0.00, 0.01, 0.0])
        ctl.set_zoom(0.1)

    # # 行驶正视角
    # if view_type == 'Front':
    #     ctl.set_front([0.0, -0.01, 0.01])
    #     ctl.set_zoom(0.1)
    #     # ctl.set_up([0.01, 0.0, 0.0]) 

    # ctr = vis.get_view_control()
    # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    # ctr.change_field_of_view(step=2)
    # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())

    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # open3d.io.write_pinhole_camera_parameters("/opt/data/private/codeN/OpenPCDet/output/video_vis/param.json", param)
    # print(param)

    # param = open3d.io.read_pinhole_camera_parameters("/opt/data/private/codeN/OpenPCDet/output/video_vis/param.json")
    # ctr = vis.get_view_control()
    # # 转换视角
    # ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(True)
    img_np = (np.array(img) * 255).astype(np.uint8)
    # print(img_np.shape)
    if view_type == 'Front':
        cv2.imwrite("/opt/data/private/codeN/OpenPCDet/output/video_vis/demo_save_img/Front_" + str(num) + ".jpg", img_np)
    if view_type == 'BEV':
        cv2.imwrite("/opt/data/private/codeN/OpenPCDet/output/video_vis/demo_save_img/BEV_" + str(num) + ".jpg", img_np)

    images.append(img_np)
    # time.sleep(1/6) # Set frame Time
    vis.run()
    vis.destroy_window()

    return vis, images


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    line_set_list= []
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            # print(ref_labels.shape, i, 111111111)
            # print(ref_labels[i])
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        line_set_list.append(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])

    return vis, line_set_list


def draw_box_gt(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    line_set_list= []
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            # print(ref_labels.shape, i, 111111111)
            # print(ref_labels[i])
            line_set.paint_uniform_color(box_colormap_gt[ref_labels[i]])

        vis.add_geometry(line_set)
        line_set_list.append(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])

    return vis, line_set_list
