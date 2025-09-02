import os

# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from dataloaders import calibration_kitti
from skimage import io
import cv2

# cmap = plt.cm.jet
# cmap2 = plt.cm.nipy_spectral

from dataloaders.my_loader import depth2pointsrgb, depth2pointsrgbp, depth2pointsrgbpm
# from dataloaders.my_loader_KittiPandasetWaymo import depth2pointsrgb, depth2pointsrgbp, depth2pointsrgbpm

def validcrop(img):
    ratio = 256/1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h-int(ratio*w):, :]

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype('uint8')

def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype('uint8')

def merge_into_row(ele, pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][0, ...]))
        img_list.append(preprocess_depth(pred[0, ...]))
    if extrargb is not None:
        img_list.append(preprocess_depth(extrargb[0, ...]))
    if predrgb is not None:
        predrgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        predrgb = np.transpose(predrgb, (1, 2, 0))
        #predrgb = predrgb.astype('uint8')
        img_list.append(predrgb)
    if predg is not None:
        predg = np.squeeze(predg[0, ...].data.cpu().numpy())
        predg = mask_vis(predg)
        predg = np.array(Image.fromarray(predg).convert('RGB'))
        #predg = predg.astype('uint8')
        img_list.append(predg)
    if extra is not None:
        extra = np.squeeze(extra[0, ...].data.cpu().numpy())
        extra = mask_vis(extra)
        extra = np.array(Image.fromarray(extra).convert('RGB'))
        img_list.append(extra)
    if extra2 is not None:
        extra2 = np.squeeze(extra2[0, ...].data.cpu().numpy())
        extra2 = mask_vis(extra2)
        extra2 = np.array(Image.fromarray(extra2).convert('RGB'))
        img_list.append(extra2)
    if 'gt' in ele:
        img_list.append(preprocess_depth(ele['gt'][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_image_torch(rgb, filename):
    #torch2numpy
    rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    #print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype('uint8')
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)

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

from XDecoder.xdecoder.BaseModel import BaseModel
from XDecoder.xdecoder import build_model
from XDecoder.utils.arguments import load_opt_command
from XDecoder.utils.distributed import init_distributed
import torch
from torchvision import transforms
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from detectron2.structures import BitMasks
import time

def save_depth_as_points(depth, idx, root_path):
    file_idx = str(idx).zfill(6)
    file_image_path = os.path.join(root_path, 'image_2', file_idx + '.png')
    file_velo_path = os.path.join(root_path, 'velodyne', file_idx + '.bin')
    file_calib = os.path.join(root_path, 'calib', file_idx + '.txt')

    calib = calibration_kitti.Calibration(file_calib)

    lidar = np.fromfile(str(file_velo_path), dtype=np.float32).reshape(-1, 4) # [point_number, 4]

    image1 = np.array(io.imread(file_image_path), dtype=np.int32)
    image = image1[:352, :1216]
    lidar_depth = None

    thing_classes, masks_img_keep, classes_keep, scores_keep, boxes2D_keep = SegSeem(file_image_path)

    pts_rect = calib.lidar_to_rect(lidar[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, image.shape, calib)
    lidar = lidar[fov_flag]

    paths = os.path.join(root_path, 'velodyne_depth')
    if not os.path.exists(paths):
        os.makedirs(paths)

    out_path = os.path.join(paths, file_idx + '.npy')
    depth = depth.cpu().detach().numpy().reshape(352, 1216, 1)[:352, :1216, :]

    final_points = depth2pointsrgbpm(depth, image, image1, calib, lidar, thing_classes, masks_img_keep, classes_keep, scores_keep, boxes2D_keep, lidar_depth, file_idx)
    final_points = final_points.astype(np.float16) # [vri_points_number, 8]
    # print('\n')
    np.save(out_path, final_points)


def SegSeem(file_image_path):
    opt, cmdline_args = load_opt_command(None)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)
    pretrained_pth = os.path.join("/opt/data/private/codeN/VirConv/tools/PENet/XDecoder/weights/X-Decoder/xdecoder_focalt_best_openseg.pt")
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    t = []
    t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    thing_classes = ["Car", "Pedestrian", "Person ride a bike"]
    # thing_classes = ['Car', 'Truck', 'Construction_vehicle', 'Bus', 'Trailer', 'Barrier', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic_cone']
    thing_colors = [[76, 76, 76], [84, 84, 84]]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes)
    with torch.no_grad():
        image_ori = cv2.imread(file_image_path)
        image_ori = image_ori[:352, :1216]
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image_ori = Image.fromarray(image_ori)
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
        batch_inputs = [{'image': images, 'height': height, 'width': width}]

        start = time.time()
        outputs = model.forward(batch_inputs)
        end = time.time()
        print("SEEM run time:", (end - start) * 1000, "ms") 
        
        predictions = outputs[0]['instances']
        masks_img = predictions.pred_masks
        classes = predictions.pred_classes
        scores_all = predictions.scores
        boxes2D = BitMasks(masks_img > 0).get_bounding_boxes()
        keep = (scores_all > 0.7).cpu()
        masks_img_keep = masks_img[keep]
        classes_keep = classes[keep]
        scores_keep = scores_all[keep]
        boxes2D_keep = boxes2D[keep]
    return thing_classes, masks_img_keep, classes_keep, scores_keep, boxes2D_keep

def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_depth_as_uint8colored(img, filename):
    #from tensor
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = depth_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if(normalized==False):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if(colored==True):
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
