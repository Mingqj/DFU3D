import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model import ENet
from model import PENet_C1_train
from model import PENet_C2_train
#from model import PENet_C4_train (Not Implemented)
from model import PENet_C1
from model import PENet_C2
from model import PENet_C4
import time
from itertools import islice

# from transformers import pipeline
import json
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.transforms import Compose
from Depth_Anything.depth_anything.dpt import DepthAnything
from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from Depth_Anything.metric_depth.zoedepth.models.builder import build_model
from Depth_Anything.metric_depth.zoedepth.utils.config import get_config

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('--command',default="evaluate", help='Command: train/evaluate/train-and-evaluate')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="pe",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='/data/dataset/kitti_depth/depth',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default='/data/dataset/kitti_raw',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('--data-folder-save',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('--detpath',
                    default='../../data/kitti/training',
                    type=str,
                    metavar='PATH',
                    help='data folder of 3D object detection')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank-metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='pe.pth.tar', type=str, metavar='PATH')

parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--test', action="store_true", default=True,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=352, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

parser.add_argument("--model", type=str, default='zoedepth', help="Name of the model to test")
# parser.add_argument("--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_outdoor.pt', help="Pretrained resource to use for fetching weights.")
parser.add_argument("--pretrained_resource", type=str, default="local::/opt/data/private/codeN/VirConv/tools/PENet/Depth_Anything/weights/depth_anything_metric_depth_outdoor.pt", help="Pretrained resource to use for fetching weights.")


parser.add_argument('--conf_files',default="/opt/data/private/codeN/VirConv/tools/PENet/XDecoder/configs/xdecoder/segvlp_focalt_lang.yaml", nargs='+', required=True, help='Path(s) to the config file(s).')
parser.add_argument('--user_dir', help='Path to the user defined module for tasks (models, criteria), optimizers, and lr schedulers.')
parser.add_argument('--config_overrides', nargs='*', help='Override parameters on config with a json style string, e.g. {"<PARAM_NAME_1>": <PARAM_VALUE_1>, "<PARAM_GROUP_2>.<PARAM_SUBGROUP_2>.<PARAM_2>": <PARAM_VALUE_2>}. A key with "." updates the object in the corresponding nested dict. Remember to escape " in command line.')
parser.add_argument('--overrides', default="/opt/data/private/codeN/VirConv/tools/PENet/XDecoder/weights/X-Decoder/xdecoder_focalt_best_openseg.pt", help='arguments that used to override the config file in cmdline', nargs=argparse.REMAINDER)


args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
# print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()

#multi batch
multi_batch_size = 1
def iterate(mode, args, loader, model, depth_anything, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    
    #########################################################
    # train idx list
    train_txt_path = "/opt/data/private/codeN/VirConv/data/kitti/ImageSets/train.txt" # kitti
    # train_txt_path = "/opt/data/private/codeN/OpenPCDet/data/pandaset_kitti/ImageSets/train.txt" # pandasetkitti
    # train_txt_path = "/opt/data/private/codeN1/mmdetection3d/data/nuscenes_kitti_format/train_28130/ImageSets/train_28130.txt" # nuscenes2kitti
    with open(train_txt_path, 'r') as f:
        lines =f.readlines()
        list_f = []
        for ele in lines:
            ele = ele.strip('\n')
            adm = ele.split(' ')
            list_f.append(int(adm[0]))
    #########################################################


    # start_deep = 1835 # -1, 1850, 3700, 5550, 7490 / -1 3745 7490 # kitti
    # end_deep = 1850

    # start_deep = 3613 # -1, 1850, 3700, 5550, 7490 / -1 3745 7490 # kitti
    # end_deep = 3700

    # start_deep = 5430 # -1, 1850, 3700, 5550, 7490 / -1 3745 7490 # kitti
    # end_deep = 5550

    # start_deep = 7228 # -1, 1850, 3700, 5550, 7490 / -1 3745 7490 # kitti
    # end_deep = 7490

    ##########################################################################
    # start_deep = 0  # nuscenes 111
    # end_deep = 7032

    # start_deep = 7031 # nuscenes 222
    # end_deep = 14062

    # start_deep = 14061 # nuscenes 333
    # end_deep = 21092

    # start_deep = 21091 # nuscenes 444
    # end_deep = 28130

    for i, batch_data in enumerate(loader):
    # for i, batch_data in islice(enumerate(loader), start_deep, end_deep + 1):
        # if i < start_deep or i > end_deep or i not in list_f: # nuscenes
        #     print('skip: ', str(i))
        #     continue
        # else:
        dstart = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        #################################################depth anything###############################################################
        model_name = args.model
        pretrained_resource = args.pretrained_resource
        DATASET = 'kitti'
        config = get_config(model_name, "eval", DATASET)
        config.pretrained_resource = pretrained_resource
        model_depth_anything = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        model_depth_anything.eval()

        image = batch_data['rgb'] / 255.0 # [1, 3, 352, 1216]
        _, _, h, w = image.shape

        start = time.time()
        depth = model_depth_anything(image, dataset=DATASET)
        end = time.time()
        print("DepthAnything run time:", (end - start) * 1000, "ms") 

        pred = F.interpolate(depth['metric_depth'], (h, w), mode='bilinear', align_corners=False)[0, 0].unsqueeze(0).unsqueeze(0)
        # pred = pred + 7 * F.normalize(pred, p=2, dim=1)
        a, b = 0, 2
        min_val = pred.min()
        max_val = pred.max()
        custom_normalized_tensor = a + ((pred - min_val) * (b - a)) / (max_val - min_val)
        # nonlinear_normalized_tensor = custom_normalized_tensor ** 2
        nonlinear_normalized_tensor = torch.exp(custom_normalized_tensor)

        # pred = pred + 1.5 * nonlinear_normalized_tensor # kitti
        pred = pred + 3 * nonlinear_normalized_tensor # nuscenes
        #############################################################################################################################

        ####################################################PENet####################################################################
        start = time.time()
        pred = model(batch_data) # PENet # 19.5 ms
        end = time.time()
        print("PENet run time:", (end - start) * 1000, "ms")
        ####################################################PENet####################################################################

        #'''
        if(args.network_model == 'e'):
            start = time.time()
            st1_pred, st2_pred, pred = model(batch_data)
        else:
            start = time.time()
            # pred = model(batch_data) # PENet

        # print(pred, 111)
        # print(pred1, 222)
        # print('\n')

        if(args.evaluate):
            gpu_time = time.time() - start
        #'''

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # inter loss_param
        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2, round3 = 1, 3, None
        if(actual_epoch <= round1):
            w_st1, w_st2 = 0.2, 0.2
        elif(actual_epoch <= round2):
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0

        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            depth_loss = depth_criterion(pred, gt)

            if args.network_model == 'e':
                st1_loss = depth_criterion(st1_pred, gt)
                st2_loss = depth_criterion(st2_pred, gt)
                loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss
            else:
                loss = depth_loss

            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            loss.backward()

            if i % multi_batch_size == (multi_batch_size-1) or i==(len(loader)-1):
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))

        if mode == "test_completion":
            # ite = i + 0 # nuscenes visual

            ite = i + 0 + 735 # nuscenes 111  
            # ite = i + 7031 + 5536 # nuscenes 222
            # ite = i + 14063 + 6871 # nuscenes 333
            # ite = i + 21094 + 520 # nuscenes 333
            vis_utils.save_depth_as_points(pred, ite, args.detpath) #################################### generate pseudo box #########################################

        if(not args.evaluate):
            gpu_time = time.time() - start
        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]

                if mode != 'train':
                    logger.conditional_print(mode, i, epoch, lr, len(loader),
                                    block_average_meter, average_meter)
                logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                epoch)
                logger.conditional_save_pred(mode, i, pred, epoch)
        end_time = time.time()-dstart
        print('iter: ', ite,'  ',  'remain time:', (len(loader)-i)*end_time//60, 'min')
    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            #args = checkpoint['args']
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True

            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
            #return

    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = None
    penet_accelerated = False
    if (args.network_model == 'e'):
        model = ENet(args).to(device)
    elif (is_eval == False):
        if (args.dilation_rate == 1):
            model = PENet_C1_train(args).to(device)
        elif (args.dilation_rate == 2):
            model = PENet_C2_train(args).to(device)
        elif (args.dilation_rate == 4):
            model = PENet_C4(args).to(device)
            penet_accelerated = True
    else:
        if (args.dilation_rate == 1):
            model = PENet_C1(args).to(device)
            penet_accelerated = True
        elif (args.dilation_rate == 2):
            model = PENet_C2(args).to(device)
            penet_accelerated = True
        elif (args.dilation_rate == 4):
            model = PENet_C4(args).to(device)
            penet_accelerated = True

    if (penet_accelerated == True):
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False

    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None

    if checkpoint is not None:
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):
            model.backbone.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.") ###############################################

    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
        del checkpoint
    print("=> logger created.")

    test_dataset = None
    test_loader = None
    if (args.test):
        test_dataset = KittiDepth('test_completion', args)
        test_dataset = torch.utils.data.Subset(
            test_dataset, 
            range(0, 7033))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True)
        ###########################################################################################################################
        # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
        # pipe = DepthAnything.from_pretrained("/opt/data/private/codeN/Depth-Anything/weights/depth_anything_vitb14.pth")
        # pipe = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').cuda().eval()

        cfg_path = "/opt/data/private/codeN/VirConv/tools/PENet/Depth_Anything/weights/config_s.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        weights = torch.load("/opt/data/private/codeN/VirConv/tools/PENet/Depth_Anything/weights/depth_anything_vits14.pth")
        depth_anything = DepthAnything(cfg).cuda().eval()
        depth_anything.load_state_dict(weights)
        ###########################################################################################################################
        iterate("test_completion", args, test_loader, model, depth_anything, None, logger, 0)
        return

    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best = iterate("val", args, val_loader, model, None, logger,
                              args.start_epoch - 1)
        return

    if (args.freeze_backbone == True):
        for p in model.backbone.parameters():
            p.requires_grad = False
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    elif (args.network_model == 'pe'):
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        ]
        model_new_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        model_new_params = list(set(model_new_params) - set(model_bone_params))
        optimizer = torch.optim.Adam([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}],
                                     lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    else:
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    print("completed.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                #    num_workers=args.workers,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))

    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch

        # validation memory reset
        for p in model.parameters():
            p.requires_grad = False
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)  # evaluate on validation set

        for p in model.parameters():
            p.requires_grad = True
        if (args.freeze_backbone == True):
            for p in model.module.backbone.parameters():
                p.requires_grad = False
        if (penet_accelerated == True):
            model.module.encoder3.requires_grad = False
            model.module.encoder5.requires_grad = False
            model.module.encoder7.requires_grad = False

        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()