# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer_id.roidb import combined_roidb
from roi_data_layer_id.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.test_model.test_utils import get_image_blob, get_rois_blob
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
# from model.faster_rcnn.vgg16 import vgg16
from model.test_model.resnet_probe import resnet_probe
from model.test_model.resnet_gallery import resnet_gallery

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--load_path', dest='load_path',
                      help='directory of load models', default=None,
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == 'psdb':
      args.imdb_name = 'psdb_train'
      args.imdbval_name = 'psdb_test'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '70']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    pass
#    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN_gallery = resnet_gallery(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN_gallery = resnet_gallery(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN_gallery = resnet_gallery(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN_gallery.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  load_dict = {k: v for k, v in checkpoint['model'].items() if k in fasterRCNN_gallery.state_dict()}
  fasterRCNN_gallery.load_state_dict(load_dict)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN_gallery.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  gallery_features = {}
  output_dir = get_output_dir(imdb, save_name)
  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN_gallery.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  for i in range(num_images):
      print(i)
      db = roidb[i]
      img_path = db['image']
      im = cv2.imread(img_path)
      im_blob, im_scales = get_image_blob(im)
      im_info = np.asarray([[im_blob.shape[2], im_blob.shape[3], im_scales[0]],])
      gt_box = np.asarray([[1,1,1,1,1], ])
      num_boxes = 0
#      im_name = img_path.split('/')[-1]
#      print(im_name, imdb.image_index[i])

      im_data = torch.from_numpy(im_blob).float().cuda()
      im_info = torch.from_numpy(im_info).float().cuda()
      gt_boxes = torch.from_numpy(gt_box).float().cuda()
      num_boxes = torch.zeros(1).long()

      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, pid_feat = fasterRCNN_gallery(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      j=1
#      tmp = {}
#      for bb, feat in zip(pred_boxes, pid_feat):
#          tmp[bb[4:]] = feat

      inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if args.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
        else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
        pid_feat = pid_feat[inds, :]
#        print(pid_feat)
            
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        pid_feat = pid_feat[order, :]
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        pid_feat = pid_feat[keep.view(-1).long(), :]
        if vis:
            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
        all_boxes[j][i] = cls_dets.cpu().numpy()
        gallery_features[i] = pid_feat.detach().cpu().numpy()
#        print(cls_dets, pid_feat)
#        for bb in cls_dets:
#            print(tmp[bb[:4]])
#        print(cls_dets.size(), pid_feat.size())
      else:
        all_boxes[j][i] = empty_array
        gallery_features[i] = empty_array

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          #pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  del fasterRCNN_gallery

  if args.net == 'vgg16':
      pass
      #    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
      fasterRCNN_probe = resnet_probe(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
      fasterRCNN_probe = resnet_probe(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
      fasterRCNN_probe = resnet_probe(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
      print("network is not defined")
      pdb.set_trace()

  fasterRCNN_probe.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  load_dict = {k: v for k, v in checkpoint['model'].items() if k in fasterRCNN_probe.state_dict()}
  fasterRCNN_probe.load_state_dict(load_dict)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  if args.cuda:
      fasterRCNN_probe.cuda()
  probes = imdb._probes
  fasterRCNN_probe.eval()

  probe_features = {}
  for i in range(len(probes)):
      print('Get %d th probe feature' % i)
      probe = probes[i]
      im_path = probe[0]
      im_name = im_path.split('/')[-1]
      roi = probe[1]
      roi_s = np.asarray([roi, ], dtype='float32')

      im = cv2.imread(im_path)
      im_blob, im_scales = get_image_blob(im)
#      roi = np.hstack((np.zeros(1), roi))* im_scales[0]
      roi_s = get_rois_blob(roi_s, im_scales)

      im_data = torch.from_numpy(im_blob).float().cuda()
      im_info = torch.from_numpy(im_scales).float().cuda()
      roi_s = torch.from_numpy(roi_s).float().cuda()

      rois, RCNN_loss_cls, RCNN_loss_bbox, feat_norm = fasterRCNN_probe(im_data, roi_s)
#      probe_features[i] = [im_name, roi, feat_norm.detach().cpu().numpy()]
      probe_features[i] = feat_norm.detach().cpu().numpy()


  with open(det_file, 'wb') as f:
      pickle.dump([all_boxes, gallery_features, probe_features], f, pickle.HIGHEST_PROTOCOL)

  with open(det_file, 'rb') as f:
      [all_boxes, gallery_features, probe_features] = pickle.load(f)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  print('Evaluating search')
  imdb.evaluate_search(all_boxes, gallery_features, probe_features, gallery_size=100)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
