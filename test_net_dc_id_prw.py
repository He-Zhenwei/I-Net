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
import math

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
from model.dcinet.resnet_probe import resnet_probe
from model.dcinet.resnet_gallery import resnet_gallery
from sklearn.metrics import average_precision_score, precision_recall_curve
import roi_data_layer_id.prw_data as prw_data
import scipy.io as sio
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch

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

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def evaluate(querys, probe_feat, out_dict, im_index):
    num_probe = len(querys)
    assert len(probe_feat) == num_probe
    aps = []
    accs = []
    topk = [1, 5, 10]

    total_tp = 0
    total_gt = 0
    all_true = []
    all_score = []

    for i in xrange(num_probe):
        y_true, y_score = [], []
        feat_query = probe_feat[i]
        query_im = querys[i][2]
        query_im = query_im.split('/')[-1]
        print(query_im)
        query_id = querys[i][0]

        count_gt = 0
        count_tp = 0

        for im_name, res in out_dict.iteritems():
#            print(im_name)
            if im_name[1] != query_im[1]:
                bbox = res[0]
                feats = res[1]

                gt_bbox = im_index[im_name]['bbox']
                gt_pid = im_index[im_name]['pids']

                sims = feat_query.ravel().dot(feats.transpose())  # .squeeze()
                #sims = -np.sum((feats - feat_query), axis=1) **2
                label = np.zeros(len(sims), dtype='int32')

                if float(query_id) in list(gt_pid):
                    count_gt = count_gt + 1
                    inds = np.argsort(sims)[::-1]
                    maxlap = bbox[:, 4]
                    max_pid = bbox[:, 5]
                    f = True

                    for i in range(len(maxlap)):
                        pid = max_pid[i]
                        iou = maxlap[i]

                        if iou > 0.5 and pid == float(query_id):
                            label[i] = 1
                            if f:
                                count_tp = count_tp + 1
                                f = False

                    '''
                    pos = np.where(gt_pid == float(query_id))[0]
                    gt = gt_bbox[pos, :]
                    [w, h] = gt[0, 2:] - gt[0, :2]
                    thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))

                    inds = np.argsort(sims)[::-1]
                    sims = sims[inds]
                    bbox = bbox[list(inds), :]

                    num_bb = bbox.shape[0]
                    for i in xrange(num_bb):
                        tmp_bb = bbox[i, :]
                        iou = _compute_iou(tmp_bb ,gt.squeeze())
                        f = True

                        if iou > thresh:
                            label[i] = 1
                            if f:
                                count_tp = count_tp + 1
                                f = False
                    '''

                y_true.extend(list(label))
                y_score.extend(list(sims))

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        total_tp = total_tp + count_tp
        total_gt = total_gt + count_gt
        recall = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * recall
        if math.isnan(ap):
            ap = 0
        aps.append(ap)

        inds = np.argsort(y_score)[::-1]
        y_score = y_score[list(inds)]
        y_true = y_true[list(inds)]

        all_true.extend(y_true)
        all_score.extend(y_score)
        accs.append([min(1, sum(y_true[:k])) for k in topk])

    accs = np.mean(accs, axis=0)
    print('mAP: %f' % np.mean(aps))

    for i, index in enumerate(topk):
        print('top%d: %f' % (index, accs[i]))

    all_true = np.asarray(all_true)
    all_score = np.asarray(all_score)
    all_recall = total_tp * 1.0 / total_gt
    all_ap = average_precision_score(all_true, all_score) * all_recall
    ps, rs, _ = precision_recall_curve(all_true, all_score)
    rs = rs * all_recall
    sio.savemat('prw_ap_inet.mat', {'p': ps, 'r': rs})
    print('The ap of all samples is %0.4f, the recall of all samples is %0.4f' % (all_ap, all_recall))

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
  elif args.dataset == 'prw':
      args.imdb_name = 'prw_train'
      args.imdbval_name = 'prw_test'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}_prw.yml".format(args.net)

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
    thresh = 0.5
  else:
    thresh = 0.5

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
  img_dict = {}
  im_index = prw_data.get_imdata('test')

  for i in range(len(im_index)):
      print(i)
      img_name = im_index.keys()[i]
      img_path = os.path.join('./data/PRW/frames', img_name + '.jpg')
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
      pid_feat = fasterRCNN_gallery(im_data, im_info, gt_boxes, num_boxes)

      gt_boxes = im_index[img_name]['bbox']
      gt_pid = im_index[img_name]['pids']
      gt_boxes = torch.from_numpy(np.asarray([gt_boxes, ])).cuda().float()

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
      if True:
#          im = cv2.imread(imdb.image_path_at(i))
          im = cv2.imread(img_path)
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
#        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
#        cls_dets = cls_dets[keep.view(-1).long()]
#        pid_feat = pid_feat[keep.view(-1).long(), :]
        if True:
            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)

#        cls_dets = cls_dets.cpu().numpy()
#        pid_feat = pid_feat.detach().cpu().numpy()
        res_bbox = cls_dets[:, :4].unsqueeze(0)
        overlaps = bbox_overlaps_batch(res_bbox, gt_boxes)

        maxlap, arg_max = torch.max(overlaps[0], dim=1)
#        print(len(maxlap), cls_dets.size(0))
        assert len(maxlap) == cls_dets.size(0)
        maxlap = maxlap.cpu().numpy()
        arg_max = arg_max.cpu().numpy()
        max_pid = gt_pid[arg_max]
#        print(max_pid, arg_max, gt_pid)

        pos1 = set(np.where(max_pid == -1)[0])
        pos2_1 = set(np.where(maxlap < 0.5)[0])
        pos2_2 = set(np.where(maxlap > 0.3)[0])
        pos2 = pos2_1 & pos2_2
        pos = pos1 | pos2
        pos = list(set(range(len(max_pid))) - pos)

        cls_dets = cls_dets.cpu().numpy()
        pid_feat = pid_feat.detach().cpu().numpy()
        res = np.hstack((cls_dets[pos, :4], maxlap[pos, np.newaxis], max_pid[pos, np.newaxis]))
        feat = pid_feat[pos, :]

        img_dict[img_name] = [res, feat]
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

      if True:
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
  model = checkpoint['model']
  for k in fasterRCNN_probe.state_dict():
      if k not in model:
          print(k)

  load_dict = {k: v for k, v in checkpoint['model'].items() if k in fasterRCNN_probe.state_dict()}
  fasterRCNN_probe.load_state_dict(load_dict)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  if args.cuda:
      fasterRCNN_probe.cuda()
  probes = imdb._probes
  fasterRCNN_probe.eval()

  probe_features = {}
  probe_metrics = []
  for i in range(len(probes)):
      print('Get %d th probe feature' % i)
      probe = probes[i]
      im_path = probe[2]
      print(im_path)
      im_name = im_path.split('/')[-1]
      roi = probe[1]
      roi_s = np.asarray([roi, ], dtype='float32')

      im = cv2.imread(im_path)
      im_blob, im_scales = get_image_blob(im)
#      roi = np.hstack((np.zeros(1), roi))* im_scales[0]
      roi_s = get_rois_blob(roi_s, im_scales)

      im_data = torch.from_numpy(im_blob).float().cuda()
      im_info = torch.from_numpy(np.asarray([im_data.size(2), im_data.size(3), im_scales[0]])).float().cuda().unsqueeze(0)
      roi_s = torch.from_numpy(roi_s).float().cuda()

      rois, cls_prob, bbox_pred, feat_norm = fasterRCNN_probe(im_data, im_info, roi_s)
      probe_metrics.append(feat_norm.detach().cpu().numpy())

  with open(det_file, 'wb') as f:
      pickle.dump([img_dict, probe_metrics], f, pickle.HIGHEST_PROTOCOL)

  with open(det_file, 'rb') as f:
      [img_dict, probe_metrics] = pickle.load(f)

  evaluate(imdb._probes, probe_metrics, img_dict, im_index)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
