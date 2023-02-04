# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import glob
from scipy import misc
import sys
import json
import argparse
from colmap_read_model import *
import shutil
import cv2
import pdb
from PIL import Image

def mkdir_s(dn):
    if not os.path.exists(dn):
        os.mkdir(dn)

def computeCropFactor(cx, cy, crop_center_x, crop_center_y):
  crop_h = 756
  crop_w = 1008  
  
  # lower left coords
  ll_x = crop_center_x - crop_w // 2
  ll_y = crop_center_y - crop_h // 2


  
  return crop_h, crop_w, cx - ll_x, cy - ll_y
  
  
def runResizeCrop(
  imagePath, 
  out_dir,
  resize_scale=756./1080., 
  crop_center_x=1344 // 2, 
  crop_center_y=756 // 2, 
  crop_h=756, 
  crop_w=1008
):
    # image = cv2.imread(imagePath)
    with Image.open(imagePath) as im:

      # resize
      wh_resize = (int(im.width * resize_scale), int(im.height * resize_scale))
      im_resized = im.resize(wh_resize, resample=Image.Resampling.LANCZOS)
      # im_resized = im.resize((width, height))


      hh = int(crop_center_y - (crop_h // 2))
      ww = int(crop_center_x - (crop_w // 2))
      # im_cropped = im_resized.crop([ hh:(hh + crop_h), ww:(ww + crop_w)])
      im_cropped = im_resized.crop([ww ,hh, ww + crop_w, hh + crop_h])

      # out_dir = os.path.dirname(os.path.dirname(os.path.dirname(imagePath))) + '/images'
      os.makedirs(out_dir, exist_ok=True)
      

      name = os.path.basename(imagePath)
      name_woe  = os.path.splitext(name)[0]
      ext  = os.path.splitext(name)[1]
      name_new = name_woe + '_crop' + ext
      fn = os.path.join(out_dir, name_new)

      # pdb.set_trace()
      
      print(fn)
      im_cropped.save(fn)
      # cv2.imwrite(fn, image)

    
def computeScaleFactor(fx, fy, cx, cy, scale=756./1080.):
    fx_s = fx * scale
    fy_s = fy * scale
    
    cx_s = cx * scale
    cy_s = cy * scale

    return fx_s, fy_s, cx_s, cy_s
    
def runScale(imagePath, cx, cy):
    image = cv2.imread(imagePath)
    scale_h = 756
    scale_w = 1008
    dim = (scale_w, scale_h)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    name = os.path.basename(imagePath)
    name_woe  = os.path.splitext(name)[0]
    ext  = os.path.splitext(name)[1]
    name_new = name_woe + '_s' + ext
    
    folder = os.path.dirname(imagePath) + '/../../images_s'
    os.makedirs(folder, exist_ok=True)
    
    fn = os.path.join(folder, name_new)
    
    print(fn)
    cv2.imwrite(fn, image)
    
def writeHeaderPLY(file_out, nv):
    file_out.write("ply\n")
    file_out.write("format ascii 1.0\n")
    file_out.write("element vertex " + str(nv) + "\n")
    file_out.write("property float x\n")
    file_out.write("property float y\n")
    file_out.write("property float z\n")
    file_out.write("element face 0\n")
    file_out.write("property list uchar int vertex_indices\n")
    file_out.write("end_header\n")
    

def cmd(s):
  print(s)
  os.system(s)

def runner(dataset):
  '''
    Use colmap
      1. feature extractor
      2. exhaustive matcher
      3. Mapper
      4. Image undistorter
  '''
  cmd("colmap feature_extractor \
   --database_path " + dataset + "/database.db \
   --image_path " + dataset + "/images \
   --ImageReader.single_camera 1 \
   --SiftExtraction.use_gpu 0 \
   --ImageReader.camera_model SIMPLE_RADIAL")

  cmd("colmap exhaustive_matcher \
   --database_path " + dataset + "/database.db " \
   "--SiftMatching.use_gpu 0 \
    --SiftMatching.guided_matching 1 " \
   )

  cmd("mkdir " + dataset + "/sparse")

  cmd("colmap mapper \
    --database_path " + dataset + "/database.db \
    --image_path " + dataset + "/images \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.num_threads 16 \
    --Mapper.extract_colors 0 \
    --output_path " + dataset + "/sparse")

  cmd("mkdir " + dataset + "/dense")

  cmd("cp -r " + dataset + "/sparse " + dataset +"/sparse_before_undistort")
  cmd("cp -r " + dataset + "/database.db " + dataset +"/sparse_before_undistort/database.db")
  cmd("colmap image_undistorter \
    --image_path " + dataset + "/images \
    --input_path " + dataset + "/sparse/0 \
    --output_path " + dataset + "/dense \
    --output_type COLMAP")
  #cmd("mv " + dataset + "/images " + dataset +"/images_distort")
  #cmd("mv " + dataset +"/dense/images" +" " + dataset +'/images')



def load_colmap_data(realdir, crop_center_x=None, crop_center_y=None):
  '''
    copy from Local light field fusion
    https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py

  '''
  colmap_path = os.path.join(realdir, 'dense/sparse/')
  # colmap_path = os.path.join(realdir, 'sparse/0/')
  camerasfile = os.path.join(colmap_path, 'cameras.bin')
  camdata = read_cameras_binary(camerasfile)

  # cam = camdata[camdata.keys()[0]]
  list_of_keys = list(camdata.keys())
  cam = camdata[list_of_keys[0]]

  h, w, fx, fy, cx, cy = cam.height, cam.width, cam.params[0], cam.params[1], cam.params[2], cam.params[3]
  print([h, w, fx, fy, cx, cy])
  print(cam.params)

  target_h, target_w = 756, 1008

  resize_scale = max(target_w / w, target_h / h)

  fx, fy, cx, cy = computeScaleFactor(fx, fy, cx, cy, scale=resize_scale)
  # hwf_cxcy_s = np.array([h_s, w_s, fx_s, fy_s, cx_s, cy_s]).reshape([6,1])

  rescale_h, rescale_w = int(h * resize_scale), int(w * resize_scale)

  if crop_center_x is None:
    crop_center_x = rescale_w // 2
  if crop_center_y is None:
    crop_center_y = rescale_h // 2

  with open(os.path.join(realdir, 'rescale_crop_center.txt'), 'w') as f:
    f.write(f'{resize_scale}, {crop_center_x}, {crop_center_y}')


  h, w, cx, cy = computeCropFactor(cx, cy, crop_center_x, crop_center_y)
  hwf_cxcy = np.array([h, w, fx, fy, cx, cy]).reshape([6,1])



  imagesfile = os.path.join(colmap_path, 'images.bin')
  imdata = read_images_binary(imagesfile)

  w2c_mats = []
  bottom = np.array([0,0,0,1.]).reshape([1,4])

  names = [imdata[k].name for k in imdata]
  print( 'Images #', len(names))
  perm = np.argsort(names)
  print(realdir)
  
  file_out = open(os.path.join(realdir, 'cameras.ply'), "w")
  writeHeaderPLY(file_out, len(imdata))
  
  # mkdir_s(os.path.join(realdir, 'images_s/'))
  
  for k in imdata:
    im = imdata[k]
    
    image_path = os.path.join(realdir, 'dense/images', im.name)
    out_dir = os.path.dirname(os.path.dirname(os.path.dirname(image_path))) + '/images'
    # image_path = os.path.join(realdir, 'images_raw', im.name)
    # out_dir = os.path.dirname(os.path.dirname(image_path)) + '/images'
    # pdb.set_trace()
    runResizeCrop(image_path, out_dir,
      resize_scale=resize_scale, crop_center_x=crop_center_x, crop_center_y=crop_center_y)
    # runScale(image_path, cx, cy)
    R = im.qvec2rotmat()
    t = im.tvec.reshape([3,1])
    file_out.write(str(float(t[0])) + " " + str(float(t[1])) + " " + str(float(t[2])) + " " + "\n")
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    w2c_mats.append(m)

  file_out.close()
  w2c_mats = np.stack(w2c_mats, 0)
  c2w_mats = np.linalg.inv(w2c_mats)

  poses = c2w_mats[:, :3, :4].transpose([1,2,0])

  #poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)

  points3dfile = os.path.join(colmap_path, 'points3D.bin')
  pts3d = read_points3d_binary(points3dfile)

  # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
  #poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
  poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :]], 1)

  return poses, pts3d, perm, hwf_cxcy

def save_poses(basedir, poses, pts3d, perm, hwf_cxcy):
  pts_arr = []
  vis_arr = []
  for k in pts3d:
    pts_arr.append(pts3d[k].xyz)
    cams = [0] * poses.shape[-1]
    for ind in pts3d[k].image_ids:
      if len(cams) < ind - 1:
        print('ERROR: the correct camera poses for current points cannot be accessed')
        return
      cams[ind-1] = 1
    vis_arr.append(cams)

  pts_arr = np.array(pts_arr)
  vis_arr = np.array(vis_arr)
  print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )

  zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
  valid_z = zvals[vis_arr==1]
  print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )

  save_arr = []
  for i in perm:
    vis = vis_arr[:, i]
    zs = zvals[:, i]
    zs = zs[vis==1]
    close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
    # print( i, close_depth, inf_depth )

    save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
  save_arr = np.array(save_arr)

  np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
  np.save(os.path.join(basedir, 'hwf_cxcy.npy'), hwf_cxcy)
  # np.save(os.path.join(basedir, 'hwf_cxcy_s.npy'), hwf_cxcy_s)

def need_run_coolmap(basedir):
  files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
  if os.path.exists(os.path.join(basedir, 'poses_bounds.npy') and os.path.join(basedir, 'hwf_cxcy.npy')):
     return False
  if os.path.exists(os.path.join(basedir, 'sparse/0')):
    files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
  elif os.path.exists(os.path.join(basedir, 'dense/sparse')):
    files_had = os.listdir(os.path.join(basedir, 'dense/sparse'))
  else:
    files_had = []
  if not all([f in files_had for f in files_needed]):
    print( 'Need to run COLMAP' )
    return True
  else:
    return False

def colmapGenPoses(dpath):
  files = os.listdir(dpath)
  #no need colmap on deepview
  if 'models.json' in files:
    return False
  #no need colmap on blender
  if 'transforms_train.json' in files:
    return False
  
  if need_run_coolmap(dpath):
    '''
      Automatically run colmap
      Get near, far planes, ref_image using code from LLFF
    '''
    if shutil.which('colmap') is None:
      print('You need to install COLMAP in this machine')
      raise Exception("No COLMAP found in this machine")

    runner(dpath)

  #post colmap
  print( 'Post-colmap')
  poses, pts3d, perm, hwf_cxcy, hwf_cxcy_s = load_colmap_data(dpath)
  print(hwf_cxcy)
  save_poses(dpath, poses, pts3d, perm, hwf_cxcy, hwf_cxcy_s)
  print( 'Done with imgs2poses' )

    
if __name__ == '__main__':
  # colmapGenPoses(sys.argv[1])

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--crop_center_x',
                        type=int,
                        default=None)
  parser.add_argument('--crop_center_y',
                        type=int,
                        default=None)

  args = parser.parse_args()

  
  if need_run_coolmap(args.data_path):
    raise Exception("No colmap result found, please run colmap first")

  #post colmap
  poses, pts3d, perm, hwf_cxcy = load_colmap_data(args.data_path, args.crop_center_x, args.crop_center_y)
  print(hwf_cxcy)
  save_poses(args.data_path, poses, pts3d, perm, hwf_cxcy)
  print( 'Done with imgs2poses' )


  # python colmap_runner.py --data_path /home/tw554/nerf_bechmark_data/data/geopards_long --crop_center_y 702
