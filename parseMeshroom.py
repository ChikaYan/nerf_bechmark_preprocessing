import json
import os
import sys
import numpy as np
import open3d as o3d

def computeCropFactor(cx, cy):
  crop_h = 756
  crop_w = 1008  
  center_h = cy #define crop center, this should be shared across all the images
  center_w = cx #define crop center this should be shared across all the images
  hh = center_h - (crop_h // 2)
  ww = center_w - (crop_w // 2)  
  cx_cropped = cx - ww
  cy_cropped = cy - hh
  return crop_h, crop_w, cx_cropped, cy_cropped

if __name__ == "__main__":
   
   f = open(sys.argv[1])
   data = json.load(f)
   
   mesh = o3d.io.read_triangle_mesh(sys.argv[2])
   pts = np.array(mesh.vertices)

   if True:#try:
      #intrinsics
      data_k = data['intrinsics'][0]
      K = np.zeros((3,3))
      p = data_k['principalPoint']

      K[0,0] = float(data_k['pxFocalLength'])
      K[1,1] = float(data_k['pxFocalLength'])
      K[0,2] = float(p[0])
      K[1,2] = float(p[1])     
      K[2,2] = 1.0

      cx = K[0,2]
      cy = K[1,2]
      fx = data_k['pxFocalLength']
      fy = data_k['pxFocalLength']
      width = float(data_k['width'])
      height = float(data_k['height'])
      
      h, w, cx, cy = computeCropFactor(cx, cy)
      hwf_cxcy = np.array([h, w, fx, fy, cx, cy]).reshape([6,1])
      np.save('hwf_cxcy.npy', hwf_cxcy)
            
      #
      #views
      #
      bottom = np.array([0,0,0,1.]).reshape([1,4])

      names = {}
      for i in data['views']:
         filename = os.path.split(i['path'])[1]
         names[filename] = i['poseId']
      #print(names)   
      out = sorted(names.items())
      
      data_poses = data['poses']
      w2c_mats = []
      vis_arr = []

      for i in range(0, len(out)):
          frame_i = out[i][1]
          match = -1
          for j in range(0, len(data_poses)):
             tmp = data_poses[j]
             if frame_i == tmp['poseId']:
                match = j
                break
          pose_i = data_poses[match]
          rot_i = pose_i['pose']['transform']['rotation']
          center_i = pose_i['pose']['transform']['center']
          
          rot = np.zeros((3,3))
          rot[0,0] = float(rot_i[0])
          rot[0,1] = float(rot_i[1])
          rot[0,2] = float(rot_i[2])
          rot[1,0] = float(rot_i[3])
          rot[1,1] = float(rot_i[4])
          rot[1,2] = float(rot_i[5])
          rot[2,0] = float(rot_i[6])
          rot[2,1] = float(rot_i[7])
          rot[2,2] = float(rot_i[8])
          
          t = np.zeros((3,1))
          t[0,0] = float(center_i[0])
          t[1,0] = float(center_i[1])
          t[2,0] = float(center_i[2])
          
          m = np.concatenate([np.concatenate([rot, t], 1), bottom], 0)
          w2c_mats.append(m)



      w2c_mats = np.stack(w2c_mats, 0)
      c2w_mats = np.linalg.inv(w2c_mats)
      poses = c2w_mats[:, :3, :4].transpose([1,2,0])
      poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :]], 1)

      zvals = np.sum(-(pts[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
      print( 'Depth stats', zvals.min(), zvals.max(), zvals.mean() )
      save_arr = []

      for i in range(0, len(out)):
         close_depth, inf_depth = np.percentile(zvals, .1), np.percentile(zvals, 99.9)
         save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
      save_arr = np.array(save_arr)
      np.save('poses_bounds.npy', save_arr)

      #np.save('poses_bounds.npy', save_arr)
  # except:
  #    print('error')

   f.close()
