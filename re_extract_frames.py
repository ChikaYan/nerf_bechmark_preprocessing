import os
from pathlib import Path
import numpy as np
import imageio
import pdb
import shutil
from util.clean_video import processOneVideo as process_vid
import json
import matplotlib.pyplot as plt
import argparse
import cv2
from util.video import *
import shutil

def cmd(s):
  print(s)
  os.system(s)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='./data')
parser.add_argument('--raw_vid_path', type=str,
                    default='./data/raw_video')
parser.add_argument('--capture_name',
                      type=str,
                      default=None)

args = parser.parse_args()


root_dir = Path(args.data_path, args.capture_name)
rgb_raw_dir = root_dir / 'images_raw'

all_frames_dir = root_dir / 'all_frames'

# Where to save the COLMAP outputs.
colmap_dir = root_dir
colmap_db_path = colmap_dir / 'database.db'
colmap_out_path = colmap_dir / 'sparse'

colmap_out_path.mkdir(exist_ok=True, parents=True)
rgb_raw_dir.mkdir(exist_ok=True, parents=True)
all_frames_dir.mkdir(exist_ok=True, parents=True)

print(f"""Directories configured:
  root_dir = {root_dir}
  rgb_raw_dir = {rgb_raw_dir}
  colmap_dir = {colmap_dir}
""")

tmp_rgb_raw_dir = rgb_raw_dir

############ process videos ############

scene_vid_path = Path(args.raw_vid_path) / args.capture_name

with (root_dir / 'split.json').open('r') as f:
  split_json = json.load(f)

imgs_names = split_json['train_imgs'] + split_json['test_imgs']

def get_vid_name(img_name):
  return img_name.split('_')[0]

def get_frame_id(img_name):
  return int(img_name.split('_')[-1])

vid_frames = {}

for img_name in imgs_names:
    vid_name = get_vid_name(img_name)
    if vid_name not in vid_frames:
        vid_frames[vid_name] = []

    vid_frames[vid_name].append(get_frame_id(img_name))


# pdb.set_trace()


for vid_name in vid_frames.keys():
   process_vid(str(scene_vid_path/f'{vid_name}.MP4'), str(all_frames_dir), sampling=1, iTarget=-1)


for img_name in imgs_names:
    shutil.copy(str(all_frames_dir / f'{img_name}.png'), str(rgb_raw_dir))


# for vid_name in vid_frames.keys():
#     cap = cv2.VideoCapture(str(scene_vid_path/f'{vid_name}.MP4'))
#     frame_ids = vid_frames[vid_name]
    
#     for i in range(0, max(frame_ids) + 1):
#         _, frame = cap.read()

#         if i in frame_ids:
#             if frame is None:
#                 pdb.set_trace()
               
#             writeCV2(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0, str(rgb_raw_dir / f"{vid_name}_{i:06d}.png"))
#             print(f'write: {str(rgb_raw_dir / f"{vid_name}_{i:06d}.png")}')

#     cap.release()


    #     v = Video(str(scene_vid_path/f'{vid_name}.MP4'))
    #     success = True
    #     i = 0
    #     while success:
    #         success, frame, j_k = v.getNextFrame(i, True)
    #         vids[vid_name].append()
    # vids[vid_name] = Video(str(scene_vid_path/f'{vid_name}.MP4'))
    # frame_id = get_frame_id(img_name)
    
    # if success:
    #     writeCV2(frame / 255.0, str(rgb_raw_dir / f"{vid_name}_{frame_id:06d}.png"))
