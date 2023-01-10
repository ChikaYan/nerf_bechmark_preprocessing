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
parser.add_argument('--train_vid_names', # needs to be passed as a list str
                      type=str,
                      default=None)
parser.add_argument('--test_vid_name',
                      type=str,
                      default=None)
parser.add_argument('--n_test',
                      type=int,
                      default=120)


args = parser.parse_args()


root_dir = Path(args.data_path, args.capture_name)
rgb_raw_dir = root_dir / 'images_raw'

# Where to save the COLMAP outputs.
colmap_dir = root_dir
colmap_db_path = colmap_dir / 'database.db'
colmap_out_path = colmap_dir / 'sparse'

colmap_out_path.mkdir(exist_ok=True, parents=True)
rgb_raw_dir.mkdir(exist_ok=True, parents=True)

print(f"""Directories configured:
  root_dir = {root_dir}
  rgb_raw_dir = {rgb_raw_dir}
  colmap_dir = {colmap_dir}
""")

tmp_rgb_raw_dir = rgb_raw_dir

############ process videos ############

scene_vid_path = Path(args.raw_vid_path) / args.capture_name
train_img_names = []
test_img_names = []

train_vid_names = args.train_vid_names.strip().strip('[').strip(']').strip().split(',')
for train_name in train_vid_names:
  process_vid(str(scene_vid_path/f'{train_name}.MP4'), str(rgb_raw_dir), sampling=-1, iTarget=30)
  train_img_names += list(rgb_raw_dir.glob(f'{train_name}_*.jpg'))
train_img_names = sorted(train_img_names)

target_num_frames = args.n_test + 30
video_path = str(scene_vid_path/f'{args.test_vid_name}.MP4')
cap = cv2.VideoCapture(video_path)
# input_fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if num_frames < target_num_frames:
  raise RuntimeError(
      'The video is too short and has fewer frames than the target.')


# fps = int(target_num_frames / num_frames * input_fps)
sampling = num_frames // target_num_frames
print(f"Auto-computed sampling = {sampling}")

process_vid(str(scene_vid_path/f'{args.test_vid_name}.MP4'), str(rgb_raw_dir), sampling=sampling, iTarget=-1)


# # @markdown Adjust `max_scale` to something smaller for faster processing.
# max_scale = 1.0  # @param {type:'number'}
# # @markdown A smaller FPS will be much faster for bundle adjustment, but at the expensive of a lower sampling density for training. For the paper we used ~15 fps but we default to something lower here to get you started faster.
# # @markdown If given an fps of -1 we will try to auto-compute it.
# fps = -1  # @param {type:'number'}
# target_num_frames = 100 # @param {type: 'number'}

# cap = cv2.VideoCapture(video_path)
# input_fps = cap.get(cv2.CAP_PROP_FPS)
# num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# if num_frames < target_num_frames:
#   raise RuntimeError(
#       'The video is too short and has fewer frames than the target.')

# if fps == -1:
#   fps = int(target_num_frames / num_frames * input_fps)
#   print(f"Auto-computed FPS = {fps}")

# # @markdown Check this if you want to reprocess the frames.

# filters = f"mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw*{max_scale}:ih*{max_scale}"
# tmp_rgb_raw_dir = str(rgb_raw_dir)
# out_pattern = str(tmp_rgb_raw_dir / f'{args.test_vid_name}_%06d.png')
# cmd(f"mkdir -p {tmp_rgb_raw_dir}")
# cmd(f"ffmpeg -i {video_path} -r $fps -vf $filters {out_pattern}")
# !mkdir -p "$rgb_raw_dir"
# !rsync -av "$tmp_rgb_raw_dir/" "$rgb_raw_dir/"

test_img_names += sorted(list(rgb_raw_dir.glob(f'{args.test_vid_name}_*.jpg')))


############ write split ############
n_test = args.n_test
if len(test_img_names) < n_test:
  print('WARNING: test frames extracted is less than required number of test')
  n_test = len(test_img_names) - 10

left_strip = (len(test_img_names) - n_test) // 2

train_img_names += test_img_names[:left_strip] + test_img_names[left_strip + n_test:]
test_img_names = test_img_names[left_strip:left_strip + n_test]


get_stem = lambda p: p.stem
train_img_names = list(map(get_stem, train_img_names))
test_img_names = list(map(get_stem, test_img_names))

split = {
    'n_imgs': len(train_img_names) + len(test_img_names),
    'n_train': len(train_img_names),
    'n_test': len(test_img_names),
    'train_imgs': train_img_names,
    'test_imgs': test_img_names,
    'id_is_train': [name in train_img_names for name in sorted(train_img_names + test_img_names)]
}

with (root_dir / 'split.json').open('w') as f:
    json.dump(split, f, indent=2)




############ run colmap ############
share_intrinsics = True 
assume_upright_cameras = True 

# @markdown This sets the scale at which we will run COLMAP. A scale of 1 will be more accurate but will be slow.
colmap_rgb_dir = rgb_raw_dir

# @markdown Check this if you want to re-process SfM.
overwrite = True 

if overwrite and colmap_db_path.exists():
  colmap_db_path.unlink()

cmd(f"colmap feature_extractor \
--SiftExtraction.use_gpu 0 \
--SiftExtraction.upright {int(assume_upright_cameras)} \
--ImageReader.camera_model OPENCV \
--ImageReader.single_camera {int(share_intrinsics)} \
--database_path {str(colmap_db_path)} \
--image_path {str(colmap_rgb_dir)}")

match_method = 'exhaustive'  # @param ["exhaustive", "vocab_tree"]

if match_method == 'exhaustive':
    cmd(f"colmap exhaustive_matcher \
        --SiftMatching.use_gpu 0 \
        --database_path {str(colmap_db_path)}")
else:
    # Use this if you have lots of frames.
    cmd("wget https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin")
    cmd(f"colmap vocab_tree_matcher \
        --VocabTreeMatching.vocab_tree_path vocab_tree_flickr100K_words32K.bin \
        --SiftMatching.use_gpu 0 \
        --database_path {str(colmap_db_path)}")

refine_principal_point = True  #@param {type:"boolean"}
min_num_matches = 32 # @param {type: 'number'}
filter_max_reproj_error = 2  # @param {type: 'number'}
tri_complete_max_reproj_error = 2  # @param {type: 'number'}

cmd(f"colmap mapper \
  --Mapper.ba_refine_principal_point {int(refine_principal_point)} \
  --Mapper.filter_max_reproj_error {filter_max_reproj_error} \
  --Mapper.tri_complete_max_reproj_error {tri_complete_max_reproj_error} \
  --Mapper.min_num_matches {min_num_matches} \
  --database_path {str(colmap_db_path)} \
  --image_path {str(colmap_rgb_dir)} \
  --output_path {str(colmap_out_path)}")

if not colmap_db_path.exists():
  raise RuntimeError(f'The COLMAP DB does not exist, did you run the reconstruction?')
elif not (colmap_dir / 'sparse/0/cameras.bin').exists():
  raise RuntimeError("""
SfM seems to have failed. Try some of the following options:
 - Increase the FPS when flattenting to images. There should be at least 50-ish images.
 - Decrease `min_num_matches`.
 - If you images aren't upright, uncheck `assume_upright_cameras`.
""")
else:
  print("Everything looks good!")

cmd(f"colmap model_analyzer --path {colmap_out_path / '0'}")

