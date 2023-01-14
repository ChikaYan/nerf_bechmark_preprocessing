from pathlib import Path
import imageio
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)

args = parser.parse_args()


scene_path = Path(args.data_path)

with (scene_path / 'split.json').open('r') as f:
    split_json = json.load(f)
    i_imgs = np.arange(split_json['n_imgs'])
    train_mask = np.array(split_json['id_is_train'])
    i_train = i_imgs[train_mask]
    i_test = i_imgs[~train_mask]

img_dir = scene_path / 'images'
test_gts = []

for i, img_path in enumerate(sorted(list(img_dir.glob('*')))):
    if not train_mask[i]:
        test_gts.append(imageio.imread(str(img_path)))

imageio.mimwrite(str(scene_path / 'test_gt.mp4'), test_gts, fps=30, quality=8)
