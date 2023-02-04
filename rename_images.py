from colmap_rw_model import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)

args = parser.parse_args()

model_path = f"{args.data_path}/sparse/0"

cameras, images, points3D = read_model(model_path)
for k in images.keys():
    images[k] =  Image(id=images[k].id, qvec=images[k].qvec, tvec=images[k].tvec,
                            camera_id=images[k].camera_id, name=images[k].name.replace('.jpg', '.png'),
                            xys=images[k].xys, point3D_ids=images[k].point3D_ids)


write_model(cameras, images, points3D, model_path)

