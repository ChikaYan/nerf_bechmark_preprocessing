DATA_PATH='./data'
RAW_VID_PATH='./data/raw_video'



SCENE_NAME='calci_museum_whale'
# TRAIN_VIDS='C0252,C0253'
# TEST_VID='C0251'

# SCENE_NAME='calci_museum_giraffe'
# TRAIN_VIDS='C0235,C0237'
# TEST_VID='C0236'

# SCENE_NAME='calci_museum_dinosaur'
# TRAIN_VIDS='C0244,C0246'
# TEST_VID='C0245'

# SCENE_NAME='calci_museum_elephant'
# TRAIN_VIDS='C0240,C0242'
# TEST_VID='C0241'


python process_all.py --data_path $DATA_PATH --raw_vid_path $RAW_VID_PATH \
    --capture_name $SCENE_NAME \
    --train_vid_names $TRAIN_VIDS --test_vid_name $TEST_VID


python gt_vid.py --data_path $DATA_PATH/$SCENE_NAME

mkdir $DATA_PATH/$SCENE_NAME/dense

colmap image_undistorter \
    --image_path $DATA_PATH/$SCENE_NAME/images_raw \
    --input_path $DATA_PATH/$SCENE_NAME/sparse/0 \
    --output_path $DATA_PATH/$SCENE_NAME/dense \
    --output_type COLMAP \
    --max_image_size 2000

python colmap_runner.py --data_path $DATA_PATH/$SCENE_NAME

rm -rf $DATA_PATH/$SCENE_NAME/images_1

cp -r $DATA_PATH/$SCENE_NAME/images $DATA_PATH/$SCENE_NAME/images_1