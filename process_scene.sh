DATA_PATH='./data'
RAW_VID_PATH='./data/raw_video'
SCENE_NAME='calci_museum_dinosaur'
TRAIN_VIDS='C0231,C0234'
TEST_VID='C0233'

CROP_CY=378 # keep top
CROP_CY=702 # keep bottom

python process_all.py --data_path $DATA_PATH --raw_vid_path $RAW_VID_PATH \
    --capture_name $SCENE_NAME \
    --train_vid_names $TRAIN_VIDS --test_vid_name $TEST_VID

python colmap_runner.py --data_path $DATA_PATH/$SCENE_NAME \
    --crop_center_y $CROP_CY