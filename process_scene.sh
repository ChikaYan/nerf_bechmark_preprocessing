DATA_PATH='./data'
RAW_VID_PATH='./data/raw_video'

SCENE_NAME='calci_museum_elephant'
TRAIN_VIDS='C0240,C0242'
TEST_VID='C0241'


python process_all.py --data_path $DATA_PATH --raw_vid_path $RAW_VID_PATH \
    --capture_name $SCENE_NAME \
    --train_vid_names $TRAIN_VIDS --test_vid_name $TEST_VID

python colmap_runner.py --data_path $DATA_PATH/$SCENE_NAME

python gt_vid.py --data_path $DATA_PATH/$SCENE_NAME

cp -r $DATA_PATH/$SCENE_NAME/images $DATA_PATH/$SCENE_NAME/images_1