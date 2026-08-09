./prepare_groundtruth.py --max_crop_size 2256 1269 dataset/original_train dataset/groundtruth_train
./prepare_groundtruth.py --max_crop_size 2256 1269 dataset/original_test dataset/groundtruth_test
./prepare_groundtruth.py --max_crop_size 752 576 --pad --prefix dataset2/original_train dataset2/groundtruth_train

