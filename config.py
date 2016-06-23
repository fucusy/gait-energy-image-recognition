
import os
class project:

    project_path = "/Users/fucus/Documents/irip/gait_recoginition/code/gait-energy-image-recognition"

    # the CASIA gait dataset B path, you can download the data from
    # CASIA website, the dirtory contains a lot sub dirtory
    # named such as 001,002...
    casia_dataset_b_path = "/Users/fucus/Documents/irip/gait_recoginition/data/DatasetB/silhouettes"

    casia_test_img = "%s/001/bg-01/000/001-bg-01-000-030.png" % casia_dataset_b_path

    casia_test_img_dir = "%s/004/nm-01/090/" % casia_dataset_b_path

    test_data_path = "%s/data" % project_path

    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)