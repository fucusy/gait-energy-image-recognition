__author__ = 'fucus'

import config
from tool import img_path_to_GEI
from skimage.io import imsave
import logging
import os

logger = logging.getLogger("data")

def load_training_validation_data():
    human_id = ["%03d" % i for i in range(1, 125)]
    training_dir = ["nm-%02d" % i for i in range(1, 6)]
    validation_dir = ["nm-06"]
    view = "090"
    # remove broken data
    broken_data_list = ["034", "046", "067", "064", "068"]
    for data in broken_data_list:
        human_id.remove(data)

    training_x = []
    training_y = []

    validation_x = []
    validation_y = []
    # check dir exists
    for id in human_id:
        for dir in training_dir:
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, view)
            if not os.path.exists(img_dir):
                logger.warning("%s do not exist" % img_dir)
        for dir in validation_dir:
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, view)
            if not os.path.exists(img_dir):
                logger.warning("%s do not exist" % img_dir)

    for id in human_id:
        for dir in training_dir:
            training_y.append(id)
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, view)
            logger.info("processing dir %s" % img_dir)
            training_x.append(img_path_to_GEI(img_dir))
        for dir in validation_dir:
            validation_y.append(id)
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, view)
            logger.info("processing dir %s" % img_dir)
            validation_x.append(img_path_to_GEI(img_dir))
    return training_x, training_y, validation_x, validation_y

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    training_x, training_y, validation_x, validation_y = load_training_validation_data()
    count = 0
    for x in training_x:
        count += 1
        imsave("%s/%03d.bmp" % (config.project.test_data_path, count), x)