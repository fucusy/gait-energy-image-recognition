__author__ = 'fucus'

import config
from tool import img_path_to_GEI
from skimage.io import imsave
import logging
import os
logger = logging.getLogger("data")

def load_training_validation_data(train_view=None, val_view=None, train_dir=None, val_dir=None):


    human_id = ["%03d" % i for i in range(1, 125)]

    if train_dir is None:
        train_dir = ["nm-%02d" % i for i in range(1, 5)]
    if val_dir is None:
        val_dir = ["nm-05", "nm-06"]

    if train_view is None:
        train_view = "090"

    if val_view is None:
        val_view = "090"


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
        for dir in train_dir:
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, train_view)
            if not os.path.exists(img_dir):
                logger.warning("%s do not exist" % img_dir)
        for dir in val_dir:
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, val_view)
            if not os.path.exists(img_dir):
                logger.warning("%s do not exist" % img_dir)

    for id in human_id:
        logger.info("processing human %s" % id)
        for dir in train_dir:
            training_y.append(id)
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, train_view)
            training_x.append(img_path_to_GEI(img_dir))
        for dir in val_dir:
            validation_y.append(id)
            img_dir = "%s/%s/%s/%s" % (config.project.casia_dataset_b_path, id, dir, val_view)
            validation_x.append(img_path_to_GEI(img_dir))
    return training_x, training_y, validation_x, validation_y


def output_result(view_list, correct_tbl):

    logger.info("every row means the validation result from different training views")
    logger.info("\t\t" + "\t\t".join(view_list))
    for val_view in view_list:
        output = "%s\t" % val_view
        for train_view in view_list:
            output += "%.2f\t" % correct_tbl["%s-%s" % (train_view, val_view)]
        logger.info(output)

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    view_list = ["%03d" % x for x in range(0, 181, 18)]
    correct_tbl = {}
    for train_view in view_list:
        for val_view in view_list:
            correct_tbl["%s-%s" % (train_view, val_view)] = 0
    output_result(view_list, correct_tbl)

    # training_x, training_y, validation_x, validation_y = load_training_validation_data()
    # count = 0
    # for x in training_x:
    #     count += 1
    #     imsave("%s/%03d.bmp" % (config.project.test_data_path, count), x)