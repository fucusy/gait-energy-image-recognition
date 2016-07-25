__author__ = 'fucus'
import logging
from data_tool import load_training_validation_data
from model.models import RandomForestClassification
from feature.hog import get_hog
import data_tool
from sklearn.metrics import classification_report

logger = logging.getLogger("main")
if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    view_list = ["%03d" % x for x in range(0, 181, 18)]
    train_dir = ["nm-%02d" % i for i in range(1, 5)]
    val_dir = ["nm-05", "nm-06"]

    # "{train_view}-{val_view}" as key, "090-072" means 090 as train data, 072 as validation data
    correct_tbl = {}

    for train_view in view_list:
        for val_view in view_list:
            training_x, training_y, validation_x, validation_y = load_training_validation_data(train_view=train_view
                                                                                               , val_view=val_view
                                                                                               , train_dir=train_dir
                                                                                               , val_dir=val_dir)
            training_feature_x = [get_hog(x) for x in training_x]
            validation_feature_x = [get_hog(x) for x in validation_x]

            model = RandomForestClassification()
            model.fit(x_train=training_feature_x, y_train=training_y)
            predict_y = model.predict(validation_feature_x)
            correct_count = sum(predict_y == validation_y)
            correct_percent = correct_count * 1.0 / len(predict_y)
            correct_tbl["%s-%s" % (train_view, val_view)] = correct_percent
            logger.info("final precision %d/%d %.3f" % (correct_count, len(predict_y), correct_percent))

    data_tool.output_result(view_list, correct_tbl)