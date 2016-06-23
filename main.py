__author__ = 'fucus'
import logging
from data import load_training_validation_data
from model.models import RandomForestClassification
logger = logging.getLogger("main")
from feature.hog import get_hog
from sklearn.metrics import classification_report

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    training_x, training_y, validation_x, validation_y = load_training_validation_data()

    training_feature_x = [get_hog(x) for x in training_x]
    validation_feature_x = [get_hog(x) for x in validation_x]

    model = RandomForestClassification()
    model.fit(x_train=training_feature_x, y_train=training_y)

    predict_y = model.predict(validation_feature_x)
    report = classification_report(predict_y, validation_y)
    logger.info("the validation report:\n %s" % report)