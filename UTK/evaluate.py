import argparse
import logging
import os
import random
import string

import pandas as pd
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model

from adversary import fgsm
from keras.metrics import categorical_accuracy, mae
import tensorflow as tf
from keras.utils import to_categorical
from PIL import Image
import glob 

DATA_DIR = "data/UTKFace"
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())

logging.basicConfig(level=logging.DEBUG)

def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        age, gender, race, _ = filename.split("_")
        return int(age), ID_GENDER_MAP[int(gender)], ID_RACE_MAP[int(race)]
    except Exception as e:
        print(filepath)
        return None, None, None

def get_adversarial_acc_metric_race(model, eps=0.3, clip_min=0.0, clip_max=1.0):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm(model, y, eps=eps, clip_min=clip_min, clip_max=clip_max)
        # Consider the attack to be constant
        x_adv = K.stop_gradient(x_adv)
        
        # Accuracy on the adversarial examples
        preds_age, preds_race, preds_gender = model(x_adv)
        return categorical_accuracy(y, preds_race)
    return adv_acc

def get_adversarial_acc_metric_age(model, eps=0.3, clip_min=0.0, clip_max=1.0):
    def adv_acc(y, _):
        # Generate adversarial examples
        # get the target tensor name by checking y in get_adversarial_acc_metric_race
        y_race = tf.get_default_graph().get_tensor_by_name("race_output_target_1:0")
        x_adv = fgsm(model, y_race, eps=eps, clip_min=clip_min, clip_max=clip_max)
        # Consider the attack to be constant
        x_adv = K.stop_gradient(x_adv)
        
        # Accuracy on the adversarial examples
        preds_age, preds_race, preds_gender = model(x_adv)
        return mae(y, preds_age)
    return adv_acc

def get_adversarial_acc_metric_gender(model, eps=0.3, clip_min=0.0, clip_max=1.0):
    def adv_acc(y, _):
        # Generate adversarial examples
        y_race = tf.get_default_graph().get_tensor_by_name("race_output_target_1:0")
        x_adv = fgsm(model, y_race, eps=eps, clip_min=clip_min, clip_max=clip_max)
        # Consider the attack to be constant
        x_adv = K.stop_gradient(x_adv)
        
        # Accuracy on the adversarial examples
        preds_age, preds_race, preds_gender = model(x_adv)
        return categorical_accuracy(y, preds_gender)
    return adv_acc

def get_data_generator(df, indices, max_age, for_training, batch_size=16):
    images, ages, races, genders = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, race, gender = r['file'], r['age'], r['race_id'], r['gender_id']
            im = Image.open(file)
            im = im.resize((IM_WIDTH, IM_HEIGHT))
            im = np.array(im) / 255.0
            images.append(im)
            ages.append(age / max_age)
            races.append(to_categorical(race, len(RACE_ID_MAP)))
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break


def get_args():
    parser = argparse.ArgumentParser(description="This script evaluates the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_file", type=str, required=True,
                        help="path to model weights file")
    parser.add_argument("--eps", type=float, required=True,
                        help="epsilon value for adversarial perturbation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    eps = args.eps
    batch_size = args.batch_size
    validation_split = args.validation_split
    model_file = args.model_file

    logging.debug("Loading data...")

    files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    attributes = list(map(parse_filepath, files))

    df = pd.DataFrame(attributes)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    df = df[(df['age'] > 10) & (df['age'] < 65)]
    df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])
    df['race_id'] = df['race'].map(lambda race: RACE_ID_MAP[race])
    max_age = df['age'].max()

    p = np.random.RandomState(10).permutation(len(df))
    print(p)
    train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
    train_idx = p[:train_up_to]
    test_idx = p[train_up_to:]
    train_up_to = int(train_up_to * 0.7)
    train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
    
    model = load_model(model_file)

    image_generator = get_data_generator(df, test_idx, max_age, for_training=False, batch_size=batch_size)

    opt = SGD(lr=0.001)
    adv_acc_metric_race = get_adversarial_acc_metric_race(model, eps=eps)
    adv_acc_metric_age = get_adversarial_acc_metric_age(model, eps=eps)
    adv_acc_metric_gender = get_adversarial_acc_metric_gender(model, eps=eps)

    model.compile(
        optimizer=opt,
        loss={'age_output': 'mse', 'race_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
        loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
        metrics={'age_output': adv_acc_metric_age, 'race_output': adv_acc_metric_race, 'gender_output': adv_acc_metric_gender}
    )
    
    eval_list = model.evaluate_generator(image_generator,
            steps=int(len(test_idx) / batch_size))

    print(eval_list)

    del(model)
    K.clear_session()
 
    image_generator2 = get_data_generator(df, test_idx, max_age, for_training=False, batch_size=batch_size)

    model = load_model(model_file)
    #model.load_weights(model_weights)
    model.compile(
        optimizer=opt,
        loss={'age_output': 'mse', 'race_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
        loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
        metrics={'age_output': 'mae', 'race_output': 'accuracy', 'gender_output': 'accuracy'}
    )

    eval_list = model.evaluate_generator(image_generator2,
            steps=int(len(test_idx) / batch_size))

    print(eval_list)


if __name__ == '__main__':
    main()
