import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
import keras.backend as K
import random 
import numpy as np
import cv2
from preprocessor import _imread as imread
from preprocessor import _imresize as imresize
from matplotlib import pyplot as plt
from keras.losses import categorical_crossentropy
import argparse
import glob
import os

DATA_DIR = "data/UTKFace"
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())

ID_GENDER_MAP, GENDER_ID_MAP, ID_RACE_MAP, RACE_ID_MAP

def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        age, gender, race, _ = filename.split("_")
        return int(age), int(gender), int(race)
    except Exception as e:
        print(filepath)
        return None, None, None

def random_image():
    files = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
    img_path = random.choice(files)

    attributes = parse_filepath(img_path)
    age, gender, race = attributes
    gender_one_hot = np_utils.to_categorical(gender, len(ID_GENDER_MAP))
    race_one_hot = np_utils.to_categorical(race, len(ID_RACE_MAP))
    return img_path, age, gender_one_hot, race_one_hot

def preprocess_image(image, input_shape):
    image = imresize(image, input_shape[:2])
    image = image.astype('float32')
    image = image/255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(model, image):
    predictions = model.predict(image)
    return predictions

def FGSM(x, race_label,model):
    sess = K.get_session()
    x_adv1 = x #+ (K.random_normal(x.shape) * 0.1)
    x_adv2 = x #+ (K.random_normal(x.shape) * 0.1)
    x_adv3 = x #+ (K.random_normal(x.shape) * 0.1)
    x_adv4 = x #+ (K.random_normal(x.shape) * 0.1)
    x_adv5 = x #+ (K.random_normal(x.shape) * 0.1)
    print(list(map(lambda x: x.name, model.layers)))
    print(model.summary())
    dense_out = model.get_layer('dense_3').output
    race_output = model.get_layer('race_output').output
    weights = model.get_layer('race_output').get_weights()
    
    loss = categorical_crossentropy(race_label, race_output)  
    grads = K.gradients(dense_out, model.input)
    delta = K.sign(grads[0])
    x_adv1 = x_adv1 + 0.1 * delta
    x_adv2 = x_adv2 + 0.2 * delta
    x_adv3 = x_adv3 + 0.3 * delta
    x_adv4 = x_adv4 + 0.4 * delta
    x_adv5 = x_adv5 + 0.5 * delta
    
    x_adv1 = K.clip(x_adv1, 0.0 ,1.0)
    x_adv2 = K.clip(x_adv2, 0.0 ,1.0)
    x_adv3 = K.clip(x_adv3, 0.0 ,1.0)
    x_adv4 = K.clip(x_adv4, 0.0 ,1.0)
    x_adv5 = K.clip(x_adv5, 0.0 ,1.0)

    gradients, x_adv1_array, x_adv2_array, x_adv3_array, x_adv4_array, x_adv5_array = sess.run([grads, x_adv1, x_adv2, x_adv3, x_adv4, x_adv5], feed_dict={model.input:x})
    print('GRADIENT SUM:{}'.format(np.sum(gradients[0])))
    return x_adv1_array, x_adv2_array, x_adv3_array, x_adv4_array, x_adv5_array

def plot_adversarial(img_list):
    plt.figure(figsize=(8,8))
    eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for n, img in enumerate(img_list):
        ax = plt.subplot(2,3,n+1)
        ax.set_title('Epsilon: {}'.format(eps[n]))
        plt.imshow(img[0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Save model path')
    #parser.add_argument('--epsilon', type=float, help='Epsilon for adversarial perturbation')

    args = parser.parse_args()
    
    input_shape = (IM_WIDTH, IM_HEIGHT, 3)
    model = load_model(args.model)
    
    img_path, age, gender_one_hot, race_one_hot = random_image()
    print(age)
    print(gender_one_hot)
    print(race_one_hot)
    image = imread(img_path)
    image = preprocess_image(image, input_shape)

    predictions = predict(model, image)
    age_pred, race_pred, gender_pred = predictions

    race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)
    print(age_pred)
    print(gender_pred)
    print(race_pred)

    x_adv1, x_adv2, x_adv3, x_adv4, x_adv5 = FGSM(image, np.expand_dims(race_one_hot, axis=0), model)

    predictions = predict(model, x_adv1)
    age_adv_img, race_adv_img, gender_adv_img = predictions

    race_adv_img, gender_adv_img = race_adv_img.argmax(axis=-1), gender_adv_img.argmax(axis=-1)
    print(age_adv_img)
    print(gender_adv_img)
    print(race_adv_img)

    predictions = predict(model, x_adv5)
    age_adv_img, race_adv_img, gender_adv_img = predictions

    race_adv_img, gender_adv_img = race_adv_img.argmax(axis=-1), gender_adv_img.argmax(axis=-1)
    print(age_adv_img)
    print(gender_adv_img)
    print(race_adv_img)

    img_list = [image, x_adv1, x_adv2, x_adv3, x_adv4, x_adv5]
    plot_adversarial(img_list)

if __name__ == '__main__':
    main()

    
