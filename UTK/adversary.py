import tensorflow as tf
import keras.backend as K

def fgsm(model, race_labels, eps=0.3, clip_min=0.0, clip_max=1.0):
    x = model.get_input_at(0)
    age_output, race_output, gender_output  = model(x)
    #y_race = tf.get_default_graph().get_tensor_by_name("race_output_target:0")
    race_loss = K.categorical_crossentropy(race_labels, race_output)
   
    dense_out = model.get_layer('dense_3').output

    #if adversary to be generated using gradients of inputs wrt race loss,
    #replace dense_out by race_loss in below line 
    grads = K.gradients(dense_out, x)
    delta = K.sign(grads[0])
    x_adv = x + eps*delta
    x_adv = K.clip(x_adv, clip_min, clip_max)

    return x_adv


    
