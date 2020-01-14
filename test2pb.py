# coding=utf-8
import cv2
import numpy as np
from keras import backend as K
from keras.models import *
from keras.layers import *
import os
import tensorflow as tf

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂",
         u"湘", u"粤", u"桂",
         u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7",
         u"8", u"9", u"A",
         u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U",
         u"V", u"W", u"X",
         u"Y", u"Z", u"港", u"学", u"使", u"警", u"澳", u"挂", u"军", u"北", u"南", u"广", u"沈", u"兰", u"成", u"济", u"海", u"民",
         u"航", u"空"
         ]


def model_seq_rec(model_path):
    width, height, n_len, n_class = 164, 48, 7, len(chars) + 1
    rnn_size = 256
    input_tensor = Input((164, 48, 3))
    x = input_tensor
    base_conv = 32
    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        x)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)
    return base_model


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util, graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    # if log_tensorboard:
    #     from tensorflow.python.tools import import_pb_to_tensorboard
    #     import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


output_dir = './'
model_path = 'model/ocr_plate_all_gru.h5'

h5_model = model_seq_rec(model_path)
h5_to_pb(h5_model, output_dir=output_dir, model_name='frozen_inference.pb')
