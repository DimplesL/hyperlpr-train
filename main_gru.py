# encoding: utf-8
import os
import argparse
import cv2
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import *
from keras.layers import *
import glob

import tensorflow as tf

from ctypes import *

CHARS = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂",
         u"湘", u"粤", u"桂",
         u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7",
         u"8", u"9", u"A",
         u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U",
         u"V", u"W", u"X",
         u"Y", u"Z", u"港", u"学", u"使", u"警", u"澳"]

CHARS_HAN = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂",
             u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"港", u"学", u"使", u"警", u"澳"]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
NUM_CHARS = len(CHARS)
indexstart = 0


# The actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, :, 0, :]
    y_pred = y_pred[:, indexstart:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(width, height, num_channels, n_class=NUM_CHARS + 1, ISTRAIN=True):
    rnn_size = 256
    input_tensor = Input(name='xinput', shape=(width, height, num_channels), dtype='float32')
    # input_tensor = Input((164, 48, 3))
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
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    return input_tensor, x


def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label


def parse_line(line):
    parts = line.split(':')
    filename = parts[0]
    try:
        label = encode_label(parts[1].strip().upper())
        return filename, label
    except Exception as e:
        return None, None


class ProImg:
    def __init__(self):
        pass

    def white_balance(self, img):
        (G, B, R) = cv2.split(img)

        G_mean = cv2.mean(G)[0]
        B_mean = cv2.mean(B)[0]
        R_mean = cv2.mean(R)[0]

        # k = (G_mean + B_mean + R_mean) / 3
        k = 100

        kg = k / (G_mean + 0.000001)
        kb = k / (B_mean + 0.000001)
        kr = k / (R_mean + 0.000001)

        G_ = cv2.addWeighted(G, kg, 0, 0, 0)
        B_ = cv2.addWeighted(B, kb, 0, 0, 0)
        R_ = cv2.addWeighted(R, kr, 0, 0, 0)
        img_ = cv2.merge([G_, B_, R_])
        return img_

    def process_addwhite(self, img):
        try:
            rnd = np.random.randint(0, 10, 1)[0]
            if rnd > 5:
                h, w = img.shape[0:2]
                posy = np.random.randint(0, h - 1, 1)[0]
                posx = np.random.randint(0, w - 1, 1)[0]

                bh = np.random.randint(1, h - posy, 1)[0]
                bw = np.random.randint(1, w - posx, 1)[0]

                whiteimg = np.random.randint(160, 255, (bh, bw, 3))
                img_clip = img[posy:posy + bh, posx:posx + bw, :]
                img_merge = whiteimg * 0.5 + img_clip * 0.5
                img_merge = img_merge.astype(np.uint8)
                img[posy:posy + bh, posx:posx + bw, :] = img_merge
        except Exception as eex:
            print(eex)
        return img

    def process(self, img):
        img = self.white_balance(img)
        img_norm = (img - 127.0) / 128.0
        return img_norm


class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, input_length, num_channels=3, label_len=5):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._input_len = input_length
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()

    def init(self):
        self.labels = []
        with open(self._label_file) as f:
            for line in f:
                filename, label = parse_line(line)
                if filename is None:
                    continue
                if len(label) != self._label_len:
                    continue

                self.filenames.append(filename)
                self.labels.append(label)
                self._num_examples += 1
        self.labels = np.float32(self.labels)

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end >= self._num_examples:
            self._next_index = 0
            self._num_epoches += 1
            end = self._num_examples
            batch_size = self._num_examples - start
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])
        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            img = cv2.imread(os.path.join(self._img_dir, fname))
            h, w = img.shape[:2]
            if h != self._img_h:
                continue
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        input_length[:] = self._input_len
        label_length[:] = self._label_len
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': images,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        return inputs, outputs

    def get_data(self):
        while True:
            yield self.next_batch()


def train(args):
    ckpt_dir = os.path.dirname(args.ck)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.log != '' and not os.path.isdir(args.log):
        os.makedirs(args.log)
    label_len = args.label_len

    input_tensor, y_pred = build_model(args.img_size[0], args.img_size[1], args.num_channels)

    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    pred_length = int(y_pred.shape[1])
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pre, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # model.load_weights('./model_h5/ocr_plate_all_gru.h5', by_name=True, skip_mismatch=True)

    train_gen = TextImageGenerator(img_dir=args.ti,
                                   label_file=args.tl,
                                   batch_size=args.b,
                                   img_size=args.img_size,
                                   input_length=pred_length,
                                   num_channels=args.num_channels,
                                   label_len=label_len)

    val_gen = TextImageGenerator(img_dir=args.vi,
                                 label_file=args.vl,
                                 batch_size=args.b,
                                 img_size=args.img_size,
                                 input_length=pred_length,
                                 num_channels=args.num_channels,
                                 label_len=label_len)

    # checkpoints_cb = ModelCheckpoint(args.c, period=1)
    # cbs = [checkpoints_cb]

    # if args.log != '':
    #     tfboard_cb = TensorBoard(log_dir=args.log, write_images=True)
    #     cbs.append(tfboard_cb)

    filepath = ckpt_dir + '/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=(train_gen._num_examples + train_gen._batch_size - 1) // train_gen._batch_size,
                        epochs=args.n,
                        validation_data=val_gen.get_data(),
                        validation_steps=(val_gen._num_examples + val_gen._batch_size - 1) // val_gen._batch_size,
                        callbacks=[checkpoint],
                        initial_epoch=args.start_epoch)


def export(args):
    """Export the model to a hdf5 file
    """
    input_tensor, y_pred = build_model(args.img_size[0], args.img_size[1], args.num_channels, ISTRAIN=False)
    model = Model(inputs=input_tensor, outputs=y_pred)
    model.load_weights('model_inc——2019-08-01.h5')
    model.save(args.m)

    print(model.output.op.name)
    print(model.input.op.name)
    sess = K.get_session()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=[model.output.op.name])
    tf.train.write_graph(frozen_graph_def, 'pbmodel', 'rec_inc_20190801.pb', as_text=False)
    print('model saved to {}'.format(args.m))


def showpb(pb):
    import tensorflow as tf
    with tf.Session() as sess:
        with open(pb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)


def main():
    parser_train = argparse.ArgumentParser()
    parser_train.add_argument('-num_channels', type=int, help='number of channels of the image', default=3)

    parser_train.add_argument('--ti', help='训练图片目录', default='/home/vip/qyr/data/ocr_data/train')
    parser_train.add_argument('--tl', help='训练标签文件', default='/home/vip/qyr/data/ocr_data/train_labels.txt')
    parser_train.add_argument('--vi', help='验证图片目录', default='/home/vip/qyr/data/ocr_data/val')
    parser_train.add_argument('--vl', help='验证标签文件', default='/home/vip/qyr/data/ocr_data/val_labels.txt')
    parser_train.add_argument('--b', type=int, help='batch size', default=16)
    parser_train.add_argument('--img_size', type=int, nargs=2, help='训练图片宽和高', default=(164, 48))
    parser_train.add_argument('--pre', help='pre trained weight file', default='')
    parser_train.add_argument('--start_epoch', type=int, default=0)
    parser_train.add_argument('--n', type=int, help='number of epochs', default=50)
    parser_train.add_argument('--label_len', type=int, help='标签长度', default=7)
    parser_train.add_argument('--ck', help='checkpoints format string', default='./models_gru', type=str)
    parser_train.add_argument('--log', help='tensorboard 日志目录, 默认为空', default='./logs_gru')

    args = parser_train.parse_args()

    train(args)


if __name__ == '__main__':
    main()

