import os
import tensorflow as tf

pb_file_path = '/Users/qiuyurui/PycharmProjects/hyperlpr-train/trans_model/model-ep019-loss2.265-val_loss2.402.pb'

from tensorflow.python.platform import gfile

sess = tf.Session()
# with gfile.FastGFile(pb_file_path + 'model.pb', 'rb') as f:
with gfile.FastGFile(pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name, '\n')

