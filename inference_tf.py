"""
create a py file for Tensorflow Inference
"""
# coding: utf-8

import argparse
parser = argparse.ArgumentParser(description='input selfDefinedFileName')
parser.add_argument('--selfDefinedFileName','-n', type=str, 
                    help='an self-defined name for an inference')
args = parser.parse_args()

selfDefinedFileName=args.selfDefinedFileName


str_ = """
import tensorflow as tf
import PIL.Image as Image
import numpy as np
from imagenet1000_clsid_to_human import cls_dict
from keras_param import models , shapes
"""

str_ += """
def get_model(model):
    if model== "inception_v3":
        from gen_code.{}_tensorflow_inception_v3 import KitModel
    elif model== "vgg16":
        from gen_code.{}_tensorflow_vgg16 import KitModel
    elif model== "vgg19":
        from gen_code.{}_tensorflow_vgg16 import KitModel
    elif model== "resnet":
        from gen_code.{}_tensorflow_resnet import KitModel
    elif model== "mobilenet":
        from gen_code.{}_tensorflow_mobilenet import KitModel
    elif model== "xception":
        from gen_code.{}_tensorflow_xception import KitModel
    else:
        return""".format(*[selfDefinedFileName]*6)

str_ += """
    npy_path="gen_pb_json_npy/{}_%s.npy" % model
    ckpt_path='gen_model/{}_tf_%s.ckpt' % model
    return KitModel, npy_path, ckpt_path
    """.format(*[selfDefinedFileName]*2)

str_ += """
def inference( img_path='cat1.jpeg', model="inception_v3" ):
    KitModel, npy_path, ckpt_path = get_model(model)
    inp, oup = KitModel(npy_path)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    im = np.array(Image.open( img_path ).resize(shapes[model]))/float(255)
    img = np.expand_dims(im,0) # just one fig
    oup_ = sess.run(oup, feed_dict={inp:img})
    oup_ = oup_.squeeze()
    obj_idx = np.argmax(oup_)
    idxs = np.argsort(oup_.squeeze())[-5:]
    str_ = ""
    for i in reversed(idxs):
        str_ += "{:05.2f}% : {}\\n".format(oup_[i]*100, cls_dict[i])
    sess.run(tf.global_variables_initializer()) # 
    tf.reset_default_graph()
    print ("Model: {}, ImgPath: {}".format(model, img_path))
    #print (str_)
    return str_"""



with open("./{}_inference_tf.py".format(selfDefinedFileName), "wb") as f:
    f.write(str_)
f.close()







