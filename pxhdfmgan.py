import os, sys, time
import tensorlayer as tl
from glob import glob
import time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import numpy as np
import tensorflow as tf
import argparse
import cv2
sys.path.insert(0,"scd")
import scd
import skimage.io

import tflib
import tflib.plot
import tflib.save_images
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
import p2phd
import fmgan
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import Activation, BatchNormalization, add, Reshape
from tensorflow.python.keras.layers import DepthwiseConv2D
slim = tf.contrib.slim
from tensorflow.python.keras import backend as K

#mean_vec=np.load("/workspace/natural-adversary/image/mean.npz")["arr_0"]
mean_vec=np.load("mean.npz")["arr_0"]

def relu6(x):
    return tf.keras.backend.relu(x, maxval=6)
def pad(x, p, mode="REFLECT"):
    if isinstance(p, int):
        h=p
        w=p
    elif len(p)==2:
        h=p[0]
        w=p[1]
    else:
        raise NotImplementedError
    return tf.pad(x, ((0,0),(h,h),(w,w),(0,0)),mode=mode)

def l1loss(x,y):
    return tf.reduce_mean(tf.abs(x-y))

def mseloss(x,y):
    return tf.reduce_mean(tf.square(x-y))

NORMALIZE_CONST=201.
TRANSPOSE=False
act_fn_switch=tf.nn.leaky_relu
initializer = tf.random_normal_initializer(stddev=0.02)
initializer2 = tf.random_normal_initializer(mean=0.,stddev=0.02)

class pxhdfmgan(object):
    def __init__(self, fw=31, fh=10, fc=512, iw=496, ih=150, ic=3, z_dim=64, nf=64,  batch_size=80,
                 output_path='./',args=None):
        self.num_D = 2
        self.n_layers=3
        self.lambda_feat=10.
        self.n_blocks = 9
        self.ngf=64
        self.ndf=64
        self.z_dim = z_dim
        self.iw = iw
        self.ih = ih
        self.ic = ic
        self.fw = fw
        self.fh = fh
        self.fc = fc

        self.nf = nf
        self.batch_size = batch_size
        self.output_path = output_path
        self.args=args

        #self.gen_params = self.dis_params = self.inv_params = None

     
        self.gamma_plh = tf.placeholder(tf.float32, shape=(), name='gammaplh')
        self.images= tf.placeholder(tf.float32, shape=[None, self.ih, self.iw, self.ic])
        self.z_noise = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        #self.z_noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])

    def build_model(self):

        self.images_real = (self.images/255.-0.5)/0.5
        self.scd_real = scd.SCD(input=self.images)
        self.feature_real = tf.image.resize_nearest_neighbor(self.scd_real.end_points["pool5"],(self.ih, self.iw))

        self.fmgan_real = fmgan.fmgan(
                            x_dim=784, z_dim=args.z_dim, w=self.fw, h=self.fh, c=self.fc, latent_dim=args.latent_dim,
                            nf=256, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
                            output_path=args.output_path,args=args)

        self.pxhdgan = p2phd.pxhdgan(
                            x_dim=784, z_dim=args.z_dim, w=self.iw, h=self.ih, c=self.ic, latent_dim=args.latent_dim,
                            nf=256, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
                            output_path=args.output_path,args=args)

        self.z_rec = self.fmgan_real.invert(self.feature_real)
        self.feature_rec = self.fmgan_real.generate(self.z_rec)
        self.images_rec = self.pxhdgan.global_generate(self.feature_rec)
        self.images_out = tf.clip_by_value((self.images_rec+1.)/2.*255.,0,255)

        self.feature_fake_noise = tf.image.resize_nearest_neighbor(self.fmgan_real.generate(self.z_noise, reuse=True),(self.ih, self.iw))
        self.images_fake_noise = self.pxhdgan.global_generate(self.feature_fake_noise, reuse=True)
        self.images_fake_noise_out = tf.clip_by_value((self.images_fake_noise+1.)/2.*255.,0,255)
        self.scd_fake_noise = scd.SCD(input=self.images_fake_noise_out,reuse=True)

        self.feature_fake = tf.image.resize_nearest_neighbor(self.scd_fake_noise.end_points["pool5"], (self.ih, self.iw))
        self.z_rec_fake = self.fmgan_real.invert(self.feature_fake, reuse=True)
        self.feature_rec_fake = tf.image.resize_nearest_neighbor(self.fmgan_real.generate(self.z_rec_fake, reuse=True), (self.ih, self.iw))
        self.images_rec_fake = self.pxhdgan.global_generate(self.feature_rec_fake, reuse=True)
        self.images_rec_fake_out = tf.clip_by_value((self.images_rec_fake+1.)/2.*255.,0,255)
        self.scd_rec_fake = scd.SCD(input=self.images_rec_fake_out, reuse=True)

        self.images_fake = self.pxhdgan.global_generate(self.feature_real, reuse=True)
        self.images_fake_out = tf.clip_by_value((self.images_fake+1.)/2.*255.,0,255)
        self.scd_fake = scd.SCD(input=self.images_fake_out, reuse=True)
        self.images_fake_uint8 = tf.cast(self.images_fake_out,tf.uint8)

        #self.images_fake_uint8 = tf.cast(self.images_fake_float,tf.uint8)

        concat_images_fake_noise = tf.concat([self.images_fake_noise, self.feature_fake_noise], axis=3)
        concat_images_rec_fake = tf.concat([self.images_rec_fake, self.feature_fake_noise], axis=3)
        concat_images_fake = tf.concat([self.images_fake, self.feature_real], axis=3)
        concat_images_real = tf.concat([self.images_real, self.feature_real], axis=3)

        self.pred_fake_noise = self.pxhdgan.multi_discriminate(concat_images_fake_noise)
        self.pred_fake = self.pxhdgan.multi_discriminate(concat_images_fake, reuse=True)
        self.pred_rec_fake = self.pxhdgan.multi_discriminate(concat_images_rec_fake, reuse=True)
        self.pred_real = self.pxhdgan.multi_discriminate(concat_images_real, reuse=True)

        #define losses
        #disc losses
        def dc(preds,oz):
            assert(oz == 1 or oz == 0)
            loss = 0
            tfoz = tf.ones_like if oz == 1 else tf.zeros_like
            for p in preds:
                pred = p[-1]
                loss += mseloss(pred, tfoz(pred))
            return loss

        self.dis_cost = dc(self.pred_fake_noise, 0) + \
                        dc(self.pred_fake, 0) + \
                        dc(self.pred_rec_fake, 0) + \
                        dc(self.pred_real, 1) * 0.5

        #gen losses
        self.gen_only_cost = dc(self.pred_fake_noise, 1) + \
                            dc(self.pred_fake, 1) + \
                            dc(self.pred_rec_fake, 1)

        self.gan_feat_cost = 0
        feat_weights = 4.0 / (self.n_layers + 1)
        D_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(self.pred_fake[i])-1):
                self.gan_feat_cost += D_weights * feat_weights * \
                    (l1loss(self.pred_fake[i][j], self.pred_real[i][j]) + \
                    l1loss(self.pred_fake_noise[i][j], self.pred_rec_fake[i][j]))# * self.lambda_feat

        self.vgg_feat_cost = 0
        #weights=[1./1.6, 1./2.3, 1./1.8, 1./2.8, 10/0.8]
        weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        
        for i in range(len(weights)):
            self.vgg_feat_cost += weights[i] * \
            (l1loss(self.scd_real.end_points["conv%d_1"%(i+1)], self.scd_fake.end_points["conv%d_1"%(i+1)]) +
            l1loss(self.scd_fake_noise.end_points["conv%d_1"%(i+1)], self.scd_rec_fake.end_points["conv%d_1"%(i+1)]))
               

        self.gen_cost = self.gen_only_cost + self.gan_feat_cost + self.vgg_feat_cost + l1loss(self.z_noise, self.z_rec_fake)*0.1



        """
        disc_reg = self.Discriminator_Regularizer(self.dis_x, self.x, self.dis_x_p, self.x_p)
        #for more than one fake dis...
        assert self.dis_cost.shape == disc_reg.shape
        self.dis_cost += (self.gamma_plh/2.0)*disc_reg
        """

        
        train_vars=tf.trainable_variables()
        self.gen_params=[v for v in train_vars if v.name.split("/")[1] == "global_generate"]
        self.dis_params=[v for v in train_vars if v.name.split("/")[1] == "multi_discriminate"]

        with tf.variable_scope('pix2pixhd'):
            genopt = tf.train.AdamOptimizer(
                learning_rate=2e-6, beta1=0.5, beta2=0.999)
            self.gen_train_op= slim.learning.create_train_op(self.gen_cost,genopt,summarize_gradients=True,variables_to_train=self.gen_params)
            disopt = tf.train.AdamOptimizer(
                learning_rate=2e-6, beta1=0.5, beta2=0.999)
            self.dis_train_op= slim.learning.create_train_op(self.dis_cost,disopt,summarize_gradients=True,variables_to_train=self.dis_params)

    def Discriminator_Regularizer(self,D1_logits, D1_arg, D2_logits, D2_arg):
        with tf.variable_scope('disc_reg'):
            D1 = tf.nn.sigmoid(D1_logits)
            D2 = tf.nn.sigmoid(D2_logits)

            grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
            grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
            grad_D1_logits_norm = tf.norm( tf.reshape(grad_D1_logits, [self.batch_size,-1]) , axis=1)
            grad_D2_logits_norm = tf.norm( tf.reshape(grad_D2_logits, [self.batch_size,-1]) , axis=1)

            

            #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
            grad_D1_logits_norm=tf.expand_dims(grad_D1_logits_norm,1)
            grad_D2_logits_norm=tf.expand_dims(grad_D2_logits_norm,1)
            print('grad_D1_logits_norm.shape {} != D1.shape {}'.format(grad_D1_logits_norm.shape, D1.shape))
            print('grad_D2_logits_norm.shape {} != D2.shape {}'.format(grad_D2_logits_norm.shape, D2.shape))
            assert grad_D1_logits_norm.shape == D1.shape
            assert grad_D2_logits_norm.shape == D2.shape
            
            reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
            reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
            disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
            
            return disc_regularizer
  


    def _residual_block(self, X, dim, norm_layer=tf.contrib.layers.instance_norm, activation=tf.nn.relu, name='p2phd_res_block'):
        with tf.variable_scope(name):

                net = pad(X, 1)
                net = slim.conv2d(net,dim ,activation_fn=None,  weights_initializer=initializer, kernel_size=(3,3), padding="VALID") 
                net = norm_layer(net, trainable=True)
                net = activation(net)
                net = pad(X, 1)
                net = slim.conv2d(net, dim, activation_fn=None,  weights_initializer=initializer, kernel_size=(3,3), padding="VALID") 
                net = norm_layer(net, trainable=True)

                return net + X

    def global_generate(self, label, norm_layer=tf.contrib.layers.instance_norm, activation=tf.nn.relu, reuse=False):
        with tf.variable_scope('pix2pixhd/global_generate', reuse=reuse):
            n_downsampling=4

            net = pad(label, 3)
            net = slim.conv2d(net,self.ngf, activation_fn=None,  weights_initializer=initializer,  kernel_size=(7,7), padding="VALID") 
            print("1 reflect conv", net)
            net = norm_layer(net, trainable=True)
            net = activation(net)

            for i in range(n_downsampling):
                mult = 2**i
                net = pad(net, 1, mode="CONSTANT")
                net = slim.conv2d(net,self.ngf*mult*2, activation_fn=None,  weights_initializer=initializer, kernel_size=(3,3), stride=(2,2), padding="VALID") 
                print("%d conv roop"%i, net)
                net = norm_layer(net, trainable=True)
                net = activation(net)

            mult = 2**n_downsampling
            for i in range(self.n_blocks):
                net = self._residual_block(net, self.ngf*mult, name="p2phd_resblock%d"%i)
                print("%d resnet roop"%i, net)
            
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                #net = pad(net, 1, mode="CONSTANT")
                net = slim.conv2d_transpose(net,self.ngf*mult//2, activation_fn=None,  weights_initializer=initializer,  kernel_size=(3,3), stride=(2,2), padding="SAME")
                #print("%d transconv roop"%i, net)
                if i == 0 or i == 2:
                    net = net[:,1:,:,:]
                #net = pad(net, (0,0) if i == 1 or i == 3 else (0,0), mode="CONSTANT")
                net = norm_layer(net, trainable=True)
                print("%d transconv pad roop"%i, net)
                net = activation(net)

            net = pad(net, 3)
            net = slim.conv2d(net, 3,activation_fn=None,  weights_initializer=initializer,  kernel_size=(7,7), padding="VALID") 
            print("last reflect conv", net)
            net = tf.nn.tanh(net)

            return net

    def multi_discriminate(self, X, norm_layer=tf.contrib.layers.instance_norm, activation=tf.nn.relu, reuse=False):
        #with tf.variable_scope('multi_discriminate', reuse=reuse):
        with tf.variable_scope('pix2pixhd/multi_discriminate',reuse=reuse):
            res=[]
            for i in range(self.num_D):
                print(i)
                res.append(self.nlayer_discriminate(X, name="nlayer_discriminate%d"%i, reuse=reuse))
                #res.append(self.nlayer_discriminate(X, reuse=None))
                if i!=(self.num_D-1):
                    X = tf.nn.avg_pool(X,strides=[1,2,2,1], ksize=[1,2,2,1],padding="VALID")
                    X = pad(X, 1, mode="CONSTANT")
        return res



    def nlayer_discriminate(self,net,name="nlayer_discriminate", norm_layer=tf.contrib.layers.instance_norm, reuse=False):
        #with tf.variable_scope(name):
        #with tf.variable_scope(tf.get_variable_scope()):
        with tf.variable_scope(name, reuse=reuse):
            res=[]
            kw = 4
            padw = int(np.ceil((kw-1.0)/2))
            #net = pad(X, padw, mode="CONSTANT")
            net = slim.conv2d(net,self.ndf, activation_fn=None,  weights_initializer=initializer,  kernel_size=(kw,kw), stride=(2,2), padding="VALID")
            net=tf.nn.leaky_relu(net)
            res.append(net)
            nf = self.ndf
            for n in range(1, self.n_layers):
                nf_prev = nf
                nf = min(nf * 2, 512)
                #net = pad(X, padw, mode="CONSTANT")
                net=slim.conv2d(net,nf, activation_fn=None,  weights_initializer=initializer, kernel_size=(kw,kw), stride=(2,2), padding="VALID")
                net = norm_layer(net, trainable=True)
                net = tf.nn.leaky_relu(net)
                res.append(net)
            nf_prev = nf
            nf = min(nf * 2, 512)
            #net = pad(X, padw, mode="CONSTANT")
            net=slim.conv2d(net,nf, activation_fn=None,  weights_initializer=initializer, kernel_size=(kw,kw), padding="VALID")
            net = norm_layer(net, trainable=True)
            net = tf.nn.leaky_relu(net)
            res.append(net)
            #net = pad(X, padw, mode="CONSTANT")
            net=slim.conv2d(net,1, activation_fn=None,  weights_initializer=initializer,  kernel_size=(kw,kw), padding="VALID")
            res.append(net)
            return res



    def train_gen(self, sess, x, noise, summary=False):
        if summary:
            _gen_cost, _, summary = sess.run([self.gen_cost, self.gen_train_op, self.merge],
                                    feed_dict={self.x: x, self.z: z})
            return _gen_cost, summary
        else:
            _gen_cost, go, gf, vf, _ = sess.run([self.gen_cost, self.gen_only_cost, self.gan_feat_cost, self.vgg_feat_cost, self.gen_train_op],
                                    feed_dict={self.images: x, self.z_noise:noise})
            return _gen_cost, go, gf, vf

    def train_dis(self, sess, x, noise, gamma, summary=False):
        if summary:
            print(gamma)
            _dis_cost, _, summary = sess.run([self.dis_cost, self.dis_train_op, self.merge],
                  feed_dict={self.real_image: x, self.gamma_plh: gamma, self.z_noise: noise})
            return _dis_cost, summary
        else:
            _dis_cost, _= sess.run([self.dis_cost, self.dis_train_op],
                                    feed_dict={self.images: x, self.gamma_plh: gamma, self.z_noise:noise})
            return _dis_cost


    def reconstruct_images(self, sess, images, frame):
        os.makedirs(os.path.join(self.args.exdir,str(frame)))
        for i in range(8):
            reconstructions,oris = sess.run([self.images_fake_uint8, self.images], feed_dict={self.images: np.expand_dims(images[i],0)})
            skimage.io.imsave(os.path.join(self.args.exdir,str(frame),"%d.jpg"%i),reconstructions[0])
            #skimage.io.imsave(os.path.join(self.args.exdir,"%d_%d_ori.jpg"%(frame,i)),np.uint8(oris[0]))
        """
        comparison = np.zeros((images.shape[0] * 2, images.shape[1],images.shape[2],images.shape[3]),
                              dtype=np.float32)

        for i in range(images.shape[0]):
            comparison[2 * i] = images[i]
            comparison[2 * i + 1] = samplesnpz[i]


        tflib.save_images.save_images(
            comparison,
            #comparison.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST,
            os.path.join(self.output_path, '{}/recs_{}.png'.format(self.args.exdir,frame)))
        return comparison
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--z_dim', type=int, default=1024, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=1024,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=2,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=.1,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    parser.add_argument('--logdir', type=str, default='./logr',
                        help='log dir')
    parser.add_argument('--dataset_test_path', type=str, default='/workspace2/kitti/testing/image_2',
                        help='dataset path')
    parser.add_argument('--dataset_train_path', type=str, default='/workspace2/kitti/training/image_2',
                        help='dataset path')
    parser.add_argument('--nf', type=int, default=128,
                        help='ooooooooooo')
    parser.add_argument('--ic', type=int, default=3,
    #parser.add_argument('--input_c', type=int, default=512,
                        help='ooooooooooo')
    parser.add_argument('--iw', type=int, default=496,
    #parser.add_argument('--input_w', type=int, default=31,
                        help='ooooooooooo')
    parser.add_argument('--ih', type=int, default=150,
    #parser.add_argument('--uuukk', type=int, default=10,
                        help='ooooooooooo')
    parser.add_argument('--fc', type=int, default=512,
    #parser.add_argument('--input_c', type=int, default=512,
                        help='ooooooooooo')
    parser.add_argument('--fw', type=int, default=31,
    #parser.add_argument('--input_w', type=int, default=31,
                        help='ooooooooooo')
    parser.add_argument('--fh', type=int, default=10,
                        help='ooooooooooo')
    parser.add_argument("--gamma",type=float,default=0.1,help="noise variance for regularizer [0.1]")
    parser.add_argument("--annealing", type=bool,default=False, help="annealing gamma_0 to decay_factor*gamma_0 [False]")
    parser.add_argument("--decay_factor",type=float, default=0.01, help="exponential annealing decay rate [0.01]")
    parser.add_argument("--doubleres",type=bool, default=False, help="exponential annealing decay rate [0.01]")
    parser.add_argument('--exdir', type=str, default='examples_unified')
    parser.add_argument('--mddir', type=str, default='models_unified')
    args = parser.parse_args()


    fixed_images = np.zeros([8,args.ih,args.iw,args.ic])
    for i in range(8):
        fixed_images[i,:,:,:]=cv2.resize(scd.imread_as_jpg(os.path.join(args.dataset_test_path,"%06d.png"%(i+7481))),(args.iw,args.ih))
    fixed_noise = np.random.randn(args.batch_size, args.z_dim)
    
    mnistWganInv = pxhdfmgan(
        z_dim=args.z_dim, iw=args.iw, ih=args.ih, ic=args.ic, 
        nf=256, batch_size=args.batch_size, 
        output_path=args.output_path,args=args)
    mnistWganInv.build_model()

    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
    p2phd_var_list=[v for v in train_vars if v.name.split("/")[0] == "pix2pixhd"]
    saver = tf.train.Saver(max_to_keep=10, var_list=p2phd_var_list)


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        """
        train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
        ssd_var_list=[v for v in train_vars if v.name.split("/")[0] == "ssd_300_vgg"]
        scd_saver = tf.train.Saver(var_list=ssd_var_list)
        """
        scd_saver = scd.get_saver()
        scd_saver.restore(session,"/workspace/imgsynth/ssd_300_kitti/ssd_model.ckpt")
        fmgan_saver = fmgan.get_saver()
        fmgan_saver.restore(session,"/workspace/natural-adversary/image/models_doubleres_constr_x_rzx/model-80999")
        p2phd_saver = p2phd.get_saver()
        p2phd_saver.restore(session,"/workspace/p2phd_tf/models/model-716999")

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        data_files = glob(os.path.join(args.dataset_train_path, "*.png"))
        data_files = sorted(data_files)
        data_files = np.array(data_files) 
        iteration=0
        npzs = {}
        batchmx = np.zeros([args.batch_size,args.ih,args.iw,args.ic])
        print("pre epoch")
        """
        with tf.name_scope("summary"):
            tf.summary.scalar("gen_cost",mnistWganInv.gen_cost)
            tf.summary.scalar("dis_cost",mnistWganInv.dis_cost)
            mnistWganInv.merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter(args.logdir,session.graph)
        """

        summary=False
        for epoch in range(args.epoch):
            try:
                minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=args.batch_size, shuffle=True)
                while True:
                    pretime=time.time()
                    if iteration % 100 == 99:
                        summary=False
                    else:
                        summary=False
                    iteration+=1
                    if args.annealing:
                        gamma = args.gamma*args.decay_factor**(iteration/(args.iterations-1))
                    else:
                        gamma = args.gamma

                    for i in range(args.dis_iter):
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        batch_files,_ = minibatch.__next__()
                        for idx, filepath in enumerate(batch_files):
                            if not filepath in npzs:
                                #npzs[filepath] = (np.load(filepath)['arr_0']/NORMALIZE_CONST-0.5)*2.
                                npzs[filepath] = cv2.resize(scd.imread_as_jpg(filepath),(args.iw,args.ih))
                            batchmx[idx,:,:,:]=npzs[filepath]
                        images = batchmx


                    if summary:
                        gen_cost, mg = mnistWganInv.train_gen(session, images, noise, summary=True)
                    else:
                        gen_cost, go, gf, vf = mnistWganInv.train_gen(session, images, noise, summary=False)

                    if summary:
                        cd, md = mnistWganInv.train_dis(session, images, noise, gamma, summary=True)
                    else:
                        cd = mnistWganInv.train_dis(session, images, noise, gamma, summary=False)
                    dis_cost = cd

                    stime=time.time()-pretime        
                    print("epoch: %d, iteration: %d, gen_cost: %f, go: %f, gf: %f, vf: %f, dis_cost: %f, time: %.2f"%(epoch, iteration, gen_cost, go, gf, vf, dis_cost, stime))

                    tflib.plot.plot('train gen cost', gen_cost)
                    tflib.plot.plot('train dis cost', dis_cost)

                    if summary:
                        writer.add_summary(mg, iteration)
                        writer.add_summary(md, iteration)
                        writer.add_summary(mi, iteration)

                    #if iteration % 100 == 99:
                    if iteration % 100==0:
                        mnistWganInv.reconstruct_images(session, fixed_images, iteration)

                    if iteration % 1000 == 999:
                        save_path = saver.save(session, os.path.join(
                            args.output_path, '{}/model'.format(args.mddir)), global_step=iteration)

                    if iteration % 1000 == 999:
                        dev_dis_cost_lst, dev_inv_cost_lst = [], []
                        dev_images = np.expand_dims(fixed_images[0],0)
                        noise = np.random.randn(1, args.z_dim)
                        dev_dis_cost = session.run(
                            mnistWganInv.dis_cost,
                            feed_dict={mnistWganInv.images: dev_images,
                                       mnistWganInv.gamma_plh:gamma,
                                       mnistWganInv.z_noise:noise})
                        dev_dis_cost_lst += [dev_dis_cost]
                        dev_inv_cost_lst += [0]
                        tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
                        tflib.plot.plot('dev inv cost', np.mean(0))

                    if iteration < 5 or iteration % 100 == 99:
                        tflib.plot.flush(os.path.join(args.output_path, args.mddir))
            except StopIteration:
                pass
                

                tflib.plot.tick()


