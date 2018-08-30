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

import tflib
import tflib.plot
import tflib.save_images_hm
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import Activation, BatchNormalization, add, Reshape
from tensorflow.python.keras.layers import DepthwiseConv2D
slim = tf.contrib.slim
def relu6(x):
    return tf.keras.backend.relu(x, maxval=6)
from tensorflow.python.keras import backend as K
NORMALIZE_CONST=201.
TRANSPOSE=False
mean_vec=np.load("mean.npz")["arr_0"]
act_fn_switch=tf.nn.leaky_relu

def get_saver():
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
    ssd_var_list=[v for v in train_vars if v.name.split("/")[0].startswith("wgan")]
    saver = tf.train.Saver(var_list=ssd_var_list)
    return saver

class fmgan(object):
    def __init__(self, x_dim=784, w=31, h=10, c=512, z_dim=64, latent_dim=64,nf=64,  batch_size=80,
                 c_gp_x=10., lamda=0.1, output_path='./',training=True,args=None):
        with tf.variable_scope('wgan'):
            self.bn_params = {
                "decay": 0.99,
                "epsilon": 1e-5,
                "scale": True,
                "is_training": training
            }
            self.x_dim = x_dim
            self.z_dim = z_dim
            self.w = w
            self.h = h
            self.c =c
            self.nf = nf
            self.restime = 3
            self.latent_dim = latent_dim
            self.batch_size = batch_size
            self.c_gp_x = c_gp_x
            self.lamda = lamda
            self.output_path = output_path
            self.args=args

            #self.gen_params = self.dis_params = self.inv_params = None

         
            self.gamma_plh = tf.placeholder(tf.float32, shape=(), name='gammaplh')
            self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
            self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.h,self.w,self.c])

    def build_model(self):
        #with tf.variable_scope('wgan'):
        if True:
            self.x_p = self.generate(self.z)

            #self.x += tf.random_normal(shape=tf.shape(self.x), mean=0.0, stddev=0.01)
            self.z_p = self.invert(self.x)

            self.dis_x = self.discriminate(self.x)
            self.dis_x_p = self.discriminate(self.x_p,reuse=True)
            self.rec_x = self.generate(self.z_p,reuse=True)
            self.rec_z = self.invert(self.x_p,reuse=True)

            self.rec_x_z_x = self.generate(self.z_p,reuse=True)

            self.gen_cost  =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_x_p, labels=tf.ones_like(self.dis_x_p)))


            self.inv_cost = tf.reduce_mean(tf.square(self.x - self.rec_x))
            self.inv_cost += self.lamda * tf.reduce_mean(tf.square(self.z - self.rec_z))
            #worked well additional loss
            self.inv_cost += tf.reduce_mean(tf.square(self.x - self.rec_x_z_x))

            dis_cost1 =  (tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_x, labels=tf.ones_like(self.dis_x)))
            self.dis_cost = tf.reduce_mean(dis_cost1)
            dis_cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_x_p, labels=tf.zeros_like(self.dis_x_p)))
            self.dis_cost += tf.reduce_mean(dis_cost2)


            self.D_strong = tf.reduce_sum(score1+score2)/2.
            disc_reg = self.Discriminator_Regularizer(self.dis_x, self.x, self.dis_x_p, self.x_p)
            #for more than one fake dis...
            assert self.dis_cost.shape == disc_reg.shape
            self.dis_cost += (self.gamma_plh/2.0)*disc_reg

            #train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
            train_vars=tf.trainable_variables()
            self.gen_params=[v for v in train_vars if v.name.split("/")[1] == "generate"]
            self.inv_params=[v for v in train_vars if v.name.split("/")[1] == "invert"]
            self.dis_params=[v for v in train_vars if v.name.split("/")[1] == "discriminate"]

        with tf.variable_scope('wgan'):
            genopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.gen_train_op= slim.learning.create_train_op(self.gen_cost,genopt,summarize_gradients=True,variables_to_train=self.gen_params)
            invopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.inv_train_op= slim.learning.create_train_op(self.inv_cost,invopt,summarize_gradients=True,variables_to_train=self.inv_params+self.gen_params)
            #self.inv_train_op= slim.learning.create_train_op(self.inv_cost,invopt,summarize_gradients=True,variables_to_train=self.inv_params)
            disopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.dis_train_op= slim.learning.create_train_op(self.dis_cost,disopt,summarize_gradients=True,variables_to_train=self.dis_params)

    #def Discriminator_Regularizer(self,D1_logits, D1_arg, D2_logits, D2_arg, D3_logits=None, D3_arg=None, D4_logits=None, D4_arg=None):
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
  


    def _residual_block(self, X, nf_output, resample=None, kernel_size=[3,3],upsample_shape=None, name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
            nf_input = input_shape[-1]
            """
            if input_shape[1]<kernel_size[0] or input_shape[2] < kernel_size[1]:
                X= tf.image.resize_image_with_crop_or_pad(X, kernel_size[0], kernel_size[1])
            """
            nf_input = input_shape[-1]
            if resample == 'down': 
                shortcut = slim.avg_pool2d(X, [2,2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) 

                net = slim.layer_norm(X, activation_fn=act_fn_switch)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None) 
                net = slim.layer_norm(net, activation_fn=act_fn_switch)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
                net = slim.avg_pool2d(net, [2,2])

                return net + shortcut
            elif resample == 'up': 
                if upsample_shape==None:
                    upsample_shape = tuple(int(x)*2 for x in input_shape[1:3])
                shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape) 
                #shortcut = tf.image.resize_bilinear(X, upsample_shape) 
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None)

                net = slim.batch_norm(X, activation_fn=act_fn_switch, **self.bn_params)
                net = tf.image.resize_nearest_neighbor(net, upsample_shape) 
                #net = tf.image.resize_bilinear(net, upsample_shape) 
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None) 
                net = slim.batch_norm(net, activation_fn=act_fn_switch, **self.bn_params)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)

                return net + shortcut
            else:

                net = slim.batch_norm(X, activation_fn=act_fn_switch, **self.bn_params)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None) 
                net = slim.batch_norm(net, activation_fn=act_fn_switch, **self.bn_params)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)

                return net + X

    def generate(self, z, reuse=False):
        with tf.variable_scope('wgan/generate', reuse=reuse):
            start_coef = 2**(self.restime-1)
            nf = self.nf
            if TRANSPOSE:
                startdimw = 2*2**(4-self.restime)#for h= 512,w=31,c=10
                startdimh = 32*2**(4-self.restime)#for h= 512,w=31,c=10
            else:
                startdimw = 4*2**(3-self.restime)
                startdimh = 4*2**(3-self.restime)
                upsample_shapes=[
                (4,6),
                (7,16),
                (10,31)
                ]
            net = slim.fully_connected(z, startdimh*startdimw*start_coef*nf, activation_fn=None) 
            net = tf.reshape(net, [-1, startdimh, startdimw, start_coef*nf])
            for i,p in enumerate(range(self.restime-1,-1,-1)):
                net = self._residual_block(net, (2**p)*nf,upsample_shape=(None if TRANSPOSE else upsample_shapes[i]), resample='up', name='res_block%d'%i)
                if self.args.doubleres:
                    net = self._residual_block(net, (2**p)*nf,name='res_block%d_2'%i)
            net = slim.batch_norm(net, activation_fn=act_fn_switch, **self.bn_params)
            net = slim.conv2d(net, self.c, kernel_size=[3,3], activation_fn=None)
            if TRANSPOSE:
                net = net[:,:,:-1,:]
            """
            else:
                net = net[:,11:21,:-1,:]
            """

            return net

    def discriminate(self, X, reuse=False):
        with tf.variable_scope('wgan/discriminate', reuse=reuse):
            nf = self.nf
            net = slim.conv2d(X, nf, [3,3], activation_fn=None) 
            for i,p in enumerate(range(1,self.restime)):
                net = self._residual_block(net, (2**p)*nf, resample='down', name='res_block%d'%i)
                if self.args.doubleres:
                    net = self._residual_block(net, (2**p)*nf,name='res_block%d_2'%i)
            net = self._residual_block(net, (2**self.restime)*nf, resample='down', name='res_block%d'%self.restime)
            if self.args.doubleres:
                net = self._residual_block(net, (2**self.restime)*nf,name='res_block%d_2'%self.restime)
            net=act_fn_switch(net)
            net=tf.reduce_mean(net,axis=[1,2])
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net
 
    def invert(self, x,reuse=False):
        with tf.variable_scope('invert', reuse=reuse):
            nf = self.nf
            net = slim.conv2d(x, nf, [3,3], activation_fn=None) 
            for i,p in enumerate(range(1,self.restime)):
                net = self._residual_block(net, (2**p)*nf, resample='down', name='res_block%d'%i)
                if self.args.doubleres:
                    net = self._residual_block(net, (2**p)*nf,name='res_block%d_2'%i)
            net=act_fn_switch(net)
            net=tf.reduce_mean(net,axis=[1,2])
            net = slim.fully_connected(net, nf*(2**(self.restime-1)), activation_fn=None)
            net = slim.fully_connected(net, self.z_dim, activation_fn=None)

            return net
 


    def train_gen(self, sess, x, z, summary=False):
        if summary:
            _gen_cost, _, summary = sess.run([self.gen_cost, self.gen_train_op, self.merge],
                                    feed_dict={self.x: x, self.z: z})
            return _gen_cost, summary
        else:
            _gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op],
                                    feed_dict={self.x: x, self.z: z})
            return _gen_cost 

    def train_dis(self, sess, x, z, gamma, summary=False):
        if summary:
            print(gamma)
            _dis_cost, _, summary = sess.run([self.dis_cost, self.dis_train_op, self.merge],
                  feed_dict={self.x: x, self.z: z, self.gamma_plh: gamma})
            return _dis_cost, summary
        else:
            _dis_cost, _d_score, _= sess.run([self.dis_cost, self.D_strong, self.dis_train_op],
                                    feed_dict={self.x: x, self.z: z, self.gamma_plh: gamma})
            return _dis_cost, _d_score

    def train_inv(self, sess, x, z, summary=False):
        if summary:
            _inv_cost, _, summary = sess.run([self.inv_cost, self.inv_train_op, self.merge],
                                    feed_dict={self.x: x, self.z: z})
            return _inv_cost, summary
        else:
            _inv_cost, _= sess.run([self.inv_cost, self.inv_train_op],
                                    feed_dict={self.x: x, self.z: z})
            return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        if TRANSPOSE:
            #samplesnpz = (samples.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1]) + 1)*NORMALIZE_CONST/2.
            samplesnpz = samples.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])
        else:
            #samplesnpz = (samples.reshape((-1, self.h,self.w,self.c)) + 1)*NORMALIZE_CONST/2.
            samplesnpz = samples.reshape((-1, self.h,self.w,self.c))
        samplesnpz+=mean_vec.reshape([1,1,1,mean_vec.shape[0]])

        np.savez(os.path.join(self.output_path, '{}/samples_{}.npz'.format(self.args.exdir,frame)),samplesnpz)

        tflib.save_images_hm.save_images(
            samplesnpz,
            os.path.join(self.output_path, '{}/samples_{}.png'.format(self.args.exdir,frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})
        if TRANSPOSE:
            samplesnpz = reconstructions.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])
            comparison = np.zeros((images.shape[0] * 2, images.shape[3],images.shape[2],images.shape[1]),
                              dtype=np.float32)
        else:
            samplesnpz = reconstructions.reshape((-1, self.h,self.w,self.c))
            comparison = np.zeros((images.shape[0] * 2, images.shape[1],images.shape[2],images.shape[3]),
                              dtype=np.float32)
        samplesnpz+=mean_vec.reshape([1,1,1,mean_vec.shape[0]])

        np.savez(os.path.join(self.output_path, '{}/recs_{}.npz'.format(self.args.exdir,frame)),samplesnpz)
        for i in range(images.shape[0]):
            if TRANSPOSE:
                comparison[2 * i] = images[i].transpose([2,1,0])+mean_vec.reshape([1,1,mean_vec.shape[0]])
            else:
                comparison[2 * i] = images[i]+mean_vec.reshape([1,1,mean_vec.shape[0]])
            comparison[2 * i + 1] = samplesnpz[i]


        tflib.save_images_hm.save_images(
            comparison,
            #comparison.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST,
            os.path.join(self.output_path, '{}/recs_{}.png'.format(self.args.exdir,frame)))
        return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
    parser.add_argument('--dataset_test_path', type=str, default='/workspace/imgsynth/hmpool5/test',
                        help='dataset path')
    parser.add_argument('--dataset_train_path', type=str, default='/workspace/imgsynth/hmpool5/train',
                        help='dataset path')
    parser.add_argument('--nf', type=int, default=128,
                        help='ooooooooooo')
    parser.add_argument('--input_c', type=int, default=512,
    #parser.add_argument('--input_c', type=int, default=512,
                        help='ooooooooooo')
    parser.add_argument('--input_w', type=int, default=31,
    #parser.add_argument('--input_w', type=int, default=31,
                        help='ooooooooooo')
    parser.add_argument('--input_h', type=int, default=10,
    #parser.add_argument('--input_h', type=int, default=10,
                        help='ooooooooooo')
    parser.add_argument("--gamma",type=float,default=0.1,help="noise variance for regularizer [0.1]")
    parser.add_argument("--annealing", type=bool,default=False, help="annealing gamma_0 to decay_factor*gamma_0 [False]")
    parser.add_argument("--decay_factor",type=float, default=0.01, help="exponential annealing decay rate [0.01]")
    parser.add_argument("--doubleres",type=bool, default=False, help="exponential annealing decay rate [0.01]")
    parser.add_argument('--exdir', type=str, default='example_doubleres_disc')
    parser.add_argument('--mddir', type=str, default='models_doubleres_disc')
    args = parser.parse_args()


    if TRANSPOSE:
        fixed_images = np.zeros([args.batch_size,args.input_h,args.input_w,args.input_c])
    else:
        fixed_images = np.zeros([args.batch_size,args.input_c,args.input_w,args.input_h])
    for i in range(args.batch_size):
        if TRANSPOSE:
            fixed_images[i,:,:,:]=(np.load(os.path.join(args.dataset_test_path,"%06d.npz"%(i+7481)))['arr_0']-mean_vec.reshape([1,1,1,mean_vec.shape[0]])).transpose(0,3,2,1)[0,:,:,:]
        else:
            #fixed_images[i,:,:,:]=np.load(os.path.join(args.dataset_path,"%06d.npz"%(i+7481)))['arr_0'][0,:,:,:]/NORMALIZE_CONST
            fixed_images[i,:,:,:]=np.load(os.path.join(args.dataset_test_path,"%06d.npz"%(i+7481)))['arr_0'][0,:,:,:]-mean_vec.reshape([1,1,mean_vec.shape[0]])
    fixed_noise = np.random.randn(args.batch_size, args.z_dim)
    
    if TRANSPOSE:
        mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, w=args.input_w, h=args.input_h, c=args.input_c, latent_dim=args.latent_dim,
        nf=args.nf, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path,args=args)
    else:
        mnistWganInv = hmgan(
        x_dim=784, z_dim=args.z_dim, w=args.input_w, h=args.input_h, c=args.input_c, latent_dim=args.latent_dim,
        nf=256, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path,args=args)

    saver = tf.train.Saver(max_to_keep=10)


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        data_files = glob(os.path.join(args.dataset_train_path, "*.npz"))
        data_files = sorted(data_files)
        data_files = np.array(data_files) 
        iteration=0
        npzs = {}
        if TRANSPOSE:
            batchmx = np.zeros([args.batch_size,args.input_h,args.input_w,args.input_c])
        else:
            batchmx = np.zeros([args.batch_size,args.input_c,args.input_w,args.input_h])
        print("pre epoch")
        with tf.name_scope("summary"):
            tf.summary.scalar("gen_cost",mnistWganInv.gen_cost)
            tf.summary.scalar("inv_cost",mnistWganInv.inv_cost)
            tf.summary.scalar("dis_cost",mnistWganInv.dis_cost)
            mnistWganInv.merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter(args.logdir,session.graph)

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
                                npzs[filepath] = np.load(filepath)['arr_0']-mean_vec.reshape([1,1,1,mean_vec.shape[0]])
                            if TRANSPOSE:
                                batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:].transpose([2,1,0])
                            else:
                                batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:]
                                
                            #batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:]
                        images = batchmx

                        """
                        if summary:
                            gen_cost, mg = mnistWganInv.train_gen(session, images, noise, summary=True)
                        else:
                            gen_cost = mnistWganInv.train_gen(session, images, noise, summary=False)

                    if summary:
                        cd, md = mnistWganInv.train_dis(session, images, noise, gamma, summary=True)
                    else:
                        cd = mnistWganInv.train_dis(session, images, noise,gamma, summary=False)
                    dis_cost_lst += [cd]

                    if summary:
                        ci, mi = mnistWganInv.train_inv(session, images, noise, summary=True)
                    else:
                        ci = mnistWganInv.train_inv(session, images, noise, summary=False)
                    inv_cost_lst += [ci]

                        """

                        if summary:
                            cd, md = mnistWganInv.train_dis(session, images, noise, gamma, summary=True)
                        else:
                            cd, d_score = mnistWganInv.train_dis(session, images, noise, gamma, summary=False)
                        dis_cost_lst += [cd]

                    if summary:
                        ci, mi = mnistWganInv.train_inv(session, images, noise, summary=True)
                    else:
                        ci = mnistWganInv.train_inv(session, images, noise, summary=False)
                    inv_cost_lst += [ci]

                    if summary:
                        gen_cost, mg = mnistWganInv.train_gen(session, images, noise, summary=True)
                    else:
                        gen_cost = mnistWganInv.train_gen(session, images, noise, summary=False)

                    dis_cost = np.mean(dis_cost_lst)
                    inv_cost = np.mean(inv_cost_lst)
                    stime=time.time()-pretime        
                    print("epoch: %d, iteration: %d, gen_cost: %f, dis_cost: %f, inv_cost: %f, d_score: %f, time: %.2f"%(epoch, iteration, gen_cost, dis_cost, inv_cost,d_score, stime))

                    tflib.plot.plot('train gen cost', gen_cost)
                    tflib.plot.plot('train dis cost', dis_cost)
                    tflib.plot.plot('train inv cost', inv_cost)

                    if summary:
                        writer.add_summary(mg, iteration)
                        writer.add_summary(md, iteration)
                        writer.add_summary(mi, iteration)

                    #if iteration % 1000 == 999:
                    if iteration % 100 == 99:
                        mnistWganInv.generate_from_noise(session, fixed_noise, iteration)
                        mnistWganInv.reconstruct_images(session, fixed_images, iteration)

                    if iteration % 1000 == 999:
                        save_path = saver.save(session, os.path.join(
                            args.output_path, '{}/model'.format(args.mddir)), global_step=iteration)

                    if iteration % 1000 == 999:
                        dev_dis_cost_lst, dev_inv_cost_lst = [], []
                        dev_images = fixed_images
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        dev_dis_cost, dev_inv_cost = session.run(
                            [mnistWganInv.dis_cost, mnistWganInv.inv_cost],
                            feed_dict={mnistWganInv.x: dev_images,
                                       mnistWganInv.z: noise,
                                       mnistWganInv.gamma_plh:gamma})
                        dev_dis_cost_lst += [dev_dis_cost]
                        dev_inv_cost_lst += [dev_inv_cost]
                        tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
                        tflib.plot.plot('dev inv cost', np.mean(dev_inv_cost_lst))

                    if iteration < 5 or iteration % 100 == 99:
                        tflib.plot.flush(os.path.join(args.output_path, args.mddir))
            except StopIteration:
                pass
                

                tflib.plot.tick()


