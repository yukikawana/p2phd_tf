import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import sys, os
import cv2
import skimage.io
sys.path.insert(0,".")
import eval_rgan
import eval_imgsynth

sys.path.insert(0,"scd")
import scd
def main():
    nf=128
    zdim=1024
    filepath="/workspace2/kitti/testing/image_2/007481.png"
    filepath="test/test0_2.jpg"
    filepath="test/test1_2.jpg"
    filepath="test/noise3.jpg"
    filepath="test/adversary.png"
    filepath="/workspace2/kitti/training/image_2/000003.png"
    filepath=sys.argv[1]
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=config)
    #Jimages = tf.placeholder(tf.float32,[1,10,31,512])
    images = tf.placeholder(tf.float32,[1,5,10,512])
    images = tf.nn.relu(images)
    imgsynth_from_feature = eval_imgsynth.ImageSynthesizer(images)

    imgsynth_saver = eval_imgsynth.get_saver()
    imgsynth_saver.restore(sess,"/workspace/imgsynth/result_kitti256p_2/model.ckpt-89")

    image = np.load(filepath)["arr_0"]
    image=image[:,0:5,0:10,:]
    ch = image.shape[0]
    for i in range(ch):
        print(i)
        reses = imgsynth_from_feature.generate_image_from_featuremap(sess,np.expand_dims(image[i],0),images)
        skimage.io.imsave("test/single_generated_from_fm.jpg",reses[0])
if __name__ == "__main__":
    main()
