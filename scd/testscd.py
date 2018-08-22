import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import sys, os
import cv2
import skimage.io

sys.path.insert(0,"scd")
import scd
def main():
    filepath="/workspace2/kitti/testing/image_2/007481.png"
    filepath="test/test0_2.jpg"
    filepath="test/test1_2.jpg"
    filepath="test/noise3.jpg"
    filepath="/workspace2/kitti/testing/image_2/007489.png"
    filepath="test/single_generated.jpg"
    filepath="test/valid_adv7.png"
    filepath="/workspace2/kitti/testing/image_2/007507.png"
    filepath="/workspace2/kitti/training/image_2/000003.png"
    filepath="/workspace2/kitti/testing/image_2/007597.png"
    filepath="/workspace2/kitti/testing/image_2/007618.png"
    filepath="test/single_generated.jpg"
    filepath="007618shadow_light2.png"
    filepath="/workspace2/kitti/testing/image_2/007619.png"
    filepath="test/adversary.png"
    filepath="/workspace2/kitti/testing/image_2/008001.png"
    filepath="test/single_generated.jpg"
    filepath="/workspace2/kitti/testing/image_2/008340.png"
    #filepath="0076182.png"
    #filepath="test/noise4.jpg"
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=config)
    images = tf.placeholder(tf.float32,[1,150,496,3])
    scdobj = scd.SCD(input=images)
    scd_saver = scd.get_saver()
    scd_saver.restore(sess,"/workspace/imgsynth/ssd_300_kitti/ssd_model.ckpt")

    image = cv2.resize(scd.imread_as_jpg(filepath),(496,150))
    image=np.expand_dims(image,0)
    reses = scdobj.get_image(sess,image,select_threshold=0.7,nms_threshold=0.3)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/single_detected%d.jpg"%i,a)
if __name__ == "__main__":
    main()
