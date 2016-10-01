

import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld

import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('max_step', 50000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size for training""")


def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs and high res input
    x_big = tf.placeholder(tf.float32, [FLAGS.batch_size, 64, 64, 3])
    x = tf.image.resize_images(x_big, 32, 32)
    print(x.get_shape())

    # create network
    # conv1
    conv1 = ld.conv_layer(x, 3, 1, 8, "encode_1")
    # conv2
    conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
    # conv3
    conv3 = ld.conv_layer(conv2, 3, 1, 16, "encode_3")
    # trans_conv4
    conv4 = ld.transpose_conv_layer(conv3, 3, 2, 3, "decode_4")
    # x_big
    x_big_pred = conv4

    # calc loss
    loss = tf.nn.l2_loss(x_big - x_big_pred)

    # save for tensorboard
    tf.scalar_summary('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in xrange(FLAGS.max_step):
      t = time.time()
      dat = b.bounce_vec(64, FLAGS.num_balls, FLAGS.batch_size)
      _, loss_r = sess.run([train_op, loss],feed_dict={x_big:dat})
      elapsed = time.time() - t
      print(elapsed)

      if step%500 == 0:
        loss_r, x_r, x_big_pred_r = sess.run([loss, x, x_big_pred],feed_dict={x_big:dat})
        summary_str = sess.run(summary_op, feed_dict={x_big:dat})
        summary_writer.add_summary(summary_str, step) 
        print("loss value at " + str(loss_r))
        print("time per batch is " + str(elapsed))
        cv2.imwrite("images/real_balls_high_res.jpg", np.uint8(dat[0, :, :, :]*255))
        cv2.imwrite("images/real_balls_los_res.jpg", np.uint8(x_r[0,:,:,:]*255))
        cv2.imwrite("images/generated_balls_high_res.jpg", np.uint8(np.minimum(np.maximum(x_big_pred_r[0,:,:,:], 0.0), 1.0)*255))
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)
        print("step " + str(step))

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()



