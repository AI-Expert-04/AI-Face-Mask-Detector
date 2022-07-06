import tensorflow as tf_new
tf = tf_new.compat.v1

g = tf.Graph()
with g.as_default() as graph:
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0, tf.float32)
    node3 = tf.add(node1, node2)
    print(node1)
    print(node2)
    print(node3)
    sess = tf.Session()
    print(sess.run(node3))
