import tensorflow as tf


class curveModel(tf.keras.Model):
    def __init__(self, n, PS, PE, dl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec = tf.Variable(tf.zeros([n], dtype=tf.float32), trainable=True)
        self.vec2 = tf.Variable(tf.zeros([n], dtype=tf.float32), trainable=True)
        self.PS = tf.constant(PS, dtype=tf.float32, trainable=False)
        self.PE = tf.constant(PE, dtype=tf.float32, trainable=False)
        self.dl = tf.constant(dl, dtype=tf.float32, trainable=False)
        self.the0 = tf.Variable([tf.constant(tf.pi / 6)], trainable=True)
        self.the02 = tf.Variable(
            [tf.constant(tf.pi + tf.math.asin(tf.ones([]) / 7.0))], trainable=True
        )
        self.unit = tf.constant([1.0, 0], dtype=tf.float32, trainable=False)

    def call(self):
        P = self.PS
        P2 = self.PE
        for i in range(thetas.shape[0]):
            pre = tf.linalg.matvec(matt1[i], pre)
            P = tf.concat((P, tf.expand_dims(P[-1] + pre, axis=0)), axis=0)
            pre2 = tf.linalg.matvec(matt2[i], pre2)
            P2 = tf.concat((P2, tf.expand_dims(P2[-1] + pre2, axis=0)), axis=0)

        return P, P2
