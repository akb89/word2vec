import tensorflow as tf

if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as session:
        x = tf.constant(['this', 'is', 'a', 'test', '_CBOW#_!MASK_'])
        is_valid_string = tf.not_equal(x, '_CBOW#_!MASK_')
        y = tf.boolean_mask(x, is_valid_string)
        print(x.eval())
        print(is_valid_string.eval())
        print(y.eval())
