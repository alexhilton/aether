#!/usr/bin/env python3

import tensorflow as tf
import os


W = tf.Variable(tf.zeros([5, 1]), name='weights')
b = tf.Variable(0., name='bias')


def combine_input(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.sigmoid(combine_input(X))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer(
        os.path.join(os.path.dirname(__file__), file_name))
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, \
    parch, ticket, fare, cabin, embarked = \
    read_csv(100, 'sigmoid.csv', [[0.0], [0.0], [0], [''], [''],
             [0.0], [0.0], [0.0], [''], [0.], [''], ['']])
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    gender = tf.to_float(tf.equal(sex, ['female']))

    features = tf.transpose(tf.pack(
        [is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])

    return features, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y)))))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_steps = 1000
    for step in range(train_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print('loss->', sess.run([total_loss]))

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()
