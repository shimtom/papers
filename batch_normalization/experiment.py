import tensorflow as tf
import numpy as np
import os
from math import ceil
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import seaborn as sns
import glob


class BasicModel:
    def __init__(self, input_dims, output_dims, learning_rate):
        self._sess = None

        w1 = tf.Variable(tf.truncated_normal(
            shape=[input_dims, 100], stddev=1. / input_dims), name='w1')
        b1 = tf.Variable(tf.zeros(shape=[100]), dtype=tf.float32, name='b1')

        w2 = tf.Variable(tf.truncated_normal(
            shape=[100, 100], stddev=1. / 100), name='w2')
        b2 = tf.Variable(tf.zeros(shape=[100]), dtype=tf.float32, name='b2')

        w3 = tf.Variable(tf.truncated_normal(
            shape=[100, 100], stddev=1. / 100), name='w3')
        b3 = tf.Variable(tf.zeros(shape=[100]), dtype=tf.float32, name='b3')

        w4 = tf.Variable(tf.truncated_normal(
            shape=[100, output_dims], stddev=1. / 100), name='w4')
        b4 = tf.Variable(
            tf.zeros(shape=[output_dims]), dtype=tf.float32, name='b4')

        self._x = tf.placeholder(
            tf.float32, shape=[None, input_dims], name='x')
        u1 = tf.matmul(self._x, w1, name='u1') + b1
        z1 = tf.nn.sigmoid(u1, name='z1')
        u2 = tf.matmul(z1, w2, name='u2') + b2
        z2 = tf.nn.sigmoid(u2, name='z2')
        u3 = tf.matmul(z2, w3, name='u3') + b3
        z3 = tf.nn.sigmoid(u3, name='z3')
        logits = tf.matmul(z3, w4, name='u4') + b4
        self._y = tf.nn.softmax(logits)
        self._d = tf.placeholder(tf.int64, shape=[None], name='d')

        self._weights = [w1, w2, w3, w4]
        self._biases = [b1, b2, b3, b4]
        self._outputs = [self._x, u1, z1, u2, z2, u3, z3, logits, self._y]

        self._loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._d), name='loss')
        self._accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self._y, 1), self._d), tf.float32))
        self._optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(self._loss)

    def initialize(self, sess):
        self._sess = sess
        return self._sess.run(tf.initialize_all_variables())

    def optimize(self, inputs, targets):
        return self._sess.run(self._optimizer, feed_dict={self._x: inputs, self._d: targets, self._is_training: True})

    def forward(self, inputs):
        return self._sess.run(self._y, feed_dict={self._x: inputs, self._is_training: False})

    def compute_loss(self, inputs, targets):
        return self._sess.run(self._loss, feed_dict={self._x: inputs, self._d: targets, self._is_training: False})

    def evaluate(self, inputs, targets):
        return self._sess.run(self._accuracy, feed_dict={self._x: inputs, self._d: targets, self._is_training: False})

    @property
    def weights(self):
        return [self._sess.run(w) for w in self._weights]

    @property
    def biases(self):
        return [self._sess.run(b) for b in self._biases]

    @property
    def outputs(self):
        return [self._sess.run(o) for o in self._outputs]


class BatchNormedModel(BasicModel):
    def __init__(self, input_dims, output_dims, learning_rate):
        self._sess = None

        w1 = tf.Variable(tf.truncated_normal(
            shape=[input_dims, 100], stddev=1. / input_dims), name='w1')
        b1 = tf.Variable(tf.zeros(shape=[100]), dtype=tf.float32, name='b1')
        w2 = tf.Variable(tf.truncated_normal(
            shape=[100, 100], stddev=1. / 100), name='w2')
        b2 = tf.Variable(tf.zeros(shape=[100]), dtype=tf.float32, name='b2')
        w3 = tf.Variable(tf.truncated_normal(
            shape=[100, 100], stddev=1. / 100), name='w3')
        b3 = tf.Variable(tf.zeros(shape=[100]), dtype=tf.float32, name='b3')
        w4 = tf.Variable(tf.truncated_normal(
            shape=[100, output_dims], stddev=1. / 100), name='w4')
        b4 = tf.Variable(
            tf.zeros(shape=[output_dims]), dtype=tf.float32, name='b4')

        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self._x = tf.placeholder(
            tf.float32, shape=[None, input_dims], name='x')
        x_ = batch_norm(self._x, self._is_training)
        u1 = tf.matmul(x_, w1, name='u1')
        z1 = tf.nn.sigmoid(u1, name='z1')
        z1_ = batch_norm(z1, self._is_training)
        u2 = tf.matmul(z1_, w2, name='u2')
        z2 = tf.nn.sigmoid(u2, name='z2')
        z2_ = batch_norm(z2, self._is_training)
        u3 = tf.matmul(z2_, w3, name='u3')
        z3 = tf.nn.sigmoid(u3, name='z3')
        z3_ = batch_norm(z3, self._is_training)
        logits = tf.matmul(z3_, w4, name='u4')
        self._y = tf.nn.softmax(logits)
        self._d = tf.placeholder(tf.int64, shape=[None], name='d')

        self._weights = [w1, w2, w3, w4]
        self._biases = [b1, b2, b3, b4]
        self._outputs = [self._x, x_, u1, z1, z1_,
                         u2, z2, z2_, u3, z3, z3_, logits, self._y]

        self._loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._d), name='loss')
        self._accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self._y, 1), self._d), tf.float32))
        self._optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(self._loss)


def batch_norm(x, is_training, decay=0.9, eps=1e-5):
    """Batch Normalizationを適用する.

    :param x: 入力. [batch_size, dim]か[batch_size, height, width, ch]の形状である必要がある.
    :param is_training: 訓練中か否かを表す.
    :param decay: 指数移動平均に使用する平滑化定数.
    :param eps: 計算を安定化させるための値.
    :return: batch normalizationを適用されたx.
    """
    shape = x.get_shape().as_list()

    if not len(shape) in [2, 4]:
        raise ValueError(
            'x must be (batch_size, dim) or (batch_size, height, width, ch) but %s' % (str(shape)))

    n_out = shape[-1]

    beta = tf.Variable(tf.zeros(n_out))
    gamma = tf.Variable(tf.ones(n_out))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_and_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training, mean_and_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


def generate_minibatch(input_set, target_set, batch_size, is_training=True):
    """指定された大きさのミニバッチを取得するジェネレータ.

    input_set: 入力データセット.
    target_set: ターゲットデータセット.
    batch_size: ミニバッチの大きさ.
    is_training: 訓練中か否かを表す.
    """
    if target_set is None:
        target_set = np.empty_like(input_set)
    index = 0
    data_size = len(input_set)
    indices = np.random.permutation(
        data_size) if is_training else np.arange(data_size)
    size = ceil(data_size / batch_size)

    for step in range(size):
        n = min(batch_size, data_size - batch_size * step)
        yield n, (input_set[indices[index:index + n]], target_set[indices[index:index + n]])
        index += n


def compute_accuracy(model, data_set, batch_size=100):
    accuracy = 0.
    data_size = 0
    input_set = data_set.images
    target_set = data_set.labels
    size = len(input_set)
    index = 0
    for i in range(ceil(size / batch_size)):
        inputs = input_set[index:index + batch_size]
        targets = target_set[index:index + batch_size]
        accuracy += model.evaluate(inputs, targets) * \
            min(batch_size, size - index)
    return accuracy / size


def learn(models, train_data_set, test_data_set, max_step, batch_size, save_dir):
    """ニューラルネットワークモデルを学習する.

    models: 同時に学習したいモデルのリスト.
    train_data_set: 訓練用のデータセット.
    test_data_set: テスト用のデータセット.
    batch_size: ミニバッチの大きさ.
    max_step: 最大学習ステップ数.
    save_dir: 結果を保存するためのディレクトリのパス.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print('save directory     : %s' % save_dir)
    print('batch size         : %d' % batch_size)
    print('max_step           : %d' % max_step)

    train_losses = [[] for _ in range(len(models))]
    test_accuracies = [[] for _ in range(len(models))]

    with tf.Session() as sess:
        for model in models:
            model.initialize(sess)

        for step in range(max_step):
            # test
            for i, model in enumerate(models):
                test_accuracy = compute_accuracy(model, test_data_set)
                test_accuracies[i].append(test_accuracy)

                # train
                inputs, targets = train_data_set.next_batch(batch_size)
                model.optimize(inputs, targets)
                train_loss = model.compute_loss(inputs, targets)
                train_losses[i].append(train_loss)

                print('Step [%d] loss %f top accuracy %f' %
                      (step, train_loss, test_accuracy))

                # save
                if step % 1000 == 0:
                    save_path = os.path.join(
                        save_dir, 'model%02d' % i, '%05d' % step)
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    for i, w in enumerate(model.weights):
                        np.save(os.path.join(save_path, 'weight-%03d' % i), w)
                    for i, b in enumerate(model.biases):
                        np.save(os.path.join(save_path, 'bias-%03d' % i), b)
                    for i, o in enumerate(model.outputs):
                        np.save(os.path.join(save_path, 'output-%03d' % i), b)

        for i, losses in enumerate(train_losses):
            np.save(os.path.join(save_dir, 'model%02d' %
                                 i, 'losses'), np.array(losses))
        for i, accuracies in enumerate(test_accuracy):
            np.save(os.path.join(save_dir, 'model%02d' %
                                 i, 'accuracies'), np.array(accuracies))


def plot_scalar(values, ylabel, xlabel, title, save_dir, show=False):
    sns.set(style='darkgrid', palette='muted', color_codes=True)
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(values)
    plt.xlim([0, len(values)])
    plt.savefig(os.path.join(save_dir, '%s.png' % title))
    if show:
        plt.show()
    plt.close()


def plot_distribution(values, title, save_dir, show=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sns.set(style='darkgrid', palette='muted', color_codes=True)
    plt.figure()
    sns.distplot(values)
    plt.savefig(os.path.join(save_dir, '%s.png' % title))
    if show:
        plt.show()
    plt.close()


def _plot_result(save_dir, last_hidden_output_index):
    save_path = os.path.join(save_dir, 'figure')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    losses = np.load(os.path.join(save_dir, 'losses.npy'))
    plot_scalar(losses, '', 'step', 'loss', save_path)
    accuracies = np.load(os.path.join(save_dir, 'accuracies.npy'))
    plot_scalar(accuracies, '', 'step', 'accuracy', save_path)

    # plot distributions of last hidden layer output
    files = [path for path in glob.glob(os.path.join(
        save_dir, '*', 'output-%d.npy' % last_hidden_output_index))]
    sorted(files)
    for i, f in enumerate(files):
        output = np.load(f)
        indices = np.argsort(output.mean(axis=1))
        plot_distribution(output[indices[int(len(indices) * 0.15)]],
                          'step %d 15th persentaile' % i, os.path.join(save_path, '15'))
        plot_distribution(output[indices[int(len(indices) * 0.5)]],
                          'step %d 50th persentaile' % i, os.path.join(save_path, '50'))
        plot_distribution(output[indices[int(len(indices) * 0.85)]],
                          'step %d 85th persentaile' % i, os.path.join(save_path, '85'))


def main(learning=True, plot=True):
    mnist_data_set = input_data.read_data_sets('./mnist')
    train_data_set = mnist_data_set.train
    test_data_set = mnist_data_set.test
    num_dim = 28 * 28
    num_class = 10

    save_dir = './result'
    batch_size = 60
    max_step = 15000
    learning_rate = 0.5
    models = [BasicModel(num_dim, num_class, learning_rate),
              BatchNormedModel(num_dim, num_class, learning_rate)]
    last_hidden_output_index = [-4, -5]
    if learning:
        learn(models, train_data_set, test_data_set,
              max_step, batch_size, save_dir)

    if plot:
        for i in range(models):
            _plot_result(os.path.join(save_dir, 'model%d' %
                                      i, last_hidden_output_index[i]))


if __name__ == '__main__':
    main()
