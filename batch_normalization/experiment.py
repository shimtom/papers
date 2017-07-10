"""Batch Normalizationの実験用ソースコード.

1. Batch Normalization が学習を加速させることを確かめる.
    * 実験方法
        Batch Normalizationを適用していないネットワークと適用したネットワークを用意する.
        同じパラメータを使用し、テストデータに対する精度が高くなるまでにかかるステップ数を比較することで加速することを検証する.
2. Batch Normalization が入力の分布を安定させることを確かめる.
    * 実験方法
        Batch Normalizationを適用していないネットワークと適用したネットワークを用意する.
        同じパラメータを使用し、最後の隠れ層の活性化関数への入力の分布を比較することで検証する.
3. Batch Normalization を使用した場合、大きな学習係数を使用できることを確かめる.
    * 実験方法
        Batch Normalizationを適用していないネットワークと適用したネットワークを用意する.
        Batch Normalizationを適用していないネットワークでの最大の学習係数を選ぶ.
        次に、その学習係数でBatch Normalizationを適用したネットワークを学習する.
        最後にBatch Normalizationを適用したネットワークでの最大の学習係数を選ぶ.
        そして、それぞれの訓練データに対する損失の変化を比較することで検証する.
4. Batch Normalization が正則化の効果を持つことを確かめる.
    * 実験方法
        Batch Normalizationを適用していないネットワークと適用したネットワークを用意する.
        Batch Normalizationを適用していないネットワークで,隠れ層のノード数を入力層よりも多くすることで過学習を起こさせる.
        次に、同じネットワークにBatch Normalizationを適用したネットワークでも学習を行い、過学習が起きないことを確かめることで検証する.
"""
import tensorflow as tf
import numpy as np
import os
from math import ceil
from tensorflow.examples.tutorials.mnist import input_data
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from abc import ABCMeta, abstractmethod


class Path:
    """データ保存に使用するパスを作成するクラス."""

    def __init__(self, base):
        self._base = base

    def model(self, model_id):
        if model_id is None:
            return ''
        return os.path.join(self._base, 'model%02d' % model_id)

    def step(self, step, model_id=None):
        if step is None:
            if model_id is not None:
                raise ValueError('model_id is not None, but step is None.')
            return ''
        return os.path.join(self.model(model_id), '%05d' % step)

    def figure(self, model_id=None):
        return os.path.join(self.model(model_id), 'figure')

    def losses(self, model_id=None,  extension=True):
        name = 'losses'
        if extension:
            name += '.npy'
        return os.path.join(self.model(model_id), name)

    def accuracies(self, model_id=None, extension=True):
        name = 'accuracies'
        if extension:
            name += '.npy'
        return os.path.join(self.model(model_id), name)

    def weight(self, layer_index, model_id=None, step=None, extension=False):
        return self._tensor('weight', layer_index, model_id=model_id, step=step, extension=extension)

    def bias(self, model_id, step, layer_index, extension=False):
        return self._tensor('bias', layer_index, model_id=model_id, step=step, extension=extension)

    def output(self, model_id, step, layer_index, extension=True):
        return self._tensor('output',layer_index, model_id=model_id, step=step, extension=extension)

    def _tensor(self, name, layer_index, model_id=None, step=None, extension=True):
        file_name = '%s-%03d' % (name, layer_index)
        if extension:
            file_name += '.npy'

        return os.path.join(self.step(step, model_id), file_name)

    def find_losses(self, model_id=None):
        return self._find(self, 'losses', model_id)

    def find_accuracies(self, model_id=None):
        return self._find(self, 'accuracies', model_id)

    def find_weight(self, model_id=None, step=None, layer_index=None):
        return self._find_tensor(self, 'weight', model_id,step, layer_index)

    def find_bias(self, model_id=None, step=None, layer_index=None):
        return self._find_tensor(self, 'bias', model_id,step, layer_index)

    def find_output(self, model_id=None, step=None, layer_index=None):
        return self._find_tensor(self, 'output', model_id,step, layer_index)

    def _find(self, name, model_id=None):
        return glob.glob(os.path.join(self.model(model_id), '%s.npy' % name))

    def _find_tensor(self, name, model_id=None, step=None, layer_index=None):
        path = '*' if model_id is None else self.model(model_id)
        path = os.path.join(path, '*' if step is None else self.step(step))
        path = os.path.join(path, '%s-*.npy' % name if layer_index is None else self._tensor(name, layer_index))

        return glob.glob(path)


class Plot:
    """データのプロットを行うクラス."""

    def __init__(self, base):
        self._base = base

    def loss(self, values, save=True, show=False):
        self.scalar(values, 'accuracy', 'step', '', xlim=[
                    0, len(values)], save=save, show=show)

    def accuracy(self, values, save=True, show=False):
        self.scalar(values, 'accuracy', 'step', '', xlim=[
                    0, len(values)], ylim=[0, 1.1], save=save, show=show)

    def distribution(self, values, title, xlim=None, ylim=None, save=True, show=False):
        sns.set(style='darkgrid', palette='muted', color_codes=True)
        plt.figure()
        sns.distplot(values)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(title)
        if save:
            if not os.path.isdir(self._base):
                os.makedirs(self._base)
            plt.savefig(os.path.join(self._base, '%s.png' % title))
        if show:
            plt.show()
        plt.close()

    def scalar(self, values, title, xlabel, ylabel, xlim=None, ylim=None, save=True, show=False):
        sns.set(style='darkgrid', palette='muted', color_codes=True)
        plt.figure()
        plt.plot(values)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if save:
            if not os.path.isdir(self._base):
                os.makedirs(self._base)
            plt.savefig(os.path.join(self._base, '%s.png' % title))
        if show:
            plt.show()
        plt.close()


class Model(metaclass=ABCMeta):
    @abstractmethod
    def initialize(self, sess):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def compute_loss(self, inputs, targets):
        pass

    @abstractmethod
    def optimize(self, inputs, targets):
        pass

    @abstractmethod
    def evaluate(self, inputs, targets):
        pass

    @abstractmethod
    def outputs(self, inputs):
        pass

    @property
    @abstractmethod
    def weights(self):
        pass

    @property
    @abstractmethod
    def biases(self):
        pass


class BasicModel(Model):
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

    def forward(self, inputs):
        return self._sess.run(self._y, feed_dict={self._x: inputs})

    def optimize(self, inputs, targets):
        return self._sess.run(self._optimizer, feed_dict={self._x: inputs, self._d: targets})

    def compute_loss(self, inputs, targets):
        return self._sess.run(self._loss, feed_dict={self._x: inputs, self._d: targets})

    def evaluate(self, inputs, targets):
        return self._sess.run(self._accuracy, feed_dict={self._x: inputs, self._d: targets})

    def outputs(self, inputs):
        return [self._sess.run(o, feed_dict={self._x: inputs}) for o in self._outputs]

    @property
    def weights(self):
        return [self._sess.run(w) for w in self._weights]

    @property
    def biases(self):
        return [self._sess.run(b) for b in self._biases]


class BatchNormedModel(Model):
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

    def initialize(self, sess):
        self._sess = sess
        return self._sess.run(tf.initialize_all_variables())

    def forward(self, inputs):
        return self._sess.run(self._y, feed_dict={self._x: inputs, self._is_training: False})

    def optimize(self, inputs, targets):
        return self._sess.run(self._optimizer, feed_dict={self._x: inputs, self._d: targets, self._is_training: True})

    def compute_loss(self, inputs, targets):
        return self._sess.run(self._loss, feed_dict={self._x: inputs, self._d: targets, self._is_training: False})

    def evaluate(self, inputs, targets):
        return self._sess.run(self._accuracy, feed_dict={self._x: inputs, self._d: targets, self._is_training: False})

    def outputs(self, inputs):
        return [self._sess.run(o, feed_dict={self._x: inputs, self._is_training: False}) for o in self._outputs]

    @property
    def weights(self):
        return [self._sess.run(w) for w in self._weights]

    @property
    def biases(self):
        return [self._sess.run(b) for b in self._biases]


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


def compute_accuracy(model, data_set, batch_size=100):
    accuracy = 0.
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


def learn(models, train_data_set, test_data_set, max_step, batch_size, save_dir, save_interval=100):
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
            for model_id, model in enumerate(models):
                test_accuracy = compute_accuracy(model, test_data_set)
                test_accuracies[model_id].append(test_accuracy)

                # train
                inputs, targets = train_data_set.next_batch(batch_size)
                model.optimize(inputs, targets)
                train_loss = model.compute_loss(inputs, targets)
                train_losses[model_id].append(train_loss)

                print('Model %d :: step [%d] loss %f accuracy %f' % (
                    model_id, step, train_loss, test_accuracy))

                # save
                if step % save_interval == 0:
                    if not os.path.isdir(Path(save_dir).step(step)):
                        os.makedirs(Path(save_dir).step(step))

                    for i, w in enumerate(model.weights):
                        np.save(Path(save_dir).weight(
                            model_id, step, i, extension=True), w)
                    for i, b in enumerate(model.biases):
                        np.save(Path(save_dir).bias(
                            model_id, step, i, extension=True), b)
                    for i, o in enumerate(model.outputs(inputs)):
                        np.save(Path(save_dir).output(
                            model_id, step, i, extension=True), o)

        for i, losses in enumerate(train_losses):
            np.save(Path(save_dir).losses(i, extension=True), np.array(losses))

        for i, accuracies in enumerate(test_accuracies):
            np.save(Path(save_dir).accuracies(
                i, extension=True), np.array(accuracies))


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
    # plt.xlim([-2,2])
    plt.title(title)
    plt.savefig(os.path.join(save_dir, '%s.png' % title))
    if show:
        plt.show()
    plt.close()


def _plot_result(save_dir, model_id, last_hidden_output_index, interval=100):
    save_path = os.path.join(save_dir, 'figure')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    path = Path(save_dir)
    plot = Plot(path.figure(model_id))

    losses = np.load(path.find_bias(model_id)[0])
    plot.loss(losses)
    accuracies = np.load(path.find_accuracies(model_id)[0])
    plot.accuracy(accuracies)

    # plot distributions of last hidden layer output
    files = path.find_output(layer_index=last_hidden_output_index)
    sorted(files)
    outputs = [np.load(f) for f in path.find_output(layer_index=last_hidden_output_index)]

    xlim, ylim = [100, -100], [100, -100]
    for output in outputs:
        y, x = np.histogram(output, normed=True)
        xlim = [min(xlim[0], x.min()), max(xlim[1], x.max())]
        ylim = [min(ylim[0], y.min()), max(ylim[1], y.max())]

    def scale_lim(lim, scale=1.5):
        interval = lim[1] - lim[0]
        scaled = interval * 0.75
        avg = interval / 2
        return [avg - scaled, avg + scaled]
    xlim = scale_lim(xlim)
    ylim = [0, ylim[1]*1.1]

    percentiles = [15, 5, 85]


    def plot_dist(step, percentile, values, indices):
        plot.distribution(values[indices[int(len(indices) * 0.15)]], 'step-%05d' % step, xlim=xlim, ylim=ylim)

    for i, output in enumerate(outputs):
        step = i * interval
        indices = np.argsort(output.mean(axis=1))
        for percentile in percentiles:
        plot.distribution(output[indices[int(len(indices) * 0.15)]], 'step-%05d' % (i * 100), xlim=xlim, ylim=ylim)

    for i, f in enumerate(files):
        output = np.load(f)
        indices = np.argsort(output.mean(axis=1))
        plot.distribution(output[indices[int(len(indices) * 0.15)]], 'step-%05d' % (i * 100), xlim=xlim, ylim=ylim)

        plot_distribution(output[indices[int(len(indices) * 0.15)]],
                          'step-%05d-15th-persentaile' % (i * 100), os.path.join(save_path, '15'))
        plot_distribution(output[indices[int(len(indices) * 0.5)]],
                          'step-%05d-50th-persentaile' % (i * 100), os.path.join(save_path, '50'))
        plot_distribution(output[indices[int(len(indices) * 0.85)]],
                          'step-%05d-85th-persentaile' % (i * 100), os.path.join(save_path, '85'))
    for i in [15, 50, 85]:
        os.system('convert -delay 10 -loop 0 %s %d.gif' %
                  (os.path.join(save_path, '%d' % i, '*.png'), i))


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
        for i in range(len(models)):
            _plot_result(os.path.join(save_dir, 'model%02d' %
                                      i, last_hidden_output_index[i]))


def test_model(learning=True, plot=True):
    num_dim = 28 * 28
    num_class = 10
    save_dir = './result'
    batch_size = 60
    max_step = 1000
    learning_rate = 0.5
    models = [BasicModel(num_dim, num_class, learning_rate),
              BatchNormedModel(num_dim, num_class, learning_rate)]
    last_hidden_output_index = [5, 8]
    if learning:
        mnist_data_set = input_data.read_data_sets('./mnist')
        train_data_set = mnist_data_set.train
        test_data_set = mnist_data_set.test

        learn(models, train_data_set, test_data_set,
              max_step, batch_size, save_dir)

    if plot:
        for i in range(len(models)):
            _plot_result(os.path.join(save_dir, 'model%02d' %
                                      i), last_hidden_output_index[i])


if __name__ == '__main__':

    test_model(learning=False)
