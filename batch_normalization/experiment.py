import tensorflow as tf
import numpy as np
import os
from collections import namedtuple
from math import ceil


Config = namedtuple('Config', ['saveDirectory', 'epochNumber', 'batchSize', 'model'])

# step 数は15kでいい
# batch normalization の有無により訓練が加速することを確かめる.
experiment1 = Config(saveDirectory='./experiment/1/', batchSize=60, epochNumber=20,model=None)

# batch normalization によりミニバッチの分布が安定することを確かめる.
experiment2 = Config(saveDirectory='./experiment/2/', batchSize=60, epochNumber=20,model=None)

# batch normalization により正則化が行われることを確かめる.
experiment3 = Config(saveDirectory='./experiment/2/', batchSize=60, epochNumber=20,model=None)


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
        raise ValueError('x must be (batch_size, dim) or (batch_size, height, width, ch) but %s' %(str(shape)))

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

def get_next_batch(input_set, target_set, batch_size, is_training=True):
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
    indices = np.random.permutation(data_size) if is_training else np.arange(data_size)
    size = ceil(data_size / batch_size)

    for step in range(size):
        n = min(batch_size, data_size - batch_size * step)
        yield n, (input_set[indices[index:index + n]], target_set[indices[index:index + n]])
        index += n


def compute_accuracy(model, input_set, target_set, batch_size=100):
    accuracy = 0.
    data_size = 0
    for n, sample in get_next_batch(input_set, target_set, batch_size, is_training=False):
        inputs, targets = sample
        accuracy += model.evaluate(inputs, targets) * n
        data_size += n

    accuracy /= data_size
    return accuracy


def main(config: Config):
    save_dir = config.saveDirectory
    batch_size = config.batchSize
    epoch_num = config.epochNumber
    model = config.model

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print('save directory     : %s' % save_dir)
    print('batch size         : %d' % batch_size)
    print('epoch number       : %d' % epoch_num)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    train_inputs = MNIST.train.images
    train_targets = MNIST.train.labels
    train_data_size = len(train_inputs)

    test_inputs = MNIST.test.images
    test_targets = MNIST.test.labels
    test_data_size = len(test_inputs)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        step = 0
        for epoch in range(epoch_num):
            train_epoch_loss = 0.
            test_top_accuracy = 0.

            for n, sample in get_next_batch(train_inputs, train_targets, batch_size):
                # test
                test_accuracy = compute_accuracy(model, test_inputs, test_targets)
                test_accuracies.append(test_accuracy)
                test_top_accuracy = max(test_top_accuracy, test_accuracy)

                # train
                inputs, targets = sample
                loss = model.compute_loss(inputs, targets)
                model.optimize(inputs, targets)
                train_losses.append(loss)
                train_epoch_loss += loss * n

                step += 1

            train_epoch_loss /= train_data_size
            print('Epoch [%d] loss %f top accuracy %f' % (epoch, train_epoch_loss, test_top_accuracy))

        np.save(os.path.join(save_dir, 'loss'), np.array(train_losses))
        np.save(os.path.join(save_dir, 'accuracy'), np.array(test_accuracies))



def main():
    pass

if __name__ == '__main__':
    main()
