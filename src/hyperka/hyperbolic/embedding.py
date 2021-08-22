import tensorflow as tf
import torch
import util

'''
    该embedding.py文件在et任务中没有作用，因此没有完成修改(但前期好像改了一点点)
'''


class Embedding(tf.keras.Model):
    def get_config(self):
        raise NotImplementedError

    def __init__(self, num_objects, dim, manifold, sparse=True):
        super(Embedding, self).__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.manifold = manifold
        self.sparse = sparse
        self.distance = manifold.distance
        self.pre_hook = None
        self.post_hook = None
        self.eps = 1e-10
        scale = 1e-4

        self.emb = 2 * scale * (torch.rand(num_objects, dim, dtype=torch.float64, requires_grad=True) - 0.5)

    def _forward(self, e):

        u = e[:, 0:1]
        v = e[:, 1:]

        changed_from = torch.nn.Embedding.from_pretrained(self.emb)(u)
        changed_to = torch.nn.Embedding.from_pretrained(self.emb)(v)

        changed_res = -self.distance(changed_from, changed_to)
        if util.DEBUG:
            origin_from = tf.nn.embedding_lookup(self.emb, u)
            origin_to = tf.nn.embedding_lookup(self.emb, v)
            origin_res = -self.distance(origin_from, origin_to)
            util.judge_change_equal(origin_res, changed_res)
        return changed_res

    def loss(self, preds, targets):
        changed_loss = torch.nn.CrossEntropyLoss(preds, targets)
        if util.DEBUG:
            origin_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=preds))
            util.judge_change_equal(origin_loss, changed_loss)
        return changed_loss

    def call(self, inputs, training=False, **kwargs):
        if self.pre_hook is not None:
            inputs = self.pre_hook(inputs)
        return self._forward(inputs)

    def add_variable(self, name, shape, dtype=None, initializer=None,
                     regularizer=None, trainable=True, constraint=None):
        pass


# some test to figure out funcs' usages
if __name__ == '__main__':
    tf_e = tf.constant([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    th_e = torch.tensor([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    print(tf_e)
    print(th_e)
    tf_u = tf.strided_slice(tf_e, [0, 0], [tf_e.shape[0], 1])
    th_u = th_e[:, 0:1]
    tf_v = tf.strided_slice(tf_e, [0, 1], [tf_e.shape[0], tf_e.shape[1]])
    th_v = th_e[:, 1:]
    print(tf_u)
    print(th_u)
    print(tf_v)
    print(th_v)
