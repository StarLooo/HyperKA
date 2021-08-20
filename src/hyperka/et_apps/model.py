import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from hyperka.et_apps.util import embed_init
# from hyperka.et_apps.util import embed_init, glorot, zeros
from hyperka.hyperbolic.poincare import PoincareManifold
from hyperka.et_funcs.test_funcs import eval_type_hyperbolic


# 这段代码是学姐给的改成torch后的版本
# TODO:里面各参数和输入输出的维度还没有全部搞清楚
class GCNLayer(nn.Module):
    def __init__(self, adj: torch.Tensor, input_dim: int, output_dim: int, layer_id: int, poincare: PoincareManifold,
                 has_bias: bool = True, activation: nn.Module = None):
        super().__init__()
        self.poincare = poincare
        self.has_bias = has_bias
        self.activation = activation
        self.adj = adj
        self.weight_matrix = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float64, requires_grad=True))
        if has_bias:
            self.bias_vec = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64, requires_grad=True))
        else:
            # TODO: 不知道这里有啥用
            self.register_parameter("bias_vec", None)

    def forward(self, inputs: torch.Tensor, drop_rate: float = 0.0):
        pre_sup_tangent = self.poincare.log_map_zero(inputs)
        if drop_rate > 0.0:
            # TODO:这里作者的代码是*(1-drop_rate),但我觉得应该是/(1-drop_rate)
            pre_sup_tangent = F.dropout(pre_sup_tangent, p=drop_rate, training=self.training) * (
                    1 - drop_rate)  # not scaled up
        # 这里应该是要求：
        # pre_sup_tangent.shape = (1,input_dim), weight_matrix.shape = (input_dim, output_dim)
        assert pre_sup_tangent.shape[1] == self.weight_matrix.shape[0]
        output = torch.mm(pre_sup_tangent, self.weight_matrix)
        # output = torch.spmm(self.adj, output) //torch.spmm稀疏矩阵乘法的位置已经移动到torch.sparse中
        # TODO:这里的self.adj是干嘛的不是很清楚
        output = torch.sparse.mm(self.adj, output)
        output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
        if self.has_bias:
            bias_vec = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.bias_vec))
            output = self.poincare.mobius_addition(output, bias_vec)
            output = self.poincare.hyperbolic_projection(output)
        if self.activation is not None:
            output = self.activation(self.poincare.log_map_zero(output))
            output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
        return output


# class GCNLayer:
#     def __init__(self, adj, input_dim, output_dim, layer_id, poincare, bias=True, act=None, name=""):
#         self.poincare = poincare
#         self.bias = bias
#         self.act = act
#         self.adj = adj
#         with tf.compat.v1.variable_scope(name + "_gcn_layer_" + str(layer_id)):
#             self.weight_mat = tf.compat.v1.get_variable("gcn_weights" + str(layer_id),
#                                                         shape=[input_dim, output_dim],
#                                                         initializer=tf.glorot_uniform_initializer(),
#                                                         dtype=tf.float64)
#             if bias:
#                 self.bias_vec = tf.compat.v1.get_variable("gcn_bias" + str(layer_id),
#                                                           shape=[1, output_dim],
#                                                           initializer=tf.zeros_initializer(),
#                                                           dtype=tf.float64)
#
#     def call(self, inputs, drop_rate=0.0):
#         pre_sup_tangent = self.poincare.log_map_zero(inputs)
#         if drop_rate > 0.0:
#             pre_sup_tangent = tf.nn.dropout(pre_sup_tangent, rate=drop_rate) * (1 - drop_rate)  # not scaled up
#         output = tf.matmul(pre_sup_tangent, self.weight_mat)
#         output = tf.sparse.sparse_dense_matmul(self.adj, output)
#         output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
#         if self.bias:
#             bias_vec = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.bias_vec))
#             output = self.poincare.mobius_addition(output, bias_vec)
#             output = self.poincare.hyperbolic_projection(output)
#         if self.act is not None:
#             output = self.act(self.poincare.log_map_zero(output))
#             output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
#         return output


class HyperKA(nn.Module):
    def __init__(self, insnet, onto, instype, ins_adj, onto_adj, args):
        super().__init__()
        self.args = args
        self.poincare = PoincareManifold()
        self.activation = torch.tanh  # 激活函数

        # insnet和onto的结构如下：
        # [triples, train_ids_triples, test_ids_triples, total_ents_num, total_rels_num, total_triples_num]
        # instype的结构如下：
        # [[train_heads_id_list, train_tails_id_list],[test_heads_id_list, test_tails_id_list, test_head_tails_id_list]]

        self.ins_ent_num = insnet[3]
        self.ins_rel_num = insnet[4]
        self.onto_ent_num = onto[3]
        self.onto_rel_num = onto[4]

        self.ins_ent_list = insnet[0].ent_list
        self.onto_ent_list = onto[0].ent_list

        # sup(train)表示训练集
        self.train_ins_head = [item[0] for item in (insnet[1])]
        self.train_ins_tail = [item[2] for item in (insnet[1])]
        self.train_onto_head = [item[0] for item in (onto[1])]
        self.train_onto_tail = [item[2] for item in (onto[1])]

        # ref(test)表示测试集
        self.test_insnet_head = [item[0] for item in (insnet[2])]
        self.test_insnet_tail = [item[2] for item in (insnet[2])]
        self.test_onto_head = [item[0] for item in (onto[2])]
        self.test_onto_tail = [item[2] for item in (onto[2])]

        # sup(train)表示训练集
        self.train_instype_head = instype[0][0]
        self.train_instype_tail = instype[0][1]
        self.train_instype_link = list()
        for i in range(len(self.train_instype_head)):
            self.train_instype_link.append((self.train_instype_head[i], self.train_instype_tail[i]))
        print("# train_instype_link len:", len(self.train_instype_link))
        # TODO: IF USELESS?
        self.train_instype_set = set(self.train_instype_link)

        # ref(test)表示测试集
        self.test_instype_head = instype[1][0]
        self.test_instype_tail = instype[1][1]
        self.test_instype_link = list()
        for i in range(len(self.test_instype_head)):
            self.test_instype_link.append((self.test_instype_head[i], self.test_instype_tail[i]))
        print("# test_instype_link len:", len(self.test_instype_link))
        # TODO: IF USELESS?
        self.test_instype_set = set(self.test_instype_link)

        self.test_all_ins_types = instype[1][2]

        # 这里的稀疏张量不知道参数对不对
        # list(zip(*--))这种写法详见torch api: https://pytorch.org/docs/1.9.0/sparse.html#sparse-uncoalesced-coo-docs
        self.ins_adj_mat = torch.sparse_coo_tensor(indices=list(zip(*ins_adj[0])), values=ins_adj[1],
                                                   size=ins_adj[2])
        self.onto_adj_mat = torch.sparse_coo_tensor(indices=list(zip(*onto_adj[0])), values=onto_adj[1],
                                                    size=onto_adj[2])
        # self.ins_adj_mat = tf.SparseTensor(indices=ins_adj[0], values=ins_adj[1], dense_shape=ins_adj[2])
        # self.onto_adj_mat = tf.SparseTensor(indices=onto_adj[0], values=onto_adj[1], dense_shape=onto_adj[2])

        self.ins_layers = list()
        self.onto_layers = list()
        self.ins_output = list()
        self.onto_output = list()
        self.ins_layer_num = args.ins_layer_num
        self.onto_layer_num = args.onto_layer_num

        # 这里是tensorflow的会话机制，torch应该不需要
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.session = tf.Session(config=config)
        # self._generate_triple_graph()
        # self._generate_mapping_graph()
        # tf.global_variables_initializer().run(session=self.session)

        self._generate_parameters()
        self.triple_optimizer = torch.optim.Adam(params=self.parameters(), lr=self.args.learning_rate)
        self.mapping_optimizer = torch.optim.Adam(params=self.parameters(), lr=self.args.learning_rate)

    # 我自己加了以下两个个函数，用于解决placeholder和feed_dict翻译的问题，但不知道合不合适
    def _trans_triple_pos_neg_batch(self, triple_pos_neg_batch):
        ins_pos, ins_neg, onto_pos, onto_neg = triple_pos_neg_batch
        self.ins_pos_h = torch.LongTensor([x[0] for x in ins_pos])
        self.ins_pos_r = torch.LongTensor([x[1] for x in ins_pos])
        self.ins_pos_t = torch.LongTensor([x[2] for x in ins_pos])
        self.ins_neg_h = torch.LongTensor([x[0] for x in ins_neg])
        self.ins_neg_r = torch.LongTensor([x[1] for x in ins_neg])
        self.ins_neg_t = torch.LongTensor([x[2] for x in ins_neg])
        self.onto_pos_h = torch.LongTensor([x[0] for x in onto_pos])
        self.onto_pos_r = torch.LongTensor([x[1] for x in onto_pos])
        self.onto_pos_t = torch.LongTensor([x[2] for x in onto_pos])
        self.onto_neg_h = torch.LongTensor([x[0] for x in onto_neg])
        self.onto_neg_r = torch.LongTensor([x[1] for x in onto_neg])
        self.onto_neg_t = torch.LongTensor([x[2] for x in onto_neg])

    def _trans_mapping_pos_neg_batch(self, mapping_pos_neg_batch):
        self.link_pos_h, self.link_pos_t, self.link_neg_h, self.link_neg_t = mapping_pos_neg_batch
        self.link_pos_h = torch.LongTensor(self.link_pos_h)
        self.link_neg_h = torch.LongTensor(self.link_neg_h)
        self.link_pos_t = torch.LongTensor(self.link_pos_t)
        self.link_neg_t = torch.LongTensor(self.link_neg_t)

    # 图卷积
    def _graph_convolution(self):
        self.ins_output_list = list()  # reset
        self.onto_output_list = list()  # reset

        # ************************* instance gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        ins_output_embeddings = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        # ins_output_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ins_ent_embeddings))
        self.ins_output_list.append(ins_output_embeddings)
        for i in range(self.ins_layer_num):
            activation = self.activation
            if i == self.ins_layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(adj=self.ins_adj_mat, input_dim=self.args.ins_dim,
                                 output_dim=self.args.ins_dim, layer_id=i, poincare=self.poincare,
                                 activation=activation)
            self.ins_layers.append(gcn_layer)
            ins_output_embeddings = gcn_layer.forward(ins_output_embeddings)
            # ins_output_embeddings = gcn_layer.call(ins_output_embeddings)
            ins_output_embeddings = self.poincare.mobius_addition(ins_output_embeddings, self.ins_output_list[-1])
            ins_output_embeddings = self.poincare.hyperbolic_projection(ins_output_embeddings)
            self.ins_output_list.append(ins_output_embeddings)

        # ************************* ontology gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        onto_output_embeddings = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        # onto_output_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.onto_ent_embeddings))
        self.onto_output_list.append(onto_output_embeddings)
        for i in range(self.onto_layer_num):
            activation = self.activation
            if i == self.onto_layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(adj=self.onto_adj_mat, input_dim=self.args.onto_dim,
                                 output_dim=self.args.onto_dim, layer_id=i, poincare=self.poincare,
                                 activation=activation)
            self.onto_layers.append(gcn_layer)
            onto_output_embeddings = gcn_layer.forward(onto_output_embeddings)
            onto_output_embeddings = self.poincare.mobius_addition(onto_output_embeddings, self.onto_output_list[-1])
            onto_output_embeddings = self.poincare.hyperbolic_projection(onto_output_embeddings)
            self.onto_output_list.append(onto_output_embeddings)

    # 生成初始化的各参数矩阵
    def _generate_parameters(self):
        # 获得初始化的ins的嵌入向量
        self.ins_ent_embeddings = embed_init(size=(self.ins_ent_num, self.args.ins_dim),
                                             name="ins_ent_embeds",
                                             method='glorot_uniform_initializer')
        self.ins_ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ins_ent_embeddings))
        self.ins_ent_embeddings = nn.Parameter(self.ins_ent_embeddings)
        # with tf.variable_scope('instance_entity' + 'embeddings'):
        #     self.ins_ent_embeddings = embed_init(self.ins_ent_num, self.params.dim, "ins_ent_embeds",
        #                                          method='glorot_uniform_initializer')
        #     self.ins_ent_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.ins_ent_embeddings))

        self.onto_ent_embeddings = embed_init(size=(self.onto_ent_num, self.args.onto_dim),
                                              name="onto_ent_embeds",
                                              method='glorot_uniform_initializer')
        self.onto_ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.onto_ent_embeddings))
        self.onto_ent_embeddings = nn.Parameter(self.onto_ent_embeddings)
        # with tf.variable_scope('ontology_entity' + 'embeddings'):
        #     self.onto_ent_embeddings = embed_init(self.onto_ent_num, self.params.onto_dim, "onto_ent_embeds",
        #                                           method='glorot_uniform_initializer')
        #     self.onto_ent_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.onto_ent_embeddings))

        self.ins_rel_embeddings = embed_init(size=(self.ins_rel_num, self.args.ins_dim),
                                             name="ins_rel_embeds",
                                             method='glorot_uniform_initializer')
        self.ins_rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ins_rel_embeddings))
        self.ins_rel_embeddings = nn.Parameter(self.ins_rel_embeddings)
        # with tf.variable_scope('instance_relation' + 'embeddings'):
        #     self.ins_rel_embeddings = embed_init(self.ins_rel_num, self.params.dim, "ins_rel_embeds",
        #                                          method='glorot_uniform_initializer')
        #     self.ins_rel_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.ins_rel_embeddings))

        self.onto_rel_embeddings = embed_init(size=(self.onto_rel_num, self.args.onto_dim),
                                              name="onto_rel_embeds",
                                              method='glorot_uniform_initializer')
        self.onto_rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.onto_rel_embeddings))
        self.onto_rel_embeddings = nn.Parameter(self.onto_rel_embeddings)
        # with tf.variable_scope('ontology_relation' + 'embeddings'):
        #     self.onto_rel_embeddings = embed_init(self.onto_rel_num, self.params.onto_dim, "onto_rel_embeds",
        #                                           method='glorot_uniform_initializer')
        #     self.onto_rel_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.onto_rel_embeddings))

        if self.args.mapping:
            # with tf.variable_scope('instance_mapping' + 'embeddings'):
            size = (self.args.ins_dim, self.args.onto_dim)
            print("init instance mapping matrix using", "orthogonal", "with size of", size)
            self.ins_mapping_matrix = nn.init.orthogonal_(
                tensor=torch.empty(size=size, dtype=torch.float64,
                                   requires_grad=True))
            self.ins_mapping_matrix = nn.Parameter(self.ins_mapping_matrix)
            # self.ins_mapping_matrix = tf.get_variable('mapping_matrix',
            #                                               dtype=tf.float64,
            #                                               shape=[self.params.dim, self.params.onto_dim],
            #                                               initializer=tf.initializers.orthogonal(dtype=tf.float64))

    # 黎曼梯度下降，Adam优化
    # TODO:这里的架构确实很迷惑
    def _adapt_riemannian_optimizer(self, loss):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.args.learning_rate)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # opt = tf.train.AdamOptimizer(self.params.learning_rate)
        # trainable_grad_vars = opt.compute_gradients(loss)
        # grad_vars = [(g, v) for g, v in trainable_grad_vars if g is not None]
        # 计算黎曼梯度
        rescaled = [(g * (1. - tf.reshape(tf.norm(v, axis=1), (-1, 1)) ** 2) ** 2 / 4., v) for g, v in grad_vars]
        # 这里貌似是梯度裁剪
        train_op = opt.apply_gradients(rescaled)
        return train_op

    # 计算triple loss的内部函数
    def _compute_triple_loss(self, phs, prs, pts, nhs, nrs, nts):
        pos_distance = self.poincare.distance(self.poincare.mobius_addition(phs, prs), pts)
        neg_distance = self.poincare.distance(self.poincare.mobius_addition(nhs, nrs), nts)
        pos_score = torch.sum(pos_distance, 1)
        # pos_score = tf.reduce_sum(pos_distance, 1)
        neg_score = torch.sum(neg_distance, 1)
        # neg_score = tf.reduce_sum(neg_distance, 1)
        pos_loss = torch.sum(torch.relu(pos_score))
        # pos_loss = tf.reduce_sum(tf.nn.relu(pos_score))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.args.neg_triple_margin, dtype=torch.float64) - neg_score))
        # neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.params.neg_triple_margin, dtype=tf.float64) - neg_score))
        triple_loss = pos_loss + neg_loss
        return triple_loss

    # 计算mapping loss的内部函数
    def _compute_mapping_loss(self, mapped_link_phs_embeds, mapped_link_nhs_embeds,
                              link_pts_embeds, link_nts_embeds):
        sup_distance = torch.sum(self.poincare.distance(mapped_link_phs_embeds, link_pts_embeds), dim=1)
        neg_distance = torch.sum(self.poincare.distance(mapped_link_nhs_embeds, link_nts_embeds), dim=1)
        pos_loss = torch.sum(torch.relu(sup_distance))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.args.neg_typing_margin, dtype=torch.float64) - neg_distance))
        mapping_loss = pos_loss + neg_loss
        return mapping_loss

    # 计算triple loss的包装函数
    def optimize_triple_loss(self, triple_pos_neg_batch):
        # 对于tf版本的代码
        # 这里需要理解一下占位符placeholder()是怎么用的
        # self.ins_pos_h = tf.placeholder(tf.int32, shape=[None], name="ins_pos_h")
        # self.ins_pos_r = tf.placeholder(tf.int32, shape=[None], name="ins_pos_r")
        # self.ins_pos_t = tf.placeholder(tf.int32, shape=[None], name="ins_pos_t")
        # self.ins_neg_h = tf.placeholder(tf.int32, shape=[None], name="ins_neg_h")
        # self.ins_neg_r = tf.placeholder(tf.int32, shape=[None], name="ins_neg_r")
        # self.ins_neg_t = tf.placeholder(tf.int32, shape=[None], name="ins_neg_t")
        # self.onto_pos_h = tf.placeholder(tf.int32, shape=[None], name="onto_pos_h")
        # self.onto_pos_r = tf.placeholder(tf.int32, shape=[None], name="onto_pos_r")
        # self.onto_pos_t = tf.placeholder(tf.int32, shape=[None], name="onto_pos_t")
        # self.onto_neg_h = tf.placeholder(tf.int32, shape=[None], name="onto_neg_h")
        # self.onto_neg_r = tf.placeholder(tf.int32, shape=[None], name="onto_neg_h")
        # self.onto_neg_t = tf.placeholder(tf.int32, shape=[None], name="onto_neg_h")
        # ***********************************************************************************

        # 这一段是不是与_generate_parameters中重复了？
        ins_ent_embeddings = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        ins_rel_embeddings = self.poincare.hyperbolic_projection(self.ins_rel_embeddings)
        onto_ent_embeddings = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        onto_rel_embeddings = self.poincare.hyperbolic_projection(self.onto_rel_embeddings)

        # 这里用torch.nn.functional中的embedding函数来替换tensorflow中的embedding_lookup不知道对不对
        self._trans_triple_pos_neg_batch(triple_pos_neg_batch)
        ins_phs_embeds = F.embedding(input=self.ins_pos_h, weight=ins_ent_embeddings)
        ins_prs_embeds = F.embedding(input=self.ins_pos_r, weight=ins_rel_embeddings)
        ins_pts_embeds = F.embedding(input=self.ins_pos_t, weight=ins_ent_embeddings)
        ins_nhs_embeds = F.embedding(input=self.ins_neg_h, weight=ins_ent_embeddings)
        ins_nrs_embeds = F.embedding(input=self.ins_neg_r, weight=ins_rel_embeddings)
        ins_nts_embeds = F.embedding(input=self.ins_neg_t, weight=ins_ent_embeddings)
        # ins_phs_embeds = tf.nn.embedding_lookup(ins_ent_embeddings, self.ins_pos_h)
        # ins_prs_embeds = tf.nn.embedding_lookup(ins_rel_embeddings, self.ins_pos_r)
        # ins_pts_embeds = tf.nn.embedding_lookup(ins_ent_embeddings, self.ins_pos_t)
        # ins_nhs_embeds = tf.nn.embedding_lookup(ins_ent_embeddings, self.ins_neg_h)
        # ins_nrs_embeds = tf.nn.embedding_lookup(ins_rel_embeddings, self.ins_neg_r)
        # ins_nts_embeds = tf.nn.embedding_lookup(ins_ent_embeddings, self.ins_neg_t)
        ins_triple_loss = self._compute_triple_loss(ins_phs_embeds, ins_prs_embeds, ins_pts_embeds,
                                                    ins_nhs_embeds, ins_nrs_embeds, ins_nts_embeds)

        onto_phs_embeds = F.embedding(input=self.onto_pos_h, weight=onto_ent_embeddings)
        onto_prs_embeds = F.embedding(input=self.onto_pos_r, weight=onto_rel_embeddings)
        onto_pts_embeds = F.embedding(input=self.onto_pos_t, weight=onto_ent_embeddings)
        onto_nhs_embeds = F.embedding(input=self.onto_neg_h, weight=onto_ent_embeddings)
        onto_nrs_embeds = F.embedding(input=self.onto_neg_r, weight=onto_rel_embeddings)
        onto_nts_embeds = F.embedding(input=self.onto_neg_t, weight=onto_ent_embeddings)
        # onto_phs_embeds = tf.nn.embedding_lookup(onto_ent_embeddings, self.onto_pos_h)
        # onto_prs_embeds = tf.nn.embedding_lookup(onto_rel_embeddings, self.onto_pos_r)
        # onto_pts_embeds = tf.nn.embedding_lookup(onto_ent_embeddings, self.onto_pos_t)
        # onto_nhs_embeds = tf.nn.embedding_lookup(onto_ent_embeddings, self.onto_neg_h)
        # onto_nrs_embeds = tf.nn.embedding_lookup(onto_rel_embeddings, self.onto_neg_r)
        # onto_nts_embeds = tf.nn.embedding_lookup(onto_ent_embeddings, self.onto_neg_t)
        onto_triple_loss = self._compute_triple_loss(onto_phs_embeds, onto_prs_embeds, onto_pts_embeds,
                                                     onto_nhs_embeds, onto_nrs_embeds, onto_nts_embeds)

        triple_loss = ins_triple_loss + onto_triple_loss

        print("all the parameters:")
        print(*[(name, param.shape) for name, param in self.named_parameters()])

        self.triple_optimizer.zero_grad()
        triple_loss.backward()
        self.triple_optimizer.step()

        print("triple_loss刚改到这里")
        os.system("pause")

        return triple_loss

        # self.triple_optimizer = self._generate_riemannian_optimizer(self.triple_loss)

    # 计算mapping loss的包装函数
    def optimize_mapping_loss(self, mapping_pos_neg_batch):
        # self.cross_pos_left = tf.placeholder(tf.int32, shape=[None], name="cross_pos_left")
        # self.cross_pos_right = tf.placeholder(tf.int32, shape=[None], name="cross_pos_right")
        # 进行论文中所说的图卷积
        self._graph_convolution()
        # ins_embeddings和onto_embeddings卷积后得到的嵌入向量
        ins_embeddings = self.ins_output_list[-1]
        onto_embeddings = self.onto_output_list[-1]
        if self.args.combine:
            ins_embeddings = self.poincare.mobius_addition(ins_embeddings, self.ins_output_list[0])
            onto_embeddings = self.poincare.mobius_addition(onto_embeddings, self.onto_output_list[0])

        self._trans_mapping_pos_neg_batch(mapping_pos_neg_batch)
        # 这一段是不是与_generate_parameters中重复了？
        ins_ent_embeddings = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        ins_rel_embeddings = self.poincare.hyperbolic_projection(self.ins_rel_embeddings)
        onto_ent_embeddings = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        onto_rel_embeddings = self.poincare.hyperbolic_projection(self.onto_rel_embeddings)
        link_phs_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.link_pos_h, weight=ins_ent_embeddings))
        link_pts_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.link_pos_t, weight=onto_ent_embeddings))

        # cross_left = tf.nn.embedding_lookup(ins_embeddings, self.cross_pos_left)
        # cross_left = self.poincare.hyperbolic_projection(cross_left)
        # cross_right = tf.nn.embedding_lookup(onto_embeddings, self.cross_pos_right)
        # cross_right = self.poincare.hyperbolic_projection(cross_right)

        mapped_link_phs_embeds = self.poincare.mobius_matmul(link_phs_embeds, self.ins_mapping_matrix)
        # mapped_sup_embeds1 = self.poincare.mobius_matmul(cross_left, self.ins_mapping_matrix)

        # sup_distance = self.poincare.distance(mapped_sup_embeds1, cross_right)
        # sup_distance = tf.reduce_sum(sup_distance, 1)

        # *****************add neg sample***********************************************
        # self.cross_neg_left = tf.placeholder(tf.int32, shape=[None], name="cross_neg_left")
        # self.cross_neg_right = tf.placeholder(tf.int32, shape=[None], name="cross_neg_right")

        link_nhs_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.link_neg_h, weight=ins_ent_embeddings))
        link_nts_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.link_neg_t, weight=onto_ent_embeddings))

        # neg_embeds1 = tf.nn.embedding_lookup(ins_embeddings, self.cross_neg_left)
        # neg_embeds1 = self.poincare.hyperbolic_projection(neg_embeds1)
        # neg_embeds2 = tf.nn.embedding_lookup(onto_embeddings, self.cross_neg_right)
        # neg_embeds2 = self.poincare.hyperbolic_projection(neg_embeds2)

        mapped_link_nhs_embeds = self.poincare.mobius_matmul(link_nhs_embeds, self.ins_mapping_matrix)
        # mapped_neg_embeds1 = self.poincare.mobius_matmul(neg_embeds1, self.ins_mapping_matrix)

        # neg_distance = self.poincare.distance(mapped_neg_embeds1, neg_embeds2)
        # neg_distance = tf.reduce_sum(neg_distance, 1)

        # pos_loss = tf.reduce_sum(tf.nn.relu(sup_distance))
        # neg_loss = tf.reduce_sum(
        #     tf.nn.relu(tf.constant(self.params.neg_typing_margin, dtype=tf.float64) - neg_distance))
        # self.mapping_loss = pos_loss + neg_loss
        # self.mapping_optimizer = self._generate_riemannian_optimizer(self.mapping_loss)

        mapping_loss = self._compute_mapping_loss(mapped_link_phs_embeds, mapped_link_nhs_embeds,
                                                  link_pts_embeds, link_nts_embeds)
        mapping_train_params = self.parameters()
        self.mapping_optimizer.zero_grad()
        mapping_loss.backward()
        self.mapping_optimizer.step()

        print("mapping_loss刚改到这里")
        os.system("pause")

    # 进行测试
    # 应该不太重要，没认真看，跳过了
    def test(self):
        t = time.time()
        ins_embeddings = self.ins_output[-1]
        onto_embeddings = self.onto_output[-1]
        if self.args.combine:
            ins_embeddings = self.poincare.mobius_addition(ins_embeddings, self.ins_output[0])
            onto_embeddings = self.poincare.mobius_addition(onto_embeddings, self.onto_output[0])

        ref_ins_embed = tf.nn.embedding_lookup(ins_embeddings, self.ref_ent1)
        ref_ins_embed = self.poincare.hyperbolic_projection(ref_ins_embed)
        ref_ins_embed = tf.matmul(self.poincare.log_map_zero(ref_ins_embed), self.ins_mapping_matrix)
        ref_ins_embed = self.poincare.exp_map_zero(ref_ins_embed)
        ref_ins_embed = self.poincare.hyperbolic_projection(ref_ins_embed)
        # ref_ins_embed = self.poincare.mobius_matmul(ref_ins_embed, self.ins_mapping_matrix)
        ref_ins_embed = ref_ins_embed.eval(session=self.session)

        onto_embed = onto_embeddings
        onto_embed = self.poincare.hyperbolic_projection(onto_embed)
        onto_embed = onto_embed.eval(session=self.session)
        hits1 = eval_type_hyperbolic(ref_ins_embed, onto_embed, self.all_ref_type,
                                     self.args.ent_top_k, self.args.nums_threads, greedy=True,
                                     mess="greedy ent typing by hyperbolic")
        eval_type_hyperbolic(ref_ins_embed, onto_embed, self.all_ref_type, self.args.ent_top_k,
                             self.args.nums_threads, greedy=False, mess="ent typing by hyperbolic")

        print("test totally costs time = {:.3f} s ".format(time.time() - t))
        return hits1

    # 应该不太重要，没认真看，跳过了
    def eval_ins_input_embed(self, is_map=False):
        embeds = tf.nn.embedding_lookup(self.ins_ent_embeddings, self.ins_entities)
        if is_map:
            embeds = self.poincare.mobius_matmul(embeds, self.ins_mapping_matrix)
        return embeds.eval(session=self.session)

    # 应该不太重要，没认真看，跳过了
    def eval_onto_input_embed(self):
        return tf.nn.embedding_lookup(self.onto_ent_embeddings, self.onto_entities).eval(session=self.session)
