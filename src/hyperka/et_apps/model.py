import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
# import tensorflow as tf
import numpy as np
from hyperka.et_apps.util import embed_init, glorot, zeros
from hyperka.hyperbolic.poincare import PoincareManifold
from hyperka.et_funcs.test_funcs import eval_type_hyperbolic


# 这段代码是学姐给的改成torch后的版本
class GCNLayer(nn.Module):
    def __init__(self, adj: torch.Tensor, input_dim: int, output_dim: int, layer_id: int, poincare: PoincareManifold,
                 bias: bool = True, act: nn.Module = None):
        super().__init__()
        self.poincare = poincare
        self.bias = bias
        self.act = act
        self.adj = adj
        self.weight_mat = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias_vec = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter("bias_vec", None)

    def forward(self, inputs: torch.Tensor, drop_rate: float = 0.0):
        pre_sup_tangent = self.poincare.log_map_zero(inputs)
        if drop_rate > 0.0:
            pre_sup_tangent = F.dropout(pre_sup_tangent, p=drop_rate) * (1 - drop_rate)  # not scaled up
        output = torch.mm(pre_sup_tangent, self.weight_mat)
        # output = torch.spmm(self.adj, output) //torch.spmm稀疏矩阵乘法的位置已经移动到torch.sparse中
        output = torch.sparse.mm(self.adj, output)
        output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
        if self.bias:
            bias_vec = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.bias_vec))
            output = self.poincare.mobius_addition(output, bias_vec)
            output = self.poincare.hyperbolic_projection(output)
        if self.act is not None:
            output = self.act(self.poincare.log_map_zero(output))
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


class HyperKA:
    def __init__(self, ins_list, onto_list, cross, ins_adj, onto_adj, params):
        # instance_list和ontology_list的结构如下：
        # [triples, train_ids_triples, test_ids_triples, total_ents_num, total_props_num, total_triples_num]
        # cross_seed的结构如下：
        # [[train_heads_id_list, train_tails_id_list],[test_heads_id_list, test_tails_id_list, test_head_tails_id_list]]

        self.ins_ent_num = ins_list[3]
        self.ins_rel_num = ins_list[4]
        self.onto_ent_num = onto_list[3]
        self.onto_rel_num = onto_list[4]

        self.ins_entities = ins_list[0].ent_list
        self.onto_entities = onto_list[0].ent_list

        # sup表示训练集
        self.ins_sup_ent1 = [item[0] for item in (ins_list[1])]
        self.ins_sup_ent2 = [item[2] for item in (ins_list[1])]
        self.onto_sup_ent1 = [item[0] for item in (onto_list[1])]
        self.onto_sup_ent2 = [item[2] for item in (onto_list[1])]

        # ref表示测试集
        self.ins_ref_ent1 = [item[0] for item in (ins_list[2])]
        self.ins_ref_ent2 = [item[2] for item in (ins_list[2])]
        self.onto_ref_ent1 = [item[0] for item in (onto_list[2])]
        self.onto_ref_ent2 = [item[2] for item in (onto_list[2])]

        # sup表示训练集
        self.seed_sup_ent1 = cross[0][0]
        self.seed_sup_ent2 = cross[0][1]
        self.seed_links = list()
        for i in range(len(self.seed_sup_ent1)):
            self.seed_links.append((self.seed_sup_ent1[i], self.seed_sup_ent2[i]))
        print("# seed associations len:", len(self.seed_links))
        self.seed_link_set = set(self.seed_links)

        # ref表示测试集
        self.ref_ent1 = cross[1][0]
        self.ref_ent2 = cross[1][1]
        self.ref_links = list()
        for i in range(len(self.ref_ent1)):
            self.ref_links.append((self.ref_ent1[i], self.ref_ent2[i]))
        print("# ref associations len:", len(self.ref_links))

        self.all_ref_type = cross[1][2]

        self.params = params
        self.poincare = PoincareManifold()
        # 这里的稀疏张量不知道参数对不对
        self.ins_adj_mat = torch.sparse_coo_tensor(indices=ins_adj[0], values=ins_adj[1], size=ins_adj[2])
        self.onto_adj_mat = torch.sparse_coo_tensor(indices=onto_adj[0], values=onto_adj[1], size=onto_adj[2])
        # self.ins_adj_mat = tf.SparseTensor(indices=ins_adj[0], values=ins_adj[1], dense_shape=ins_adj[2])
        # self.onto_adj_mat = tf.SparseTensor(indices=onto_adj[0], values=onto_adj[1], dense_shape=onto_adj[2])
        self.activation = torch.tanh  # 激活函数
        # self.activation = tf.tanh

        self.ins_layers = list()
        self.onto_layers = list()
        self.ins_output = list()
        self.onto_output = list()
        self.ins_layer_num = params.ins_layer_num
        self.onto_layer_num = params.onto_layer_num

        # 这里是tensorflow的会话机制，torch应该不需要
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.session = tf.Session(config=config)

        self._generate_variables()
        self._generate_triple_graph()
        self._generate_mapping_graph()

        # tf.global_variables_initializer().run(session=self.session)

    # 我自己加了以下两个个函数，用于解决placeholder和feed_dict翻译的问题，但不知道合不合适
    def _trans_triple_feed_dict(self, ins_pos, ins_neg, onto_pos, onto_neg):
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

    def _trans_mapping_feed_dict(self, pos_entities1, pos_entities2, neg_entities1, neg_entities2):
        self.cross_pos_left = torch.LongTensor(pos_entities1)
        self.cross_pos_right = torch.LongTensor(pos_entities2)
        self.cross_neg_left = torch.LongTensor(neg_entities1)
        self.cross_neg_right = torch.LongTensor(neg_entities2)

    # 图卷积
    def _graph_convolution(self):
        self.ins_output = list()  # reset
        self.onto_output = list()
        # ************************* instance gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        ins_output_embeddings = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        # ins_output_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ins_ent_embeddings))
        self.ins_output.append(ins_output_embeddings)
        for i in range(self.ins_layer_num):
            activation = self.activation
            if i == self.ins_layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(self.ins_adj_mat, self.params.dim, self.params.dim, i, self.poincare, act=activation)
            self.ins_layers.append(gcn_layer)
            ins_output_embeddings = gcn_layer.forward(ins_output_embeddings)
            # ins_output_embeddings = gcn_layer.call(ins_output_embeddings)
            ins_output_embeddings = self.poincare.mobius_addition(ins_output_embeddings, self.ins_output[-1])
            ins_output_embeddings = self.poincare.hyperbolic_projection(ins_output_embeddings)
            self.ins_output.append(ins_output_embeddings)

        # ************************* ontology gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        onto_output_embeddings = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        # onto_output_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.onto_ent_embeddings))
        self.onto_output.append(onto_output_embeddings)
        for i in range(self.onto_layer_num):
            activation = self.activation
            if i == self.onto_layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(self.onto_adj_mat, self.params.onto_dim, self.params.onto_dim, i, self.poincare,
                                 act=activation)
            self.onto_layers.append(gcn_layer)
            onto_output_embeddings = gcn_layer.forward(onto_output_embeddings)
            onto_output_embeddings = self.poincare.mobius_addition(onto_output_embeddings, self.onto_output[-1])
            onto_output_embeddings = self.poincare.hyperbolic_projection(onto_output_embeddings)
            self.onto_output.append(onto_output_embeddings)

    # 生成初始化的各参数矩阵
    def _generate_variables(self):
        self.ins_ent_embeddings = embed_init(self.ins_ent_num, self.params.dim, "ins_ent_embeds",
                                             method='glorot_uniform_initializer')
        self.ins_ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ins_ent_embeddings))
        # with tf.variable_scope('instance_entity' + 'embeddings'):
        #     self.ins_ent_embeddings = embed_init(self.ins_ent_num, self.params.dim, "ins_ent_embeds",
        #                                          method='glorot_uniform_initializer')
        #     self.ins_ent_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.ins_ent_embeddings))

        self.onto_ent_embeddings = embed_init(self.onto_ent_num, self.params.onto_dim, "onto_ent_embeds",
                                              method='glorot_uniform_initializer')
        self.onto_ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.onto_ent_embeddings))
        # with tf.variable_scope('ontology_entity' + 'embeddings'):
        #     self.onto_ent_embeddings = embed_init(self.onto_ent_num, self.params.onto_dim, "onto_ent_embeds",
        #                                           method='glorot_uniform_initializer')
        #     self.onto_ent_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.onto_ent_embeddings))

        self.ins_rel_embeddings = embed_init(self.ins_rel_num, self.params.dim, "ins_rel_embeds",
                                             method='glorot_uniform_initializer')
        self.ins_rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ins_rel_embeddings))
        # with tf.variable_scope('instance_relation' + 'embeddings'):
        #     self.ins_rel_embeddings = embed_init(self.ins_rel_num, self.params.dim, "ins_rel_embeds",
        #                                          method='glorot_uniform_initializer')
        #     self.ins_rel_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.ins_rel_embeddings))

        self.onto_rel_embeddings = embed_init(self.onto_rel_num, self.params.onto_dim, "onto_rel_embeds",
                                              method='glorot_uniform_initializer')
        self.onto_rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.onto_rel_embeddings))
        # with tf.variable_scope('ontology_relation' + 'embeddings'):
        #     self.onto_rel_embeddings = embed_init(self.onto_rel_num, self.params.onto_dim, "onto_rel_embeds",
        #                                           method='glorot_uniform_initializer')
        #     self.onto_rel_embeddings = self.poincare.hyperbolic_projection(
        #         self.poincare.exp_map_zero(self.onto_rel_embeddings))

        if self.params.mapping:
            # with tf.variable_scope('instance_mapping' + 'embeddings'):
            print("init instance mapping matrix using", "orthogonal", "with dim of", self.params.dim)
            self.ins_mapping_matrix = nn.init.orthogonal_(
                tensor=torch.empty(size=(self.params.dim, self.params.onto_dim), dtype=torch.float64,
                                   requires_grad=True))
            # self.ins_mapping_matrix = tf.get_variable('mapping_matrix',
            #                                               dtype=tf.float64,
            #                                               shape=[self.params.dim, self.params.onto_dim],
            #                                               initializer=tf.initializers.orthogonal(dtype=tf.float64))

    # 黎曼梯度下降，Adam优化
    def _generate_riemannian_optimizer(self, loss):
        opt = tf.train.AdamOptimizer(self.params.learning_rate)
        trainable_grad_vars = opt.compute_gradients(loss)
        grad_vars = [(g, v) for g, v in trainable_grad_vars if g is not None]
        # 计算黎曼梯度
        rescaled = [(g * (1. - tf.reshape(tf.norm(v, axis=1), (-1, 1)) ** 2) ** 2 / 4., v) for g, v in grad_vars]
        # 这里貌似是梯度裁剪
        train_op = opt.apply_gradients(rescaled)
        return train_op

    # 计算triple loss
    def _generate_triple_loss(self, phs, prs, pts, nhs, nrs, nts):
        pos_distance = self.poincare.distance(self.poincare.mobius_addition(phs, prs), pts)
        neg_distance = self.poincare.distance(self.poincare.mobius_addition(nhs, nrs), nts)
        pos_score = torch.sum(pos_distance, 1)
        # pos_score = tf.reduce_sum(pos_distance, 1)
        neg_score = torch.sum(neg_distance, 1)
        # neg_score = tf.reduce_sum(neg_distance, 1)
        pos_loss = torch.sum(torch.relu(pos_score))
        # pos_loss = tf.reduce_sum(tf.nn.relu(pos_score))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.params.neg_triple_margin, dtype=torch.float64) - neg_score))
        # neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.params.neg_triple_margin, dtype=tf.float64) - neg_score))
        return pos_loss + neg_loss

    # 计算mapping loss
    # TODO:命名太混乱了，之后得统一一下
    def _generate_mapping_loss(self, mapped_sup_embeds1, mapped_neg_embeds1, cross_right, neg_embeds2):
        sup_distance = torch.sum(self.poincare.distance(mapped_sup_embeds1, cross_right), 1)
        neg_distance = torch.sum(self.poincare.distance(mapped_neg_embeds1, neg_embeds2), 1)
        pos_loss = torch.sum(torch.relu(sup_distance))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.params.neg_typing_margin, dtype=torch.float64) - neg_distance))
        return pos_loss + neg_loss

    # 生成triple计算图？
    def _generate_triple_graph(self):
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
        # 这一段是不是与_generate_variables中重复了？
        ins_ent_embeddings = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        ins_rel_embeddings = self.poincare.hyperbolic_projection(self.ins_rel_embeddings)
        onto_ent_embeddings = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        onto_rel_embeddings = self.poincare.hyperbolic_projection(self.onto_rel_embeddings)

        # ins_ent_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ins_ent_embeddings))
        # ins_rel_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ins_rel_embeddings))
        # onto_ent_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.onto_ent_embeddings))
        # onto_rel_embeddings =
        # self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.onto_rel_embeddings))

        # 这里用torch.nn.functional中的embedding函数来替换tensorflow中的embedding_lookup不知道对不对
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
        self.ins_triple_loss = self._generate_triple_loss(ins_phs_embeds, ins_prs_embeds, ins_pts_embeds,
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
        self.onto_triple_loss = self._generate_triple_loss(onto_phs_embeds, onto_prs_embeds, onto_pts_embeds,
                                                           onto_nhs_embeds, onto_nrs_embeds, onto_nts_embeds)

        self.triple_loss = self.ins_triple_loss + self.onto_triple_loss
        self.triple_optimizer = self._generate_riemannian_optimizer(self.triple_loss)

    # 生成mapping计算图？
    # 这里面涉及loss计算的地方可以拆分出来，整合成一个_generate_mapping_loss函数
    def _generate_mapping_graph(self):
        # self.cross_pos_left = tf.placeholder(tf.int32, shape=[None], name="cross_pos_left")
        # self.cross_pos_right = tf.placeholder(tf.int32, shape=[None], name="cross_pos_right")
        self._graph_convolution()
        ins_embeddings = self.ins_output[-1]
        onto_embeddings = self.onto_output[-1]
        if self.params.combine:
            ins_embeddings = self.poincare.mobius_addition(ins_embeddings, self.ins_output[0])
            onto_embeddings = self.poincare.mobius_addition(onto_embeddings, self.onto_output[0])

        cross_left = F.embedding(input=self.cross_pos_left, weight=ins_embeddings)
        cross_left = self.poincare.hyperbolic_projection(cross_left)
        cross_right = F.embedding(input=self.cross_pos_right, weight=onto_embeddings)
        cross_right = self.poincare.hyperbolic_projection(cross_right)

        # cross_left = tf.nn.embedding_lookup(ins_embeddings, self.cross_pos_left)
        # cross_left = self.poincare.hyperbolic_projection(cross_left)
        # cross_right = tf.nn.embedding_lookup(onto_embeddings, self.cross_pos_right)
        # cross_right = self.poincare.hyperbolic_projection(cross_right)

        mapped_sup_embeds1 = torch.mm(self.poincare.log_map_zero(cross_left), self.ins_mapping_matrix)
        # mapped_sup_embeds1 = tf.matmul(self.poincare.log_map_zero(cross_left), self.ins_mapping_matrix)
        mapped_sup_embeds1 = self.poincare.exp_map_zero(mapped_sup_embeds1)
        mapped_sup_embeds1 = self.poincare.hyperbolic_projection(mapped_sup_embeds1)
        # mapped_sup_embeds1 = self.poincare.mobius_matmul(cross_left, self.ins_mapping_matrix)

        # sup_distance = self.poincare.distance(mapped_sup_embeds1, cross_right)
        # sup_distance = tf.reduce_sum(sup_distance, 1)

        # *****************add neg sample***********************************************
        # self.cross_neg_left = tf.placeholder(tf.int32, shape=[None], name="cross_neg_left")
        # self.cross_neg_right = tf.placeholder(tf.int32, shape=[None], name="cross_neg_right")
        neg_embeds1 = F.embedding(input=self.cross_neg_left, weight=ins_embeddings)
        # neg_embeds1 = tf.nn.embedding_lookup(ins_embeddings, self.cross_neg_left)
        neg_embeds1 = self.poincare.hyperbolic_projection(neg_embeds1)
        neg_embeds2 = F.embedding(input=self.cross_neg_right, weight=onto_embeddings)
        # neg_embeds2 = tf.nn.embedding_lookup(onto_embeddings, self.cross_neg_right)
        neg_embeds2 = self.poincare.hyperbolic_projection(neg_embeds2)

        mapped_neg_embeds1 = torch.mm(self.poincare.log_map_zero(neg_embeds1), self.ins_mapping_matrix)
        # mapped_neg_embeds1 = tf.matmul(self.poincare.log_map_zero(neg_embeds1), self.ins_mapping_matrix)
        mapped_neg_embeds1 = self.poincare.exp_map_zero(mapped_neg_embeds1)
        mapped_neg_embeds1 = self.poincare.hyperbolic_projection(mapped_neg_embeds1)
        # mapped_neg_embeds1 = self.poincare.mobius_matmul(neg_embeds1, self.ins_mapping_matrix)

        # neg_distance = self.poincare.distance(mapped_neg_embeds1, neg_embeds2)
        # neg_distance = tf.reduce_sum(neg_distance, 1)

        # pos_loss = tf.reduce_sum(tf.nn.relu(sup_distance))
        # neg_loss = tf.reduce_sum(
        #     tf.nn.relu(tf.constant(self.params.neg_typing_margin, dtype=tf.float64) - neg_distance))
        # self.mapping_loss = pos_loss + neg_loss

        self.mapping_loss = self._generate_mapping_loss(mapped_sup_embeds1, mapped_neg_embeds1, cross_right,
                                                        neg_embeds2)
        self.mapping_optimizer = self._generate_riemannian_optimizer(self.mapping_loss)

    # 进行测试
    # 应该不太重要，没认真看，跳过了
    def test(self):
        t = time.time()
        ins_embeddings = self.ins_output[-1]
        onto_embeddings = self.onto_output[-1]
        if self.params.combine:
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
                                     self.params.ent_top_k, self.params.nums_threads, greedy=True,
                                     mess="greedy ent typing by hyperbolic")
        eval_type_hyperbolic(ref_ins_embed, onto_embed, self.all_ref_type, self.params.ent_top_k,
                             self.params.nums_threads, greedy=False, mess="ent typing by hyperbolic")

        print("test totally costs time = {:.3f} s ".format(time.time() - t))
        return hits1

    def eval_ins_input_embed(self, is_map=False):
        embeds = tf.nn.embedding_lookup(self.ins_ent_embeddings, self.ins_entities)
        if is_map:
            embeds = self.poincare.mobius_matmul(embeds, self.ins_mapping_matrix)
        return embeds.eval(session=self.session)

    def eval_onto_input_embed(self):
        return tf.nn.embedding_lookup(self.onto_ent_embeddings, self.onto_entities).eval(session=self.session)
