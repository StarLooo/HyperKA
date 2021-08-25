# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperka.et_apps.util import embed_init
from hyperka.et_funcs.test_funcs import eval_type_hyperbolic
from hyperka.hyperbolic.poincare import PoincareManifold


# TODO:因为没有理解adj所以对卷图积层的作用不是很明白
class GCNLayer(nn.Module):
    def __init__(self, adj: torch.Tensor, input_dim: int, output_dim: int, layer_id: int, poincare: PoincareManifold,
                 has_bias: bool = True, activation: nn.Module = None):
        super().__init__()
        self.poincare = poincare
        self.has_bias = has_bias
        self.activation = activation
        self.adj = adj
        self.weight_matrix = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, output_dim, dtype=torch.float64, requires_grad=True)))
        if has_bias:
            self.bias_vec = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64, requires_grad=True))
        else:
            # TODO: 不知道这里register_parameter是否是多余的
            self.register_parameter("bias_vec", None)

    def forward(self, inputs: torch.Tensor, drop_rate: float = 0.0):
        assert 0.0 <= drop_rate < 1.0
        pre_sup_tangent = self.poincare.log_map_zero(inputs)
        if drop_rate > 0.0:
            # TODO:这里作者的代码是*(1-drop_rate),但我觉得应该是/(1-drop_rate)才能使得drop之后期望保持不变
            pre_sup_tangent = F.dropout(pre_sup_tangent, p=drop_rate, training=self.training) / (
                    1 - drop_rate)  # not scaled up
        assert pre_sup_tangent.shape[1] == self.weight_matrix.shape[0]
        output = torch.mm(pre_sup_tangent, self.weight_matrix)
        # output = torch.spmm(self.adj, output) //torch.spmm稀疏矩阵乘法的位置已经移动到torch.sparse中(使用的torch版本：1.9.0)
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


class HyperKA(nn.Module):
    def __init__(self, insnet, onto, instype, ins_adj, onto_adj, args):
        super().__init__()
        self.args = args
        self.poincare = PoincareManifold()
        self.activation = torch.tanh

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
        self.train_instype_set = set(self.train_instype_link)

        # ref(test)表示测试集
        self.test_instype_head = instype[1][0]
        self.test_instype_tail = instype[1][1]
        self.test_instype_link = list()
        for i in range(len(self.test_instype_head)):
            self.test_instype_link.append((self.test_instype_head[i], self.test_instype_tail[i]))
        print("# test_instype_link len:", len(self.test_instype_link))

        self.test_all_ins_types = instype[1][2]

        # list(zip(*--))这种写法详见torch api: https://pytorch.org/docs/1.9.0/sparse.html#sparse-uncoalesced-coo-docs
        self.ins_adj_mat = torch.sparse_coo_tensor(indices=list(zip(*ins_adj[0])), values=ins_adj[1],
                                                   size=ins_adj[2])
        self.onto_adj_mat = torch.sparse_coo_tensor(indices=list(zip(*onto_adj[0])), values=onto_adj[1],
                                                    size=onto_adj[2])

        self._generate_base_parameters()
        self.all_named_train_parameters_list = []
        self.all_train_parameters_list = []
        for name, param in self.named_parameters():
            self.all_named_train_parameters_list.append((name, param))
            self.all_train_parameters_list.append(param)

        self.ins_layer_num = args.ins_layer_num
        self.onto_layer_num = args.onto_layer_num
        # ************************* instance gnn ***************************
        self.ins_gcn_layers_list = []
        for ins_layer_id in range(self.ins_layer_num):
            activation = self.activation
            if ins_layer_id == self.ins_layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(adj=self.ins_adj_mat, input_dim=self.args.ins_dim,
                                 output_dim=self.args.ins_dim, layer_id=ins_layer_id, poincare=self.poincare,
                                 activation=activation)
            self.ins_gcn_layers_list.append(gcn_layer)
            for name, param in gcn_layer.named_parameters():
                self.all_named_train_parameters_list.append((name, param))
                self.all_train_parameters_list.append(param)

        # ************************* ontology gnn ***************************
        self.onto_gcn_layers_list = []
        for onto_layer_id in range(self.onto_layer_num):
            activation = self.activation
            if onto_layer_id == self.onto_layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(adj=self.onto_adj_mat, input_dim=self.args.onto_dim,
                                 output_dim=self.args.onto_dim, layer_id=onto_layer_id, poincare=self.poincare,
                                 activation=activation)
            self.onto_gcn_layers_list.append(gcn_layer)
            for name, param in gcn_layer.named_parameters():
                self.all_named_train_parameters_list.append((name, param))
                self.all_train_parameters_list.append(param)

        self.optimizer = torch.optim.Adam(self.all_train_parameters_list, lr=self.args.learning_rate)

    # 生成初始化的基本参数
    def _generate_base_parameters(self):
        # 获得初始化的ins的嵌入向量
        self.ins_ent_embeddings = embed_init(size=(self.ins_ent_num, self.args.ins_dim),
                                             name="ins_ent_embeds",
                                             method='xavier_uniform')
        self.ins_ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ins_ent_embeddings))
        self.ins_ent_embeddings = nn.Parameter(self.ins_ent_embeddings)

        self.onto_ent_embeddings = embed_init(size=(self.onto_ent_num, self.args.onto_dim),
                                              name="onto_ent_embeds",
                                              method='xavier_uniform')
        self.onto_ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.onto_ent_embeddings))
        self.onto_ent_embeddings = nn.Parameter(self.onto_ent_embeddings)

        self.ins_rel_embeddings = embed_init(size=(self.ins_rel_num, self.args.ins_dim),
                                             name="ins_rel_embeds",
                                             method='xavier_uniform')
        self.ins_rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ins_rel_embeddings))
        self.ins_rel_embeddings = nn.Parameter(self.ins_rel_embeddings)

        self.onto_rel_embeddings = embed_init(size=(self.onto_rel_num, self.args.onto_dim),
                                              name="onto_rel_embeds",
                                              method='xavier_uniform')
        self.onto_rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.onto_rel_embeddings))
        self.onto_rel_embeddings = nn.Parameter(self.onto_rel_embeddings)

        if self.args.mapping:
            size = (self.args.ins_dim, self.args.onto_dim)
            print("init instance mapping matrix using", "orthogonal", "with size of", size)
            self.ins_mapping_matrix = nn.init.orthogonal_(
                tensor=torch.empty(size=size, dtype=torch.float64,
                                   requires_grad=True))
            self.ins_mapping_matrix = nn.Parameter(self.ins_mapping_matrix)

    # 我自己加了这个函数，用于解决tf版本代码中placeholder和feed_dict翻译的问题
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

    # 我自己加了这个函数，用于解决tf版本代码中placeholder和feed_dict翻译的问题
    def _trans_mapping_pos_neg_batch(self, mapping_pos_neg_batch):
        self.link_pos_h, self.link_pos_t, self.link_neg_h, self.link_neg_t = mapping_pos_neg_batch
        self.link_pos_h = torch.LongTensor(self.link_pos_h)
        self.link_neg_h = torch.LongTensor(self.link_neg_h)
        self.link_pos_t = torch.LongTensor(self.link_pos_t)
        self.link_neg_t = torch.LongTensor(self.link_neg_t)

    # 图卷积
    def _graph_convolution(self):
        self.ins_ent_embeddings_output_list = list()  # reset
        self.onto_ent_embeddings_output_list = list()  # reset

        # ************************* instance gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        ins_ent_embeddings_output = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        self.ins_ent_embeddings_output_list.append(ins_ent_embeddings_output)
        for ins_layer_id in range(self.ins_layer_num):
            gcn_layer = self.ins_gcn_layers_list[ins_layer_id]
            ins_ent_embeddings_output = gcn_layer.forward(ins_ent_embeddings_output)
            ins_ent_embeddings_output = self.poincare.mobius_addition(ins_ent_embeddings_output,
                                                                      self.ins_ent_embeddings_output_list[-1])
            ins_ent_embeddings_output = self.poincare.hyperbolic_projection(ins_ent_embeddings_output)
            self.ins_ent_embeddings_output_list.append(ins_ent_embeddings_output)

        # ************************* ontology gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        onto_ent_embeddings_output = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        self.onto_ent_embeddings_output_list.append(onto_ent_embeddings_output)
        for onto_layer_id in range(self.onto_layer_num):
            gcn_layer = self.onto_gcn_layers_list[onto_layer_id]
            onto_ent_embeddings_output = gcn_layer.forward(onto_ent_embeddings_output)
            onto_ent_embeddings_output = self.poincare.mobius_addition(onto_ent_embeddings_output,
                                                                       self.onto_ent_embeddings_output_list[-1])
            onto_ent_embeddings_output = self.poincare.hyperbolic_projection(onto_ent_embeddings_output)
            self.onto_ent_embeddings_output_list.append(onto_ent_embeddings_output)

    # 黎曼梯度下降，Adam优化
    # TODO:不知道这里写的对不对
    def _adapt_riemannian_optimizer(self, loss, named_train_params):
        optimizer = self.optimizer
        optimizer.zero_grad()
        loss.backward()
        # 不知道这样间接计算Riemannian梯度并重新赋值给参数的.grad是否合适
        for name, train_param in named_train_params:
            if train_param.grad is None:
                # print("skip")
                continue
            riemannian_grad = train_param.grad * (1. - torch.norm(train_param, dim=1).reshape((-1, 1)) ** 2) ** 2 / 4
            train_param.grad = riemannian_grad
        optimizer.step()

    # 计算triple loss的内部函数
    def _compute_triple_loss(self, phs, prs, pts, nhs, nrs, nts):
        pos_distance = self.poincare.distance(self.poincare.mobius_addition(phs, prs), pts)
        neg_distance = self.poincare.distance(self.poincare.mobius_addition(nhs, nrs), nts)
        pos_score = torch.sum(pos_distance, 1)
        neg_score = torch.sum(neg_distance, 1)
        pos_loss = torch.sum(torch.relu(pos_score))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.args.neg_triple_margin, dtype=torch.float64) - neg_score))
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

    # 根据triple loss优化参数
    def optimize_triple_loss(self, triple_pos_neg_batch):
        # 这一段是不是与_generate_parameters中重复了？
        ins_ent_embeddings = self.poincare.hyperbolic_projection(self.ins_ent_embeddings)
        ins_rel_embeddings = self.poincare.hyperbolic_projection(self.ins_rel_embeddings)
        onto_ent_embeddings = self.poincare.hyperbolic_projection(self.onto_ent_embeddings)
        onto_rel_embeddings = self.poincare.hyperbolic_projection(self.onto_rel_embeddings)

        # 这里用torch.nn.functional中的embedding函数来替换tensorflow中的embedding_lookup
        self._trans_triple_pos_neg_batch(triple_pos_neg_batch)
        ins_phs_embeds = F.embedding(input=self.ins_pos_h, weight=ins_ent_embeddings)
        ins_prs_embeds = F.embedding(input=self.ins_pos_r, weight=ins_rel_embeddings)
        ins_pts_embeds = F.embedding(input=self.ins_pos_t, weight=ins_ent_embeddings)
        ins_nhs_embeds = F.embedding(input=self.ins_neg_h, weight=ins_ent_embeddings)
        ins_nrs_embeds = F.embedding(input=self.ins_neg_r, weight=ins_rel_embeddings)
        ins_nts_embeds = F.embedding(input=self.ins_neg_t, weight=ins_ent_embeddings)

        ins_triple_loss = self._compute_triple_loss(ins_phs_embeds, ins_prs_embeds, ins_pts_embeds,
                                                    ins_nhs_embeds, ins_nrs_embeds, ins_nts_embeds)

        onto_phs_embeds = F.embedding(input=self.onto_pos_h, weight=onto_ent_embeddings)
        onto_prs_embeds = F.embedding(input=self.onto_pos_r, weight=onto_rel_embeddings)
        onto_pts_embeds = F.embedding(input=self.onto_pos_t, weight=onto_ent_embeddings)
        onto_nhs_embeds = F.embedding(input=self.onto_neg_h, weight=onto_ent_embeddings)
        onto_nrs_embeds = F.embedding(input=self.onto_neg_r, weight=onto_rel_embeddings)
        onto_nts_embeds = F.embedding(input=self.onto_neg_t, weight=onto_ent_embeddings)

        onto_triple_loss = self._compute_triple_loss(onto_phs_embeds, onto_prs_embeds, onto_pts_embeds,
                                                     onto_nhs_embeds, onto_nrs_embeds, onto_nts_embeds)

        triple_loss = ins_triple_loss + onto_triple_loss

        self._adapt_riemannian_optimizer(triple_loss, self.all_named_train_parameters_list)

        return triple_loss

    # 根据mapping loss优化参数
    def optimize_mapping_loss(self, mapping_pos_neg_batch):
        # 进行论文中所说的图卷积
        self._graph_convolution()
        # ins_embeddings和onto_embeddings卷积后得到的嵌入向量
        ins_embeddings = self.ins_ent_embeddings_output_list[-1]
        onto_embeddings = self.onto_ent_embeddings_output_list[-1]
        if self.args.combine:
            ins_embeddings = self.poincare.mobius_addition(ins_embeddings, self.ins_ent_embeddings_output_list[0])
            onto_embeddings = self.poincare.mobius_addition(onto_embeddings, self.onto_ent_embeddings_output_list[0])

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

        mapped_link_phs_embeds = self.poincare.mobius_matmul(link_phs_embeds, self.ins_mapping_matrix)

        link_nhs_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.link_neg_h, weight=ins_ent_embeddings))
        link_nts_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.link_neg_t, weight=onto_ent_embeddings))

        mapped_link_nhs_embeds = self.poincare.mobius_matmul(link_nhs_embeds, self.ins_mapping_matrix)

        mapping_loss = self._compute_mapping_loss(mapped_link_phs_embeds, mapped_link_nhs_embeds,
                                                  link_pts_embeds, link_nts_embeds)

        self._adapt_riemannian_optimizer(mapping_loss, self.all_named_train_parameters_list)

        return mapping_loss

    # 进行测试
    # TODO:测试的逻辑还不是很明白
    def test(self):
        start = time.time()
        ins_embeddings = self.ins_ent_embeddings_output_list[-1]
        onto_embeddings = self.onto_ent_embeddings_output_list[-1]
        if self.args.combine:
            ins_embeddings = self.poincare.mobius_addition(ins_embeddings, self.ins_ent_embeddings_output_list[0])
            onto_embeddings = self.poincare.mobius_addition(onto_embeddings, self.onto_ent_embeddings_output_list[0])

        test_ins_embeddings = F.embedding(input=torch.LongTensor(self.test_instype_head), weight=ins_embeddings)
        test_ins_embeddings = self.poincare.hyperbolic_projection(test_ins_embeddings)
        test_ins_embeddings = torch.matmul(self.poincare.log_map_zero(test_ins_embeddings), self.ins_mapping_matrix)
        test_ins_embeddings = self.poincare.exp_map_zero(test_ins_embeddings)
        test_ins_embeddings = self.poincare.hyperbolic_projection(test_ins_embeddings)

        onto_embeddings = self.poincare.hyperbolic_projection(onto_embeddings)

        hits1 = eval_type_hyperbolic(test_ins_embeddings, onto_embeddings, self.test_all_ins_types,
                                     self.args.ent_top_k, self.args.nums_threads, greedy=True,
                                     mess="greedy ent typing by hyperbolic")
        eval_type_hyperbolic(test_ins_embeddings, onto_embeddings, self.test_all_ins_types, self.args.ent_top_k,
                             self.args.nums_threads, greedy=False, mess="ent typing by hyperbolic")

        end = time.time()
        print("test totally costs time = {:.3f} s ".format(end - start))
        return hits1

    # 生成评估用的instance的嵌入向量
    def eval_ins_input_embed(self, is_map=False):
        ins_embeddings = F.embedding(input=self.ins_ent_list, weight=self.ins_ent_embeddings)
        if is_map:
            ins_embeddings = self.poincare.mobius_matmul(ins_embeddings, self.ins_mapping_matri)
        return ins_embeddings

    # 生成评估用的ontology的嵌入向量
    def eval_onto_input_embed(self):
        onto_embeddings = F.embedding(input=self.onto_ent_list, weight=self.onto_ent_embeddings)
        return onto_embeddings
