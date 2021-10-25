# -*- coding: utf-8 -*-
# import tensorflow as tf
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.hyperka.ea_apps.util as ut
from src.hyperka.ea_apps.util import embed_init
from src.hyperka.hyperbolic.euclidean import EuclideanManifold
from src.hyperka.hyperbolic.poincare import PoincareManifold
from src.hyperka.ea_funcs.test_funcs import eval_alignment_hyperbolic_multi

g = 1024 * 1024


class GATLayer(nn.Module):
    def __init__(self, near_ents_adj, near_rels_adj, ents_near_ents_num, ents_near_rels_num, rels_near_ents_num,
                 input_dim, output_dim, layer_id, poincare: PoincareManifold, has_bias: bool = True,
                 activation: nn.Module = None, another_attention_mode: bool = False):
        super().__init__()
        self.poincare = poincare
        self.has_bias = has_bias
        self.activation = activation
        self.near_ents_adj = near_ents_adj
        self.ents_near_ents_num = ents_near_ents_num
        self.near_rels_adj = near_rels_adj
        self.ents_near_rels_num = ents_near_rels_num
        self.rels_near_ents_num = rels_near_ents_num
        self.n_entities, self.n_rels = near_rels_adj.shape
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 两个线性变换
        self.W_ent = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(input_dim, output_dim, dtype=torch.float64, requires_grad=True, device=ut.try_gpu())))
        self.W_rel = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(input_dim, output_dim, dtype=torch.float64, requires_grad=True, device=ut.try_gpu())))
        # attention method 2
        if another_attention_mode:
            self.w = nn.Parameter(
                torch.rand(output_dim, 1, dtype=torch.float64, requires_grad=True, device=ut.try_gpu()))
        if has_bias:
            self.bias_vec = nn.Parameter(
                torch.zeros(1, output_dim, dtype=torch.float64, requires_grad=True, device=ut.try_gpu()))
        else:
            self.register_parameter("bias_vec", None)
        self.another_attention_mode = another_attention_mode

    def forward(self, ents_embed_input: torch.Tensor, rels_embed_input: torch.Tensor, drop_rate: float = 0.0,
                combine_rels_weight: float = 0.1):
        assert 0.0 <= drop_rate < 1.0
        # TODO:映射到欧氏空间
        ents_pre_sup_tangent = self.poincare.log_map_zero(ents_embed_input)
        rels_pre_sup_tangent = self.poincare.log_map_zero(rels_embed_input)
        if drop_rate > 0.0:
            # TODO:这里作者的代码是*(1-drop_rate),但我觉得应该是/(1-drop_rate)才能使得drop之后期望保持不变
            # TODO: 不过貌似实际上并没有drop_out
            ents_pre_sup_tangent = F.dropout(ents_pre_sup_tangent, p=drop_rate, training=self.training) * (
                    1 - drop_rate)  # not scaled up
            rels_pre_sup_tangent = F.dropout(rels_pre_sup_tangent, p=drop_rate, training=self.training) * (
                    1 - drop_rate)  # not scaled up
        assert ents_pre_sup_tangent.shape[1] == self.W_ent.shape[0]
        assert rels_pre_sup_tangent.shape[1] == self.W_rel.shape[0]
        ents_embed_mapped = torch.mm(ents_pre_sup_tangent, self.W_ent)
        rels_embed_mapped = torch.mm(rels_pre_sup_tangent, self.W_rel)
        rels_embed_origin = rels_pre_sup_tangent

        # 实体的邻域实体信息
        # attention method 2
        if self.another_attention_mode:
            alpha_matrix = F.leaky_relu(torch.matmul(ents_embed_mapped.detach(), self.w)).t().expand(
                (self.n_entities, self.n_entities)).sparse_mask(self.near_ents_adj)
            alpha_matrix = torch.sparse.softmax(alpha_matrix, dim=1)
            assert alpha_matrix.requires_grad is True
        # attention method 1
        else:
            alpha_matrix = torch.matmul(ents_embed_mapped.detach(), ents_embed_mapped.detach().t()).sparse_mask(
                self.near_ents_adj)
            alpha_matrix = torch.sparse.softmax(alpha_matrix, dim=1)
            assert alpha_matrix.requires_grad is False
        ents_near_ents_embeddings = torch.sparse.mm(alpha_matrix, ents_embed_mapped)
        ents_near_ents_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(ents_near_ents_embeddings))

        # 实体的邻域边信息
        mapped_edge_embeddings = rels_embed_mapped[self.near_rels_adj.values()]
        ents_near_rels_embed_adj = torch.sparse_coo_tensor(indices=self.near_rels_adj.indices(),
                                                           values=mapped_edge_embeddings,
                                                           size=(self.n_entities, self.n_rels, self.output_dim),
                                                           device=ut.try_gpu())
        ents_near_rels_embeddings = torch.sparse.sum(ents_near_rels_embed_adj, dim=1).to_dense()
        ents_near_rels_embeddings = ents_near_rels_embeddings / self.ents_near_rels_num.unsqueeze(1)
        ents_near_rels_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(ents_near_rels_embeddings))

        # 边的邻域边信息，通过实体作为中转间接传递
        edge_embeddings = rels_embed_origin[self.near_rels_adj.values()]
        rels_near_ens_embed_adj = torch.sparse_coo_tensor(indices=self.near_rels_adj.indices(),
                                                          values=edge_embeddings,
                                                          size=(self.n_entities, self.n_rels, self.output_dim),
                                                          device=ut.try_gpu()).transpose(0, 1)
        rels_near_rels_embeddings = torch.sparse.sum(rels_near_ens_embed_adj, dim=1).to_dense()
        # TODO:这行代码的除法的分母有问题
        rels_near_rels_embeddings = rels_near_rels_embeddings / self.rels_near_ents_num.unsqueeze(1)
        rels_near_rels_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(rels_near_rels_embeddings))

        # "-------------------代码刚刚改到这里--------------------"

        assert ents_near_rels_embeddings.shape == ents_near_ents_embeddings.shape == ents_embed_input.shape
        assert rels_near_rels_embeddings.shape == rels_embed_input.shape

        # TODO: 结合实体消息传递和关系消息传递
        ents_near_embeddings_output = self.poincare.hyperbolic_projection(
            self.poincare.mobius_addition(ents_near_ents_embeddings, combine_rels_weight * ents_near_rels_embeddings))
        rels_near_embeddings_output = rels_near_rels_embeddings
        # TODO: 是否还需要bias
        if self.has_bias:
            bias_vec = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.bias_vec))
            ents_near_embeddings_output = self.poincare.mobius_addition(ents_near_embeddings_output, bias_vec)
            ents_near_embeddings_output = self.poincare.hyperbolic_projection(ents_near_embeddings_output)
        if self.activation is not None:
            ents_near_embeddings_output = self.activation(self.poincare.log_map_zero(ents_near_embeddings_output))
            ents_near_embeddings_output = self.poincare.hyperbolic_projection(
                self.poincare.exp_map_zero(ents_near_embeddings_output))
            rels_near_embeddings_output = self.activation(self.poincare.log_map_zero(rels_near_embeddings_output))
            rels_near_embeddings_output = self.poincare.hyperbolic_projection(
                self.poincare.exp_map_zero(rels_near_embeddings_output))
        return ents_near_embeddings_output, rels_near_embeddings_output


# 我自己的变量名与源代码的变量名的对应关系如下：
# total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
#                       ref_source_aligned_ents, ref_target_aligned_ents, source_triples.ent_list,
#                       target_triples.ent_list, adj, args
# ent_num, rel_num, sup_ent1, sup_ent2,ref_ent1, ref_ent2, kb1_entities, kb2_entities, adj, params
# TODO: 与et中还是有一些重要的差别，包括bootstrapping在内还有一些不是很明白的东西
# TODO: 需要搞清楚为什么两个KG用了同一张图，以及GAT层为什么只有一种(不用分ins和onto)
class HyperKA(nn.Module):
    def __init__(self, total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
                 ref_source_aligned_ents, ref_target_aligned_ents, source_triples_list, target_triples_list,
                 near_ents_graph, near_rels_graph, args):

        super().__init__()

        self.args = args
        self.poincare = PoincareManifold()
        self.euclidean = EuclideanManifold()
        self.activation = torch.tanh
        self.learning_rate = args.learning_rate
        self.dim = args.dim

        self.total_ents_num = total_ents_num
        self.total_rels_num = total_rels_num

        self.sup_source_aligned_ents = sup_source_aligned_ents
        self.sup_target_aligned_ents = sup_target_aligned_ents
        self.sup_aligned_ents_pairs = [(sup_source_aligned_ents[i], sup_target_aligned_ents[i]) for i in
                                       range(len(sup_source_aligned_ents))]
        self.ref_source_aligned_ents = ref_source_aligned_ents
        self.ref_target_aligned_ents = ref_target_aligned_ents
        self.ref_aligned_ents_pairs = [(ref_source_aligned_ents[i], ref_target_aligned_ents[i]) for i in
                                       range(len(ref_source_aligned_ents))]
        self.source_triples_list = source_triples_list
        self.target_triples_list = target_triples_list
        all_ents = sup_source_aligned_ents + sup_target_aligned_ents + ref_source_aligned_ents + ref_target_aligned_ents
        # TODO: 自己和自己对齐？
        self.self_aligned_ents_pairs = [(all_ents[i], all_ents[i]) for i in range(len(all_ents))]

        self.near_ents_adj, self.ents_near_ents_num = near_ents_graph
        self.near_rels_adj, self.ents_near_rels_num, self.rels_near_ens_num = near_rels_graph

        # TODO: 这两个东西都含义需要搞清楚
        self.new_alignment = list()
        self.new_alignment_pairs = list()

        self._generate_base_parameters()
        self.all_named_train_parameters_list = []
        self.all_train_parameters_list = []
        for name, param in self.named_parameters():
            self.all_named_train_parameters_list.append((name, param))
            self.all_train_parameters_list.append(param)

        self.layer_num = args.gat_layer_num
        self.ent_embeddings_output_list = list()
        self.rel_embeddings_output_list = list()
        self.gat_layers_list = []
        for layer_id in range(self.layer_num):
            activation = self.activation
            if layer_id == self.layer_num - 1:
                activation = None
            gat_layer = GATLayer(near_ents_adj=self.near_ents_adj, near_rels_adj=self.near_rels_adj,
                                 ents_near_ents_num=self.ents_near_ents_num,
                                 ents_near_rels_num=self.ents_near_rels_num,
                                 rels_near_ents_num=self.rels_near_ens_num, input_dim=self.args.dim,
                                 output_dim=self.args.dim, layer_id=layer_id, poincare=self.poincare,
                                 activation=activation)

            self.gat_layers_list.append(gat_layer)
            for name, param in gat_layer.named_parameters():
                self.all_named_train_parameters_list.append((name, param))
                self.all_train_parameters_list.append(param)

        self.triple_optimizer = torch.optim.Adam(self.all_train_parameters_list, lr=self.args.learning_rate)
        self.mapping_optimizer = torch.optim.Adam(self.all_train_parameters_list, lr=self.args.learning_rate)

    # 生成初始化的基本参数
    def _generate_base_parameters(self):
        # 获得初始化的ent的嵌入向量
        self.ent_embeddings = embed_init(size=(self.total_ents_num, self.args.dim),
                                         name="ent_embeds",
                                         method='xavier_uniform')
        self.ent_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.ent_embeddings))
        self.ent_embeddings = nn.Parameter(self.ent_embeddings)

        # 获得初始化的rel的嵌入向量
        self.rel_embeddings = embed_init(size=(self.total_rels_num, self.args.dim),
                                         name="rel_embeds",
                                         method='xavier_uniform')
        self.rel_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(self.rel_embeddings))
        self.rel_embeddings = nn.Parameter(self.rel_embeddings)

        if self.args.mapping:
            size = (self.args.dim, self.args.dim)
            print("init mapping matrix using", "orthogonal", "with size of", size)
            self.mapping_matrix = nn.init.orthogonal_(
                tensor=torch.empty(size=size, dtype=torch.float64, requires_grad=True, device=ut.try_gpu()))
            self.mapping_matrix = nn.Parameter(self.mapping_matrix)

    # 我自己加了这个函数，用于解决tf版本代码中placeholder和feed_dict翻译的问题
    def _trans_triple_pos_neg_batch(self, triple_pos_neg_batch):
        pos_batch, neg_batch, self.learning_rate = triple_pos_neg_batch
        self.triple_pos_h = torch.tensor(data=[x[0] for x in pos_batch], dtype=torch.long, device=ut.try_gpu())
        self.triple_pos_r = torch.tensor(data=[x[1] for x in pos_batch], dtype=torch.long, device=ut.try_gpu())
        self.triple_pos_t = torch.tensor(data=[x[2] for x in pos_batch], dtype=torch.long, device=ut.try_gpu())
        self.triple_neg_h = torch.tensor(data=[x[0] for x in neg_batch], dtype=torch.long, device=ut.try_gpu())
        self.triple_neg_r = torch.tensor(data=[x[1] for x in neg_batch], dtype=torch.long, device=ut.try_gpu())
        self.triple_neg_t = torch.tensor(data=[x[2] for x in neg_batch], dtype=torch.long, device=ut.try_gpu())

    # 我自己加了这个函数，用于解决tf版本代码中placeholder和feed_dict翻译的问题
    def _trans_mapping_pos_neg_batch(self, mapping_pos_neg_batch):
        mapping_pos_h, mapping_pos_t, mapping_neg_h, mapping_neg_t, \
        mapping_new_pos_h, mapping_new_pos_t, self.learning_rate = mapping_pos_neg_batch
        self.mapping_pos_h = torch.tensor(mapping_pos_h, dtype=torch.long, device=ut.try_gpu())
        self.mapping_pos_t = torch.tensor(mapping_pos_t, dtype=torch.long, device=ut.try_gpu())
        self.mapping_neg_h = torch.tensor(mapping_neg_h, dtype=torch.long, device=ut.try_gpu())
        self.mapping_neg_t = torch.tensor(mapping_neg_t, dtype=torch.long, device=ut.try_gpu())

    # 图注意力
    def _graph_attention(self, drop_rate):
        self.ent_embeddings_output_list = list()  # reset
        self.rel_embeddings_output_list = list()  # reset

        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        ent_embeddings_output = self.poincare.hyperbolic_projection(self.ent_embeddings)
        rel_embeddings_output = self.poincare.hyperbolic_projection(self.rel_embeddings)
        self.ent_embeddings_output_list.append(ent_embeddings_output)
        self.rel_embeddings_output_list.append(rel_embeddings_output)
        for layer_id in range(self.layer_num):
            gat_layer = self.gat_layers_list[layer_id]
            ent_near_embeddings_output, rel_near_embeddings_output = gat_layer.forward(ent_embeddings_output,
                                                                                       rel_embeddings_output, drop_rate)
            ent_embeddings_output = self.poincare.mobius_addition(ent_embeddings_output,
                                                                  self.ent_embeddings_output_list[-1])
            rel_embeddings_output = self.poincare.mobius_addition(rel_embeddings_output,
                                                                  self.rel_embeddings_output_list[-1])
            ent_embeddings_output = self.poincare.hyperbolic_projection(ent_embeddings_output)
            rel_embeddings_output = self.poincare.hyperbolic_projection(rel_embeddings_output)
            self.ent_embeddings_output_list.append(ent_embeddings_output)
            self.rel_embeddings_output_list.append(rel_embeddings_output)

    # 黎曼梯度下降，Adam优化
    # TODO:不知道这里写的对不对
    def _adapt_riemannian_optimizer(self, optimizer, loss, named_train_params):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        # 不知道这样间接计算Riemannian梯度并重新赋值给参数的.grad是否合适
        for name, train_param in named_train_params:
            if train_param.grad is None:
                # print("skip")
                continue
            # print("name:", name, "shape:", train_param.shape)
            riemannian_grad = train_param.grad * (1. - torch.norm(train_param, dim=1).reshape((-1, 1)) ** 2) ** 2 / 4
            train_param.grad = riemannian_grad
        optimizer.step()

    # 计算triple loss的内部函数
    def _compute_triple_loss(self, phs, prs, pts, nhs, nrs, nts):
        pos_distance = self.poincare.distance(self.poincare.mobius_addition(phs, prs), pts)
        neg_distance = self.poincare.distance(self.poincare.mobius_addition(nhs, nrs), nts)
        pos_score = torch.sum(pos_distance, dim=1)
        neg_score = torch.sum(neg_distance, dim=1)
        pos_loss = torch.sum(torch.relu(pos_score))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.args.neg_triple_margin, dtype=torch.float64) - neg_score))
        triple_loss = pos_loss + neg_loss
        return triple_loss

    # 计算mapping loss的内部函数
    def _compute_mapping_loss(self, mapped_mapping_phs_embeds, mapping_pts_embeds, mapped_mapping_nhs_embeds,
                              mapping_nts_embeds, mapped_mapping_new_phs_embeds, mapping_new_pts_embeds):
        pos_distance = torch.sum(self.poincare.distance(mapped_mapping_phs_embeds, mapping_pts_embeds), dim=1)
        neg_distance = torch.sum(self.poincare.distance(mapped_mapping_nhs_embeds, mapping_nts_embeds), dim=1)
        new_pos_distance = torch.sum(self.poincare.distance(mapped_mapping_new_phs_embeds, mapping_pts_embeds), dim=1)

        pos_loss = torch.sum(torch.relu(pos_distance))
        neg_loss = torch.sum(
            torch.relu(torch.tensor(data=self.args.neg_mapping_margin, dtype=torch.float64) - neg_distance))
        new_pos_loss = torch.sum(torch.relu(pos_distance))

        mapping_loss = pos_loss + neg_loss + self.args.bp_param * new_pos_loss
        return mapping_loss

    # 根据triple loss优化参数
    def optimize_triple_loss(self, triple_pos_neg_batch):
        # 这一段是不是与_generate_parameters中重复了？
        ent_embeddings = self.poincare.hyperbolic_projection(self.ent_embeddings)
        rel_embeddings = self.poincare.hyperbolic_projection(self.rel_embeddings)

        # 这里用torch.nn.functional中的embedding函数来替换tensorflow中的embedding_lookup
        self._trans_triple_pos_neg_batch(triple_pos_neg_batch)
        phs_embeds = F.embedding(input=self.triple_pos_h, weight=ent_embeddings)
        prs_embeds = F.embedding(input=self.triple_pos_r, weight=rel_embeddings)
        pts_embeds = F.embedding(input=self.triple_pos_t, weight=ent_embeddings)
        nhs_embeds = F.embedding(input=self.triple_neg_h, weight=ent_embeddings)
        nrs_embeds = F.embedding(input=self.triple_neg_r, weight=rel_embeddings)
        nts_embeds = F.embedding(input=self.triple_neg_t, weight=ent_embeddings)

        triple_loss = self._compute_triple_loss(phs_embeds, prs_embeds, pts_embeds,
                                                nhs_embeds, nrs_embeds, nts_embeds)

        self._adapt_riemannian_optimizer(self.triple_optimizer, triple_loss, self.all_named_train_parameters_list)

        return triple_loss

    # 根据mapping loss优化参数
    def optimize_mapping_loss(self, mapping_pos_neg_batch):
        # 进行论文中所说的图卷积
        self._graph_attention(self.args.drop_rate)
        # 卷积后得到的嵌入向量
        ent_embeddings = self.ent_embeddings_output_list[-1]
        if self.args.combine:
            ent_embeddings = self.poincare.mobius_addition(ent_embeddings, self.ent_embeddings_output_list[0])
            ent_embeddings = self.poincare.hyperbolic_projection(ent_embeddings)

        self._trans_mapping_pos_neg_batch(mapping_pos_neg_batch)

        mapping_phs_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.mapping_pos_h, weight=ent_embeddings))
        mapping_pts_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.mapping_pos_t, weight=ent_embeddings))
        mapping_nhs_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.mapping_neg_h, weight=ent_embeddings))
        mapping_nts_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.mapping_neg_t, weight=ent_embeddings))

        mapping_new_phs_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.mapping_new_pos_h, weight=ent_embeddings))
        mapping_new_pts_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=self.mapping_new_pos_t, weight=ent_embeddings))

        mapped_mapping_phs_embeds = self.poincare.mobius_matmul(mapping_phs_embeds, self.mapping_matrix)
        mapped_mapping_nhs_embeds = self.poincare.mobius_matmul(mapping_nhs_embeds, self.mapping_matrix)
        mapped_mapping_new_phs_embeds = self.poincare.mobius_matmul(mapping_new_phs_embeds, self.mapping_matrix)

        mapping_loss = self._compute_mapping_loss(mapped_mapping_phs_embeds, mapping_pts_embeds,
                                                  mapped_mapping_nhs_embeds, mapping_nts_embeds,
                                                  mapped_mapping_new_phs_embeds, mapping_new_pts_embeds)

        self._adapt_riemannian_optimizer(self.mapping_optimizer, mapping_loss, self.all_named_train_parameters_list)

        return mapping_loss

    # 获取测试用的source KG中的ent embedding
    def eval_source_input_embed(self, is_map=False):

        source_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=torch.LongTensor(self.source_triples_list), weight=self.ent_embeddings))
        if is_map:
            source_embeds = self.poincare.hyperbolic_projection(
                self.poincare.mobius_matmul(source_embeds, self.mapping_matrix))
        return source_embeds.detach()

    # 获取测试用的target KG中的ent embedding
    def eval_target_input_embed(self, is_map=False):

        target_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=torch.LongTensor(self.target_triples_list), weight=self.ent_embeddings))
        if is_map:
            target_embeds = self.poincare.hyperbolic_projection(
                self.poincare.mobius_matmul(target_embeds, self.mapping_matrix))
        return target_embeds.detach()

    # 为了测试的图注意力
    def _graph_attention_for_evaluation(self):
        eval_ent_embeddings_output_list = list()
        eval_rel_embeddings_output_list = list()
        ent_embeddings = self.ent_embeddings.detach()
        rel_embeddings = self.rel_embeddings.detach()
        ent_embeddings_output = self.poincare.hyperbolic_projection(ent_embeddings)
        rel_embeddings_output = self.poincare.hyperbolic_projection(rel_embeddings)
        assert ent_embeddings_output.requires_grad is False and rel_embeddings_output.requires_grad is False
        eval_ent_embeddings_output_list.append(ent_embeddings_output)
        eval_rel_embeddings_output_list.append(rel_embeddings_output)
        for layer_id in range(self.layer_num):
            gat_layer = self.gat_layers_list[layer_id]
            ent_near_embeddings, rel_near_embeddings = gat_layer.forward(ent_embeddings_output, rel_embeddings_output)
            ent_embeddings_output = self.poincare.mobius_addition(ent_near_embeddings.detach(),
                                                                  eval_ent_embeddings_output_list[-1])
            rel_embeddings_output = self.poincare.mobius_addition(rel_near_embeddings.detach(),
                                                                  eval_rel_embeddings_output_list[-1])
            ent_embeddings_output = self.poincare.hyperbolic_projection(ent_embeddings_output)
            rel_embeddings_output = self.poincare.hyperbolic_projection(rel_embeddings_output)
            assert ent_embeddings_output.requires_grad is False and rel_embeddings_output.requires_grad is False
            eval_ent_embeddings_output_list.append(ent_embeddings_output)
            eval_rel_embeddings_output_list.append(rel_embeddings_output)

        if self.args.combine:
            ent_embeddings_output = self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(ent_embeddings_output, eval_ent_embeddings_output_list[0]))
            rel_embeddings_output = self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(rel_embeddings_output, eval_rel_embeddings_output_list[0]))
        assert ent_embeddings_output.requires_grad is False and rel_embeddings_output.requires_grad is False
        return ent_embeddings_output, rel_embeddings_output

    # 进行测试
    # TODO:测试的逻辑还不是很明白
    def test(self, k=10):
        start = time.time()
        ent_embeddings_output, rel_embeddings_output = self._graph_attention_for_evaluation()
        ref_source_aligned_ents_embed = F.embedding(weight=ent_embeddings_output,
                                                    input=torch.LongTensor(self.ref_source_aligned_ents))
        ref_target_aligned_ents_embed = F.embedding(weight=ent_embeddings_output,
                                                    input=torch.LongTensor(self.ref_target_aligned_ents))
        mapped_ref_source_aligned_ents_embed = self.poincare.hyperbolic_projection(
            self.poincare.mobius_matmul(ref_source_aligned_ents_embed, self.mapping_matrix.detach()))

        assert ref_source_aligned_ents_embed.requires_grad is False
        assert ref_target_aligned_ents_embed.requires_grad is False
        assert mapped_ref_source_aligned_ents_embed.requires_grad is False

        # TODO: TO BE FINISHED
        # if k > 0:
        #     message = "ent alignment by hyperbolic and csls"
        #     sim = sim_handler_hyperbolic(mapped_ref_source_aligned_ents_embed, ref_target_aligned_ents_embed, k,
        #                                  self.args.nums_threads)
        #     hits1 = eval_alignment_mul(sim, self.args.ent_top_k, self.args.nums_threads, message)
        # else:
        message = "fast ent alignment by hyperbolic"
        hits1 = eval_alignment_hyperbolic_multi(mapped_ref_source_aligned_ents_embed, ref_target_aligned_ents_embed,
                                                self.args.ent_top_k, self.args.nums_threads, message)

        end = time.time()
        print("test totally costs {:.3f} s ".format(end - start))
        del ref_source_aligned_ents_embed, ref_target_aligned_ents_embed, mapped_ref_source_aligned_ents_embed
        gc.collect()
        return hits1

    def eval_output_embed(self, ents, is_map=False):
        output_embeddings = self._graph_convolution_for_evaluation()
        assert output_embeddings.requires_grad is False
        embeds = F.embedding(input=torch.LongTensor(ents), weight=output_embeddings)
        if is_map:
            embeds = self.poincare.mobius_matmul(embeds, self.mapping_matrix.detach())
        assert embeds.requires_grad is False
        return embeds

# def eval_ent_embeddings(self):
#     ent_embeddings = self._graph_convolution_for_evaluation()
#     return ent_embeddings.eval(session=self.session)
#
# def eval_kb12_embed(self):
#     ent_embeddings = self._graph_convolution_for_evaluation()
#     embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.kb1_entities)
#     embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.kb2_entities)
#     return embeds1.eval(session=self.session), embeds2.eval(session=self.session)
