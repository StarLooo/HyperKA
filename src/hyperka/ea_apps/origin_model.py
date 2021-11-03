# -*- coding: utf-8 -*-
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.hyperka.ea_apps.util as ut
from src.hyperka.ea_apps.util import embed_init
from src.hyperka.ea_funcs.test_funcs import eval_alignment_hyperbolic_multi, sim_handler_hyperbolic, eval_alignment_mul
from src.hyperka.hyperbolic.euclidean import EuclideanManifold
from src.hyperka.hyperbolic.poincare import PoincareManifold

g = 1024 * 1024


class GCNLayer(nn.Module):
    def __init__(self, adj, input_dim, output_dim, layer_id, poincare: PoincareManifold, has_bias: bool = True,
                 activation: nn.Module = None):
        super().__init__()
        self.poincare = poincare
        self.has_bias = has_bias
        self.activation = activation
        self.adj = adj
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 两个线性变换
        self.W_ent = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(input_dim, output_dim, dtype=torch.float64, requires_grad=True, device=ut.try_gpu())))
        if has_bias:
            self.bias_vec = nn.Parameter(
                torch.zeros(1, output_dim, dtype=torch.float64, requires_grad=True, device=ut.try_gpu()))
        else:
            self.register_parameter("bias_vec", None)

    def forward(self, ents_embed_input: torch.Tensor, drop_rate: float = 0.0, ):
        assert 0.0 <= drop_rate < 1.0
        # TODO:映射到欧氏空间
        ents_pre_sup_tangent = self.poincare.log_map_zero(ents_embed_input)
        if drop_rate > 0.0:
            # TODO:这里作者的代码是*(1-drop_rate),但我觉得应该是/(1-drop_rate)才能使得drop之后期望保持不变
            # TODO: 不过貌似实际上并没有drop_out
            ents_pre_sup_tangent = F.dropout(ents_pre_sup_tangent, p=drop_rate, training=self.training) * (
                    1 - drop_rate)  # not scaled up
        assert ents_pre_sup_tangent.shape[1] == self.W_ent.shape[0]
        ents_embed_mapped = torch.mm(ents_pre_sup_tangent, self.W_ent)

        # 实体的邻域实体信息
        ents_near_ents_embeddings = torch.sparse.mm(self.adj, ents_embed_mapped)
        ents_near_ents_embeddings = self.poincare.hyperbolic_projection(
            self.poincare.exp_map_zero(ents_near_ents_embeddings))

        if self.has_bias:
            bias_vec = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.bias_vec))
            ents_near_embeddings_output = self.poincare.mobius_addition(ents_near_ents_embeddings, bias_vec)
            ents_near_embeddings_output = self.poincare.hyperbolic_projection(ents_near_embeddings_output)
        else:
            ents_near_embeddings_output = ents_near_ents_embeddings

        if self.activation is not None:
            ents_near_embeddings_output = self.activation(self.poincare.log_map_zero(ents_near_embeddings_output))
            ents_near_embeddings_output = self.poincare.hyperbolic_projection(
                self.poincare.exp_map_zero(ents_near_embeddings_output))

        return ents_near_embeddings_output


# 我自己的变量名与源代码的变量名的对应关系如下：
# total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
#                       ref_source_aligned_ents, ref_target_aligned_ents, source_triples.ent_list,
#                       target_triples.ent_list, adj, args
# ent_num, rel_num, sup_ent1, sup_ent2,ref_ent1, ref_ent2, kb1_entities, kb2_entities, adj, params
# TODO: 与et中还是有一些重要的差别，包括bootstrapping在内还有一些不是很明白的东西
class Origin_HyperKA(nn.Module):
    def __init__(self, total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
                 ref_source_aligned_ents, ref_target_aligned_ents, source_triples_list, target_triples_list,
                 adj, args):

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

        self.adj = adj

        # TODO: 这两个东西都含义需要搞清楚
        self.new_alignment = list()
        self.new_alignment_pairs = list()

        self._generate_base_parameters()
        self.all_named_train_parameters_list = []
        self.all_train_parameters_list = []
        for name, param in self.named_parameters():
            self.all_named_train_parameters_list.append((name, param))
            self.all_train_parameters_list.append(param)

        self.layer_num = args.layer_num
        self.ent_embeddings_output_list = list()
        self.gcn_layers_list = []
        for layer_id in range(self.layer_num):
            activation = self.activation
            if layer_id == self.layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(adj=self.adj, input_dim=self.args.dim, output_dim=self.args.dim, layer_id=layer_id,
                                 poincare=self.poincare, activation=activation)
            self.gcn_layers_list.append(gcn_layer)
            for name, param in gcn_layer.named_parameters():
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
        self.mapping_new_pos_h = torch.tensor(mapping_new_pos_h, dtype=torch.long, device=ut.try_gpu())
        self.mapping_new_pos_t = torch.tensor(mapping_new_pos_t, dtype=torch.long, device=ut.try_gpu())

    # 图注意力
    def _graph_convolution(self, drop_rate):
        self.ent_embeddings_output_list = list()  # reset

        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        ent_embeddings_output = self.poincare.hyperbolic_projection(self.ent_embeddings)
        self.ent_embeddings_output_list.append(ent_embeddings_output)
        for layer_id in range(self.layer_num):
            gcn_layer = self.gcn_layers_list[layer_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ent_near_embeddings_output = gcn_layer.forward(ent_embeddings_output, drop_rate)
            ent_embeddings_output = self.poincare.mobius_addition(ent_embeddings_output,
                                                                  self.ent_embeddings_output_list[-1])
            ent_embeddings_output = self.poincare.hyperbolic_projection(ent_embeddings_output)
            self.ent_embeddings_output_list.append(ent_embeddings_output)

    # 黎曼梯度下降，Adam优化
    # TODO:不知道这里写的对不对
    def _adapt_riemannian_optimizer(self, optimizer, loss, named_train_params):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        for name, train_param in named_train_params:
            if train_param.grad is None:
                # print("skip")
                continue
            if "emb" in name:
                # print("hyperbolic param:", name)
                riemannian_grad = train_param.grad * (
                        1. - torch.norm(train_param, dim=1).reshape((-1, 1)) ** 2) ** 2 / 4
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
        self._graph_convolution(self.args.drop_rate)
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
            F.embedding(input=torch.tensor(data=self.source_triples_list, dtype=torch.int, device=ut.try_gpu()),
                        weight=self.ent_embeddings))
        if is_map:
            source_embeds = self.poincare.hyperbolic_projection(
                self.poincare.mobius_matmul(source_embeds, self.mapping_matrix))
        return source_embeds.detach()

    # 获取测试用的target KG中的ent embedding
    def eval_target_input_embed(self, is_map=False):
        target_embeds = self.poincare.hyperbolic_projection(
            F.embedding(input=torch.tensor(data=self.target_triples_list, dtype=torch.int, device=ut.try_gpu()),
                        weight=self.ent_embeddings))
        if is_map:
            target_embeds = self.poincare.hyperbolic_projection(
                self.poincare.mobius_matmul(target_embeds, self.mapping_matrix))
        return target_embeds.detach()

    # 为了测试的图注意力
    def _graph_convolution_for_evaluation(self):
        eval_ent_embeddings_output_list = list()
        ent_embeddings = self.ent_embeddings.detach()
        ent_embeddings_output = self.poincare.hyperbolic_projection(ent_embeddings)
        assert ent_embeddings_output.requires_grad is False
        eval_ent_embeddings_output_list.append(ent_embeddings_output)
        for layer_id in range(self.layer_num):
            gcn_layer = self.gcn_layers_list[layer_id]
            ent_near_embeddings = gcn_layer.forward(ent_embeddings_output)
            ent_embeddings_output = self.poincare.mobius_addition(ent_near_embeddings.detach(),
                                                                  eval_ent_embeddings_output_list[-1])
            ent_embeddings_output = self.poincare.hyperbolic_projection(ent_embeddings_output)
            assert ent_embeddings_output.requires_grad is False
            eval_ent_embeddings_output_list.append(ent_embeddings_output)

        if self.args.combine:
            ent_embeddings_output = self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(ent_embeddings_output, eval_ent_embeddings_output_list[0]))
        assert ent_embeddings_output.requires_grad is False
        return ent_embeddings_output

    # 进行测试
    # TODO:测试的逻辑还不是很明白
    def test(self, k=10):
        start = time.time()
        # ent_embeddings_output = self._graph_convolution_for_evaluation()
        ent_embeddings_output = self.ent_embeddings_output_list[-1].detach()
        ref_source_aligned_ents_embed = F.embedding(weight=ent_embeddings_output,
                                                    input=torch.tensor(data=self.ref_source_aligned_ents,
                                                                       dtype=torch.int, device=ut.try_gpu()))
        ref_target_aligned_ents_embed = F.embedding(weight=ent_embeddings_output,
                                                    input=torch.tensor(data=self.ref_target_aligned_ents,
                                                                       dtype=torch.int, device=ut.try_gpu()))
        mapped_ref_source_aligned_ents_embed = self.poincare.hyperbolic_projection(
            self.poincare.mobius_matmul(ref_source_aligned_ents_embed, self.mapping_matrix.detach()))

        assert ref_source_aligned_ents_embed.requires_grad is False
        assert ref_target_aligned_ents_embed.requires_grad is False
        assert mapped_ref_source_aligned_ents_embed.requires_grad is False

        # TODO: TO BE FINISHED
        if k > 0:
            message = "ent alignment by hyperbolic and csls"
            sim = sim_handler_hyperbolic(mapped_ref_source_aligned_ents_embed, ref_target_aligned_ents_embed, k,
                                         self.args.nums_threads)
            hits1 = eval_alignment_mul(sim, self.args.ent_top_k, self.args.nums_threads, message)
        else:
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
