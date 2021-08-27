# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import time
import sys
import gc
import ast
# import ray

from hyperka.ea_funcs.train_funcs import get_model
from hyperka.ea_apps.model import HyperKA

# from hyperka.ea_funcs.test_funcs import sim_handler_hyperbolic
# from hyperka.ea_funcs.train_bp import bootstrapping
# from hyperka.hyperbolic.metric import normalization

# from hyperka.ea_funcs.train_funcs import generate_pos_neg_batch, generate_batch_via_neighbour
# from hyperka.ea_funcs.train_funcs import get_model, find_neighbours_multi

g = 1024 * 1024
# ray.init()

parser = argparse.ArgumentParser(description='HyperKE4EA')
parser.add_argument('--input', type=str, default='../../../dataset/dbp15k/zh_en/mtranse/0_3/')  # 路径
parser.add_argument('--output', type=str, default='../../../output/results/')  # 路径

parser.add_argument('--dim', type=int, default=75)  # 嵌入向量的维度
parser.add_argument('--gcn_layer_num', type=int, default=2)  # gcn层数

parser.add_argument('--neg_align_margin', type=float, default=0.4)  # 计算mapping loss的margin
parser.add_argument('--neg_triple_margin', type=float, default=0.1)  # 计算triple loss的margin

parser.add_argument('--learning_rate', type=float, default=0.0002)  # 学习率
parser.add_argument('--batch_size', type=int, default=20000)  # batch_size
parser.add_argument('--epochs', type=int, default=800)  # epochs
parser.add_argument('--drop_rate', type=float, default=0.2)  # 丢弃率

parser.add_argument('--epsilon4triple', type=float, default=0.98)  # TODO: 这个参数的含义不是很清楚

parser.add_argument('--ent_top_k', type=list, default=[1, 5, 10, 50])  # 应当是选取作为输出的预测列表的

parser.add_argument('--triple_neg_nums', type=int, default=40)  # 计算triple loss时每个正例对应多少个负例
parser.add_argument('--mapping_neg_nums', type=int, default=40)  # 计算mapping loss时每个正例对应多少个负例

parser.add_argument('--nums_threads', type=int, default=1)  # TODO: 多线程数，这里本来默认值为8，但在本机上不支持，所以直接改为1
parser.add_argument('--test_interval', type=int, default=4)

parser.add_argument('--sim_th', type=float, default=0.75)
parser.add_argument('--nearest_k', type=int, default=10)
parser.add_argument('--start_bp', type=int, default=40)
parser.add_argument('--bp_param', type=float, default=0.05)
parser.add_argument('--is_bp', type=ast.literal_eval, default=False)  # TODO:是否采用bootstrapping?
parser.add_argument('--heuristic', type=ast.literal_eval, default=True)
parser.add_argument('--combine', type=ast.literal_eval, default=True)  # 是否结合第0层和最后一层的嵌入

if __name__ == '__main__':
    args = parser.parse_args()
    print("show the args:")
    print(args)
    print()

    print("get model...")
    source_triples, target_triples, model = get_model(args.input, HyperKA, args)
    print("get model finished\n")
    os.system("pause")  # 2021/8/27 14:09改代码至能够正确运行到此处

    # hits1, old_hits1 = None, None
    # trunc_ent_num1 = int(len(triples1.ent_list) * (1.0 - args.epsilon4triple))
    # print("trunc ent num for triples:", trunc_ent_num1)
    # if args.is_bp:
    #     epochs_each_iteration = 5
    # else:
    #     epochs_each_iteration = 10
    # total_iteration = args.epochs // epochs_each_iteration
    # for iteration in range(1, total_iteration + 1):
    #     print("iteration", iteration)
    #     train_k_epochs(iteration, model, triples1, triples2, epochs_each_iteration, trunc_ent_num1, args)
    #     if iteration % args.test_interval == 0:
    #         model.test(k=0)
    #     if iteration >= args.start_bp and args.is_bp:
    #         semi_alignment(model, args)
    # model.test()

# def generate_link_batch(model: HyperKA, align_batch_size, nums_neg):
#     assert align_batch_size <= len(model.sup_ent1)
#     pos_links = random.sample(model.sup_links, align_batch_size)
#     neg_links = list()
#
#     for i in range(nums_neg // 2):
#         neg_ent1 = random.sample(model.sup_ent1 + model.ref_ent1, align_batch_size)
#         neg_ent2 = random.sample(model.sup_ent2 + model.ref_ent2, align_batch_size)
#         neg_links.extend([(pos_links[i][0], neg_ent2[i]) for i in range(align_batch_size)])
#         neg_links.extend([(neg_ent1[i], pos_links[i][1]) for i in range(align_batch_size)])
#
#     neg_links = set(neg_links) - set(model.sup_links) - set(model.self_links)
#     return pos_links, list(neg_links)
#
#
# def train_k_epochs(iteration, model: HyperKA, triples1, triples2, k, trunc_ent_num, params):
#     neighbours4triple1, neighbours4triple2 = None, None
#     t1 = time.time()
#     if trunc_ent_num > 0.1:
#         kb1_embeds = model.eval_kb1_input_embed()
#         kb2_embeds = model.eval_kb2_input_embed()
#         neighbours4triple1 = find_neighbours_multi(kb1_embeds, model.kb1_entities, trunc_ent_num, params.nums_threads)
#         neighbours4triple2 = find_neighbours_multi(kb2_embeds, model.kb2_entities, trunc_ent_num, params.nums_threads)
#         print("generate nearest-{} neighbours: {:.3f} s, size: {:.6f} G".format(trunc_ent_num, time.time() - t1,
#                                                                                 sys.getsizeof(neighbours4triple1) / g))
#     total_time = 0.0
#     for i in range(k):
#         loss1, loss2, t2 = train_1epoch(iteration, model, triples1, triples2,
#                                         neighbours4triple1, neighbours4triple2, params)
#         total_time += t2
#         print("triple_loss = {:.3f}, mapping_loss = {:.3f}, time = {:.3f} s".format(loss1, loss2, t2))
#     print("average time for each epoch training = {:.3f} s".format(round(total_time / k, 5)))
#     if neighbours4triple1 is not None:
#         del neighbours4triple1, neighbours4triple2
#         gc.collect()
#
#
# def train_1epoch(iteration, model: HyperKA, triples1, triples2, neighbours1, neighbours2, params, burn_in=5):
#     triple_loss = 0
#     mapping_loss = 0
#     total_time = 0.0
#     lr = params.learning_rate
#     if iteration <= burn_in:
#         lr /= 5
#     steps = math.ceil((triples1.triples_num + triples2.triples_num) / params.batch_size)
#     link_batch_size = math.ceil(len(model.sup_ent1) / steps)
#     for step in range(steps):
#         loss1, t1 = train_triple_1step(model, triples1, triples2, neighbours1, neighbours2, step, params, lr)
#         triple_loss += loss1
#         total_time += t1
#         loss2, t2 = train_alignment_1step(model, link_batch_size, params.nums_neg, lr)
#         mapping_loss += loss2
#         total_time += t2
#     triple_loss /= steps
#     mapping_loss /= steps
#     random.shuffle(triples1.triple_list)
#     random.shuffle(triples2.triple_list)
#     return triple_loss, mapping_loss, total_time
#
#
# def train_alignment_1step(model: HyperKA, batch_size, neg_num, lr):
#     fetches = {"link_loss": model.mapping_loss, "train_op": model.mapping_optimizer}
#     pos_links, neg_links = generate_link_batch(model, batch_size, neg_num)
#     pos_entities1 = [p[0] for p in pos_links]
#     pos_entities2 = [p[1] for p in pos_links]
#     neg_entities1 = [n[0] for n in neg_links]
#     neg_entities2 = [n[1] for n in neg_links]
#     if len(model.new_alignment_pairs) > 0:
#         new_batch_size = math.ceil(len(model.new_alignment_pairs) / len(model.sup_ent1) * batch_size)
#         samples = random.sample(model.new_alignment_pairs, new_batch_size)
#         new_pos_entities1 = [pair[0] for pair in samples]
#         new_pos_entities2 = [pair[1] for pair in samples]
#     else:
#         new_pos_entities1 = [pos_entities1[0]]
#         new_pos_entities2 = [pos_entities2[0]]
#     start = time.time()  # for training time
#     feed_dict = {model.pos_entities1: pos_entities1, model.pos_entities2: pos_entities2,
#                  model.neg_entities1: neg_entities1, model.neg_entities2: neg_entities2,
#                  model.new_pos_entities1: new_pos_entities1, model.new_pos_entities2: new_pos_entities2,
#                  model.lr: lr}
#     results = model.session.run(fetches=fetches, feed_dict=feed_dict)
#     mapping_loss = results["link_loss"]
#     end = time.time()
#     return mapping_loss, round(end - start, 2)
#
#
# def train_triple_1step(model, triples1, triples2, neighbours1, neighbours2, step, params, lr):
#     triple_fetches = {"triple_loss": model.triple_loss, "train_op": model.triple_optimizer}
#     if neighbours2 is None:
#         batch_pos, batch_neg = generate_pos_neg_batch(triples1, triples2, step, params.batch_size,
#                                                       multi=params.triple_nums_neg)
#     else:
#         batch_pos, batch_neg = generate_batch_via_neighbour(triples1, triples2, step, params.batch_size,
#                                                             neighbours1, neighbours2, multi=params.triple_nums_neg)
#     start = time.time()
#     triple_feed_dict = {model.pos_hs: [x[0] for x in batch_pos],
#                         model.pos_rs: [x[1] for x in batch_pos],
#                         model.pos_ts: [x[2] for x in batch_pos],
#                         model.neg_hs: [x[0] for x in batch_neg],
#                         model.neg_rs: [x[1] for x in batch_neg],
#                         model.neg_ts: [x[2] for x in batch_neg],
#                         model.lr: lr}
#     results = model.session.run(fetches=triple_fetches, feed_dict=triple_feed_dict)
#     triple_loss = results["triple_loss"]
#     end = time.time()
#     return triple_loss, round(end - start, 2)
#
#
# def semi_alignment(model: HyperKA, params):
#     print()
#     t = time.time()
#     refs1_embed = model.eval_output_embed(model.ref_ent1, is_map=True)
#     refs2_embed = model.eval_output_embed(model.ref_ent2, is_map=False)
#     sim_mat = sim_handler_hyperbolic(refs1_embed, refs2_embed, 5, params.nums_threads)
#     sim_mat = normalization(sim_mat)
#     # temp_sim_th = (np.mean(sim_mat) + np.max(sim_mat)) / 2
#     # sim_th = (params.sim_th + temp_sim_th) / 2
#     # print("min, mean, and max of sim mat, sim_th = ", np.min(sim_mat), np.mean(sim_mat), np.max(sim_mat), sim_th)
#     sim_th = params.sim_th
#     new_alignment, entities1, entities2 = bootstrapping(sim_mat, model.ref_ent1, model.ref_ent2, model.new_alignment,
#                                                         sim_th, params.nearest_k, is_edit=True,
#                                                         heuristic=params.heuristic)
#     model.new_alignment = list(new_alignment)
#     model.new_alignment_pairs = [(entities1[i], entities2[i]) for i in range(len(entities1))]
#     print("semi-supervised alignment costs time = {:.3f} s\n".format(time.time() - t))
