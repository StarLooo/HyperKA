# -*- coding: utf-8 -*-
import gc
import os
import time
import ray
import numpy as np
import random
from sklearn import preprocessing
import sys
from hyperka.ea_apps.util import generate_adjacent_graph
import hyperka.ea_funcs.utils as ut
from hyperka.hyperbolic.metric import compute_hyperbolic_similarity
from hyperka.ea_funcs.test_funcs import sim_handler_hyperbolic

g = 1000000000


def get_model(folder, kge_model, args):
    print("data folder:", folder)

    print("read_input begin...")
    if "15" in folder:
        read_func = ut.read_dbp15k_input
    else:
        read_func = ut.read_other_input

    source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents, \
    ref_source_aligned_ents, ref_target_aligned_ents, total_ents_num, total_rels_num = read_func(folder)

    # TODO: linked_entities内涵为未知
    linked_entities = set(
        sup_source_aligned_ents + sup_target_aligned_ents + ref_source_aligned_ents + ref_target_aligned_ents)

    # TODO: 增强triples中的数据
    print("enhance triples begin:")
    enhanced_source_triples_list, enhanced_target_triples_list = enhance_triples(source_triples, target_triples,
                                                                                 sup_source_aligned_ents,
                                                                                 sup_target_aligned_ents)
    print("enhance triples finished")

    # 删除掉单独存在于source KG或target KG中而没有与之对齐的实体所在的三元组
    triples_list = remove_unlinked_triples(source_triples.triple_list + target_triples.triple_list +
                                           enhanced_source_triples_list + enhanced_target_triples_list, linked_entities)

    # 这里应该与et里是一样的
    adj = generate_adjacent_graph(total_ents_num, triples_list)

    model = kge_model(total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
                      ref_source_aligned_ents, ref_target_aligned_ents, source_triples.ent_list,
                      target_triples.ent_list, adj, args)

    return source_triples, target_triples, model


# 数据增强
def enhance_triples(source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents):
    assert len(sup_source_aligned_ents) == len(sup_target_aligned_ents)

    print("before enhanced:")
    print("source KG's triples num:", len(source_triples.triples))
    print("target KG's triples num:", len(target_triples.triples))

    # 这里的逻辑是：对source KG中的每一个triple(包含source_head, source_rel和source_tail)的头(source_head)和尾(source_tail)，
    # 寻找在训练集中给出的target KG中与source_head和source_tail对齐的实体(分别记为enhanced_head和enhanced_tail)。
    # 如果enhanced_head和enhanced_tail均存在且source KG中没有(enhanced_head, source_rel, enhanced_tail)这一triple，
    # 则添将其加到enhanced_source_triples_set中。
    enhanced_source_triples_set = set()
    links_from_source_to_target = dict(zip(sup_source_aligned_ents, sup_target_aligned_ents))
    for source_head, source_rel, source_tail in source_triples.triples:
        enhanced_head = links_from_source_to_target.get(source_head, None)
        enhanced_tail = links_from_source_to_target.get(source_tail, None)

        if enhanced_head is not None and enhanced_tail is not None \
                and enhanced_tail not in target_triples.out_related_ents_dict.get(enhanced_head, set()):
            enhanced_source_triples_set.add((enhanced_head, source_rel, enhanced_tail))

    # 同理，此略
    enhanced_target_triples_set = set()
    links_from_target_to_source = dict(zip(sup_target_aligned_ents, sup_source_aligned_ents))
    for source_head, target_rel, source_tail in target_triples.triples:
        enhanced_head = links_from_source_to_target.get(source_head, None)
        enhanced_tail = links_from_source_to_target.get(source_tail, None)

        if enhanced_head is not None and enhanced_tail is not None \
                and enhanced_tail not in source_triples.out_related_ents_dict.get(enhanced_head, set()):
            enhanced_target_triples_set.add((enhanced_head, target_rel, enhanced_tail))

    print("after enhanced finished")
    print("source KG's triples num:", len(enhanced_source_triples_set))
    print("target KG's triples num:", len(enhanced_target_triples_set))

    return list(enhanced_source_triples_set), list(enhanced_target_triples_set)


# 删除掉单独存在于source KG或target KG中而没有与之对齐的实体所在的三元组
def remove_unlinked_triples(triples, linked_ents):
    print("before removing unlinked triples num:", len(triples))
    new_triples_set = set()
    for head, rel, tail in triples:
        if head in linked_ents and tail in linked_ents:
            new_triples_set.add((head, rel, tail))
    print("after removing unlinked triples num:", len(new_triples_set))
    return list(new_triples_set)


# 训练k个epoch
def train_k_epochs(model, source_triples, target_triples, k, args, trunc_source_ent_num):
    neighbours_of_source_triples, neighbours_of_target_triples = dict(), dict()
    start = time.time()
    # TODO:trunc_source_ent_num的作用不是很明白，下面这个if语句内的逻辑不清楚
    if trunc_source_ent_num > 0.1:
        source_embeds = model.eval_source_input_embed()
        target_embeds = model.eval_target_input_embed()
        neighbours_of_source_triples = find_neighbours_multi(source_embeds, model.source_triples_list,
                                                             trunc_source_ent_num,
                                                             args.nums_threads)
        neighbours_of_target_triples = find_neighbours_multi(target_embeds, model.target_triples_list,
                                                             trunc_source_ent_num,
                                                             args.nums_threads)
        end = time.time()
        print("generate nearest-{} neighbours: {:.3f} s, size: {:.6f} G".format(trunc_source_ent_num, end - start,
                                                                                sys.getsizeof(
                                                                                    neighbours_of_source_triples) / g))
    for epoch in range(k):
        print("epoch:", epoch + 1)
        triple_loss, mapping_loss, time_cost = train_1_epoch(model, source_triples, target_triples, args,
                                                             neighbours_of_source_triples,
                                                             neighbours_of_target_triples)
        print("triple_loss(L1) = {:.3f}, mapping_loss(L2) = {:.3f}, "
              "time = {:.3f} s".format(triple_loss, mapping_loss, time_cost))

    end = time.time()
    print("train k epochs finished, time cost:", round(end - start, 2), "s")
    if neighbours_of_source_triples is {}:
        del neighbours_of_source_triples, neighbours_of_target_triples
        gc.collect()


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
#
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

def get_transe_model(folder, kge_model, params):
    print("data folder:", folder)
    if "15" in folder:
        read_func = ut.read_dbp15k_input
    else:
        read_func = ut.read_other_input
    ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, _, ent_n, rel_n = read_func(folder)
    model = kge_model(ent_n, rel_n, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2,
                      ori_triples1.ent_list, ori_triples2.ent_list, params)
    return ori_triples1, ori_triples2, model


@ray.remote(num_cpus=1)
def find_neighbours(sub_ent_list, ent_list, sub_ent_embed, ent_embed, k, metric):
    dic = dict()
    if metric == 'euclidean':
        sim_mat = np.matmul(sub_ent_embed, ent_embed.T)
        for i in range(sim_mat.shape[0]):
            sort_index = np.argpartition(-sim_mat[i, :], k + 1)
            dic[sub_ent_list[i]] = ent_list[sort_index[0:k + 1]].tolist()
    else:
        sim_mat = compute_hyperbolic_similarity(sub_ent_embed, ent_embed)
        for i in range(sim_mat.shape[0]):
            sort_index = np.argpartition(-sim_mat[i, :], k + 1)
            dic[sub_ent_list[i]] = ent_list[sort_index[0:k + 1]].tolist()
    del sim_mat
    return dic


def find_neighbours_multi_4link_(embed1, embed2, ent_list1, ent_list2, k, params, metric='euclidean'):
    if metric == 'euclidean':
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    sub_ent_list1 = ut.div_list(np.array(ent_list1), params.nums_threads)
    sub_ent_embed_indexes = ut.div_list(np.array(range(len(ent_list1))), params.nums_threads)
    results = list()
    for i in range(len(sub_ent_list1)):
        res = find_neighbours.remote(sub_ent_list1[i], np.array(ent_list2),
                                     embed1[sub_ent_embed_indexes[i], :], embed2, k, metric)
        results.append(res)
    dic = dict()
    for res in ray.get(results):
        dic = ut.merge_dic(dic, res)
    gc.collect()
    return dic


@ray.remote(num_cpus=1)
def find_neighbours_from_sim_mat(ent_list_x, ent_list_y, sim_mat, k):
    dic = dict()
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k + 1)
        dic[ent_list_x[i]] = ent_list_y[sort_index[0:k + 1]].tolist()
    return dic


def find_neighbours_multi_4link_from_sim(ent_list_x, ent_list_y, sim_mat, k, nums_threads):
    ent_list_x_tasks = ut.div_list(np.array(ent_list_x), nums_threads)
    ent_list_x_indexes = ut.div_list(np.array(range(len(ent_list_x))), nums_threads)
    dic = dict()
    rest = []
    for i in range(len(ent_list_x_tasks)):
        res = find_neighbours_from_sim_mat.remote(ent_list_x_tasks[i], np.array(ent_list_y),
                                                  sim_mat[ent_list_x_indexes[i], :], k)
        rest.append(res)
    for res in ray.get(rest):
        dic = ut.merge_dic(dic, res)
    return dic


def find_neighbours_multi_4link(embed1, embed2, ent_list1, ent_list2, k, params, metric='euclidean', is_one=False):
    if metric == 'euclidean':
        sim_mat = np.matmul(embed1, embed2.T)
    else:
        sim_mat = sim_handler_hyperbolic(embed1, embed2, 0, params.nums_neg)
    neighbors1 = find_neighbours_multi_4link_from_sim(ent_list1, ent_list2, sim_mat, k, params.nums_neg)
    if is_one:
        return neighbors1, None
    neighbors2 = find_neighbours_multi_4link_from_sim(ent_list2, ent_list1, sim_mat.T, k, params.nums_neg)
    return neighbors1, neighbors2


# TODO:不是很清楚这里是干嘛的
def find_neighbours_multi(embed, ent_list, k, nums_threads, metric='euclidean'):
    if nums_threads > 1:
        ent_frags = ut.div_list(np.array(ent_list), nums_threads)
        ent_frag_indexes = ut.div_list(np.array(range(len(ent_list))), nums_threads)
        dic = dict()
        rest = []
        for i in range(len(ent_frags)):
            res = find_neighbours.remote(ent_frags[i], np.array(ent_list), embed[ent_frag_indexes[i], :], embed,
                                         k, metric)
            rest.append(res)
        for res in ray.get(rest):
            dic = ut.merge_dic(dic, res)
    else:
        dic = find_neighbours(np.array(ent_list), np.array(ent_list), embed, embed, k, metric)
    del embed
    gc.collect()
    return dic


def trunc_sampling(pos_triples, all_triples, dic, ent_list):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                candidates = dic.get(h, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                h2 = candidates[index]
            elif choice >= 500:
                candidates = dic.get(t, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                t2 = candidates[index]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    return neg_triples


def trunc_sampling_multi(pos_triples, all_triples, dic, ent_list, multi):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = dic.get(h, ent_list)
            h2s = random.sample(candidates, multi)
            temp_neg_triples = [(h2, r, t) for h2 in h2s]
            neg_triples.extend(temp_neg_triples)
        elif choice >= 500:
            candidates = dic.get(t, ent_list)
            t2s = random.sample(candidates, multi)
            temp_neg_triples = [(h, r, t2) for t2 in t2s]
            neg_triples.extend(temp_neg_triples)
    # neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


def generate_batch_via_neighbour(triples1, triples2, step, batch_size, neighbours_dic1, neighbours_dic2, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    neg_triples.extend(trunc_sampling_multi(pos_triples1, triples1.triples, neighbours_dic1, triples1.ent_list, multi))
    neg_triples.extend(trunc_sampling_multi(pos_triples2, triples2.triples, neighbours_dic2, triples2.ent_list, multi))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def generate_neg_triples(pos_triples, triples_data):
    all_triples = triples_data.triples
    entities = triples_data.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                h2 = random.sample(entities, 1)[0]
            elif choice >= 500:
                t2 = random.sample(entities, 1)[0]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    assert len(neg_triples) == len(pos_triples)
    return neg_triples


def generate_neg_triples_multi(pos_triples, triples_data, multi):
    all_triples = triples_data.triples
    entities = triples_data.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            h2s = random.sample(entities, multi)
            neg_triples.extend([(h2, r, t) for h2 in h2s])
        elif choice >= 500:
            t2s = random.sample(entities, multi)
            neg_triples.extend([(h, r, t2) for t2 in t2s])
    neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


def generate_pos_neg_batch(triples1, triples2, step, batch_size, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    # for i in range(multi):
    #     choice = random.randint(0, 999)
    #     if choice < 500:
    #         h = True
    #     else:
    #         h = False
    #     # neg_triples.extend(generate_neg_triples_batch(pos_triples1, triples1, h))
    #     # neg_triples.extend(generate_neg_triples_batch(pos_triples2, triples2, h))
    #     neg_triples.extend(generate_neg_triples(pos_triples1, triples1))
    #     neg_triples.extend(generate_neg_triples(pos_triples2, triples2))
    neg_triples.extend(generate_neg_triples_multi(pos_triples1, triples1, multi))
    neg_triples.extend(generate_neg_triples_multi(pos_triples2, triples2, multi))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_triples_of_latent_entities(triples1, triples2, entities1, entites2):
    assert len(entities1) == len(entites2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(entities1)):
        newly_triples1.extend(generate_newly_triples(entities1[i], entites2[i], triples1.rt_dict, triples1.hr_dict))
        newly_triples2.extend(generate_newly_triples(entites2[i], entities1[i], triples2.rt_dict, triples2.hr_dict))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples
