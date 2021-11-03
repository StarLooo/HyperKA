# -*- coding: utf-8 -*-
import gc
import math
import random
import sys
import time

import numpy as np
import ray
from sklearn import preprocessing

import src.hyperka.ea_funcs.utils as ut
from src.hyperka.ea_apps.util import generate_adjacent_graph
from src.hyperka.ea_funcs.test_funcs import sim_handler_hyperbolic
from src.hyperka.ea_funcs.train_bp import bootstrapping
from src.hyperka.hyperbolic.metric import compute_hyperbolic_similarity, normalization

g = 1000000000


# 获取模型
def get_model(folder, kge_model, args):
    print("data folder:", folder)

    print("read_input begin...")
    if "15" in folder:
        read_func = ut.read_dbp15k_input
    else:
        read_func = ut.read_other_input

    source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents, \
    ref_source_aligned_ents, ref_target_aligned_ents, total_ents_num, total_rels_num = read_func(folder)

    # inked_entities中的ent为已经对齐的ent
    linked_entities = set(
        sup_source_aligned_ents + sup_target_aligned_ents + ref_source_aligned_ents + ref_target_aligned_ents)

    # TODO: 增强triples中的数据
    print("enhance triples begin:")
    enhanced_source_triples_list, enhanced_target_triples_list = enhance_triples(source_triples, target_triples,
                                                                                 sup_source_aligned_ents,
                                                                                 sup_target_aligned_ents)
    print("enhance triples finished")

    # 删除掉单独存在于source KG或target KG中而没有与之对齐的实体所在的三元组
    all_triples_list = source_triples.triple_list + target_triples.triple_list + \
                       enhanced_source_triples_list + enhanced_target_triples_list
    # TODO: 暂时先跳过remove_unlinked_triples()这一步
    # all_aligned_triples_list = remove_unlinked_triples(all_triples_list=all_triples_list,
    #                                                    linked_entities=linked_entities)
    all_aligned_triples_list = all_triples_list

    # 这里应该与et里是一样的
    near_ents_graph, near_rels_graph = generate_adjacent_graph(total_ents_num=total_ents_num,
                                                               total_rels_num=total_rels_num,
                                                               triples=all_aligned_triples_list)
    print("near_ents_adj shape:", near_ents_graph[0].shape)
    print("near_rels_adj shape:", near_rels_graph[0].shape)
    print("generate_adjacent_graph finished\n")

    model = kge_model(total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
                      ref_source_aligned_ents, ref_target_aligned_ents, source_triples.ent_list,
                      target_triples.ent_list, near_ents_graph, near_rels_graph, args)

    # 注意这里的source_triples和target_triples并没有去掉无法对齐的triples
    return source_triples, target_triples, model


# 获取模型
def get_origin_model(folder, kge_model, args):
    print("data folder:", folder)

    print("read_input begin...")
    if "15" in folder:
        read_func = ut.read_dbp15k_input
    else:
        read_func = ut.read_other_input

    source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents, \
    ref_source_aligned_ents, ref_target_aligned_ents, total_ents_num, total_rels_num = read_func(folder)

    # inked_entities中的ent为已经对齐的ent
    linked_entities = set(
        sup_source_aligned_ents + sup_target_aligned_ents + ref_source_aligned_ents + ref_target_aligned_ents)

    # TODO: 增强triples中的数据
    print("enhance triples begin:")
    enhanced_source_triples_list, enhanced_target_triples_list = enhance_triples(source_triples, target_triples,
                                                                                 sup_source_aligned_ents,
                                                                                 sup_target_aligned_ents)
    print("enhance triples finished")

    # 删除掉单独存在于source KG或target KG中而没有与之对齐的实体所在的三元组
    all_triples_list = source_triples.triple_list + target_triples.triple_list + \
                       enhanced_source_triples_list + enhanced_target_triples_list
    # TODO: 暂时先跳过remove_unlinked_triples()这一步
    # all_aligned_triples_list = remove_unlinked_triples(all_triples_list=all_triples_list,
    #                                                    linked_entities=linked_entities)
    all_aligned_triples_list = all_triples_list

    # 这里应该与et里是一样的
    adj = generate_adjacent_graph(total_ents_num=total_ents_num, total_rels_num=total_rels_num,
                                  triples=all_aligned_triples_list, origin=True)

    print("adj shape:", adj.shape)
    print("generate_adjacent_graph finished\n")

    model = kge_model(total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
                      ref_source_aligned_ents, ref_target_aligned_ents, source_triples.ent_list,
                      target_triples.ent_list, adj, args)

    # 注意这里的source_triples和target_triples并没有去掉无法对齐的triples
    return source_triples, target_triples, model


# 数据增强
def enhance_triples(source_triples, target_triples, sup_source_aligned_ents, sup_target_aligned_ents):
    assert len(sup_source_aligned_ents) == len(sup_target_aligned_ents)

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
        enhanced_head = links_from_target_to_source.get(source_head, None)
        enhanced_tail = links_from_target_to_source.get(source_tail, None)

        if enhanced_head is not None and enhanced_tail is not None \
                and enhanced_tail not in source_triples.out_related_ents_dict.get(enhanced_head, set()):
            enhanced_target_triples_set.add((enhanced_head, target_rel, enhanced_tail))

    print("enhanced source KG's triples num:", len(enhanced_source_triples_set))
    print("enhanced target KG's triples num:", len(enhanced_target_triples_set))

    return list(enhanced_source_triples_set), list(enhanced_target_triples_set)


# 删除掉单独存在于source KG或target KG中而没有与之对齐的实体所在的三元组
def remove_unlinked_triples(all_triples_list, linked_entities):
    print("before removing unlinked triples num:", len(all_triples_list))
    new_triples_set = set()
    for head, rel, tail in all_triples_list:
        if head in linked_entities and tail in linked_entities:
            new_triples_set.add((head, rel, tail))
    print("after removing unlinked triples num:", len(new_triples_set))
    return list(new_triples_set)


# 训练k个epoch
def train_k_epochs(model, source_triples, target_triples, k, args, trunc_source_ent_num, iteration):
    neighbours_of_source_triples, neighbours_of_target_triples = dict(), dict()
    start = time.time()
    # TODO:trunc_source_ent_num的作用不是很明白，下面这个if语句内的逻辑不清楚
    if trunc_source_ent_num > 0.1:
        print("begin generate nearest-{} neighbours".format(trunc_source_ent_num))
        source_embeds = model.eval_source_input_embed()
        target_embeds = model.eval_target_input_embed()
        neighbours_of_source_triples = find_neighbours_multi(embeds=source_embeds, ent_list=model.source_triples_list,
                                                             k=trunc_source_ent_num, nums_threads=args.nums_threads, )
        neighbours_of_target_triples = find_neighbours_multi(embeds=target_embeds, ent_list=model.target_triples_list,
                                                             k=trunc_source_ent_num, nums_threads=args.nums_threads, )
        end = time.time()
        print("generate nearest-{} neighbours: {:.3f} s, size: {:.6f} G".format(trunc_source_ent_num, end - start,
                                                                                sys.getsizeof(
                                                                                    neighbours_of_source_triples) / g))

    for epoch in range(k):
        print("epoch:", epoch + 1)
        triple_loss, mapping_loss, time_cost = train_1_epoch(model, source_triples, target_triples, args,
                                                             neighbours_of_source_triples,
                                                             neighbours_of_target_triples, iteration)
        print("triple_loss(L1) = {:.3f}, mapping_loss(L2) = {:.3f}, "
              "time = {:.3f} s".format(triple_loss, mapping_loss, time_cost))

    end = time.time()
    print("train k epochs finished, time cost:", round(end - start, 2), "s")
    if neighbours_of_source_triples is {}:
        del neighbours_of_source_triples, neighbours_of_target_triples
        gc.collect()


# 训练1个epoch
def train_1_epoch(model, source_triples, target_triples, args, neighbours_of_source_triples,
                  neighbours_of_target_triples, iteration, burn_in=5):
    lr = args.learning_rate
    if iteration <= burn_in:
        lr /= 5  # 该开始的几个iteration学习率要调小点
    triple_loss, mapping_loss = 0, 0
    start = time.time()
    # 一个epoch需要跑steps部‍，每一步跑batch_size大小的数据
    steps = math.ceil((source_triples.triples_num + target_triples.triples_num) / args.batch_size)
    # print("steps per epoch:", steps)
    mapping_batch_size = math.ceil(len(model.sup_source_aligned_ents) / steps)
    for step in range(steps):
        # if step % 5 == 0:
        #     print("\tstep:", step + 1)
        triple_step_loss, triple_step_time = train_triple_1_step(model, source_triples, target_triples,
                                                                 neighbours_of_source_triples,
                                                                 neighbours_of_target_triples, step, args, lr)
        triple_loss += triple_step_loss
        mapping_step_loss, mapping_step_time = train_mapping_1_step(model, mapping_batch_size,
                                                                    args.mapping_neg_nums,
                                                                    lr)
        mapping_loss += mapping_step_loss
    triple_loss /= steps
    mapping_loss /= steps
    random.shuffle(source_triples.triple_list)
    random.shuffle(target_triples.triple_list)
    end = time.time()
    return triple_loss, mapping_loss, round(end - start, 2)


# 根据triple loss训练一步
def train_triple_1_step(model, source_triples, target_triples, neighbours_of_source_triples,
                        neighbours_of_target_triples, step, args, lr):
    start = time.time()
    if neighbours_of_target_triples is None:
        pos_batch, neg_batch = generate_triple_pos_neg_batch(source_triples, target_triples, step, args.batch_size,
                                                             args.triple_neg_nums)
    else:
        pos_batch, neg_batch = generate_triple_batch_via_neighbour(source_triples, target_triples, step,
                                                                   args.batch_size,
                                                                   neighbours_of_source_triples,
                                                                   neighbours_of_target_triples,
                                                                   args.triple_neg_nums)

    triple_pos_neg_batch = [pos_batch, neg_batch, lr]
    triple_loss = model.optimize_triple_loss(triple_pos_neg_batch).data

    end = time.time()
    return triple_loss, round(end - start, 2)


# 根据mapping loss训练一步
def train_mapping_1_step(model, mapping_batch_size, mapping_neg_nums, lr):
    start = time.time()
    pos_batch, neg_batch = generate_mapping_pos_neg_batch(model, mapping_batch_size, mapping_neg_nums)

    pos_h = [p[0] for p in pos_batch]
    pos_t = [p[1] for p in pos_batch]
    neg_h = [n[0] for n in neg_batch]
    neg_t = [n[1] for n in neg_batch]

    # TODO: new_alignment_pairs作用未知
    if len(model.new_alignment_pairs) > 0:
        new_batch_size = math.ceil(
            len(model.new_alignment_pairs) / len(model.sup_source_aligned_ents) * mapping_batch_size)
        samples = random.sample(model.new_alignment_pairs, new_batch_size)
        new_pos_h = [pair[0] for pair in samples]
        new_pos_t = [pair[1] for pair in samples]
    else:
        new_pos_h = [pos_h[0]]
        new_pos_t = [pos_t[0]]

    mapping_pos_neg_batch = [pos_h, pos_t, neg_h, neg_t, new_pos_h, new_pos_h, lr]

    mapping_loss = model.optimize_mapping_loss(mapping_pos_neg_batch).data

    end = time.time()
    return mapping_loss, round(end - start, 2)


# 通过邻居信息生成triple batch(包含neg负例和pos正例)
def generate_triple_batch_via_neighbour(source_triples, target_triples, step, batch_size, neighbours_of_source_triples,
                                        neighbours_of_target_triples, triple_neg_nums):
    assert triple_neg_nums >= 1
    pos_triples_1, pos_triples_2 = generate_pos_triples(source_triples.triple_list, target_triples.triple_list, step,
                                                        batch_size)

    pos_batch = pos_triples_1 + pos_triples_2

    neg_triples = list()
    neg_triples.extend(trunc_sampling_multi(pos_triples_1, source_triples.triples, neighbours_of_source_triples,
                                            source_triples.ent_list, triple_neg_nums))
    neg_triples.extend(trunc_sampling_multi(pos_triples_2, target_triples.triples, neighbours_of_target_triples,
                                            target_triples.ent_list, triple_neg_nums))
    neg_batch = neg_triples
    return pos_batch, neg_batch


# 生成triple batch(包含neg负例和pos正例)
def generate_triple_pos_neg_batch(source_triples, target_triples, step, batch_size, triple_neg_nums=1):
    assert triple_neg_nums >= 1

    pos_triples_1, pos_triples_2 = generate_pos_triples(source_triples.triple_list, target_triples.triple_list, step,
                                                        batch_size)
    pos_batch = pos_triples_1 + pos_triples_2

    neg_triples = list()
    neg_triples.extend(generate_neg_triples(pos_triples_1, source_triples, triple_neg_nums))
    neg_triples.extend(generate_neg_triples(pos_triples_2, target_triples, triple_neg_nums))

    neg_batch = neg_triples
    return pos_batch, neg_batch


# 生成pos triples
def generate_pos_triples(source_triples_list, target_triples_list, step, batch_size):
    num1 = int(len(source_triples_list) / (len(source_triples_list) + len(target_triples_list)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(source_triples_list):
        end1 = len(source_triples_list)
    if end2 > len(target_triples_list):
        end2 = len(target_triples_list)
    pos_triples_1 = source_triples_list[start1: end1]
    pos_triples_2 = target_triples_list[start2: end2]
    return pos_triples_1, pos_triples_2


# 生成neg triples
def generate_neg_triples(pos_triples, all_triples, triple_neg_nums):
    all_triples_set = all_triples.triples
    ent_list = all_triples.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            neg_h_list = random.sample(ent_list, triple_neg_nums)
            neg_triples.extend([(neg_h, r, t) for neg_h in neg_h_list])
        elif choice >= 500:
            neg_t_list = random.sample(ent_list, triple_neg_nums)
            neg_triples.extend([(h, r, neg_t) for neg_t in neg_t_list])
    neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


# 生成neg triples
def trunc_sampling_multi(pos_triples, all_triples, neighbours_of_triples, ent_list, triple_nums_neg):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = neighbours_of_triples.get(h, ent_list)
            neg_h_list = random.sample(candidates, triple_nums_neg)
            temp_neg_triples = [(neg_h, r, t) for neg_h in neg_h_list]
            neg_triples.extend(temp_neg_triples)
        elif choice >= 500:
            candidates = neighbours_of_triples.get(t, ent_list)
            neg_t_list = random.sample(candidates, triple_nums_neg)
            temp_neg_triples = [(h, r, neg_t) for neg_t in neg_t_list]
            neg_triples.extend(temp_neg_triples)
    # TODO:为什么源码中这里不需要去除掉all_triples呢？
    neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


# 生成mapping batch(包含neg负例和pos正例)
def generate_mapping_pos_neg_batch(model, mapping_batch_size, mapping_neg_nums):
    assert mapping_batch_size <= len(model.sup_source_aligned_ents)
    pos_batch = random.sample(model.sup_aligned_ents_pairs, mapping_batch_size)
    neg_batch = list()
    # sup_source_aligned_ents, sup_target_aligned_ents,ref_source_aligned_ents, ref_target_aligned_ents
    for i in range(mapping_neg_nums // 2):
        neg_h_list = random.sample(model.sup_source_aligned_ents + model.ref_source_aligned_ents, mapping_batch_size)
        neg_t_list = random.sample(model.sup_target_aligned_ents + model.ref_target_aligned_ents, mapping_batch_size)
        neg_batch.extend([(pos_batch[i][0], neg_t_list[i]) for i in range(mapping_batch_size)])
        neg_batch.extend([(neg_h_list[i], pos_batch[i][1]) for i in range(mapping_batch_size)])

    neg_batch = list(set(neg_batch) - set(model.sup_aligned_ents_pairs) - set(model.self_aligned_ents_pairs))
    return pos_batch, neg_batch


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


# TODO:不是很清楚这里是干嘛的
def find_neighbours_multi(embeds, ent_list, k, nums_threads, metric='euclidean'):
    embeds = embeds.cpu()
    if nums_threads > 1:
        ent_frags = ut.div_list(np.array(ent_list), nums_threads)
        ent_frag_indexes = ut.div_list(np.array(range(len(ent_list))), nums_threads)
        dic = dict()
        results = []
        for i in range(len(ent_frags)):
            res = find_neighbours(sub_ent_list=ent_frags[i], ent_list=np.array(ent_list),
                                  sub_ent_embed=embeds[ent_frag_indexes[i], :], ent_embed=embeds, k=k, metric=metric)
            results.append(res)
        for res in ray.get(results):
            dic = ut.merge_dic(dic, res)
    else:
        dic = find_neighbours(sub_ent_list=np.array(ent_list), ent_list=np.array(ent_list), sub_ent_embed=embeds,
                              ent_embed=embeds, k=k, metric=metric)
    del embeds
    gc.collect()
    return dic


# @ray.remote(num_cpus=1)
def find_neighbours(sub_ent_list, ent_list, sub_ent_embed, ent_embed, k, metric):
    dic = dict()
    if metric == 'euclidean':
        sim_mat = np.matmul(sub_ent_embed, ent_embed.t())
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
        res = find_neighbours(sub_ent_list1[i], np.array(ent_list2),
                              embed1[sub_ent_embed_indexes[i], :], embed2, k, metric)
        results.append(res)
    dic = dict()
    for res in ray.get(results):
        dic = ut.merge_dic(dic, res)
    gc.collect()
    return dic


# @ray.remote(num_cpus=1)
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
        res = find_neighbours_from_sim_mat(ent_list_x_tasks[i], np.array(ent_list_y),
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


# bootstrapping
def semi_alignment(model, args):
    print("semi_alignment begin...")
    start = time.time()
    ref_source_aligned_ents_embed = model.eval_output_embed(model.ref_source_aligned_ents, is_map=True)
    ref_target_aligned_ents_embed = model.eval_output_embed(model.ref_target_aligned_ents, is_map=False)
    sim_matrix = sim_handler_hyperbolic(ref_source_aligned_ents_embed, ref_target_aligned_ents_embed, 5,
                                        args.nums_threads)
    sim_matrix = normalization(sim_matrix)
    # temp_sim_th = (np.mean(sim_mat) + np.max(sim_mat)) / 2
    # sim_th = (params.sim_th + temp_sim_th) / 2
    # print("min, mean, and max of sim mat, sim_th = ", np.min(sim_mat), np.mean(sim_mat), np.max(sim_mat), sim_th)
    sim_th = args.sim_th
    new_alignment, entities1, entities2 = bootstrapping(sim_matrix, model.ref_source_aligned_ents,
                                                        model.ref_target_aligned_ents, model.new_alignment,
                                                        sim_th, args.nearest_k, is_edit=True,
                                                        heuristic=args.heuristic)
    model.new_alignment = list(new_alignment)
    model.new_alignment_pairs = [(entities1[i], entities2[i]) for i in range(len(entities1))]
    end = time.time()
    print("semi-supervised alignment costs time = {:.3f} s\n".format(end - start))
    print("semi_alignment end.")
