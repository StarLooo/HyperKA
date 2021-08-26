# -*- coding: utf-8 -*-
import gc
import os

import ray
import numpy as np
import random
from sklearn import preprocessing

from hyperka.ea_apps.util import gen_adj
import hyperka.ea_funcs.utils as ut
from hyperka.hyperbolic.metric import compute_hyperbolic_similarity
from hyperka.ea_funcs.test_funcs import sim_handler_hyperbolic

g = 1000000000


def get_model(folder, kge_model, params):
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
    enhanced_source_triples, enhanced_target_triples = enhance_triples(source_triples, target_triples,
                                                                       sup_source_aligned_ents,
                                                                       sup_target_aligned_ents)
    print("enhance triples finished")
    os.system("pause")

    triples = remove_unlinked_triples(source_triples.triple_list + target_triples.triple_list +
                                      list(enhanced_source_triples) + list(enhanced_target_triples), linked_entities)
    adj = gen_adj(total_ents_num, triples)
    model = kge_model(total_ents_num, total_rels_num, sup_source_aligned_ents, sup_target_aligned_ents,
                      ref_source_aligned_ents, ref_target_aligned_ents, source_triples.ent_list,
                      target_triples.ent_list, adj, params)
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

    return enhanced_source_triples_set, enhanced_target_triples_set


def remove_unlinked_triples(triples, linked_ents):
    print("before removing unlinked triples:", len(triples))
    new_triples = set()
    for h, r, t in triples:
        if h in linked_ents and t in linked_ents:
            new_triples.add((h, r, t))
    print("after removing unlinked triples:", len(new_triples))
    return list(new_triples)


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
