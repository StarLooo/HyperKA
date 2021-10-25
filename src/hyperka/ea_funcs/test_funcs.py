import gc
import numpy as np
import time
import ray

from src.hyperka.ea_funcs.utils import div_list
from src.hyperka.hyperbolic.metric import compute_hyperbolic_distances, normalization

g = 1000000000


# TODO:封装测试函数，内部逻辑不是很清楚，暂时当成黑箱处理
def eval_alignment_hyperbolic_multi(embed1, embed2, top_k, nums_threads, message=""):
    assert embed1.requires_grad is False and embed2.requires_grad is False
    assert embed1.shape == embed2.shape
    start = time.time()
    ref_num = embed1.shape[0]
    # initial hits list
    hits = np.array([0 for k in top_k])
    # Mean Rank(MR):
    # 平均到第多少个才能匹配到正确的结果
    # Mean Reciprocal Rank(MRR):
    # 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，以此类推
    mr = 0
    mrr = 0
    total_alignment_set = set()
    # frag应该是fragment的缩写，这部分代码涉及多线程，本机测试中默认nums_threads=1，也就是不进行多线程测试
    frags = div_list(np.array(range(ref_num)), nums_threads)
    results_list = list()
    for frag in frags:
        sub_embed1 = embed1[frag, :]
        result = cal_rank_multi_embed_hyperbolic(frag=frag, sub_embed1=sub_embed1, embed2=embed2, top_k=top_k)
        results_list.append(result)

    # TODO:暂时不考虑ray库的代码，想办法替代
    # for res in ray.get(results_list):
    for res in results_list:
        sub_mr, sub_mrr, sub_hits, sub_alignment_set = res
        mr += sub_mr
        mrr += sub_mrr
        hits += sub_hits
        total_alignment_set |= sub_alignment_set

    assert len(total_alignment_set) == ref_num

    hits = hits / ref_num
    for i in range(len(hits)):
        hits[i] = round(hits[i], 4)  # 舍入
    mr /= ref_num
    mrr /= ref_num
    end = time.time()
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(message, top_k, hits, mr, mrr,
                                                                                 end - start))
    gc.collect()
    return hits[0]


# TODO:被eval_alignment_hyperbolic_multi封装调用的多线程测试函数，内部逻辑不是很清楚，暂时当成黑箱处理
# @ray.remote(num_cpus=1)
def cal_rank_multi_embed_hyperbolic(frag, sub_embed1, embed2, top_k):
    print("cal_rank_multi_embed_hyperbolic begin...")
    sub_mr = 0
    sub_mrr = 0
    sub_hits = np.array([0 for k in top_k])
    sim_matrix = compute_hyperbolic_similarity_single(sub_embed1, embed2)
    results_set = set()
    for i in range(frag.size):
        ref = frag[i]
        rank = (-sim_matrix[i, :]).argsort()  # default ascending
        aligned_ent = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        sub_mr += (rank_index + 1)
        sub_mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                sub_hits[j] += 1
        results_set.add((ref, aligned_ent))
    del sim_matrix
    print("cal_rank_multi_embed_hyperbolic finished.\n")
    return sub_mr, sub_mrr, sub_hits, results_set


# TODO:被cal_rank_multi_embed_hyperbolic调用的计算sim_matrix的函数，内部逻辑不是很清楚，暂时当成黑箱处理
def compute_hyperbolic_similarity_single(sub_embed1, embed2):
    sub_embed1 = sub_embed1.cpu()
    embed2 = sub_embed1.cpu()
    print("compute_hyperbolic_similarity_single begin...")
    x1, y1 = sub_embed1.shape  # <class 'numpy.ndarray'>
    x2, y2 = embed2.shape
    assert y1 == y2
    distance_vector_list = list()
    for i in range(x1):
        sub_embeds1_line = sub_embed1[i, :]  # <class 'numpy.ndarray'> (y1,)
        sub_embeds1_line = np.reshape(sub_embeds1_line, (1, y1))  # (1, y1)
        sub_embeds1_line = np.repeat(sub_embeds1_line, x2, axis=0)  # (x2, y1)
        dist_vec = compute_hyperbolic_distances(sub_embeds1_line, embed2)
        distance_vector_list.append(dist_vec)
    dis_mat = np.row_stack(distance_vector_list)  # (x1, x2)
    print("compute_hyperbolic_similarity_single finished.\n")
    return normalization(-dis_mat)


# TODO：用于bootstrapping, 被semi_alignment函数调用，内部逻辑不是很清楚，暂时当成黑箱处理
def sim_handler_hyperbolic(embed1, embed2, k, nums_threads):
    print("sim_handler_hyperbolic begin...")
    assert embed1.requires_grad is False and embed2.requires_grad is False
    tasks = div_list(np.array(range(embed1.shape[0])), nums_threads)
    # results_list = list()
    # for task in tasks:
    #     result = compute_hyperbolic_similarity(embed1[task, :], embed2)
    #     results_list.append(result)
    # sim_lists = list()
    # for result in ray.get(results_list):
    #     sim_lists.append(result)
    sim_lists = list()
    for task in tasks:
        sim = compute_hyperbolic_similarity(embed1[task, :], embed2)
        sim_lists.append(sim)
    sim_matrix = np.concatenate(sim_lists, axis=0)
    if k == 0:
        return sim_matrix
    csls1 = csls_neighbor_sim(sim_matrix, k, nums_threads)
    csls2 = csls_neighbor_sim(sim_matrix.T, k, nums_threads)
    csls_sim_mat = 2 * sim_matrix.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_matrix
    gc.collect()
    print("sim_handler_hyperbolic end")
    return csls_sim_mat


# TODO:被sim_handler_hyperbolic调用的计算双曲相似度的函数，内部逻辑不是很清楚，暂时当成黑箱处理
# @ray.remote(num_cpus=1)
def compute_hyperbolic_similarity(embeds1, embeds2):
    print("compute_hyperbolic_similarity begin...")
    x1, y1 = embeds1.shape  # <class 'numpy.ndarray'>
    x2, y2 = embeds2.shape
    assert y1 == y2
    dist_vec_list = list()
    for i in range(x1):
        embed1 = embeds1[i, :]  # <class 'numpy.ndarray'> (y1,)
        embed1 = np.reshape(embed1, (1, y1))  # (1, y1)
        embed1 = np.repeat(embed1, x2, axis=0)  # (x2, y1)
        dist_vec = compute_hyperbolic_distances(embed1, embeds2)
        dist_vec_list.append(dist_vec)
    dis_mat = np.row_stack(dist_vec_list)  # (x1, x2)
    print("compute_hyperbolic_similarity end.")
    return normalization(-dis_mat)


# @ray.remote(num_cpus=1)
def cal_rank(task, sim, top_k):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    for i in range(len(task)):
        ref = task[i]
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num


def eval_alignment_mul(sim_mat, top_k, nums_threads, mess=""):
    t = time.time()
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0

    tasks = div_list(np.array(range(ref_num)), nums_threads)
    results = list()
    for task in tasks:
        res = cal_rank(task, sim_mat[task, :], top_k)
        results.append(res)
    for res in ray.get(results):
        mean, mrr, num = res
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)

    acc = np.array(t_num) / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    t_mean /= ref_num
    t_mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, t_mean, t_mrr,
                                                                                 time.time() - t))
    return acc[0]


def cal_rank_multi_embed(frags, dic, sub_embed, embed, top_k):
    print("cal_rank_multi_embed begin...")
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    sim_mat = np.matmul(sub_embed, embed.T)  # ndarray
    # print("matmul sim mat type:", type(sim_mat))
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]
        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))

    del sim_mat
    gc.collect()
    print("cal_rank_multi_embed end.")
    return mean, mrr, num, mean1, mrr1, num1, prec_set


# @ray.remote(num_cpus=1)
def cal_csls_neighbor_sim(sim_mat, k):
    print("cal_csls_neighbor_sim begin...")
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    print("cal_csls_neighbor_sim end.")
    return sim_values


def csls_neighbor_sim(sim_mat, k, nums_threads):
    print("csls_neighbor_sim begin...")
    tasks = div_list(np.array(range(sim_mat.shape[0])), nums_threads)
    results = list()
    for task in tasks:
        res = cal_csls_neighbor_sim(sim_mat[task, :], k)
        results.append(res)
    sim_values = None
    for res in results:
        val = res
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat.shape[0]
    print("csls_neighbor_sim end.")
    return sim_values
