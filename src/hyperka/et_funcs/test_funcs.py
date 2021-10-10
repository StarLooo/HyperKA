# -*- coding: utf-8 -*-
import multiprocessing
import gc
import numpy as np
import time

from hyperka.et_funcs.utils import div_list
from hyperka.hyperbolic.metric import compute_hyperbolic_similarity

'''
    由于该test_funcs.py文件中并未涉及到tf的内容，所以未作修改，并且也没有仔细理解流程(多线程编程确实之前从未涉及过，看起来比较头大)
'''


def cal_rank_hyperbolic(frags, sub_embed, embed, multi_types_list, top_k, greedy):
    onto_number = embed.shape[0]
    mr = 0
    mrr = 0
    hits = np.array([0 for _ in top_k])
    sim_mat = compute_hyperbolic_similarity(sub_embed, embed)
    results = set()
    test_num = sub_embed.shape[0]
    for i in range(len(frags)):
        ref = frags[i]
        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        results.add((ref, aligned_e))
        multi_types = multi_types_list[ref]
        if greedy:
            rank_index = onto_number
            for item in multi_types:
                temp_rank_index = np.where(rank == item)[0][0]
                rank_index = min(temp_rank_index, rank_index)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    hits[j] += 1
        else:
            for item in multi_types:
                rank_index = np.where(rank == item)[0][0]
                mr += (rank_index + 1)
                mrr += 1 / (rank_index + 1)
                for j in range(len(top_k)):
                    if rank_index < top_k[j]:
                        hits[j] += 1
            test_num += (len(multi_types) - 1)
    del sim_mat
    gc.collect()
    return mr, mrr, hits, results, test_num


def eval_type_hyperbolic(embed1, embed2, ent_type, top_k, nums_threads, greedy=True, mess=""):
    t = time.time()
    ref_num = embed1.shape[0]
    hits = np.array([0 for _ in top_k])
    mr = 0
    mrr = 0
    total_test_num = 0
    total_alignment = set()

    # TODO：这里会报错：
    #  RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad,
    #  since autograd does not support crossing process boundaries.
    #  If you just want to transfer the data, call detach() on the tensor before serializing
    #  (e.g., putting it on the queue).
    frags = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(frags))
    results = list()
    for frag in frags:
        results.append(
            pool.apply_async(cal_rank_hyperbolic,
                             (frag, embed1[frag, :].detach(), embed2.detach(), ent_type, top_k, greedy)))
    pool.close()
    pool.join()

    for res in results:
        mr1, mrr1, hits1, alignment, test_num = res.get()
        mr += mr1
        mrr += mrr1
        hits += hits1
        total_test_num += test_num
        total_alignment |= alignment

    if greedy:
        assert total_test_num == ref_num
    else:
        print("multi types:", total_test_num - ref_num)

    hits = hits / total_test_num
    for i in range(len(hits)):
        hits[i] = round(hits[i], 4)
    mr /= total_test_num
    mrr /= total_test_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, hits, mr, mrr,
                                                                                 time.time() - t))
    return hits[0]
