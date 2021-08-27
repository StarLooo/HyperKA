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

from hyperka.ea_apps.model import HyperKA
from hyperka.ea_funcs.train_funcs import get_model, train_k_epochs

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

# TODO:由于不明白bootstrapping，所以暂且只修改了不进行bootstrapping下的代码
if __name__ == '__main__':
    args = parser.parse_args()
    print("show the args:")
    print(args)
    print()

    print("get model...")
    source_triples, target_triples, model = get_model(args.input, HyperKA, args)
    print("get model finished\n")
    os.system("pause")  # 2021/8/27 14:09改代码至能够正确运行到此处

    hits1, old_hits1 = None, None
    trunc_source_ent_num = int(len(source_triples.ent_list) * (1.0 - args.epsilon4triple))
    print("trunc ent num for triples:", trunc_source_ent_num)
    if args.is_bp:
        epochs_each_iteration = 5
    else:
        epochs_each_iteration = 10
    assert args.epochs % epochs_each_iteration == 0
    num_iteration = args.epochs // epochs_each_iteration  # 循环次数
    for iteration in range(num_iteration):
        print("iteration", iteration + 1)
        train_k_epochs(model, source_triples, target_triples, epochs_each_iteration, args, trunc_source_ent_num)
        if iteration % args.test_interval == 0:
            model.test(k=0)
        if iteration >= args.start_bp and args.is_bp:
            semi_alignment(model, args)
    model.test()
