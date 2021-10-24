# -*- coding: utf-8 -*-
import argparse
import ast
import torch
from src.hyperka.et_apps.model import HyperKA
from src.hyperka.et_funcs.train_funcs import get_model, train_k_epochs

parser = argparse.ArgumentParser(description='HyperKA_ET')
# parser.add_argument('--input', type=str, default='./dataset/joie/yago/')  # 路径
# parser.add_argument('--output', type=str, default='./output/results/')  # 路径
parser.add_argument('--input', type=str, default='../../../dataset/joie/yago/')  # 路径
parser.add_argument('--output', type=str, default='../../../output/results/')  # 路径

parser.add_argument('--ins_dim', type=int, default=75)  # instance嵌入向量的维度
parser.add_argument('--onto_dim', type=int, default=15)  # ontology嵌入向量的维度
parser.add_argument('--ins_layer_num', type=int, default=3)  # instance gcn层数
parser.add_argument('--onto_layer_num', type=int, default=3)  # ontology gcn层数

parser.add_argument('--neg_typing_margin', type=float, default=0.1)  # 计算mapping loss的margin
parser.add_argument('--neg_triple_margin', type=float, default=0.2)  # 计算triple loss的margin

parser.add_argument('--triple_neg_nums', type=int, default=40)  # 计算triple loss时每个正例对应多少个负例
parser.add_argument('--mapping_neg_nums', type=int, default=40)  # 计算mapping loss时每个正例对应多少个负例

parser.add_argument('--learning_rate', type=float, default=5e-4)  # learning_rate
parser.add_argument('--batch_size', type=int, default=2000)  # batch_size
parser.add_argument('--epochs', type=int, default=100)  # epochs

parser.add_argument('--epsilon4triple', type=float, default=1.0)  # TODO: 这个参数的含义不是很清楚
parser.add_argument('--mapping', type=bool, default=True)  # 是否采用mapping_matrix投影
parser.add_argument('--combine', type=ast.literal_eval, default=True)  # gcn输出的时候是否结合第0层和最后1层的嵌入向量
parser.add_argument('--ent_top_k', type=list, default=[1, 3, 5, 10])  # 用作评价指标的预测列表的下标
parser.add_argument('--nums_threads', type=int, default=1)  # TODO: 多线程数，这里本来默认值为8，但在本机上不支持，所以直接改为1

'''
    注意使用GPU前还需将torch.tensor的初始化中加入device项
    use pytorch version: 1.9.0
'''

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    args = parser.parse_args()
    print("show the args:")
    print(args)

    print("get model...")
    ins_triples, onto_triples, model = get_model(args.input, HyperKA, args)
    print("get model finished\n")

    # TODO:truncated_ins_num和truncated_onto_num的作用不是很清楚
    truncated_ins_num = int(len(model.ins_ent_list) * (1.0 - args.epsilon4triple))
    truncated_onto_num = int(len(model.onto_ent_list) * (1.0 - args.epsilon4triple))

    k = 5
    assert args.epochs % k == 0
    num_iteration = args.epochs // k  # 循环次数
    for iteration in range(1, num_iteration + 1):
        print("iteration:", iteration)
        # 每个iteration训练k个epochs
        train_k_epochs(model, ins_triples, onto_triples, k, args, truncated_ins_num, truncated_onto_num)
        h1 = model.test()
        print("h1:", h1, '\n')
    print("stop")
