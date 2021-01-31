# -*- coding: utf-8 -*-
# written by mark zeng 2018-11-14
# modified by Yao Zhao 2019-10-30
# re-modified by Yiming Chen 2020-11-04
import math
import random
import argparse
import numpy as np
import multiprocessing as mp
import time
import sys

mode = 'LT'
core = 8
task = 2000

def generate_ic(graph,node):
    total = [node]
    active = np.zeros(len(graph), int)
    activity_set = [node]
    for i in range(0, len(activity_set)):
        active[activity_set[i] - 1] = 1
    while len(activity_set) != 0:
        new_activity_set = []
        for i in activity_set:
            for j in graph[i - 1]:
                if active[j[0] - 1] == 1:
                    continue
                r = np.random.random()
                if r < j[1]:
                    active[j[0] - 1] = 1
                    new_activity_set.append(j[0])
        for p in new_activity_set:
            total.append(p)
        activity_set = new_activity_set.copy()
    return total

def generate_lt(graph,node):
    active = np.zeros(len(graph), int)
    active[node-1] = 1
    total = [node]
    next_node = node
    while 1:
        if len(graph[next_node-1])!=0:
                j = random.randint(0, len(graph[next_node-1]) - 1)
                if active[graph[next_node-1][j][0]-1]==1:
                    return total
                else:
                    active[((graph[next_node-1])[j])[0]-1] = 1
                    total.append(graph[next_node-1][j][0])
                    next_node = graph[next_node-1][j][0]
        else:
            break
    return total

def lognk(n, k):
    a = 0
    for i in range(n - k + 1, n + 1):
        a = a + math.log(i)
    for i in range(1, k + 1):
        a = a - math.log(i)
    return a

def node_selection(R,k,n):
    node_index = {}
    rr_cnt = np.zeros(n+1, int)
    count = 0
    S = set()
    for i in range(0, len(R)):
        rr = R[i]
        for rr_node in rr:
            rr_cnt[rr_node] = rr_cnt[rr_node] +  1
            if rr_node not in node_index.keys():
                node_index[rr_node] = []
            node_index[rr_node].append(i)

    for i in range(k):
        max_index = 0
        max_temp = -1
        for w in range(0,len(rr_cnt)):
            if rr_cnt[w]>max_temp:
                max_temp = rr_cnt[w]
                max_index = w
        max_index = max_index
        S.add(max_index)
        count = count + len(node_index[max_index])
        index_set = [rr for rr in node_index[max_index]]
        for j in index_set:
            rr = R[j]
            for rr_node in rr:
                rr_cnt[rr_node] -= 1
                node_index[rr_node].remove(j)
    return S, count / len(R)


def sampling(graph,k,e,l):
    R = []
    LB = 1
    e1 = math.sqrt(2)*e
    n = len(graph)
    for i in range(1,int(math.log2(n))):
        x = n/(math.pow(2,i))
        lambda1 = (n*(2+e1*2/3)*(lognk(n,k)+l*math.log(n)+math.log(math.log2(n))))/(math.pow(e1,2))
        c1 = lambda1/x
        while len(R)<=c1:
            if mode=='IC':
                R.append(generate_ic(graph,random.randint(1, n)))
            else:
                R.append(generate_lt(graph, random.randint(1, n)))
        S,F = node_selection(R,k,n)
        if n*F>=(1+e1)*x:
            LB = n*F/(1+e1)
            break
    alpha = math.sqrt(l*math.log(n)+math.log(2))
    beta = math.sqrt((1-1/math.e)*(lognk(n,k)+l*math.log(n)+math.log(2)))
    lambda2 = 2*n*math.pow(((1-1/math.e)*alpha+beta),2)*math.pow(e,-2)
    c = lambda2/LB
    while len(R)<c:
        if mode == 'IC':
            R.append(generate_ic(graph, random.randint(1, n)))
        else:
            R.append(generate_lt(graph, random.randint(1, n)))
    return R


def imm(graph,k,e,l,n):
    l = l*(1+math.log(2)/math.log(n))
    R = sampling(graph,k,e,l)
    seeds = node_selection(R,k,n)[0]
    return seeds


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default="C:\\Users\\17392\PycharmProjects\imp\\NetHEPT.txt")
    parser.add_argument('-k', '--seed_size', type=int, default=5)
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=30)

    args = parser.parse_args()
    file_name = args.file_name
    seed_size = args.seed_size
    model = args.model
    mode = model
    time_limit = args.time_limit

    graph = []

    # 读取图信息
    with open(file_name, encoding='utf-8') as info:
        lines = info.readlines()
    based = lines[0].rstrip().split()
    graph = [[] for i in range(int(based[0]))]
    for i in range(1, int(based[1]) + 1):
        line = lines[i].rstrip().split()
        graph[int(line[1]) - 1].append([int(line[0]), float(line[2])])

    e = 0.5
    l = 1
    pool = mp.Pool(core)
    re = []
    final = {}
    method = imm

    for i in range(core):
        for j in range(0, task):
            pool.apply_async(method, args=(graph,seed_size,e,l,int(based[0])), callback=re.append)
    last_task = pool.apply_async(method, args=(graph,seed_size,e,l,int(based[0])), callback=re.append)

    pool.close()
    while time.time() - start_time < time_limit - 5 and not last_task.ready():
        time.sleep(1)
    pool.terminate()

    for seeds in re:
        for seed in seeds:
            if final.get(seed,-1)!=-1:
                final[seed] += 1
            else:
                final[seed] = 1
    final_order = sorted(final.items(),key=lambda x:x[1],reverse=True)
    for i in range(0,seed_size):
        print(final_order[i][0])

    sys.stdout.flush()
    exit(0)

