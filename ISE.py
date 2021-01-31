# -*- coding: utf-8 -*-
# written by mark zeng 2018-11-14
# modified by Yao Zhao 2019-10-30
# re-modified by Yiming Chen 2020-11-04

import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np

core = 8
n = 2000


def IC(graph,seeds):
    active = np.zeros(len(graph),int)
    activity_set = seeds.copy()
    for i in range(0, len(activity_set)):
        active[activity_set[i] - 1] = 1
    count = len(activity_set)
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
        count = count + len(new_activity_set)
        activity_set = new_activity_set.copy()
    return count

def LT(graph,seeds):
    active = np.zeros(len(graph), int)
    activity_set = seeds.copy()
    influence = [0 for i in range(len(graph))]
    threshold = [np.random.random() for i in range(len(graph))]
    for i in range(0, len(activity_set)):
        active[activity_set[i] - 1] = 1
    count = len(activity_set)
    while len(activity_set)!=0:
        new_activity_set = []

        for i in activity_set:
            for j in graph[i-1]:
                k = influence[j[0] - 1] + j[1]
                influence[j[0] - 1] = k

        for i in activity_set:
            for j in graph[i-1]:
                if active[j[0] - 1] == 1:
                    continue
                if influence[j[0] - 1]>=threshold[j[0] - 1]:
                    active[j[0] - 1] = 1
                    new_activity_set.append(j[0])
        count = count + len(new_activity_set)
        activity_set = new_activity_set.copy()
    return count


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default="C:\\Users\\17392\PycharmProjects\imp\\NetHEPT.txt")
    parser.add_argument('-s', '--seed', type=str, default="C:\\Users\\17392\PycharmProjects\imp\\network_seeds.txt")
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=30)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit

    graph = []
    seeds = []

    # 读取图信息
    with open(file_name, encoding='utf-8') as info:
        lines = info.readlines()
    based = lines[0].rstrip().split()
    graph = [[] for i in range(int(based[0]))]
    for i in range(1, int(based[1]) + 1):
        line = lines[i].rstrip().split()
        graph[int(line[0]) - 1].append([int(line[1]), float(line[2])])

    # 读取seed
    with open(seed, encoding='utf-8') as seed_info:
        lines = seed_info.readlines()
        for line in lines:
            a = line.rstrip()
            seeds.append(int(a))

    np.random.seed(0)
    pool = mp.Pool(core)

    if model == "IC":
        method = IC
    else:
        method = LT
    re = []

    for i in range(core):
        for j in range(0, n):
            pool.apply_async(method, args=(graph,seeds), callback=re.append)
    last_task = pool.apply_async(method, args=(graph, seeds), callback=re.append)

    pool.close()
    while time.time() - start_time < time_limit - 5 and not last_task.ready():
        time.sleep(1)
    pool.terminate()

    sum = 0
    for i in re:
        sum += i/len(re)
    print(sum)

    sys.stdout.flush()
    exit(0)
