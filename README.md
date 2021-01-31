# IMP&ISE Algorithm

### Report of CS303A Project-2

### 11812520 张淘月

------

#### 1. preliminary

##### 1.1 Problem Description

The influence maximazition problem are given the number of initial active nodes in a given network, the problem of maximizing influence is to find a set of fixed number of active nodes, and through a specific propagation model, the number of final active nodes is maximized.

In the research process of this problem, we can divide into two parts: ISE problem and imp problem. The ISE problem computes the maximum active degree of these seeds for a given graph and seed. The imp problem is to find the seed which can make the most influence active point for a given graph.

For the above two experimental stages, LT and IC are based on two different experimental modelsTherefore, we need to design and test the two models respectively in the experiment, and finally compare their efficiency and results.In the whole experiment process, because there are some random factors in the model, I use multi process to improve the calculation efficiency and synthesize the results of different processes to get more accurate results

##### 1.2 Problem Applications

Imp can solve the problem of information transmission in a variety of media, such as advertisement, advertisement structure and so on.

For example , In modern society, information is the most important. If we have a piece of information at present, and we want to maximize the dissemination of information, we have to consider how to choose the entry point and who to choose as the initial dissemination node, so that the whole network can be most affected by the information. A company wants to market a new product in the hope that it will be accepted by most people on the network. The company plans to initially target a small group of people, and then send them free product samples (products are very expensive, so the company has to limit the budget and only select a small group of people to distribute). The company hopes that these initial users can recommend these products to their friends, who will influence their friends. If this goes on, many individuals will eventually be affected to accept these new products. Through the powerful influence of the word of mouth world, or virus market. The problem is how to choose a collection of individuals to distribute free items so that they can influence the largest number of people on social networks.

------

#### 2. Methodology

##### 2.1 Notation

- **n**: the number of nodes in graph

- **m**: the number of edges in graph

- **k**: the number of the seed set for influence maximization

- **θ**: the number of rr sets in R

- **R**: the set of rr sets sampled

- **FR(S)**: the fraction of rr sets in R covered by node set S

##### 2.2 Data Structure

- ```python
  file_name
  # File path to store graph information
  ```

- ```python
  seed_size
  # Seed size to be generated
  ```

- ```python
  model
  # Based on which model to calculate, it can be lt or IC
  ```

- ```python
  time_limit
  # The time limit for the execution of the entire program
  ```

- ```python
  core
  # The core of the operation
  ```

- ```python
  task
  # Num of operations performed
  ```

- ```python
  graph
  # The graph executed by the algorithm is stored in the form of three-dimensional array, and all the pointed edges are stored in the index position represented by each node
  ```

- ```python
  node
  # Nodes requiring lt or IC operations are stored in the form of integers
  ```

- ```python
  activity_set
  # It is used to record the active nodes in the current layer that need BFS
  ```

- ```python
  new_activity_set
  # It is used to record the active nodes in the next layer that need BFS
  ```

- ```python
  next_node
  # The next node recorded when unidirectional DFS is removed in the lt of imp
  ```

- ```python
  total
  # All activated nodes that need to be returned
  ```

- ```python
  active
  # An array used to record which nodes have been activated. 1 is activated and 0 is not activated
  ```

##### 2.3 Model Design

###### 2.3.1. Functions

- **ISE Functions**

  ```python
  def IC(graph,seeds):
      #The number of nodes that can be activated is calculated by IC model. BFs of the whole graph is started with a certain number of seeds. The number between 0 and 1 is randomly generated to try to activate new nodes
      
  def LT(graph,seeds):
      #When the active nodes point to a certain number of active nodes, that is to say, when all the active nodes point to a certain number of active nodes
      
  if __name__ == '__main__': 
      #Read the input information and select the model to be executed according to different instructions. Control the process pool operation and time control, force to stop the unexecuted process in the remaining 5 seconds, and synthesize the final results
  ```

- **IMP Functions**

  ```python
  def generate_ic(graph,node):
      # One node is used as the initial node for BFS, and a random value of 0 to 1 is taken each time to try to activate the next node. Finally, the activated point set is returned
      
  def generate_lt(graph,node):
      # Take a node as the initial node, select one neighbor randomly at a time. If it is not activated, it will activate it and repeat the process. If it has been activated, it will exit and return all the activated nodes
      
  def node_selection(R,k,n):
      # According to the advantages of the nodes selected by R, K nodes with the highest marginal benefit are selected
      
  def sampling(graph,k,e,l):
      # Sampling function, sample the graph to get the list of rr set R.
      
  def imm(graph,k,e,l,n):
      # Sampling the graph to generate a list R containing the rr set, and then use the NodeSelection function to select the nodes in it to obtain the final result set.
      
  if __name__ == '__main__':
      # Read the input information and select the model to be executed according to different instructions. Control the process pool operation and time control, force to stop the unexecuted process in the remaining 5 seconds, and synthesize the final results
  ```

###### 2.3.2. Program processing

- **ISE Processing**

  After reading the relevant information, the main program will open the progress pool according to the selected calculation model, and call IC or LT respectively. Finally, all current calculation results will be counted and the average value will be returned when the remaining 5 s from the time limit

- **IMP Processing**

  After a separate calculation, IMM is set for the main process. Step into the process pool and repeat the execution of the IMM method, call the sampling method in IMM, and generate RR sets according to the different methods set in sampling, and then return to the final seeds set after nodeselection. Finally, select the seed with the most frequent occurrences and output according to all the calculation results.

###### 2.3.3. Assess method

$$
λ^{\prime} =\frac{(2+\frac{2}{3}\epsilon^2)*(\log\binom{n}{k}+l*\log{n}+\log{\log_2{n}})*n}{{{\epsilon}^{\prime}}^2}
$$

$$
\alpha = \sqrt{l*\log{n}+\log{2}}
$$

$$
\beta = \sqrt{(1-1/e)*(\log\binom{n}{k}+l*\log{n}+\log2)}
$$

$$
\lambda^{*} = 2n*((1-1/e)*\alpha+\beta)^2*\epsilon^{-2}
$$

##### 2.4 Detail of Algorithms

- **IMM**

  Sampling the graph to generate a list R containing the rr set, and then use the NodeSelection function to select the nodes in it to obtain the final result set.

```pseudocode
Function imm(graph,k,e,l,n)
    l <- l*(1+math.log(2)/math.log(n))
    R <- sampling(graph,k,e,l)
    seeds <- node_selection(R,k,n)[0]
    return seeds
```

- **Sampling**

  Sampling function, sample the graph to get the list of rr set R.

```pseudocode
Function sampling(graph,k,e,l)
	R <- []
    LB <- 1
    e1 <- math.sqrt(2)*e
    n <- len(graph)
    for i in range(1,int(math.log2(n))) then
        x <- n/(math.pow(2,i))
        lambda1 <- (n*(2+e1*2/3)*(lognk(n,k)+l*math.log(n)+math.log(math.log2(n))))/(math.pow(e1,2))
        c1 <- lambda1/x
        while len(R)<=c1 then
            if mode=='IC'then
                R.append(generate_ic(graph,random.randint(1, n)))
            else then
                R.append(generate_lt(graph, random.randint(1, n)))
        S,F = node_selection(R,k,n)
        if n*F>=(1+e1)*x then
            LB <- n*F/(1+e1)
            break
    alpha <- math.sqrt(l*math.log(n)+math.log(2))
    beta <- math.sqrt((1-1/math.e)*(lognk(n,k)+l*math.log(n)+math.log(2)))
    lambda2 <- 2*n*math.pow(((1-1/math.e)*alpha+beta),2)*math.pow(e,-2)
    c <- lambda2/LB
    while len(R)<c then
        if mode == 'IC' then
            R.append(generate_ic(graph, random.randint(1, n)))
        else then
            R.append(generate_lt(graph, random.randint(1, n)))
    return R
```

- **NodeSelection**

   Iteratively calculate the marginal benefits of the selected nodes according to R, and select the k nodes with the highest marginal benefits

```pseudocode
Function node_selection(R,k,n) 
 	node_index <- {}
    rr_cnt <- np.zeros(n+1, int)
    count <- 0
    S <- set()
    for i in range(0, len(R)) then
        rr <- R[i]
        for rr_node in rr 
            rr_cnt[rr_node] <- rr_cnt[rr_node] +  1
            if rr_node not in node_index.keys() then
                node_index[rr_node] <- []
            node_index[rr_node].append(i)

    for i in range(k) 
        max_index <- 0
        max_temp <- -1
        for w in range(0,len(rr_cnt))
            if rr_cnt[w]>max_temp then
                max_temp <- rr_cnt[w]
                max_index <- w
        S.add(max_index)
        count <- count + len(node_index[max_index])
        index_set <- [rr for rr in node_index[max_index]]
        for j in index_set
            rr <- R[j]
            for rr_node in rr
                rr_cnt[rr_node] - 1
                node_index[rr_node].remove(j)
    return S, count/len(R)
```

- **IC**

  One node is used as the initial node for BFS, and a random value of 0 to 1 is taken each time to try to activate the next node. Finally, the activated point set is returned

```pseudocode
Function generate_ic(graph,node)	
 	total <- [node]
    active <- np.zeros(len(graph), int)
    activity_set <- [node]
    for i in range(0, len(activity_set))
        active[activity_set[i] - 1] <- 1
    while len(activity_set) != 0
        new_activity_set <- []
        for i in activity_set
            for j in graph[i - 1]
                if active[j[0] - 1] == 1 then
                    continue
                r <- np.random.random()
                if r < j[1] then
                    active[j[0] - 1] <- 1
                    new_activity_set.append(j[0])
        for p in new_activity_set
            total.append(p)
        activity_set <- new_activity_set
    return total
```

- **LT**

  Take a node as the initial node, select one neighbor randomly at a time. If it is not activated, it will activate it and repeat the process. If it has been activated, it will exit and return all the activated nodes

```pseudocode
Function generate_lt(graph,node):
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
```

------

#### 3. Empirical Verification

##### 3.1 Dataset

In the local test of the experiment, I used network. TXT, Nethept.txt and Network seeds. TXT as the test data. 

During the process of submission, the network diagram of the submitted platform is used as test data to test the program

##### 3.2 Performance Measure

Because the program is executed with 8 cores, 2000 processes are run on each core, and in case of insufficient time, it is set to exit the program before 5 seconds of the time limit. Therefore, the accuracy of program execution results can be guaranteed, but for smaller models, it may still take a long time.

- **ISE**

  | Dataset                                  | Runtime | Result   |
  | ---------------------------------------- | ------- | -------- |
  | random-graph50-50-seeds-5-IC-new         | 4.06    | 30.658   |
  | random-graph50-50-seeds-5-LT-new         | 4.02    | 41.274   |
  | random-graph500-500-seeds-10-IC-new      | 15.21   | 167.821  |
  | random-graph500-500-seeds-10-LT-new      | 14.11   | 209.085  |
  | random-graph5000-5000-seeds-10-IC-new    | 57.85   | 944.755  |
  | random-graph5000-5000-seeds-10-LT-new    | 57.78   | 1034.756 |
  | random-graph15000-15000-seeds-100-IC-new | 57.55   | 1595.457 |
  | random-graph50000-50000-seeds-100-IC-new | 117.65  | 4106.152 |
  | random-graph50000-50000-seeds-100-LT-new | 118.40  | 4334.920 |

- **IMP**

  | Dataset        | Runtime | Result   |
  | -------------- | ------- | -------- |
  | network-5-IC   | 34.93   | 30.675   |
  | network-5-LT   | 27.97   | 37.5412  |
  | NetHEPT-5-IC   | 57.64   | 323.488  |
  | NetHEPT-5-LT   | 57.75   | 392.975  |
  | NetHEPT-50-IC  | 117.64  | 1296.324 |
  | NetHEPT-50-LT  | 117.69  | 1701.995 |
  | NetHEPT-500-IC | 117.55  | 4320.300 |
  | NetHEPT-500-LT | 117.76  | 5578.619 |

  

##### 3.3 Hyper Parameters

 **ε, l:**  used to control the length of R in each iteration, in order to ensure that each rr set is sufficiently effective and does not take too much time, so 

ε = 0.5, l = 1 to obtain better results.

##### 3.4 Experimental results

In ISE passed all test

In IMP all test case are seems good

##### 3.5 Conclusion

In this experiment, I experienced and learned the idea and method of solving NP hard problem by stochastic model, and mastered IC and lt model, and solved imp model by imm method. In addition, I became more familiar with Python Programming and multiprocessing to improve efficiency.

I believe that with this experimental experience, I will be more handy in the related problems later

------



#### 4. References

[1] Kempe, D., Kleinberg, J., and Tardos, E.. Maximizing the spread of ´ influence through a social network. In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 137-146,2003.

[2] Tang, Y., Shi, Y., and Xiao, X.. Influence maximization in nearlinear time: A martingale approach. In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data, pages 1539-1554,2015. 

[3] S. Bharathi, D. Kempe, and M. Salek. Competitive influence maximization in social networks. In WINE, pages 306–311,2007. 
