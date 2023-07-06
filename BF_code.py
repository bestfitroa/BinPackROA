import networkx as nx
from functools import cmp_to_key
import copy
import numpy as np
 
## Code computes the markov chain for Best-Fit in the IID model, and outputs the asymptotic performance.

"----------------------------------------- Defining functions-----------------------------------"
one = 1 + 1e-14 ## custom "1" to prevent rounding errors.



def compare(item1, item2):                      # custom comparator for bins, so as to sort bin configurations in decreasing order of load.
    return -sum(item1) + sum(item2)


def strconv(convconfig):                        # converting a bin configuration description into a string of numbers, these strings will be states in the markov chain.
    str1 = ""                                   # string format: number of bins, followed by a description of each bin. if the item list is [1/4,1/3,1/2],
                                                #  then 210 means a bin with 2 1/4s, one 1/3, 0 1/2s.
    str1+=str(convconfig[0]) 
    for i in range(convconfig[0]):
        for j in range(len(convconfig[1][i])):
            str1+=str(convconfig[1][i][j])
    return str1



def convert(config,L):                          # describing the bin configuration (an array of arrays, i.e., a collection of open bins) in terms of item counts
    n = len(config)
    config1 = copy.deepcopy(config)
    config1.sort(key=cmp_to_key(compare))
    arr = []
    for i in range(n):
        arr1 = [0]*len(L)
        for j in range(len(L)):
            arr1[j] = config1[i].count(L[j])
        arr.append(arr1)
    return (n,arr)


def BF(config,item,L):                    # simulating best fit for a particular collection of open bins so as to find reachable states, we use this to compute the markov chain
    n = len(config)
    config1 = copy.deepcopy(config)
    maxival = -100
    maxiind = 0
    for i in range(n):
        sum1 = sum(config1[i])
        if(sum1+ item <one  ):
            if(maxival < sum1 ):
                maxival = sum1
                maxiind = i         
    if(maxival == -100):
        config1.append([item])
        config1.sort(key=cmp_to_key(compare))
        return config1
    else:
        config1[maxiind].append(item)
        
        if(sum(config1[maxiind])+ min(L) > one):
            config1[maxiind] = [-1]
            
            config1.sort(key=cmp_to_key(compare))
            config1.pop()
            
        config1.sort(key=cmp_to_key(compare))    
        return config1  
    


def AsymBF(L,p):                              # finds asymptotic performance of BF for the list L with given probabilities p. i.e., returns alpha where alpha*n 
   n = len(L)                                 # is the expected number of bins used by BF for (L,p)
   G = ConfigGraphGen(L,p) 
   newbinedges = NewBinEdges(G)
   M = ConfigTransitionMatrix(G)
   m = len(M)
   val = np.zeros((m))
   for i in range(m):
        val[i]  = 0
   val[m-1] = 1
   x= np.linalg.lstsq(M, val,rcond=None)     # solves transition eqs. for Markov chain
      
   BF = 0
    
   for edge in newbinedges:                   #adding up the transitions that create new bins
        ind1 = edge[0]
        ind2 = edge[1]
        BF+= x[0][ind2]*M[ind1][ind2]
   
   return BF



         
def ConfigGraphGen(L,p):                  # generates the markov chain for the list L with probabilities p
    n = len(L)
    flag = 0
    queue = [[]]                          # states that need to be explored
    encountered = []                      # already seen states
    encounteredconv = []                  # already seen states in converted format
    G = nx.DiGraph()
    
    while(flag == 0):
            
        for i in range(n):
            if(i==0):
                encountered.append(queue[0])                        # initialization with empty bin
                encounteredconv.append(convert(queue[0],L))
            newconfig = BF(queue[0],L[i],L)                       # simulating BF on the state to find neighbouring state
            newconfigconv = convert(newconfig,L)
            queueconfigconv = convert(queue[0],L)
            strnewconfigconv = strconv(newconfigconv)
            strqueueconfigconv = strconv(queueconfigconv)
            bincheck = 0
            
            if(queueconfigconv[0] < newconfigconv[0]):  
                bincheck = 1                                # marks the transitions that create new bins
            
            if(G.has_edge(strqueueconfigconv,strnewconfigconv)):
                G[strqueueconfigconv][strnewconfigconv]["weight"]+=p[i]
            else:
                G.add_edge(strqueueconfigconv, strnewconfigconv,  weight= p[i], newbin = bincheck)
            
            if(newconfigconv not in encounteredconv):
                encountered.append(newconfig)
                encounteredconv.append(newconfigconv) 
                queue.append(newconfig)
                
        del queue[0]
        if(len(queue) == 0):
            flag = 1
    return G



def ConfigTransitionMatrix(G):      # generates the markov transition matrix from the markov chain graph
    n = len(G.nodes())
    nodes  = list(G.nodes())
    edges = list(G.edges())
    
    
    M = [ [0] * n for _ in range(n+1)]
    for i in range(n):
        M[n][i] = 1
        M[i][i] = -1
        
    for edge in edges:
        edgedict = G.get_edge_data(edge[0], edge[1])
        ind1 = nodes.index(edge[0])
        ind2 = nodes.index(edge[1])
        M[ind2][ind1] = edgedict['weight'] 
    return M



def NewBinEdges(G):                # lists the transitions that create new bins
    edges = list(G.edges())
    nodes = list(G.nodes()) 
    arr = []
    for edge in edges:
        edgedict = G.get_edge_data(edge[0], edge[1])
        ind1 = nodes.index(edge[0])
        ind2 = nodes.index(edge[1])
        if(edgedict['newbin'] == 1):
            arr.append([ind2,ind1])
    return arr  

"-----------------------------------------End of Defining functions-----------------------------------"      

    
print(AsymBF([1/3,1/4], [0.5, 0.5]))     # sample use of code.
    
    
    
    
    
    
    
    
    

