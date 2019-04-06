import sys
readline = sys.stdin.readline

# Lowlink で橋と関節点を求める
sys.setrecursionlimit(10**7)
class LowLinks:
    def __init__(self, edges, edges_num:int):
        """edges[u]: all vertexes connected with vertex 'u'
        edges_num: number of edges of graph
        the root of DFS-tree is vertex 0
        O(V + E)
        """
        self.edges = edges
        self.V = len(edges)
        self.order = [-1]*V
        self.low = [float('inf')]*V
        self.bridges = []
        # if degreee(root) > 1 and graph is tree: root is articulation
        self.articulations = []
        if len(edges[0]) > 1 and edges_num == self.V-1:
            self.articulations.append(0)
        self.k = 0

    def build(self):
        self.dfs(0, 0)

    def get_bridges(self)->list:
        return self.bridges

    def get_articulations(self)->list:
        return self.articulations

    def dfs(self, v:int, prev:int):
        self.order[v] = self.k
        self.low[v] = self.k
        self.k += 1
        is_articulation = False
        for to in self.edges[v]:
            if self.order[to] < 0: # not visited
                self.dfs(to, v)
                self.low[v] = min(self.low[v], self.low[to])
                if self.order[v] < self.low[to]:
                    self.bridges.append((v, to) if v < to else (to, v))
                is_articulation |= self.order[v] <= self.low[to]
            elif to != prev: # back edge
                self.low[v] = min(self.low[v], self.order[to])
        if v>0 and is_articulation:
            self.articulations.append(v)

if __name__ == "__main__":
    V,E = map(int, readline().split())
    edges = [[] for _ in range(V)]
    for _ in range(E):
        s,t = map(int, readline().split())
        edges[s].append(t)
        edges[t].append(s)
    lowlinks = LowLinks(edges, E)
    lowlinks.build()
    bridges = lowlinks.get_bridges()
    bridges.sort()
    if bridges:
        for s,t in bridges:
            print(s, t)
