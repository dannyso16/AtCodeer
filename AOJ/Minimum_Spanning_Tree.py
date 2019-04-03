class UnionFind:
    def __init__(self, v_count: int):
        self.par = [-1]*v_count # 親の番号
        for i in range(v_count):
            self.par[i] = i  # はじめすべての頂点が根

    def root(self, x: int) -> int:
        """根を求める"""
        if self.par[x] == x: # if root
            return x
        else:
            self.par[x] = self.root(self.par[x]) # 経路圧縮
            return self.par[x]

    def is_same(self, x:int, y:int)->bool:
        """x と y が同じ集合に属するか否か"""
        return self.root(x)==self.root(y)

    def unite(self, x:int, y:int):
        """x と y の属する集合を併合"""
        x = self.root(x)
        y = self.root(y)
        if x == y: return
        self.par[x] = y

def kruskal(v_count: int) -> int:
    """O(|E| log|V|)
    return the minimum cost of Spanning Tree"""
    global edges # (w, s, t)
    uf = UnionFind(v_count)

    cost = 0
    for w,s,t in sorted(edges): # costが小さい順に
        if not uf.is_same(s, t):
            uf.unite(s, t)
            cost += w
    return cost


if __name__ == "__main__":
    V,E = map(int, input().split())
    edges = []
    for _ in range(E):
        s,t,w = map(int, input().split())
        edges.append((w,s,t))
    ans = kruskal(V)
    print(ans)
