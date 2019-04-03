import sys

class UnionFind:
    def __init__(self, N:int):
        """N: 大きさ
        size: 連結成分の大きさ
        """
        self.par = [i for i in range(N)]
        self.size = [1]*N
        self.rank = [1]*N

    def root(self, x:int)->int:
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
        """y を x の属する集合に併合"""
        root_x = self.root(x)
        root_y = self.root(y)
        if root_x == root_y: return
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        elif self.rank[root_x]<self.rank[root_y]:
            root_x,root_y = root_y,root_x
        # 短いほう(y)を長いほう(x)にくっつける
        self.par[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.size[root_y] = 0 # もういらない

    def get_size(self, x:int)->int:
        """xの属するグループのサイズ"""
        return self.size[self.root(x)]

if __name__ == "__main__":
    N,Q = map(int, sys.stdin.readline().split())
    UNITE,SAME = 0,1
    uf = UnionFind(N)
    for _ in range(Q):
        com,x,y = map(int, sys.stdin.readline().split())
        if com==UNITE:
            uf.unite(x, y)
        else: # same
            print(int(uf.is_same(x, y)))
