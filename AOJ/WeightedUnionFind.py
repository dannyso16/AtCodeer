import sys


class WeightedUnionFind:
    def __init__(self, N: int):
        """N: 大きさ
        size: 連結成分の大きさ
        weight: rootからの重み
        """
        self.par = [i for i in range(N)]
        #self.size = [1]*N
        self.rank = [0]*N
        self.weight = [0]*N

    def root(self, x: int) -> int:
        """根を求める"""
        if self.par[x] == x:  # if root
            return x
        else:
            root_x = self.root(self.par[x])  # 経路圧縮
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = root_x
            return root_x

    def get_weight(self, x: int) -> int:
        """x の重みを取得"""
        self.root(x)  # 経路圧縮
        return self.weight[x]

    def get_diff(self, x: int, y: int) -> int:
        """x と y の差分を取得"""
        if not self.is_same(x, y):
            return "?"  # 判定不可
        return self.get_weight(y) - self.get_weight(x)

    def is_same(self, x: int, y: int) -> bool:
        """x と y が同じ集合に属するか否か"""
        return self.root(x) == self.root(y)

    def unite(self, x: int, y: int, w: int) -> bool:
        """weight[y] = weight[x]+w となるようにmerge(x が y の親)
        return True if merge 可能"""
        if self.is_same(x, y):
            return False
        root_x = self.root(x)
        root_y = self.root(y)
        w += (self.weight[x] - self.weight[y])  # 重みを補正
        if self.rank[root_x] < self.rank[root_y]:
            self.par[root_x] = root_y
            self.weight[root_x] = -w
        else:
            self.par[root_y] = root_x
            self.weight[root_y] = w
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        return True

    def get_size(self, x: int) -> int:
        """xの属するグループのサイズ"""
        return self.size[self.root(x)]


if __name__ == "__main__":
    N, Q = map(int, sys.stdin.readline().split())
    UNITE, DIFF = 0, 1
    wuf = WeightedUnionFind(N)
    for _ in range(Q):
        com, *x = map(int, sys.stdin.readline().split())
        if com == UNITE:
            wuf.unite(*x)
        else:  # same
            print(wuf.get_diff(*x))
