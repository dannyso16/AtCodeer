# 目次

- [探索](#探索)
- [グラフ](#グラフ)
- [データ構造](#データ構造)
- [動的計画法](#動的計画法)
- [文字列](#文字列)
- [数学](#数学)
- [貪欲](#貪欲)
- [テクニック](#テクニック)
- [構築](#構築)
- [ゲーム](#ゲーム)
- [フロー](#フロー)

米AtCoder tags 参考

[HOME](https://dannyso16.github.io/AtCoderMemo/)



## 便利なサイト

[GPATH x xGPATH](https://hello-world-494ec.firebaseapp.com/): グラフを描画してくれる！

## 気になる

- 最長共通部分列

  

# 探索

## 深さ優先探索　再帰

パスを全列挙する→dfs呼び出し後探索済みをfalseにする

- ABC054 C - One-stroke Path: 以下のコード

```python
def main():
    N, M = map(int, input().split())
    edges = [[] for _ in range(N)]
    for _ in range(M):
        a, b = map(lambda x: int(x)-1, input().split())
        edges[a].append(b)
        edges[b].append(a)

    visited = [False]*N
    visited[0] = True
    ans = 0

    def dfs(v: int):
        nonlocal visited
        nonlocal ans
        if all(visited):
            ans += 1
            return
        for n in edges[v]:
            if visited[n]:
                continue
            visited[n] = True
            dfs(n)
            visited[n] = False
        return

    dfs(0)
```



## DFS　stack

全頂点を探索したりする。再帰よりはやい

- ABC138 D - Ki(400)

```python
from collections import deque

N, Q = map(int, input().split())
edges = [[] for _ in range(N)]
for _ in range(N-1):
    a, b = map(lambda x: int(x)-1, input().split())
    edges[a].append(b)
    edges[b].append(a)

visited = [False]*N
visited[0] = True
q = deque([0])
while q:
    v = q.pop()
    for n in edges[v]:
        if visited[n]:
            continue
        visited[n] = True
        
        # ナンカスル
        
        q.append(n)

```



## 幅優先探索　BFS

- ABC146: 方針はたつが実装が難しい典型問題
- ABC067 D - Fennec VS. Snuke(400)

```python
from collections import deque

def bfs(edges, s)->list:
    """edges: 隣接リスト
    """
    V = len(edges)
    dist = [float('inf')]*V
    visited = [False]*V
    dist[s] = 0
    visited[s] = True
    q = deque([s])
    while q:
        cur = q.popleft()
        for nex in edges[cur]:
            if visited[nex]:
                continue
            q.append(nex)
            visited[nex] = True
            dist[nex] = dist[cur] + 1
    return dist
 
dist1 = bfs(edges, 0)

```



## 二部探索：bisect

- ソート状態を保ったまま要素を挿入できたりする
- 条件を満たす最小値を探索する

ソート済みのリストに挿入

```python
import bisect

l = [1, 3, 4]
a = 2
idx = bisect.bisect(l, a) # 挿入するindexを取得
bisect.insort(l, a)       # 挿入
# l: [1, 2, 3, 4]
```

同じ要素があるときに右か左かも指定できる

```python
import bisect

idx_r = bisect.bisect_right(l, a) # bisect.bisectと同じ
idx_l = bisect.bisect_left(l, a)

bisect.insort_right(l, a) # insort()と同じ
bisect.insort_left(l, a)
```



### じぶんで実装

平均の最大化→答え（目標）を決めてしまう。つまり「平均の最大値を求める」と考えず、「つくれる最大の閾値を求める」と考える

- ABC034

```python
def check(m: int) -> bool:
    """m 以上で条件を満たすかどうか
    mで条件を満たす → return True
    """
    # なんか
    return bool


high = 100 # 必ず条件を満たす
low = 0    # 必ず条件を満たさない
while high - low > 1e-6:
    mid = (high + low) / 2
    if check(mid):
        low = mid
    else:
        high = mid
print(low)

```

## 

### 近い発想の問題

- ABC093 C - Same Integers

操作によって合計は２しか増えない→偶奇は不変。操作を積み上げる思考ではなく、最後等しくなった値がいくつになるか考える



## しゃくとり法

- ABC032
- ABC038
- ABC154 D - Dice in Line(400)

```python
s = list(map(int, input().split()))

#  [left, right)で考えることに注意!
right = 0
for left in range(N):
    while (right < N) and (right をひとつ進めても条件を満たす ex. sum_+s[right] <= K):
        # right ++ の処理
        sum_ += s[right]
        right += 1

    # この時点でright は条件を満たす最大値 [left, right)
    # ans の更新とかする

    # left++ する準備
    if right == left:
        right += 1
    else:
        sum_ -= s[left]

```



## 三部探索

- 二部探索：bool値の変わる境界を探す
- 三部探索：凸関数の極値を探す

三部探索で整数解を求めたいときは，

```python
def f(x:float)->int:
    x = int(x+0.5) # round
```

のようにfloatで受けて，intにするといい．
答えは誤差を考慮して，前後の値も候補に入れておくと安心

```python
def tri_search(f: "f(x:float)->float", left: float, right: float,
               is_convex_downward=True, iter=100) -> float:
    """is_convex_downward: 下に凸 return minimum
    else: 上に凸 return Maximum
    f: convex upward -> -f: convex downward
    """
    for _ in range(iter):
        ml = (left*2 + right) / 3
        mr = (left + right*2) / 3
        if is_convex_downward:
            f_ml, f_mr = f(ml), f(mr)
        else:
            f_ml, f_mr = -f(ml), -f(mr)

        if f_ml < f_mr:
            right = mr
        else:
            left = ml
    print(left, right)
    return (right + left) / 2
```



## Counterで数を数えて最頻値を昇順に出力

- ABC155

```python
from collections import Counter
 
N = int(input())
s = ['a', 'z', 'z', 'b', 'b']
 
c = Counter(s)
c = c.most_common() # これでリストになる
				  # [('a', 1), ('z', 2), ('b', 2)]
c.sort(key=lambda x: x[0])  # [('a', 1), ('b', 2), ('z', 2)]
c.sort(key=lambda x: x[1], reverse=True) # 出現回数順
 
most_cnt = c[0][1]
for key, cnt in c:
    if cnt == most_cnt:
        print(key)
    else:
```





## いもす法

- 区間 $[l, r]$ に値 $x_i$ を足す

上の操作を$O(1)$で保存、計算に$O(N)$でできる

- ABC127 C - Prison(300): そのまま

```python
from itertools import accumulate

N  # length of list
Q  # Number of Query

imos = [0]*N
for _ in range(Q):
    l, r = map(lambda x: int(x)-1, input().split())
    imos[l] += 1
    if r < N-1:
        imos[r+1] -= 1
imos = accumulate(imos)

```



### 累積和

- ABC138 D - Ki(400) : 応用

```python
# 自分でかく
a = list(range(10))
accum = [a[0]]
for ai in a[1:]:
    accum.append(accum[-1] + ai)

    
# itertools.accumlate
from itertools import accumulate

a = range(10)
accum = list(accumulate(a))
```



## 最長増加部分列の長さ LIS

> 数列 $A= a_0,a_1,…,a_{n−1}$ の最長増加部分列 の長さを求めてください。
> 数列 $A$の増加部分列は$ 0≤i_0<i_1<…<i_k<n $かつ
> 　$a_{i_0}<a_{i_1}<…<a_{i_k} $を満たす部分列 $a_{i_0}, a_{i_1}, …, a_{i_k} $です。最長増加部分列はその中で最も $*k*$　が大きいものです。

- [AOJ](http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=DPL_1_D&lang=ja)
- ABC006 D - トランプ挿入ソート: そのまま
- ABC134 E - Sequence Decomposing(500): わりとそのまま

二次元いれこ構造を一次元化

- ABC038 D - プレゼント: (w,h)のいれこ構造をwを固定して求める。
  - wを昇順ソートし、wが同じものはhで降順にしておく→LISに帰着
- Chokudai speedrun L - 長方形 β(600): 上とほぼ同じ。w,hで大きい方をwにするだけ

三次元（やってない）

- [AOJ Longest Chain](http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1341&lang=en)

長さのみは$O(NlogN)$ で求まる

```python
def LIS(seq: list) -> int:
    """狭義単調増加
    param: seq
    return:LISの長さ（a_i < a_j）
    """
    from bisect import bisect_left

    L = [seq[0]]
    for i in range(len(seq)):
        if seq[i] > L[-1]:
            L.append(seq[i])
        else:
            idx = bisect_left(L, seq[i])
            L[idx] = seq[i]
    return len(L)
```

以下広義

```python
def LIS(seq: list) -> int:
    """広義単調増加
    param: seq
    return:LISの長さ（a_i =< a_j）
    """
    from bisect import bisect_right
    N = len(seq)
    L = [seq[0]]
    for i in range(1, N):
        if seq[i] > L[-1]:
            L.append(seq[i])
        else:
            idx = bisect_right(L, seq[i])
            if idx == len(L):
                L.append(None)  # avoid Out-of-Index-Error
            L[idx] = seq[i]
    return len(L)
```



# グラフ

## 隣接行列

```python
# infで隣接行列を初期化
matrix = [[float('inf')]*V for _ in range(V)]

# 隣接行列を作成
for i in range(10):
    c = map(int, input().split())
    for j, cj in enumerate(c):
        matrix[i][j] = cj
```



## 隣接リスト

```python
edges = [[] for _ in range(N)]
for _ in range(N-1):
    s,t = map(lambda x:int(x)-1, readline().split())
    edges[s].append(t)
    edges[t].append(s)
```



## ワーシャルフロイド法

- ABC079- D - Wall  やや応用

全頂点間の最短距離を $O(V^3)$ で求める

```python
import sys
from math import isinf
from typing import List  # New in python 3.5


def warshall_floyd(matrix: List[List]) -> List[List]:
    """ ワーシャルフロイド
    :param matrix:  隣接行列(到達不能はfloat("inf"))
    :return matrix
    """
    # 到達不能をfloat("inf")にしておけば余計なチェックを入れなくても
    # inf > inf+(-1) のような到達不能＋負辺が繋がってしまうことはない
    V = len(matrix)  # 頂点数
    for i in range(V):
        for j, c2 in enumerate(row[i] for row in matrix):
            for k, (c1, c3) in enumerate(zip(matrix[j], matrix[i])):
                if c1 > c2+c3:
                    matrix[j][k] = c2+c3  # min(c1, c2+c3)
    return matrix


# infで隣接行列を初期化
matrix = [[float('inf')]*V for _ in range(V)]
# 隣接行列を作成
for i in range(10):
    c = map(int, input().split())
    for j, cj in enumerate(c):
        matrix[i][j] = cj

# 全点間の最短距離を求める
matrix = warshall_floyd(matrix)

# 例
matrix[0][1] # 点0から点1の最短距離

```

## ある始点から各頂点への最短経路をだす（ダイクストラ）

$頂点数 < 10^3, 辺数 < 10^3$ なら、全点間最短距離を出せる（$O(N^2)$程度）

- ABC160: 制約が$頂点数 < 10^3$ならまず2重ループから考えてもいい
  - 近道があるなら近道を使う時と使わないときを考えて、minをとったり

### 

```python
from collections import deque


def dijkstra(start: int, edges: list):
    """単一始点最短距離  O(|E|log|V|)
    start: int  始点
    edges: list 隣接リスト
    """
    dist = [float('inf') for _ in range(N)]  # list dist[|V|]
    dist[start] = 0
    prev_v = [-1]*(len(dist))  # 最短経路でのひとつ前の頂点
    q = deque([(0, start)])    # （暫定的な距離，頂点）
    while q:
        d_cur, cur = q.pop()
        if dist[cur] < d_cur:  # すでに探索済み
            continue
        for nex, cost in edges[cur]:
            if dist[nex] > dist[cur]+cost:
                dist[nex] = dist[cur]+cost
                q.append((dist[nex], nex))
                prev_v[nex] = cur
    return dist


if __name__ == "__main__":
    # 入力（0-indexed）
    edges = [[] for _ in range(N)]

    # 隣接リストを定義

    # 全点間で計算
    for i in range(N):
        d = dijkstra(i, edges)
```



## 最短経路を復元（ダイクストラ）

```python
from collections import deque


def get_shortest_path(start: int, goal: int, edges: list):
    """単一始点最短距離  O(|E|log|V|)
    start: int  始点
    goal: int   終点
    edges: list 隣接リスト
    """
    dist = [float('inf') for _ in range(N)]  # list dist[|V|]
    dist[start] = 0
    prev_v = [-1]*(len(dist))  # 最短経路でのひとつ前の頂点
    q = deque([(0, start)])    # （暫定的な距離，頂点）
    while q:
        d_cur, cur = q.pop()
        if dist[cur] < d_cur:  # すでに探索済み
            continue
        for nex, cost in edges[cur]:
            if dist[nex] > dist[cur]+cost:
                dist[nex] = dist[cur]+cost
                q.append((dist[nex], nex))
                prev_v[nex] = cur
    
    # 以下 path の復元
    path = []
    v = goal
    while v != -1:
        path.append(v)
        v = prev_v[v]
    path = path[::-1] # reverse
    return path       # [start, ., ., goal]


if __name__ == "__main__":
    # 入力（0-indexed）
    edges = [[] for _ in range(N)]

    # 隣接リストを定義

    # 最短経路を出力
    path = get_shortest_path(i, edges)
    print(path)
```



# データ構造

## collections.deque

- ABC158

先頭や末尾にデータを挿入したい場合はこれ。listだと先頭への挿入が遅いので。なお、`deque`には、両端以外の要素へのアクセスが遅いというデメリットもあるので注意。QueueもStackもこれでいい。

[python時間計算量ドキュメント](https://wiki.python.org/moin/TimeComplexity)

| **Operation** | **Average Case** | **Amortized Worst Case** |
| ------------- | ---------------- | ------------------------ |
| Copy          | O(n)             | O(n)                     |
| append        | O(1)             | O(1)                     |
| appendleft    | O(1)             | O(1)                     |
| pop           | O(1)             | O(1)                     |
| popleft       | O(1)             | O(1)                     |
| extend        | O(k)             | O(k)                     |
| extendleft    | O(k)             | O(k)                     |
| rotate        | O(k)             | O(k)                     |
| remove        | O(n)             | O(n)                     |

ただし添え字参照では中央付近のアクセスに`O(N)`程度かかることも示唆されている

[python-deque](https://docs.python.org/ja/3/library/collections.html#deque-objects)

初期化の際にstringを渡すとバラバラにしてくれる

```python
d = deque("string")
# deque(['s', 't', 'r', 'i', 'n', 'g'])
```





## 優先度付きキュー（Priority queue）:heapq

- 最小値（最大値）を `O(logN)`で取り出す
- 要素を `O(logN)`で挿入する

通常は`O(N)`かかるから早い。注意点としてはデータは常にソートされた状態で保持されているわけではない。

```python
import heapq 

a = [1, 6, 8, 0, -1]
heapq.heapify(a)  # リストを優先度付きキューに変換

_min = heapq.heappop(a)  # 最小値の取り出し

heapq.heappush(a, -2)  # 要素の挿入
```

-1をかけておけば最大値も取り出せる

```python
import heapq

a = [1, 6, 8, 0, -1]
a = list(map(lambda x: x*(-1), a))  # 各要素を-1倍

heapq.heapify(a)
_max = heapq.heappop(a)*(-1)  # 最大値の取り出し

heapq.heappush(a, -1*(-2)) # 要素の挿入
```

## 

## 重みなしDisjoint set (Union Find)

- ARC032

```python
import sys


class UnionFind:
    def __init__(self, N: int):
        """N: 大きさ
        size: 連結成分の大きさ
        """
        self.par = [i for i in range(N)]
        self.size = [1]*N
        self.rank = [1]*N

    def root(self, x: int) -> int:
        """根を求める"""
        if self.par[x] == x:  # if root
            return x
        else:
            self.par[x] = self.root(self.par[x])  # 経路圧縮
            return self.par[x]

    def is_same(self, x: int, y: int) -> bool:
        """x と y が同じ集合に属するか否か"""
        return self.root(x) == self.root(y)

    def unite(self, x: int, y: int):
        """y を x の属する集合に併合"""
        root_x = self.root(x)
        root_y = self.root(y)
        if root_x == root_y:
            return
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        elif self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        # 短いほう(y)を長いほう(x)にくっつける
        self.par[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.size[root_y] = 0  # もういらない

    def get_size(self, x: int) -> int:
        """xの属するグループのサイズ"""
        return self.size[self.root(x)]
```

連結成分の数はrootが自分のものの数 `sum([i == uf.root(i) for i in range(N)])`

## 

## 重み付きUnion Find

- ABC087

普通の UnionFind 木のサポートする処理は

| クエリ       | 処理内容                                        |
| :----------- | :---------------------------------------------- |
| merge(x, y)  | x を含むグループと y を含むグループをマージする |
| issame(x, y) | x と y が同じグループにいるかどうかを判定する   |

ですが、重みつきUnionFind木は少し発展させて、各ノード v に重み weight(v) を持たせ、**ノード間の距離**も管理するようなものになっています。

| クエリ         | 処理内容                                                     |
| :------------- | :----------------------------------------------------------- |
| merge(x, y, w) | weight(y) = weight(x) + w となるように x と y をマージする   |
| issame(x, y)   | x と y が同じグループにいるかどうかを判定する                |
| diff(x, y)     | x と y とが同じグループにいるとき、weight(y) - weight(x) をリターンする |

```python 
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


```



## Binary Indexed Tree (BIT)

[参考](https://ikatakos.com/pot/programming_algorithm/data_structure/binary_indexed_tree)

### Range Sum Query（区間の和）

数列 $a$ に対して以下の操作が $O(log N)$  でできる

- $a_i$ に $x$ を加算
- $a_s$ から $a_t$ までの合計を得る

例

- [AOJ Range Sum Query](https://onlinejudge.u-aizu.ac.jp/problems/DSL_2_B)
- 

```python
class BIT():
    """Range Sum Query（区間の和）
    """

    def __init__(self, n: int):
        self.n = n
        self.bit = [0]*(n+1)  # 1-indexed

    def add(self, i: int, x: int):
        """i番目(1-idexed)にxを加算"""
        while i <= self.n:
            self.bit[i] += x
            i += i & -i

    def sum_1_to_i(self, i: int) -> int:
        """1番目からi番目(含む)までの総和"""
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def sum_i_to_j(self, i: int, j: int):
        """i番目からj番目(含む)までの総和"""
        return self.sum_1_to_i(j) - self.sum_1_to_i(i-1)
```



### 区間の最大値・最小値

Segment Treeほどの柔軟性は無いが、いくらかの制約された条件下で、区間最大値・最小値の管理にも使える（以下は最大値の例）

- update(i,x): ai を x で更新する。
- getmax(i): a1～ai の最大値を取得する。必ず1からの最大値であり、途中からは取得できない。

例

- ABC038 D - プレゼント: LISやけどな！

```python
class BIT():
    """区間max, min
    f に max or min を指定
    """

    def __init__(self, n: int, f: 'max or min'):
        ok = (f == max or f == min)
        assert ok, "f は max または min"

        self.n = n
        fill = 0 if f == max else 10**18
        self.bit = [fill]*(n+1)  # 1-indexed
        self.f = f

    def range_f_1_to_i(self, i) -> int:
        """区間[1, i]での最大値または最小値
        """
        ret = 0 if self.f == max else 10**18
        while i:
            ret = self.f(ret, self.bit[i])
            i ^= i & -i
        return ret

    def update(self, i, x):
        """ i 番目を x で更新
        """
        while i <= self.n:
            self.bit[i] = self.f(self.bit[i], x)
            i += i & -i
        return

```



# 動的計画法

## 桁DP

- ABC155

## bitDP

bitDP は「ある集合の部分集合を添字とした DP」。順列を全探索したりできる $O(M2^N)$

- [ABC 142 E - Get Everything (500 点)](https://drken1215.hatenablog.com/entry/2019/09/29/103500)

# 文字列

## ランレングス圧縮

- ABC019B - 高橋くんと文字列圧縮 そのまま

- ABC136 D - Gathering Children 

周期が2なのではじめ偶数番目にいた子供は最後も偶数番目にいるのがポイント

連続した文字を（文字＋連続する数）で圧縮する手法

例：AAAABBCCCCCCC →　A4B2C7

```python
def rle(s: str) -> str:
    """ランレングス圧縮 RLE(Run Length Encoding)
    例：AAABCCCC　→　A3B1C4
    """
    cur = s[0]
    count = 1
    compressed = ""
    for i in range(1, len(s)):
        if cur == s[i]:
            count += 1
        else:
            compressed += cur+str(count)
            cur = s[i]
            count = 1
    compressed += cur+str(count)
    return compressed
```

## f文字列(python3.6～)

`.format`なしで簡潔にかけるよ。

```python
x = 1
print(f"x={x}")    # x=1
print(f"x={x:04}") # x=0001 ゼロ埋め

y = 0.1234
print(f"y={y:.2f}") # y=0.12 桁指定
```

## 回文判定

- ABC159

```python
S: str = input()
if S == S[::-1]:
    print("回文")
else:
    print("回文でない")
```

## 大文字や小文字にそろえる

```python
s = "abcAA"
s.upper()
s.lower()
```

## stringモジュールの文字列定数

小文字や大文字などの一覧が取れる

```python
print(string.ascii_lowercase)
# abcdefghijklmnopqrstuvwxyz

print(string.ascii_uppercase)
# ABCDEFGHIJKLMNOPQRSTUVWXYZ

print(string.ascii_letters)
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ

print(string.digits)
# 0123456789

print(string.hexdigits)
# 0123456789abcdefABCDEF
```

## 文字とasciiの変換

```python
ord("a")  # 97
chr("97") # 'a'
```



# 数学

## 素因数分解

- ABC142

```python
def factorize(a:int)->dict:
    """素因数分解 O(sqrt(N))
    return 素因数のdict
    """
    ps = {}
    i = 2
    nokori = a
    while i*i <= nokori:
        if nokori%i==0:
            cnt = 0
            while nokori%i==0:
                cnt += 1
                nokori //= i
            ps[i] = cnt
        i += 1
    if nokori != 1:
        ps[nokori] = 1
    return ps
```

## 素数列挙　エラトステネスのふるい

```python
def sieve(n:int)->list:
    """エラトステネスの篩 O(N loglog N)
    return 区間 (0, n] で素数かどうかのbool
    """
    assert n >= 1, "don't input negative value"

    is_prime = [True]*(n+1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n+1):
        if is_prime[i]:
            # 素数の倍数は合成数
            for j in range(i*2, n+1, i):
                is_prime[j] = False
    return is_prime

```

## 組み合わせ

```python
# nCr = n*(n-1)*...*(n-r+1) / r*(r-1)*...*1
# 割り算があるのでmod計算が厄介
# →　a / b (mod p) != a (mod p) / b (mod p)

from math import factorial
MOD = 10**9 + 7


# 1. python らしいごり押し (あまり大きくないときに使おう)
def comb_naive(n: int, r: int) -> int:
    ret = factorial(n) // (factorial(n-r)*factorial(r))
    return ret % MOD


### 
# 2. modありの組み合わせの数を高速に計算
MOD = 10**9 + 7
# mod計算を含めて大きな数でもできるようにしたもの
def comb_fermat(n: int, r: int) -> int:
    """mod MOD を法とした組み合わせの数
    フェルマー小定理を利用: O(r)
    a^(-1) ≡ a^(p-2) (mod p), p : 素数
    割り算を逆元の掛け算に変形できる
    return nCr (mod MOD)	
    """
    if r > n:
        return 0
    if r > n-r:
        return comb_fermat(n, n-r)
    mul, div = 1, 1
    for i in range(r):
        mul *= n-i
        mul %= MOD
        div *= i+1
        div %= MOD

    ret = mul * pow(div, MOD-2, MOD) % MOD
    return ret


### 
# 3. 動的計画法でパスカル三角形を使う
com = [[0]*2000 for _ in range(2000)] # com[2000][2000]
com[0][0] = 1

def calc_comb():
    # k,n <= 2000
    # dp (Pascal's triangle): O(n*k)
    global com
    for i in range(1, len(com)):
        com[i][0] = 1
        for j in range(1, len(com)):
            com[i][j] = com[i-1][j-1] + com[i-1][j]
            com[i][j] %= MOD

            
# 4. 逆元を前計算(O(MAX_N))しておくことでクエリをO(1)で返す
"""使用例
comb_init()
calc_comb()
print(com[10][3]) # 10C3
"""
MAX = 10**5
fac = [0]*MAX  # fac[n]:  (n!) mod p
finv = [0]*MAX  # finv[n]: (n!)^-1 mod p
inv = [0]*MAX  # inv[n]:  (n)^-1 mod -p

def comb_init():
    global fac, finv, inv
    fac[0] = fac[1] = 1
    finv[0] = finv[1] = 1
    inv[1] = 1
    for i in range(2, MAX):
        fac[i] = fac[i-1] * i % MOD
        inv[i] = MOD - inv[MOD % i] * (MOD//i) % MOD
        finv[i] = finv[i-1] * inv[i] % MOD

def comb(n: int, r: int) -> int:
    global fac, finv
    if n < r:
        return 0
    if n < 0 or r < 0:
        return 0
    return fac[n] * (finv[r] * finv[n-r] % MOD) % MOD

```



## くり返し二乗法

pythonだと`pow(a, b, M)`で高速に $a^b \ mod \ M$ が計算できるが、下に実装例も載せておく

- ABC156

```python
MOD = 10**9 + 7

def modpow(a: int, p: int, mod: int) -> int:
    # return a**p (mod MOD) O(log p)
    if p == 0:
        return 1
    if p % 2 == 0:
        half = modpow(a, p//2, mod)
        return half*half % mod
    else:
        return a * modpow(a, p-1, mod) % mod
```

以下もくりかえし二乗法だが文字化け・・・

```python
# a^p (mod MOD) 繧帝ｫ倬溘↓豎ゅａ繧区婿?�ｿｽ�ｿｽ?
# 譎ｮ騾壹↓險育ｮ励☆繧九→縲O(p)縲?�ｿｽ�ｿｽ?縺鯉ｼ薫(log p) 縺ｧ豎ゅ∪?�ｿｽ�ｿｽ?
# p is even:  a^p = a^(p/2) * a^(p/2)
# p is odd:   a^p = a * a^(p-1)
# p is 0:     a^p = 1
# python 縺ｮ蝣ｴ蜷茨ｼ鯉ｿｽ?縺ｿ霎ｼ縺ｿ縺ｮ pow(a, p, MOD) 縺ｨ縺吶ｌ縺ｰ?�ｿｽ�ｿｽ??�ｿｽ�ｿｽ???�ｿｽ�ｿｽ?

MOD = 10**9 + 7

def modpow_bitwise(a: int, p: int, mod: int) -> int:
    # return a**p (mod MOD) O(log p)
    # 陝ｻ譛ｬp115??�ｿｽ�ｿｽ?�ｿｽ�ｿｽ莠御ｹ励＠縺ｦ?�ｿｽ�ｿｽ?縺阪↑縺後ｉ 1 縺檎ｫ九▲縺ｦ?�ｿｽ�ｿｽ?繧九→縺薙ｍ?�ｿｽ�ｿｽ?縺代ｒ菴ｿ?�ｿｽ�ｿｽ?
    res = 1
    while p > 0:
        if p & 1 > 0:
            res = res * a % mod
        a = a**2 % mod
        p >>= 1
    return res

###### 陦鯉ｿｽ??�ｿｽ�ｿｽver #####


def mat_pow(base, p):
    # base: matrix
    # return base^p
    # 縺滂ｿｽ??�ｿｽ�ｿｽ??�ｿｽ�ｿｽ?�ｿｽ�ｿｽ縺ゅ▲縺ｦ?�ｿｽ�ｿｽ??�ｿｽ�ｿｽ?
    ret = None
    mag = base
    while p > 0:
        if p & 1:
            ret = mag if ret is None else mat_dot(mag, ret)
        mag = mat_dot(mag, mag)
        p >>= 1
    return ret


def mat_dot(m1, m2):
    # a,b: matrix
    # (A x B) @ (B x C) ??�ｿｽ�ｿｽ?(A x C)
    # verified ABC021C
    if len(m1[0]) != len(m2):
        raise ValueError('Check matrix shape.')
    A = len(m1)
    C = len(m2[0])
    m2_t = list(zip(*m2))  # m2 ?�ｿｽ�ｿｽ]?�ｿｽ�ｿｽu
    ret = [[None]*C for _ in range(A)]
    for row in range(A):
        for col in range(C):
            v = 0
            for a, b in zip(m1[row], m2_t[col]):
                v += a*b
            ret[row][col] = v
    return ret


if __name__ == '__main__':
    MOD = 10**9 + 7
    print(modpow(100, 30000, MOD))
    print(modpow_bitwise(100, 30000, MOD))
    m1 = [[1, 2, 3], [2, 3, 4]]
    m2 = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
    print(mat_dot(m1, m2))
    m3 = [[1, 0], [0, 1]]
    print(mat_pow(m3, 5))

```



## 再帰のメモ化

`functools.lru_cache`で再帰関数に`@lru_cache`デコレータをつけるだけ！ただし普通のループで計算できる漸化式などはループで書いたらいい。

```python
from functools import lru_cache

# max_size: 128 (default)
# 2 の累乗であるときが最も効率的に動作
# noneを指定するとキャッシュは際限なく大きくなる
@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)


print([fib(n) for n in range(16)])
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

print(fib.cache_info())
# CacheInfo(hits=28, misses=16, maxsize=None, currsize=16)

```



## 漸化式の前計算

$a_0=1, a_i = 2a_{i-1} + 3$ の計算結果を保持したいとき

```python
a = [1]   # a_0
for i in range(100):  # a_100まで計算
    a.append(2*a[-1] + 3)
```





## 再帰上限の引き上げ

```python
import sys
sys.setrecursionlimit(1_000_000)
```

## 最大公約数GCDと最小公倍数LCM

二項演算

```python
from math import gcd
print(gcd(2, 3))

def lcm(a, b):
    return a*b // gcd(a, b)
```

複数

```python
import math
from functools import reduce


def gcd(*numbers):
    return reduce(math.gcd, numbers)


def lcm_base(x, y):
    return (x * y) // math.gcd(x, y)

def lcm(*numbers):
    return reduce(lcm_base, numbers, 1)


print(gcd(27, 18, 9))
# 9
print(gcd(*[27, 18, 9, 3]))
# 3

print(lcm(27, 18, 9, 3))
# 54
print(lcm(*[27, 9, 3]))
# 27
```



## その他小さなtips

### 便利な関数たち

```python
# 累乗、階乗
from math import factorial
f = factorial(5)


# 商とあまりを同時に取得
d, m = divmod(a, b)
d, m = a//b, a % b

# 距離の計算など平方和のルート
from math import sqrt
from math import hypot
ax,ay = 1, 2
bx,by = 4, 5
distance = hypot(ax-bx, ay-by)
distance = sqrt((ax-bx)**2 + (ay-by)**2)

```



### 切り上げ

```python
from math import ceil

ceil(0.5)
ceil(a / b)
(a+b-1) / b

```



### 四捨五入(a / b)

```python
int(x+0.5) # round
(a + (b / 2)) / b
```



### 10進数の一番下の桁で条件分岐したり桁を増やしたり

10進数は10倍、10で割るでシフトできる

```python
k = 12345678
k1 = k%10         # 1の位は10で割ったあまり

kk = 10*k + k%10  # 1の位を追加して左にシフト
```



# 貪欲

# テクニック

## インデックスが煩雑なら変数に置く

`atusa[i-1]`を繰り返し使うようなら、`a = atusa[i-1]`と置くと可読性が増す。あとアクセスを減らせるので早い。

```python
# 変数におくとき
def f(n: int, x: int) -> int:
    a = atusa[n-1]
    p = pathi[n-1]
    if x <= 2*a + 2:
        return p + 1 + f(n-1, x - 2 - a)

# 愚直に書いたとき
def f(n: int, x: int) -> int:
    if x <= 2*atusa[n-1] + 2:
        return pathi[n-1] + 1 + f(n-1, x - (2+atusa[n-1]))
```



# 構築

# ゲーム

- ABC067 D - Fennec VS. Snuke(400)

ゲームの言い換えを考えて、「hogeが多いプレイヤーの勝ち」とすると考えやすい。

# フロー









# 文法など

## 入力関係

`input`より`stdin.readline`のが速い（これでTLEなどはないけど）。改行文字が厄介なので、`rstrip`を使う

```python
from sys import stdin
input = stdin.readline().rstrip
```



```python
N
N = int(input())-1

a_1, ..., a_N
a = list(map(lambda x: int(x)-1, input().split())) # int -> lambdaに変える

a0 b0
a1 b1
...
aN bN
ab = [tuple(map(int, input().split())) for _ in range(N)]
```



## 出力

改行しながら

```python
ans = [1, 2, 3]
print(*ans, sep='\n')
# 1
# 2
# 3
```

flushする

```python
print("hello", flush=True)
```



## python vs pypy

基本的にpypyのが明らかに早い

ただし `+=` のような文字列結合に弱い。**pypyでは`string.join('a')`を使用する。**

あと**再帰が弱い**。遅いし上限も小さい

```python
import sys
sys.setrecursionlimit(10**7)

def dp(N) :
    if N == 0 :
        return 0
    else :
        return 1 + dp(N - 1)

print(dp(600000))
```

|          | Python  | PyPy    |
| :------- | :------ | ------- |
| **最高** | 915 ms  | 2680 ms |
| **最低** | 1270 ms | 2705 ms |



## リスト

### 条件を満たす個数を求める

```python
data = range(1, 100)
cnt = len([x for x in data if (条件)])
```

### 行列のprintを見やすく

```python
data = [[1, 2, 3],
        [4, 5, 6]]
print(*data, sep='\n')
# [1, 2, 3]
# [4, 5, 6]
```



### 行列の転置

```python
data = [[1, 2, 3],
        [4, 5, 6]]
data_T = [x for x in zip(*data)]
print(*data_T, sep='\n')
# [1, 4]
# [2, 5]
# [3, 6]
```

### 行列をフラットに

```python
data = [[1, 2, 3],
        [4, 5, 6]]

"""1. itertools.chain.from_iterable
1万行で500 us
"""
import itertools
flat = list(itertools.chain.from_iterable(data))

"""2. 初期値を[]にして要素をsumで+していく
1万行で400 ms
"""
flat = sum(data, [])

```

### 削除

```python
a = [1, 2, 3, 4, 5]

del a[1]        # インデックス指定で削除したい場合

del a[1:3]      # スライスで部分リストを指定して削除も可能

x = a.pop(1)    # popでもインデックス指定で削除可能

a.remove(3)     # オブジェクトを指定して削除したい場合
```

### ソート

```python
a = [1, 3, 2, 4,]

"""基本
"""
sorted(a)   # ソート結果を返す
a.sort()    # 破壊的ソート

"""keyを指定する
"""
a = [(1, 'One', '1'), (1, 'One', '01'),
     (2, 'Two', '2'), (2, 'Two', '02'),
     (3, 'Three', '3'), (3, 'Three', '03')]
print(sorted(a, key=lambda x: x[1]))


"""keyを複数指定
"""
a = [(1, 'One', '1'), (1, 'One', '01'),
     (2, 'Two', '2'), (2, 'Two', '02'),
     (3, 'Three', '3'), (3, 'Three', '03')]
print(sorted(a, key=lambda x: (x[1], x[2], x[0])))


"""keyを複数指定（片方は降順）
"""
a = [(1, 'One', '1'), (1, 'One', '01'),
     (2, 'Two', '2'), (2, 'Two', '02'),
     (3, 'Three', '3'), (3, 'Three', '03')]
print(sorted(a, key=lambda x: (x[1], -x[2], -x[0])))
```

### deepcopy

```python
a = [1, 2, 3, 4, 5]

"""一次元なら以下でできる
"""
b = a[:]    # 終点始点を指定してすべてをスライスで取得（結果コピーと同じ）

"""copy モジュールだと多重リストでも安心
"""
from copy import deepcopy
b = deepcopy(a)
```

### 反転

```python
a = [1, 2, 3]
reverse = a[::-1]
```

### anyとall

すべて探索したか、とかで便利

```python
a = [True, False, True]
if any(a):
    実行される
if all(a):
    実行されない
```

### 行列の転置

- ABC159

```python
mat = [[0, 1, 2], 
       [3, 4, 5]]
matT_tuple = list(zip(*mat))
# [(0, 3), 
#  (1, 4), 
#  (2, 5)]

matT_list = [list(x) for x in zip(*mat)]
# [[0, 3], 
#  [1, 4], 
#  [2, 5]]
```

### リスト各要素のカウント →collections.Counter

- ABC159

```python
from collections import Counter

l = ['a', 'a', 'a', 'a', 'b', 'c', 'c']
cnts = Counter(l)

# dictのように使える
cnts.items()
cnts.keys()
cnts.values()
cnts['a']

# 出現回数順
cnts.most_common() # [('a', 4), ('c', 2), ('b', 1)]

```



## itertoolsまとめ

itertoolsの戻り値はイテレータなのでリストなどに変換しよう



### 累積和：`itertools.accumulate`

```python
from itertools import accumulate

a = list(range(1, 11))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

b = list(accumulate(a))    
# [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
```

### 条件が真である限り除外／取り出し：`dropwhile`、`takewhile`

```python
from itertools import dropwhile, takewhile

a = [3, 6, 1, 7, 2, 5]

b = dropwhile(lambda x: x != 1, a)  # 1が出るまでを除外する
# [1, 7, 2, 5]

c = takewhile(lambda x: x != 1, a)  # 1が出るまでを取り出す
# [3, 6]
```

### 連続する要素のグループ化：`groupby`

あらかじめソートしとくといいかも

```python
from itertools import groupby

a = [1, 1, 2, 3, 3, 3, 1, 2, 2]

for key, value in groupby(a):
    print(key, list(value))
1 [1, 1]
2 [2]
3 [3, 3, 3]
1 [1]
2 [2, 2]

# keyの指定ができる
# 以下は偶奇でわけてる
for key, value in groupby(a, key=lambda x: x % 2):
    print(key, list(value))
1 [1, 3]
0 [2, 4]
1 [3, 1, 1]
0 [2, 4]
```

### 順列　permutation

- ABC054 C - One-stroke Path
- ABC145 C – Average Length (300点)

```python
from itertools import permutations

print(*permutations(range(3)))     # 3!　通り
print(*permutations(range(3), 2))  # 3P2
```

### 長い方に合わせるzip - itertools.zip_longest

- ABC058-B- ∵∴∵(200)

```python
a = [1, 2, 3]
b = ["a", "b"]

# 通常
for ai,bi in zip(a, b):
    print(ai,bi)
# 1 a
# 2 b


# zip_longest
# fillvalueを指定可能
from itertools import zip_longest
for ai,bi in zip_longest(a,b, fillvalue=None):
    print(ai,bi)
# 1 a
# 2 b
# 3 None
```

# 参考

[Qiita-Pythonで競プロやるときによく書くコードをまとめてみた]([https://qiita.com/y-tsutsu/items/aa7e8e809d6ac167d6a1#%E7%B4%AF%E7%A9%8D%E5%92%8C](https://qiita.com/y-tsutsu/items/aa7e8e809d6ac167d6a1#累積和))

[itertools](https://docs.python.org/ja/3/library/itertools.html#itertools.chain.from_iterable)