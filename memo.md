# Memo

## ABC160

- $頂点数 < 10^3, 辺数 < 10^3$ なら、全点間最短距離を出せる（$O(N^2)$程度）
- 制約が$頂点数 < 10^3$ならまず2重ループから考えてもいい
- 近道がある
  - 近道を使う時と使わないときを考えて、minをとったり

### ある始点から各頂点への最短経路をだす（ダイクストラ）

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

### 最短経路を復元（ダイクストラ）

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



## ABC159

### 回文判定

```python
S: str = input()
if S == S[::-1]:
    print("回文")
else:
    print("回文でない")
```

### リスト各要素のカウント →collections.Counter

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

### 行列の転置

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

E:
実装が難しそう…[例](http://kmjp.hatenablog.jp/entry/2020/03/22/0900)



## ABC158

### collections.deque

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



## ABC087

### 重み付きUnion Find

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





## ABC156

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

### modありの組み合わせの数を高速に計算

```python
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
```

### 繰り返し二乗法

pythonだと`pow(a, b, M)`で高速に $a^b \ mod \ M$ が計算できるが、下に実装例も載せておく

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

## ABC142

### 素因数分解

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



## ABC034

平均の最大化→答え（目標）を決めてしまう。つまり「平均の最大値を求める」と考えず、「つくれる最大の閾値を求める」と考える

### 二部探索

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



## ARC32

### 重みなしDisjoint set (Union Find)

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

# ライブラリ

## 素数関係

```python
"""
素因数分解に関係するものたち
- エラトステネスの篩 O(N loglog N)
  --sieve(n:int)->list

- 素因数分解 O(sqrt(N))
  --factorize(a:int)->dict
"""

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



## 組み合わせ関係

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



## 繰り返し二乗法

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

# python vs pypy

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

