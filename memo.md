# Memo

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
(a + (b / 2)) / b
```



フェルマーの定理





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


# 2. mod計算を含めて大きな数でもできるようにしたもの
def comb_fermat(n: int, r: int) -> int:
    """"mod MOD を法とした組み合わせの数
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

```



