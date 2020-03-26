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
