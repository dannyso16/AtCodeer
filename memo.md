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

