""" リストSにある値xがあるか効率的に探索する
"""
N = int(input())
S = list(map(int, input().split()))
# 両端に必ずFalse ，True になるものを追加しておく
S.extend([-1, float('inf')])
S.sort()
# print(S)
Q = int(input())
T = list(map(int, input().split()))
ans = 0
for ti in T:
    # S[x]<=tiなる最大のxを求める
    ok = 0
    ng = len(S)
    while ng - ok > 1:
        mid = (ng+ok)//2
        if S[mid]<=ti: ok=mid
        else:          ng=mid
    if S[ok]==ti:
        ans += 1

print(ans)
