"""f: function(x:int)->bool
f(s): NG(False), f(t)=OK(True)，f は広義単調増加
return : [s, t) での f(x)=True なる最小のx を求める． O(log(t-s))
"""

def f(x:int)->bool:
    """return (積載量ｘ),k台でいけるか
    """
    global K, ws
    track_cnt = 1
    weight = 0
    for wi in ws:
        if wi > x: return False
        if weight+wi > x:
            track_cnt += 1
            weight = wi
        else:
            weight += wi
    return track_cnt <= K

def binary_search(f, s:int, t:int)->int:
    """f: function(x:int)->bool
    f(s): NG(False), f(t)=OK(True)，f は広義単調増加
    return : [s, t) での f(x)=True なる最小のx
    """
    ng = s-1
    ok = t
    while ok - ng > 1:
        mid = (ok + ng)//2
        if f(mid): ok = mid
        else:      ng = mid
        # print(ng, ok, ok-ng)

    if ok==s-1:
        raise(ValueError("For all x, f(x)=False"))
    return ok

if __name__ == "__main__":
    N,K = map(int, input().split())
    ws = [int(input()) for _ in range(N)]
    MAX_P = 10**10
    MIN_P = 0
    print(binary_search(f, MIN_P, MAX_P))
