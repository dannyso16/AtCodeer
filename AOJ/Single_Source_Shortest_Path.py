import sys
from math import isinf
# どれが終点の時に負の閉路があるかをnegative[v]にいれる
# dists = [float('inf')] * V # 到達不能はinf
# negatives = [False] * V

def bellman_ford(start:int)->bool:
    """ ベルマンフォード O(|V|*|E|)
    :param V:      頂点数
    :param edges:  辺 [(始点，終点，重み)]
    :param start:  始点
    """
    global dists, negatives
    global edges
    dists[start] = 0
    V = len(dists)
    for i in range(V):
        for (s, t, w) in edges:
            if isinf(dists[s]): continue
            if dists[t] > dists[s] + w:
                dists[t] = dists[s] + w
                if i==V-1:
                    negatives[t] = True


if __name__ == "__main__":
    V,E,start = map(int, input().split())
    edges = []
    for _ in range(E):
        s,t,d = map(int, input().split())
        edges.append((s,t,d))

    dists = [float('inf')] * V # 到達不能はinf
    negatives = [False] * V
    bellman_ford(start)
    if any(neg for neg in negatives)==True:
        print("NEGATIVE CYCLE")
        sys.exit()
    for di in dists:
        if di==float('inf'):
            print("INF")
        else:
            print(di)
