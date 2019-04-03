import sys
from math import isinf
def warshall_floyd(v_count: int):
    """ ワーシャルフロイド　(user:htkb)
    :param v_count: 頂点数
    :param matrix:  隣接行列(到達不能はfloat("inf"))
    """
    # 到達不能をfloat("inf")にしておけば余計なチェックを入れなくても
    # inf > inf+(-1) のような到達不能＋負辺が繋がってしまうことはない
    global matrix
    for i in range(v_count):
        for j, c2 in enumerate(row[i] for row in matrix):
            for k, (c1, c3) in enumerate(zip(matrix[j], matrix[i])):
                if c1 > c2+c3:
                    matrix[j][k] = c2+c3 # min(c1, c2+c3)

if __name__ == "__main__":
    V,E = map(int, input().split())
    matrix = [[float('inf')]*V for _ in range(V)]
    for i in range(V):
        matrix[i][i] = 0
    for _ in range(E):
        s,t,d = map(int, input().split())
        matrix[s][t] = d

    warshall_floyd(V)
    for i in range(V):
        if matrix[i][i] < 0:
            print("NEGATIVE CYCLE")
            sys.exit()
    for mi in matrix:
        print(' '.join( ('INF' if isinf(dij) else str(dij) for dij in mi)))
