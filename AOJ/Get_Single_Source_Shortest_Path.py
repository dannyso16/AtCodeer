import heapq
def dijkstra(start:int, goal:int):
    """print the shortet path  O(|E|log|V|)
    maybe verified
    """
    global dist # list dist[|V|]
    global edges # 隣接リスト
    dist[start] = 0
    prev_v = [-1]*(len(dist)) # 最短経路でのひとつ前の頂点
    q = [(0,start)] #（暫定的な距離，頂点）
    while q:
        d_cur,cur = heapq.heappop(q)
        if dist[cur] < d_cur: # すでに探索済み
            continue
        for nex,cost in edges[cur]:
            if dist[nex] > dist[cur]+cost:
                dist[nex] = dist[cur]+cost
                heapq.heappush(q, (dist[nex], nex))
                prev_v[nex] = cur

    # 以下 path の復元
    path = []
    v = goal
    while v != -1:
        path.append(v)
        v = prev_v[v]
    path = path[::-1] # reverse
    print("Shortest path: ", path)


if __name__ == "__main__":
    V,E,start = map(int, input().split())
    goal = V-1
    edges = [[] for _ in range(V)]
    for _ in range(E):
        s,t,d = map(int, input().split())
        edges[s].append((t, d))
    dist = [float('inf') for _ in range(V)]
    dijkstra(start, goal)
