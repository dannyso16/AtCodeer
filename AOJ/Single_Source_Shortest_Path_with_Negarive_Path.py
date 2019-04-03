import heapq
def dijkstra(start:int):
    """priority_queue ver. O(|E|log|V|)
    """
    global dist # list dist[|V|]
    global edges # 隣接リスト
    dist[start] = 0
    q = [(0,start)] #（暫定的な距離，頂点）
    while q:
        d_cur,cur = heapq.heappop(q)
        if dist[cur] < d_cur: # すでに探索済み
            continue
        for nex,cost in edges[cur]:
            if dist[nex] > dist[cur]+cost:
                dist[nex] = dist[cur]+cost
                heapq.heappush(q, (dist[nex], nex))


if __name__ == "__main__":
    V,E,start = map(int, input().split())
    edges = [[] for _ in range(V)]
    for _ in range(E):
        s,t,d = map(int, input().split())
        edges[s].append((t, d))
    dist = [float('inf') for _ in range(V)]
    dijkstra(start)
    for di in dist:
        if di==float('inf'):
            print("INF")
        else:
            print(di)
