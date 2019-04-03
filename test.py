# b = [list(map(int, input().split())) for _ in range(2)]
# c = [list(map(int, input().split())) for _ in range(3)]

for cond in range(2**10):
    c = cond
    cnt = 0
    while c>0:
        if c&1==1:
            cnt += 1
        c >>= 1
    if cnt != 4: continue
    choku = 0
    naoko = 0
    for i in range(2):
        for j in range(3):
            if 
