# 尺取り法 pseudo code
N,K = int(input())
s = list(map(int, input().split()))

right = 0
for left in range(N):
    while (right < N) and (right をひとつ進めても条件を満たす ex. sum_+s[right] <= K):
        # right ++ の処理
        sum_ += s[right]
        right += 1

    # right は条件を満たす最大値 [left, right)
    # ans の更新とかする

    # left++ する準備
    if right == left:
        right += 1
    else:
        sum_ -= s[left]
