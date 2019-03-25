# a^p (mod MOD) を高速に求める方法
# 普通に計算すると　O(p)　だが，O(log p) で求まる
# p is even:  a^p = a^(p/2) * a^(p/2)
# p is odd:   a^p = a * a^(p-1)
# p is 0:     a^p = 1
# python の場合，組み込みの pow(a, p, MOD) とすればいい．

MOD = 10**9 + 7
def modpow(a:int, p:int)->int:
    # return a**p (mod MOD)
    if p==0: return 1
    if p%2==0:
        half = modpow(a, p//2)
        return half*half % MOD
    else:
        return a * modpow(a, p-1) % MOD


if __name__ == '__main__':
    print(modpow(10, 3))
