# nCr = n*(n-1)*...*(n-r+1) / r*(r-1)*...*1
# 割り算があるのでmod計算が厄介 a / b (mod p) != a (mod p) / b (mod p)
# Fermat's little theorem より，
# a^(-1) ≡ a^(p-2) (mod p), p is prime number
# 割り算 -> 逆元の掛け算に変形させる

from math import factorial
MOD = 10**9 + 7


def comb_fermat(n: int, r: int) -> int:
    # Fermat's little theorem: O(r)
    # return nCr (mod MOD)
    if r > n:
        return 0
    if r > n-r:
        return comb_fermat(n, n-r)
    mul, div = 1, 1
    for i in range(r):
        mul *= n-i
        mul %= MOD
        div *= i+1
        div %= MOD

    ret = mul * pow(div, MOD-2, MOD) % MOD
    return ret


# python らしいごり押し (あまり大きくないときに使おう)
def comb_naive(n: int, r: int) -> int:
    ret = factorial(n) // (factorial(n-r)*factorial(r))
    return ret % MOD


# com[2000][2000]
com = [[0]*2000 for _ in range(2000)]
com[0][0] = 1


def calc_comb():
    # k,n <= 2000
    # dp (Pascal's triangle): O(n*k)
    global com
    for i in range(1, len(com)):
        com[i][0] = 1
        for j in range(1, len(com)):
            com[i][j] = com[i-1][j-1] + com[i-1][j]
            com[i][j] %= MOD


# 逆元を前計算(O(MAX_N))しておくことでクエリをO(1)で返す
MAX = 10**5
fac = [0]*MAX  # fac[n]:  (n!) mod p
finv = [0]*MAX  # finv[n]: (n!)^-1 mod p
inv = [0]*MAX  # inv[n]:  (n)^-1 mod -p


def comb_init():
    global fac, finv, inv
    fac[0] = fac[1] = 1
    finv[0] = finv[1] = 1
    inv[1] = 1
    for i in range(2, MAX):
        fac[i] = fac[i-1] * i % MOD
        inv[i] = MOD - inv[MOD % i] * (MOD//i) % MOD
        finv[i] = finv[i-1] * inv[i] % MOD


def comb(n: int, r: int) -> int:
    global fac, finv
    if n < r:
        return 0
    if n < 0 or r < 0:
        return 0
    return fac[n] * (finv[r] * finv[n-r] % MOD) % MOD


if __name__ == '__main__':
    print(comb_fermat(10, 3))
    print(comb_naive(10, 3))
    comb_init()
    print(comb(10, 3))
    calc_comb()
    print(com[10][3])
