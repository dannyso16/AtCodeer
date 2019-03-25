# nCr = n*(n-1)*...*(n-r+1) / r*(r-1)*...*1
# 割り算があるのでmod計算が厄介 a / b (mod p) != a (mod p) / b (mod p)
# Fermat's little theorem より，
# a^(-1) ≡ a^(p-2) (mod p), p is prime number
# 割り算 -> 逆元の掛け算に変形させる


MOD = 10**9 + 7
def comb_fermat(n:int, r:int)->int:
    # return nCr (mod MOD)
    if r > n-r: return comb(n, n-r)
    mul,div = 1,1
    for i in range(r):
        mul *= n-i
        mul %= MOD
        div *= i+1
        div %= MOD

    ret = mul * pow(div, MOD-2, MOD) % MOD
    return ret

# python らしいごり押し (あまり大きくないときに使おう)
from math import factorial
def comb(n:int, r:int)->int:
    ret = factorial(n) // (factorial(n-r)*factorial(r))
    return ret % MOD


if __name__ == '__main__':
    print(comb_fermat(10, 3))
    print(comb(10, 3))
