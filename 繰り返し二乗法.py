# a^p (mod MOD) を高速に求める方?��?
# 普通に計算すると　O(p)　?��?が，O(log p) で求ま?��?
# p is even:  a^p = a^(p/2) * a^(p/2)
# p is odd:   a^p = a * a^(p-1)
# p is 0:     a^p = 1
# python の場合，�?み込みの pow(a, p, MOD) とすれば?��??��???��?

MOD = 10**9 + 7


def modpow(a: int, p: int, mod: int) -> int:
    # return a**p (mod MOD) O(log p)
    if p == 0:
        return 1
    if p % 2 == 0:
        half = modpow(a, p//2, mod)
        return half*half % mod
    else:
        return a * modpow(a, p-1, mod) % mod


def modpow_bitwise(a: int, p: int, mod: int) -> int:
    # return a**p (mod MOD) O(log p)
    # 蟻本p115??��?��二乗して?��?きながら 1 が立って?��?るところ?��?けを使?��?
    res = 1
    while p > 0:
        if p & 1 > 0:
            res = res * a % mod
        a = a**2 % mod
        p >>= 1
    return res

###### 行�??��ver #####


def mat_pow(base, p):
    # base: matrix
    # return base^p
    # た�??��??��?��あって?��??��?
    ret = None
    mag = base
    while p > 0:
        if p & 1:
            ret = mag if ret is None else mat_dot(mag, ret)
        mag = mat_dot(mag, mag)
        p >>= 1
    return ret


def mat_dot(m1, m2):
    # a,b: matrix
    # (A x B) @ (B x C) ??��?(A x C)
    # verified ABC021C
    if len(m1[0]) != len(m2):
        raise ValueError('Check matrix shape.')
    A = len(m1)
    C = len(m2[0])
    m2_t = list(zip(*m2))  # m2 ?��]?��u
    ret = [[None]*C for _ in range(A)]
    for row in range(A):
        for col in range(C):
            v = 0
            for a, b in zip(m1[row], m2_t[col]):
                v += a*b
            ret[row][col] = v
    return ret


if __name__ == '__main__':
    MOD = 10**9 + 7
    print(modpow(100, 30000, MOD))
    print(modpow_bitwise(100, 30000, MOD))
    m1 = [[1, 2, 3], [2, 3, 4]]
    m2 = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
    print(mat_dot(m1, m2))
    m3 = [[1, 0], [0, 1]]
    print(mat_pow(m3, 5))
