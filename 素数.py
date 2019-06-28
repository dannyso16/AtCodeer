"""
素因数分解に関係するものたち
- エラトステネスの篩 O(N loglog N)
  --sieve(n:int)->list

- 素因数分解 O(sqrt(N))
  --factorize(a:int)->dict

"""
def sieve(n:int)->list:
    """Sieve of Eratoshenes O(N loglog N)
    return the list of is_prime (0, n]
    """
    assert n >= 1, "don't input negative value"

    is_prime = [True]*(n+1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n+1):
        if is_prime[i]:
            # 素数の倍数は合成数
            for j in range(i*2, n+1, i):
                is_prime[j] = False
    return is_prime


def factorize(a:int)->dict:
    """prime factorization O(sqrt(N))
    return dict of prime factor
    """
    ps = {}
    i = 2
    nokori = a
    while i*i <= nokori:
        if nokori%i==0:
            cnt = 0
            while nokori%i==0:
                cnt += 1
                nokori //= i
            ps[i] = cnt
        i += 1
    if nokori != 1:
        ps[nokori] = 1
    return ps


if __name__ == '__main__':
    N = 10
    print(sieve(N))
    print(factorize(N))
