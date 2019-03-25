# find prime number O(N loglog N)

N = 100

is_prime = [True]*(N+1)
is_prime[0] = is_prime[1] = False
def sieve(n:int)->int:
    """Sieve of Eratoshenes
    return the number of prime number (0, n]"""
    global is_prime
    if n<2: return 0
    cnt = 0
    for i in range(2, n+1):
        if is_prime[i]:
            cnt += 1
            # 素数の倍数は合成数
            for j in range(i*2, n+1, i):
                is_prime[j] = False
    return cnt


if __name__ == '__main__':
    N = 100
    is_prime = [True]*(N+1)
    is_prime[0] = is_prime[1] = False
    print(sieve(100))
    print(is_prime)
