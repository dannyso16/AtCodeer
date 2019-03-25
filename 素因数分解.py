# prime factorization O(sqrt(N))

def factorize(a:int):
    """print dict of prime factor"""
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
        ps[i] = 1
    print(ps)


if __name__ == '__main__':
    factorize(1025)
