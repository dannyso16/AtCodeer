"""
二部探索と三部探索
- 二部探索：bool値の変わる境界を探す
- 三部探索：凸関数の極値を探す

三部探索で整数解を求めたいときは，
def f(x:float)->int:
    x = int(x+0.5) # round
のようにfloatで受けて，intにするといい．
答えは誤差を考慮して，前後の値も候補に入れておくと安心．

"""


def tri_search(f: "f(x:float)->float", left: float, right: float,
               is_convex_downward=True, iter=100) -> float:
    """is_convex_downward: 下に凸 return minimum
    else: 上に凸 return Maximum
    f: convex upward -> -f: convex downward
    """
    for _ in range(iter):
        ml = (left*2 + right) / 3
        mr = (left + right*2) / 3
        if is_convex_downward:
            f_ml, f_mr = f(ml), f(mr)
        else:
            f_ml, f_mr = -f(ml), -f(mr)

        if f_ml < f_mr:
            right = mr
        else:
            left = ml
    print(left, right)
    return (right + left) / 2


def f_conv_up(x: float) -> float:
    return -(x-5)**2 + 10


def f_conv_down(x: float) -> float:
    return (x-5)**2 + 10


if __name__ == "__main__":
    x = tri_search(f_conv_down, 0, 100, is_convex_downward=True)
    print(x, f_conv_down(x))

    x = tri_search(f_conv_up, 0, 100, is_convex_downward=False)
    print(x, f_conv_up(x))
