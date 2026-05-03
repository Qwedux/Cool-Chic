import random
from typing import assert_never

for _ in range(10):
    A: bool = bool(random.randint(0, 2))
    match A:
        case True:
            print("True A")
        case False:
            print("False A")
        case _:
            assert_never(A)

print("# " + "=" * 78)
