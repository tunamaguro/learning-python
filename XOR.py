from AND import AND
from OR import OR
from NAND import NAND


def XOR(x1: int, x2: int) -> int:
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
