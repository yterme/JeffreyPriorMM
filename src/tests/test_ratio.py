from tools import ratio

num = [4, 0.0000001]
den = [0.00000000000000000000000000000000000000000000000000001, 1]
rat = ratio(num, den, log=True)
print(rat)