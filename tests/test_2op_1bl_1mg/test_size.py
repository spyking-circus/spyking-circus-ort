from main import repeat_test
import settings



sizes = [2, 3, 4]

for size in sizes:
    t_comp = repeat_test(size=size)
    print("size {}: {}".format(size, t_comp))
