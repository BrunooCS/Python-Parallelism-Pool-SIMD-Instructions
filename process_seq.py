import random
import sys

import time

from funcs import dprint, is_prime, flatten_list_of_lists, to_time_human, CHUNK_SIZE, \
    read_args, len_of, get_step, sci_not_str, INT_LOWER, INT_UPPER, reverse_number, factorize, count_digit_occurrences


def process_data_1(data):
    start = time.time()
    chunk_size = CHUNK_SIZE
    i = 0
    results = list()
    while i < len(data):
        split1 = data[i:i + chunk_size]
        split2 = data[i + chunk_size:i + 2 * chunk_size]
        split3 = data[i + 2 * chunk_size:i + 3 * chunk_size]
        split4 = data[i + 3 * chunk_size:i + 4 * chunk_size]

        for n1, n2, n3, n4 in zip(split1, split2, split3, split4):
            res1 = [n1 + n3]
            res2 = [int((n1 - n3) / 13)]
            res3 = [(n1 * n3) % 125]
            res4 = [7 * (n1 - n3)]
            res5 = [(n1 * n3) % 333]
            res6 = [int((n1 + n3) * 66.89)]
            res7 = [int((n1 + n3) / 16)]
            results.append(res1)
            results.append(res2)
            results.append(res3)
            results.append(res4)
            results.append(res5)
            results.append(res6)
            results.append(res7)
            res1 = [n2 + n4]
            res2 = [int((n4 - n2) / 17)]
            res3 = [(n2 * n4) % 250]
            res4 = [7 * (n2 - n4)]
            res5 = [int((n2 - n4) / 11)]
            res6 = [(n4 * n2) % 123]
            res7 = [int((n2 + n4) * 87.67)]
            res8 = [int((n2 + n4) / 17)]
            results.append(res1)
            results.append(res2)
            results.append(res3)
            results.append(res4)
            results.append(res5)
            results.append(res6)
            results.append(res7)
            results.append(res8)
            res1 = [n2 + n3 + n4 + 1]
            res2 = [n2 * (n3 + n4 % 1000)]
            res3 = [int(n2 / (1 + n3 + n4 % 1000))]
            results.append(res1)
            results.append(res2)
            results.append(res3)

        i = i + 4 * chunk_size
    results = flatten_list_of_lists(results)
    results = halve(results)
    results = halve(results)
    # Clip
    results = [n % int(1e5) for n in results]
    end = time.time()
    return results, to_time_human(start, end)


def process_data_2(data):
    def op_mod0(n):
        return sum([i for i in [*range(0, n, get_step(n))]]) % 3 == 1

    def op_mod1(n):
        return sum([i for i in [*range(0, n, get_step(n))]]) % 7 == 4

    def op_mod2(n):
        return sum([i for i in [*range(0, n, get_step(n))]]) % 10 == 2

    def op_mod3(n):
        return sum([i for i in [*range(0, n, get_step(n))]]) % 5 == 3

    ops = [op_mod0, op_mod1, op_mod2, op_mod3]

    def process(l):
        res = list()
        prev_n = l[0]
        res.append(prev_n)
        mod = 0
        for n in l[1:]:
            if ops[mod](prev_n):
                res.append(n * 7)
            else:
                res.append(prev_n * 3)
            prev_n = n
            mod = (mod + 1) % 4
        return res

    start = time.time()

    results = process(data)
    results = process(results)
    results = process(results)
    results = process(results)

    end = time.time()
    return results, to_time_human(start, end)


def process_data_3(data):
    start = time.time()
    results = list()
    for i in range(0, len(data)):
        n = data[i]
        random.seed(i % 300)
        digit = random.randint(1, 9)
        range_limit = 15 * (1 + count_digit_occurrences(n, digit))
        acum = n
        for x in range(0, n % range_limit):
            acum *= (x + 1)
            acum %= random.randint(int(2.5e2), int(1e3))
        results.append(acum)
    end = time.time()
    return results, to_time_human(start, end)


def process_data_4(data):
    start = time.time()
    results = list()
    for n in data:
        primes = factorize(n)
        results += primes
    results_final = list()
    for n in results:
        results_final.append(reverse_number(n))
    results_final = sorted(results_final)
    end = time.time()
    return results_final, to_time_human(start, end)


def process_data_5(data):
    start = time.time()
    results = list()
    for n in data:
        if is_prime(n):
            results.append(n * 52)
        else:
            results.append(n * 12)
    end = time.time()
    return results, to_time_human(start, end)


def halve(data):
    results = list()
    quarter = int(len(data) / 4)
    for x, y, z, w in zip(data[:quarter], data[1 * quarter:2 * quarter], data[2 * quarter:3 * quarter],
                          data[3 * quarter:]):
        # results.append(x + y - z + w)
        results.append(x + y)
        results.append(z - w)
    return results


def sequential(BLOCK_SIZE, N_BLOCKS):
    data, t0 = generate_data(BLOCK_SIZE, N_BLOCKS)
    dprint("DATA: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(data)), len_of(data), t0))

    results1, t1 = process_data_1(data)
    dprint("RES1: hash -> {0:22d} | len -> {1:4} | time -> {2:8}".format(hash(tuple(results1)), len_of(results1), t1))

    results2, t2 = process_data_2(results1)
    dprint("RES2: hash -> {0:22d} | len -> {1:4} | time -> {2:8}".format(hash(tuple(results2)), len_of(results2), t2))

    results3, t3 = process_data_3(results2)
    dprint("RES3: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(results3)), len_of(results3), t3))

    results4, t4 = process_data_4(results3)
    dprint("#### CHECKPOINT, RES4 MUST BE EXACTLY EQUAL ####")
    dprint("RES4: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(results4)), len_of(results4), t4))

    results5, t5 = process_data_5(results4)
    dprint("RES5: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(results5)), len_of(results5), t5))

    return data, results4, results5


def generate_data(BLOCK_SIZE, N_BLOCKS):
    start = time.time()
    blocks = []
    for s in [*range(0, N_BLOCKS)]:
        random.seed(1234 * s)
        l = list()
        for n in [*range(0, BLOCK_SIZE)]:
            x = random.randint(INT_LOWER, INT_UPPER)
            x = reverse_number(x)
            x *= n % int(1e2)
            l.append(x)
        blocks.append(l)
    data = flatten_list_of_lists(blocks)
    end = time.time()
    return data, to_time_human(start, end)


if __name__ == "__main__":
    BLOCK_SIZE, N_BLOCKS = read_args(sys.argv)
    print("Processing {0} blocks of len {1}".format(N_BLOCKS, sci_not_str(BLOCK_SIZE)))
    start = time.time()
    sequential(BLOCK_SIZE, N_BLOCKS)
    end = time.time()
    print("It took {0} to process the data".format(to_time_human(start, end)))
