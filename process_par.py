# Imports
##############################################################################
import concurrent.futures
import numpy as np
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import math

import random
import sys

import time

from funcs import dprint, is_prime, flatten_list_of_lists, to_time_human, CHUNK_SIZE, \
    read_args, len_of, get_step, sci_not_str, INT_LOWER, INT_UPPER, reverse_number, factorize, count_digit_occurrences
##############################################################################



# ************************* Funciones mejoradas ************************
##############################################################################
def factorize_optim(n):
    """
    En lugar de probar todos los números hasta n, podemos limitarnos a probar solo 
    los números hasta la raíz cuadrada de n. 
    Esto se debe a que si n tiene un factor mayor que su raíz cuadrada, 
    entonces necesariamente tiene un factor menor que su raíz cuadrada.
    Complejidad -> O(log(n))
    """
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            yield i
    if n > 1:
        yield n


def is_prime_test(num):
    """ Implementación de  Test de Primalidad"""
    """ los primos mayores a 6, siguen la fórmula 6k+-1 
        Forma más eficiente de saber si un número es primo o no
        Complejidad O(sqrt(n)/log(sqrt(n))) caso medio,  O(sqrt(n)) en el peor de los casos"""
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True
##############################################################################



# Función 1
##############################################################################

def process_data_1(data):
    start = time.time()
    chunk_size = CHUNK_SIZE

    # División de los datos en cuatro chunks con cuatro splits cada uno
    chunks = [(
        data[i:i + chunk_size],
        data[i + chunk_size:i + 2 * chunk_size],
        data[i + 2 * chunk_size:i + 3 * chunk_size],
        data[i + 3 * chunk_size:i + 4 * chunk_size]
    ) for i in range(0, len(data), CHUNK_SIZE *4)]
    

    # Ejecución secuecnial de los 4 chunk (más rápido que un pool con 4 procesos)
    res = []
    for split1, split2, split3, split4 in chunks:
        res.append(process_chunk_np(split1, split2, split3, split4))

    
    results = np.concatenate(res).flatten() # Concatenar y aplanar los resultados de los chunk 
    results = halve(results)
    results = halve(results)
    results = results % int(1e5)

    end = time.time()

    return results, to_time_human(start, end)

def process_chunk_np(split1, split2, split3, split4 ):
    # Procesamiento de los chunks de forma vectorial (Operaciones SIMD)
    res1_7 = np.array([split1 + split3, 
                       ((split1 - split3) / 13).astype(int), 
                       (split1 * split3) % 125, 
                       7 * (split1 - split3), 
                       (split1 * split3) % 333, 
                       ((split1 + split3) * 66.89).astype(int), 
                       ((split1 + split3) / 16).astype(int)]).T

    res8_15 = np.array([split2 + split4, 
                        ((split4 - split2) / 17).astype(int), 
                        (split2 * split4) % 250, 
                        7 * (split2 - split4), 
                        ((split2 - split4) / 11).astype(int), 
                        (split4 * split2) % 123, 
                        ((split2 + split4) * 87.67).astype(int), 
                        ((split2 + split4) / 17).astype(int)]).T

    res16_18 = np.array([(split2 + split3) + (split4 + 1), 
                         split2 * (split3 + split4 % 1000), 
                         (split2 / (1 + split3 + split4 % 1000)).astype(int)]).T
    
    return np.concatenate((res1_7, res8_15, res16_18), axis=1)








# Función 2
##############################################################################

def sum_range(n, step):
    # Suma de una secuencia aricmética
    # s = m/2 * (m-1)*step   (siendo m el número de términos)
    m = (n+step-1)//step
    return ((m*(m-1)//2 )*step)

def op_mod0(n):
    return sum_range(n,get_step(n)) % 3 == 1

def op_mod1(n):
    return sum_range(n,get_step(n)) % 7 == 4

def op_mod2(n):
    return sum_range(n,get_step(n)) % 10 == 2

def op_mod3(n):
    return sum_range(n,get_step(n)) % 5 == 3

# Lista de funciones op_mod
ops = [op_mod0, op_mod1, op_mod2, op_mod3]


def process_single_element(mod,prev_n):
    """ Procesamiento de op_mod a un elemento"""
    return ops[mod](prev_n)


def process_chunk_2(chunk, mods_start):
  """ Procesamiento de cada Chunk  """
  l = chunk
  # Generación de todos los módulos del chunk
  mods = [i%4 for i in range(mods_start, len(chunk)+mods_start-1)] 

  # Aplicación de forma vectorizada la función "process_single_element" a todos los elementos del chunk
  bools = np.vectorize(process_single_element)(mods,l[:-1])

  return np.where(bools,l[1:] * 7,l[:-1] * 3) # Los resultados "True" de "process_single_element" se multiplican por 7, los "False" por 3




def process_data_2(data):

    def process(l):
        with concurrent.futures.ProcessPoolExecutor() as executor:

            for _ in range(4): # Se aplica cuatro veces el "process" a los datos 
                prev = l[0] # Primer dato

                # División de los datos en en numero de procesadores disponibles
                num_processes = multiprocessing.cpu_count()
                chunk_size = len(data) // num_processes

                # Generación del primer mod de cada chunk (Dependecnia de datoss)
                mods_start = [i % 4 for i in range(0,len(l)-1, chunk_size)]

                # División de los datos en tantos chunks como procesadores
                chunks = [l[i:i+chunk_size+1] for i in range(0, len(l), chunk_size)]
                
                # Pool de "process_chunk_2" con dependencias (mods)
                results =  list()
                for chunk_results in executor.map(process_chunk_2, chunks, mods_start):
                    results.append(chunk_results)

                l = np.insert(np.concatenate(results),0, prev) # Añade el primer dato al principio de los resultados del pool

        return l
    
    start = time.time()

    results = process(data)

    end = time.time()
    return results, to_time_human(start, end)







# Función 3
##############################################################################



def process_chunk_3(chunk, start_index, results):
    """Procesamiento chunk aplicando "process_element_3" de forma vectorial a todos los datos """
    
    seeds = np.mod(np.arange(start_index, len(chunk)+start_index), 300) # Generación de semillas para cada dato

    res = np.vectorize(process_element_3)(chunk, seeds) # Vectorización  de "process_element_3" 

    # El resultado del chunk se añade a la lista "results" global en la posición que la corresponda según el index
    results[start_index:len(res)+start_index] = res 


def process_element_3(n,seed):
    """ Procesamiento de cada dato """
    random.seed(int(seed))
    digit = random.randint(1, 9)
    range_limit =  15 * (1 + count_digit_occurrences(n, digit))
    acum = n
    for x in range(n % range_limit):
        acum = (acum * (x + 1)) % random.randint(int(2.5e2), int(1e3))
    return acum


def process_data_3(data):
    start = time.time()

    results = multiprocessing.Array('i', len(data)) # Array global en el que cada proceso añade el resutlado en la posición correspondiente

    # División de los datos en tantos chunks como procesadores disponibles
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(data) // num_processes
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


    # Crea cada proceso con el chunk correspondiente y el índice correspondiente (Dependencia de datos)
    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=process_chunk_3, args=(chunk, i * chunk_size, results))
        processes.append(p)
        p.start()

    # Esperar a que todos los procesos terminen
    for p in processes:
        p.join()
    
    end = time.time()
    return results, to_time_human(start, end)







# Función 4
##############################################################################



def chunks(lst, n):
    """ División de los datos en n chunks con generadores.
        Más eficiente en el caso de tener que recorrer cada elemento 
        del chunk en process_chunk_4 y no poder aplicar funciones de forma vectorizada"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_chunk_4(chunk):
    """ Procesamiento del chunk de forma secuencial"""

    # Aplicación de "factorize_optim" a cada dato
    results = list()
    for n in chunk:
        primes = factorize_optim(n)
        results += primes

    # Aplicación de "reverse_number" a cada dato
    results_final = list()
    for n in results:
        results_final.append(reverse_number(n))

    return results_final


def process_data_4(data):
    start = time.time()

    # Cálculo tamaño de chunk para el numero de procesadores disponibles
    chunk_size = CHUNK_SIZE*8


    # Se envía a procesar cada chunk a un proceso.
    results_final = list()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chunk_results in executor.map(process_chunk_4, chunks(data, chunk_size)):
            results_final += chunk_results

    # Ordena los resultados 
    results_final.sort()

    end = time.time()
    return results_final, to_time_human(start, end)






# Función 4
##############################################################################
 

def process_chunk_5(chunk):
    """ Procesamiento chunks de "process_data_5"""
    # Vectrorización de "is_prime_eras" y multiplicadicón de los datos primos por 52 y por 12 los no primos
    return np.where(np.vectorize(is_prime_test, otypes=[object])(chunk),  np.multiply(chunk , 52), np.multiply(chunk , 12))


def process_data_5(data):
    start = time.time()

    # Creación de chunk (en este caso particular es más eficiente chunks más pequeños)
    chunk_size = CHUNK_SIZE*8
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Ejecución de los chunk en los procesos 
    results =  list()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chunk_results in executor.map(process_chunk_5, chunks):
            results.extend(chunk_results)

    #results = flatten_list_of_lists(results)

    end = time.time()
    return results, to_time_human(start, end)


def halve(data):
    """ Implementación vectorial de halve"""
    quarter = len(data) // 4

    x = data[:quarter]
    y = data[quarter:2 * quarter]
    z = data[2 * quarter:3 * quarter]
    w = data[3 * quarter:]

    results = np.empty(2 * quarter, dtype=data.dtype)
    results[::2] = x + y
    results[1::2] = z - w

    return results

def parallel(BLOCK_SIZE, N_BLOCKS):
    data, t0 = generate_data_parallel(BLOCK_SIZE, N_BLOCKS)
    dprint("DATA: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(data.tolist())), len_of(data), t0))

    results1, t1 = process_data_1(data)
    dprint("RES1: hash -> {0:22d} | len -> {1:4} | time -> {2:8}".format(hash(tuple(results1.tolist())), len_of(results1), t1))

    results2, t2 = process_data_2(results1)
    dprint("RES2: hash -> {0:22d} | len -> {1:4} | time -> {2:8}".format(hash(tuple(results2.tolist())), len_of(results2), t2))

    results3, t3 = process_data_3(results2)
    dprint("RES3: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(results3)), len_of(results3), t3))

    results4, t4 = process_data_4(results3)
    dprint("#### CHECKPOINT, RES4 MUST BE EXACTLY EQUAL ####")
    dprint("RES4: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(results4)), len_of(results4), t4))

    results5, t5 = process_data_5(results4)
    dprint("RES5: hash -> {0:22d} | len -> {1:3} | time -> {2:8}".format(hash(tuple(results5)), len_of(results5), t5))

    return data, results4, results5


def generate_block(seed, BLOCK_SIZE):
    """ Procesamiento de cada bloque de "generate_data_parallel" """
    random.seed(seed)
    return [reverse_number(random.randint(INT_LOWER, INT_UPPER)) * (i % int(1e2)) for i in range(BLOCK_SIZE)]


def generate_data_parallel(BLOCK_SIZE, N_BLOCKS):
    start = time.time()
    seeds = [1234 * s for s in range(N_BLOCKS)]

    # Ejecución de cada bloque en paralelo
    with concurrent.futures.ProcessPoolExecutor() as executor:
        blocks = list(executor.map(generate_block, seeds, [BLOCK_SIZE] * N_BLOCKS))

    data = np.concatenate(blocks)
    
    end = time.time()
    return data, to_time_human(start, end)



if __name__ == "__main__":
    BLOCK_SIZE, N_BLOCKS = read_args(sys.argv)
    print("Processing {0} blocks of len {1}".format(N_BLOCKS, sci_not_str(BLOCK_SIZE)))
    start = time.time()
    parallel(BLOCK_SIZE, N_BLOCKS)
    end = time.time()
    print("It took {0} to process the data".format(to_time_human(start, end)))
