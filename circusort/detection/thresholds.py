import multiprocessing as mp
import numpy as np
import sys

import circusort



path = "/home/pierre/tmp/synthetic.raw"


# def read(mem, path=None):
#     # TODO read chunks from file, split chunk into subchunks and send subchunks
#     # to the correponding workers.
#     # data = get_data(path)
#     mem[:] = range(0, len(mem))
#     return


def read():
    global buf
    buf = np.random.standard_normal(buf.shape)
    # TODO complete...
    return True

def work(mem, i):
    # TODO read subchunks
    mem[i] = 10 + i
    pass

def estimate_median(id):
    '''
    Parameters
    ----------
    id: int
        Channel identifier.
    '''
    global med
    # TODO complete...
    med[id] = id
    return

def estimate_mad(id):
    '''
    Parameters
    ----------
    id: int
        Channel identifier.
    '''
    global mad
    # TODO complete...
    mad[id] = id
    return

def detect_peaks(id):
    '''
    Parameters
    ----------
    id: int
        Channel identifier.
    '''
    global buf
    # TODO complete...
    return

if __name__ == '__main__':

    nb_cpu = mp.cpu_count()
    print("Number of CPUs: {}".format(nb_cpu))

    nb_workers = nb_cpu / 2 - 1
    nb_channels = 4
    size = 32

    def initializer(cbuf, cmed, cmad):
        global buf, med, mad
        buf = np.ctypeslib.as_array(cbuf)
        med = np.ctypeslib.as_array(cmed)
        mad = np.ctypeslib.as_array(cmad)
        return

    # Set up shared memory
    ## Data
    buf = np.zeros(size)
    cbuf = np.ctypeslib.as_ctypes(buf)
    cbuf = mp.Array(cbuf._type_, cbuf, lock=False)
    buf = np.ctypeslib.as_array(cbuf)
    ## Medians
    med = np.zeros(nb_channels)
    cmed = np.ctypeslib.as_ctypes(med)
    cmed = mp.Array(cmed._type_, cmed, lock=False)
    med = np.ctypeslib.as_array(cmed)
    ## MADs
    mad = np.zeros(nb_channels)
    cmad = np.ctypeslib.as_ctypes(mad)
    cmad = mp.Array(cmad._type_, cmad, lock=False)
    mad = np.ctypeslib.as_array(cmad)
    # Set up initial arguments
    initargs = (cbuf, cmed, cmad,)

    # Create pool of workers
    pool = mp.Pool(processes=nb_workers, initializer=initializer, initargs=initargs)

    # while True:
    for _ in range(1):
        # Update the input buffer
        ans = read()
        if not ans:
            break
        # Estimate the median for each channel
        pool.map(estimate_median, range(nb_channels))
        # Estimate the MAD for each channel
        pool.map(estimate_mad, range(nb_channels))
        # Detect peaks for each channel
        pool.map(detect_peaks, range(nb_channels))

    print(med)

    # # TODO remove old version...
    # # Set up shared memory
    # mem = mp.Array('d', 10)
    # # Launch the reader
    # reader = mp.Process(target=read, args=(mem,))
    # # Launch the workers
    # workers = [mp.Process(target=work, args=(mem, i,)) for i in range(0, nb_workers)]
    # # TODO use a pool of workers instead
    # reader.start()
    # [worker.start() for worker in workers]
    # reader.join()
    # [worker.join() for worker in workers]
    # print(mem[:])
