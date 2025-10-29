from scipy import stats
import numpy as np
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count

def mode(x: NDArray) -> NDArray:
    return stats.mode(x, axis=2, keepdims=False).mode

def mode_multiprocessed(arr: NDArray, num_processes: int = cpu_count()):
    '''
    multiprocess computation of mode along 3rd axis
    '''
    
    # create iterable
    chunk_size = int(arr.shape[0] / num_processes) 
    chunks = [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)] 

    # distribute work
    with Pool(processes=num_processes) as pool:
        res = pool.map(mode, chunks)

    # reshape result
    out = np.vstack(res)
    out.reshape(arr.shape[:-1])

    return out
