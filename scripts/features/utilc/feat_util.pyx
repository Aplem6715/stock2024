import pandas as pd
import numpy as np
import cython
cimport numpy as cnp

ctypedef cnp.float32_t DTYPE_f32_t
ctypedef cnp.int32_t DTYPE_i32_t

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_shift_diff(df: pd.DataFrame, cnp.ndarray[DTYPE_f32_t, ndim=1] arr, list shifts, str col_name):
    cdef int shift, i
    cdef str col
    for i in range(len(shifts)):
        shift = shifts[i]
        col = '{}_Shift{}'.format(col_name, shift)
        df[col] = arr
        df[col].diff(shift)
        
    return df