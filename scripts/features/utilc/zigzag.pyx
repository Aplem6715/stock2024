# distutils: language=c++
#cython: language_level=3

# # cython: profile=True
# # cython: linetrace=True
# # cython: binding=True
# # distutils: define_macros=CYTHON_TRACE_NOGIL=1

from typing import List, Tuple
import pandas as pd
import numpy as np
import cython
cimport numpy as cnp
from libcpp.vector cimport vector

cdef int ZIGZAG_UPUP = 1
cdef int ZIGZAG_DOWN = -1
cdef int ZIGZAG_EMPTY = 0

ctypedef cnp.float32_t DTYPE_f32_t
ctypedef cnp.float64_t DTYPE_f64_t
ctypedef cnp.int32_t DTYPE_i32_t


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_zigzag(
    cnp.ndarray[DTYPE_f32_t, ndim=1] high,
    cnp.ndarray[DTYPE_f32_t, ndim=1] low,
    int zz_period,
    int num_shifts):
    cdef:
        size_t i, i_limit
        int shift
        cnp.ndarray[DTYPE_f32_t, ndim=1] buf, last_price
        cnp.ndarray[DTYPE_f32_t, ndim=2] h_shifts, l_shifts
        cnp.ndarray[DTYPE_i32_t, ndim=1] direction
        # list high_pivots, low_pivots

    buf = np.zeros_like(high, np.float32)  # type: np.ndarray
    last_price = np.empty_like(high, np.float32)  # type: np.ndarray
    direction = np.empty_like(high, np.int32)  # type: np.ndarray

    h_shifts = np.zeros(shape=(len(high), num_shifts),
                        dtype=np.float32)  # type: np.ndarray
    l_shifts = np.zeros(shape=(len(high), num_shifts),
                        dtype=np.float32)  # type: np.ndarray

    # 初期化
    i_limit = len(high)
    buf[0] = high[0]
    last_price[0] = high[0]
    direction[0] = ZIGZAG_UPUP

    high_pivots: List[int] = [0]
    low_pivots: List[int] = []

    for i in range(1, i_limit):
        last_price[i] = last_price[i-1]
        direction[i] = direction[i-1]

        is_change_dir = False
        is_modify_price = False

        if direction[i] == ZIGZAG_UPUP:
            min_low = 999999
            for j in range(max(0, i-zz_period), i):
                if low[j] < min_low:
                    min_low = low[j]
            # min_low = low[max(0, i-zz_period):i].min()
            if low[i] < min_low:
                is_change_dir = True
            if last_price[i] < high[i]:
                is_modify_price = True

            if not (is_change_dir and is_modify_price):

                if is_change_dir:
                    buf[i] = low[i]
                    last_price[i] = low[i]
                    direction[i] = ZIGZAG_DOWN
                    low_pivots.append(i)

                if is_modify_price:
                    if len(high_pivots) > 0:
                        # １つ前の頂点価格をEmptyに
                        buf[high_pivots.pop()] = ZIGZAG_EMPTY
                    buf[i] = high[i]
                    last_price[i] = high[i]
                    high_pivots.append(i)

        elif direction[i] == ZIGZAG_DOWN:
            max_high = 0
            for j in range(max(0, i-zz_period), i):
                if high[j] > max_high:
                    max_high = high[j]
            # max_high = high[max(0, i-zz_period):i].max()
            if high[i] > max_high:
                is_change_dir = True
            if last_price[i] > low[i]:
                is_modify_price = True

            if not (is_change_dir and is_modify_price):

                if is_change_dir:
                    buf[i] = high[i]
                    last_price[i] = high[i]
                    direction[i] = ZIGZAG_UPUP
                    high_pivots.append(i)

                if is_modify_price:
                    if len(low_pivots) > 0:
                        # １つ前の頂点価格をEmptyに
                        buf[low_pivots.pop()] = ZIGZAG_EMPTY
                    buf[i] = low[i]
                    last_price[i] = buf[i]
                    low_pivots.append(i)

        for shift in range(num_shifts):
            if len(high_pivots) >= shift+2:
                h_shifts[i][shift] = high[high_pivots[len(high_pivots)-shift-2]]
            # h_shifts[i, 1] = high[high_pivots[-3]]
            # h_shifts[i, 2] = high[high_pivots[-4]]

        for shift in range(num_shifts):
            if len(low_pivots) >= shift+2:
                l_shifts[i][shift] = low[low_pivots[len(low_pivots)-shift-2]]

    return h_shifts, l_shifts, high_pivots, low_pivots

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def get_zigzag_turning(df: pd.DataFrame, cnp.ndarray[DTYPE_f32_t, ndim=1] pivots, cnp.ndarray[DTYPE_i32_t, ndim=1] vertex, int shift_min, int shift_max, str suffix) -> pd.DataFrame:
#    cdef:
#        int shift, i
#        cnp.ndarray[DTYPE_f32_t, ndim=1] data
#    zigzag_times = df.index[pivots > 0]
#    zigzag_price = pivots[pivots > 0]
#    zigzag_dir = vertex[pivots > 0]
#    dx = zigzag_times[zigzag_dir == ZIGZAG_DOWN]
#    ux = zigzag_times[zigzag_dir == ZIGZAG_UPUP]
#    dy = zigzag_price[zigzag_dir == ZIGZAG_DOWN]
#    uy = zigzag_price[zigzag_dir == ZIGZAG_UPUP]
#
#    # 記録したデータを1つのDataFrameにまとめる
#    fdf = pd.DataFrame(index=df.index)
#    # シフトして記録
#    for shift in range(shift_min, shift_max+1):
#        up_col_name = 'ZigzagUp'+str(shift)+'_'+suffix
#        down_col_name = 'ZigzagDown'+str(shift)+'_'+suffix
#        data = np.ndarray(shape=(len(ux)), dtype=np.float32)
#        for i in range(len(ux)):
#            if i < shift:
#                data[i] = 0
#            else:
#                data[i] = uy[i-shift]
#        fdf.loc[ux, up_col_name] = data
#        data = np.ndarray(shape=(len(dx)), dtype=np.float32)
#        for i in range(len(dx)):
#            if i < shift:
#                data[i] = 0
#            else:
#                data[i] = dy[i-shift]
#        fdf.loc[dx, down_col_name] = data
#    fdf = fdf.fillna(method='ffill')
#    return fdf
#    
#@cython.boundscheck(False)
#@cython.wraparound(False)
#def calc_zigzag(df: pd.DataFrame, int zz_period = 12):
#    cdef:
#        size_t i, j, j_lim, i_limit
#        float min_low, max_high
#        cnp.ndarray[DTYPE_i32_t, ndim=1] vertex
#    cdef cnp.ndarray[DTYPE_f32_t, ndim=1] high = df['High'].values
#    cdef cnp.ndarray[DTYPE_f32_t, ndim=1] low = df['Low'].values
#    cdef cnp.ndarray[DTYPE_f32_t, ndim=1] buf = np.zeros_like(high, np.float32)  # type: np.ndarray
#    cdef cnp.ndarray[DTYPE_f32_t, ndim=1] last_price = np.empty_like(high, np.float32)  # type: np.ndarray
#    cdef cnp.ndarray[DTYPE_i32_t, ndim=1] direction = np.empty_like(high, np.int32)  # type: np.ndarray
#    # 初期化
#    i_limit = len(df)
#    buf[0] = high[0]
#    last_price[0] = high[0]
#    direction[0] = ZIGZAG_UPUP
#    for i in range(1, i_limit):
#        last_price[i] = last_price[i-1]
#        direction[i] = direction[i-1]
#        is_change_dir = False
#        is_modify_price = False
#        if direction[i] == ZIGZAG_UPUP:
#            min_low = 10000
#            j_lim = i-zz_period
#            if j_lim < 0:
#                j_lim = 0
#            for j in range(j_lim, i):
#                if low[j] < min_low:
#                    min_low = low[j]
#            if low[i] < min_low:
#                is_change_dir = True
#            if last_price[i] < high[i]:
#                is_modify_price = True
#            if is_change_dir and is_modify_price:
#                continue
#            if is_change_dir:
#                buf[i] = low[i]
#                last_price[i] = low[i]
#                direction[i] = ZIGZAG_DOWN
#            if is_modify_price:
#                buf[i] = high[i]
#                last_price[i] = high[i]
#                for j in range(i-1, 0, -1):
#                    if buf[j] != 0:
#                        buf[j] = 0
#                        break
#            continue
#        elif direction[i] == ZIGZAG_DOWN:
#            max_high = -10000
#            j_lim = i-zz_period
#            if j_lim < 0:
#                j_lim = 0
#            for j in range(j_lim, i):
#                if high[j] > max_high:
#                    max_high = high[j]
#            if high[i] > max_high:
#                is_change_dir = True
#            if last_price[i] > low[i]:
#                is_modify_price = True
#            if is_change_dir and is_modify_price:
#                continue
#            if is_change_dir:
#                buf[i] = high[i]
#                last_price[i] = high[i]
#                direction[i] = ZIGZAG_UPUP
#            if is_modify_price:
#                buf[i] = low[i]
#                last_price[i] = buf[i]
#                for j in range(i-1, 0, -1):
#                    if buf[j] != 0:
#                        buf[j] = 0
#                        break
#    vertex = direction.copy()
#    vertex[buf == 0] = ZIGZAG_EMPTY
#    return buf, vertex

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_peak(df: pd.DataFrame, cnp.ndarray[DTYPE_f32_t, ndim=1] close, cnp.ndarray[DTYPE_f32_t, ndim=1] high, cnp.ndarray[DTYPE_f32_t, ndim=1] low, unsigned int period, list shifts: List[int], str suffix: str):
    cdef list peak_high = []
    cdef list peak_low = []
    cdef size_t i_limit = len(high)
    cdef size_t shift_idx_limit = len(shifts)

    cdef cnp.ndarray[DTYPE_f32_t, ndim=2] high_buf = np.zeros(shape=(len(high), len(shifts)), dtype=np.float32)
    cdef cnp.ndarray[DTYPE_f32_t, ndim=2] low_buf = np.zeros(shape=(len(low), len(shifts)), dtype=np.float32)

    cdef DTYPE_f32_t current
    cdef unsigned int cursor, i, offset, i_shift, shift

    i = 0
    while i < i_limit:
        current = high[i]

        cursor = i-period
        found_peak = True
        if i+period < i_limit:
            while cursor <= i+period:
                if high[cursor] > current:
                    found_peak = False
                    break
                cursor += 1
        else:
            found_peak = False

        if found_peak:
            # 新しいピーク出現後，periodの間はそのピークを無視する
            # （出現ピークは未来period分にピークがない前提＝未来データを参照しているため）
            for offset in range(period):
                for i_shift in range(shift_idx_limit):
                    if len(peak_high) <= shifts[i_shift]:
                        continue
                    high_buf[i+offset,
                             i_shift] = peak_high[-(shifts[i_shift]+1)]
            peak_high.append(current)
            # ここiがpeakなので，右にi+periodまでの区間には新しいpeakは出現しない
            i += period-1
            if i >= i_limit:
                break
        else:
            for i_shift in range(shift_idx_limit):
                if len(peak_high) <= shifts[i_shift]:
                    continue
                high_buf[i, i_shift] = peak_high[-(shifts[i_shift]+1)]

        i += 1

    i = 0
    while i < i_limit:
        current = low[i]

        cursor = i-period
        found_peak = True
        if i+period < i_limit:
            while cursor <= i+period:
                if low[cursor] < current:
                    found_peak = False
                    break
                cursor += 1
        else:
            found_peak = False

        if found_peak:
            # 新しいピーク出現後，periodの間はそのピークを無視する
            # （出現ピークは未来period分にピークがない前提＝未来データを参照しているため）
            for offset in range(period):
                for i_shift in range(shift_idx_limit):
                    if len(peak_low) <= shifts[i_shift]:
                        continue
                    low_buf[i+offset, i_shift] = peak_low[-(shifts[i_shift]+1)]
            peak_low.append(current)
            # ここiがpeakなので，右にi+periodまでの区間には新しいpeakは出現しない
            i += period-1
            if i >= i_limit:
                break
        else:
            for i_shift in range(shift_idx_limit):
                if len(peak_low) <= shifts[i_shift]:
                    continue
                low_buf[i, i_shift] = peak_low[-(shifts[i_shift]+1)]
        i += 1

    high_cols = ['PeakHigh{}_Shift{}_{}'.format(
        period, shift, suffix) for shift in shifts]
    low_cols = ['PeakLow{}_Shift{}_{}'.format(
        period, shift, suffix) for shift in shifts]
    df[high_cols] = high_buf - close.reshape(-1, 1)
    df[low_cols] = low_buf - close.reshape(-1, 1)

    return df

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef peek_shifted(df: pd.DataFrame, peek_results: Tuple[List, List, List, List], int shift_min, int shift_max, int period, str suffix):
#     cdef vector[float] peek_high
#     cdef vector[int] idx_high
#     cdef vector[float] peek_low
#     cdef vector[int] idx_low
# 
#     cdef list new_cols
#     cdef str col
#     cdef int shift, i, idx, cursor
#     cdef float current
# 
#     idx_high, peek_high, idx_low, peek_low = peek_results
#     new_cols = []
# 
#     cdef size_t df_len = len(df)
# 
#     cdef cnp.ndarray[DTYPE_f32_t, ndim=1] high_buf = np.zeros(df_len, dtype=np.float32)
#     cdef cnp.ndarray[DTYPE_f32_t, ndim=1] low_buf = np.zeros(df_len, dtype=np.float32)
# 
#     cdef size_t idx_high_limit = len(idx_high)
#     cdef size_t idx_low_limit = len(idx_low)
# 
#     # シフト1未満だとリークの可能性がある？
#     for shift in range(shift_min, shift_max+1):
#         col = 'HighPeek{}_Shift{}_{}'.format(period, shift, suffix)
#         new_cols.append(col)
#         df[col] = np.nan
#         if shift < idx_high_limit:
#             # cursor = 0
#             # current = 0
#             # for idx in range(df_len):
#             #     if cursor < idx_high_limit and idx_high[cursor] == idx:
#             #         if cursor-shift > 0:
#             #             current = peek_high[cursor-shift]
#             #         cursor += 1
#             #     high_buf[idx] = current
#             df.loc[df.index[idx_high], col] = [np.nan]*shift + peek_high[:-shift]
# 
# 
#         col = 'LowPeek{}_Shift{}_{}'.format(period, shift, suffix)
#         new_cols.append(col)
#         df[col] = np.nan
#         if shift < idx_low_limit:
#             # cursor = 0
#             # current = 0
#             # for idx in range(df_len):
#             #     if cursor < idx_low_limit and idx_low[cursor] == idx:
#             #         if cursor-shift > 0:
#             #             current = peek_low[cursor-shift]
#             #         cursor += 1
#             #     low_buf[idx] = current
#             df.loc[df.index[idx_low], col] = [np.nan]*shift + peek_low[:-shift]
# 
#     df = df[new_cols].fillna(method='ffill')
# 
#     return df
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def peek_completed(out_df: pd.DataFrame, cnp.ndarray[DTYPE_f32_t, ndim=1] high, cnp.ndarray[DTYPE_f32_t, ndim=1] low, int period, int shift_min, int shift_max, str suffix):
#     return peek_shifted(
#         out_df,
#         calc_peek(high, low, period),
#         shift_min,
#         shift_max,
#         period,
#         suffix
#     )

    # return profile(peek_shifted)(
    #     out_df,
    #     profile(calc_peek)(high, low, period),
    #     shift_min,
    #     shift_max,
    #     period,
    #     suffix
    # )
