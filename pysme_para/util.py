import numpy as np
import spectres, scipy
import pandas as pd

def log_wav_raster(wave):
    a = wave[1] / wave[0]
    length = int(np.ceil(np.log10(wave[-1]/wave[0]) / np.log10(a)))
    return wave[0] * a ** np.arange(length)

def log_wav2rv(wav):
    '''
    Convert the log wavelangth scale to RV.

    :param wav:
        A numpy.array, containing the logarithmic wavelength scale of the spectra.
    :return:
        The RV array. 
    '''
    
    # check if the a value is constant for all the pixels
    if not np.all(wav):
        raise ValueError('The wavelength array is not in constant log scale.')
        
    c = 299792458.0 / 1000
    a = wav[1] / wav[0]
    if len(wav) % 2 == 1:
        # Odd case
        multiplicative_shift = a**(np.arange(1, (len(wav)-1) // 2 + 1) / 1)
        rv_array = c * (a**np.arange(1, len(wav) // 2 + 1) - 1)
        rv_array = c * (multiplicative_shift**2 - 1) / (multiplicative_shift**2 + 1)
        rv_array = np.concatenate([-rv_array[::-1], [0], rv_array])
    else:
        # Even case
        wav_temp = np.append(wav, wav[0] * a**len(wav))
        rv_array = log_wav2rv(wav_temp)[:-1]
            
    return rv_array

def interpolation_quadratic_dcf(x_vals, y_vals):
    """
    Use quadratic interpolation to find the sub-pixel position of the peak of the CCF. 
    This is Dominic's version which uses analytic solution, modified by Mingjie to do the fitting in lambda scale.

    :param x_vals:
        If the CCF is y(x), this is the array of the x values of the data points supplied to interpolate.
    :param y_vals:
        If the CCF is y(x), this is the array of the y values of the data points supplied to interpolate.
    :return:
        Our best estimate of the position of the peak.
    """

    assert len(x_vals) == 3  # This analytic solution only works with three input points
    assert x_vals[1] == 0  # Three points must be centred around x==0

    def quadratic_peak_x(p):
        return -p[1] / (2 * p[0])

    p0 = (y_vals[0] + y_vals[2]) / (x_vals[0]**2 - x_vals[0]*x_vals[2]) + y_vals[1] / (x_vals[0]*x_vals[2])

    p1 = -(y_vals[0]*x_vals[2]) / (x_vals[0]**2 - x_vals[0]*x_vals[2]) 
    p1 -= y_vals[1]*(x_vals[0]+x_vals[2]) / (x_vals[0]*x_vals[2])
    p1 -= x_vals[0]*y_vals[2] / (x_vals[2]**2 - x_vals[0]*x_vals[2])
    
    p2 = y_vals[1]

    peak_x = quadratic_peak_x(p=(p0, p1, p2))

    return p0, p1, p2, peak_x

def measure_rv_from_ccf(rv_raster, cross_correlation, interpolation_pixels=3):
    
    max_position = np.where(cross_correlation == np.max(cross_correlation))[0][0]
    N = len(rv_raster)
    
    # Now make three points which straddle the maximum
    x_min = max_position - interpolation_pixels // 2
    x_max = x_min + interpolation_pixels - 1
    x_indices = np.array(range(x_min, x_max + 1))
    x_vals = rv_raster[x_indices]
    y_vals = cross_correlation[x_indices]

    # Put peak close to zero for numerical stability
    x_vals = x_vals - rv_raster[max_position]
    y_peak = y_vals[1]

    # Do interpolation
    if len(x_vals) == 3:
        dcf_para = interpolation_quadratic_dcf(x_vals=x_vals, y_vals=y_vals)
        peak_x = dcf_para[-1] + rv_raster[max_position]
        rv_err = np.sqrt(- 1 / N * y_peak / (dcf_para[0] * 2) * (1 - y_peak**2) / y_peak**2)
    return peak_x, rv_err

def measure_rv(wav_template, flux_template, wav_spec, flux_spec, ccf_out=False):
    log_wave = log_wav_raster(wav_template)
    flux_rv_template = spectres.spectres(log_wave, wav_template, flux_template, fill=1)
    flux_rv_template -= np.nanmean(flux_rv_template)
    flux_rv_template /= np.nanstd(flux_rv_template)
    flux_rv_template[np.isnan(flux_rv_template)] = 0
    rv_raster = log_wav2rv(log_wave)

    flux_rv_obs = spectres.spectres(log_wave, wav_spec, flux_spec, fill=1)
    flux_rv_obs -= np.nanmean(flux_rv_obs)
    flux_rv_obs/= np.nanstd(flux_rv_obs)
    flux_rv_obs[np.isnan(flux_rv_obs)] = 0
    ccf = scipy.signal.correlate(flux_rv_obs, flux_rv_template, mode='same') / len(log_wave)
    star_rv, rv_err = measure_rv_from_ccf(rv_raster, ccf)
    if ccf_out:
        return star_rv, rv_err, rv_raster, ccf
    else:
        return star_rv, rv_err

def get_vmac(Teff, sp_type):
    '''
    Get the empirical Vmac from Gray 1984 (dwarf) and Gray 1982 (giant).
    '''
    Teff = np.asarray(Teff)
    if sp_type == 'dwarf':
        result = np.where(Teff > 4873, 3.95 * Teff/1000 - 19.25, 0)
    elif sp_type == 'giant':
        result = np.where(Teff < 5500, 7 - (5.5 - Teff/1000)**2.6, 7 + (Teff/1000 - 5.5)**2.6)
    result[result < 0] = 0
    return result

def sigma_clip(data, sigma=3.0, max_iters=None, return_mask=False):
    """
    对数据进行 Sigma Clipping 处理。

    参数:
    - data: 输入数据 (NumPy 数组)
    - sigma: sigma clipping 阈值 (默认 3.0)
    - max_iters: 最大迭代次数 (如果为 None，则直到收敛)
    - return_mask: 如果为 True，则返回掩码数组，而不是裁剪后的数据

    返回:
    - 经过 sigma clipping 处理后的数据 或 掩码数组
    """
    data = np.asarray(data)  # 确保是 NumPy 数组
    mask = np.ones_like(data, dtype=bool)  # 初始化掩码（全部为 True）
    
    previous_mask = np.zeros_like(mask)  # 记录上一次的掩码状态
    iteration = 0

    while not np.array_equal(mask, previous_mask):  # 直到掩码不再变化
        if max_iters is not None and iteration >= max_iters:
            break  # 达到最大迭代次数
        
        previous_mask = mask.copy()  # 记录当前掩码
        
        mean = np.nanmean(data[mask])  # 计算均值（仅对未被剪除的数据）
        std = np.nanstd(data[mask])  # 计算标准差
        
        # 更新掩码：剔除超出 sigma*std 范围的数据
        mask = np.abs(data - mean) <= sigma * std

        iteration += 1  # 迭代计数

    if return_mask:
        return mask  # 返回布尔掩码
    else:
        return data[mask]  # 返回剔除异常值后的数据

def detect_peak_regions(data, peak_thresh, valley_thresh):
    """
    当某个峰值高于 peak_thresh，就从该峰向两边扩展直到数据低于 valley_thresh。
    输出一个 bool mask：峰值范围为 False，其他为 True。
    """
    data = np.asarray(data)
    mask = np.ones_like(data, dtype=bool)  # 初始化全部为 True
    
    i = 0
    while i < len(data):
        if data[i] > peak_thresh:
            # 向左扩展
            left = i
            while left > 0 and data[left] > valley_thresh:
                left -= 1
            # 向右扩展
            right = i
            while right < len(data) - 1 and data[right] > valley_thresh:
                right += 1
            # 设置峰值区域为 False
            mask[left+1:right] = False
            i = right  # 跳过这个区域，加快处理
        else:
            i += 1

    return mask

def generate_consistency_mask(wave, flux_obs, flux_syn, line_mask, sigma_thresh=2, valley_thresh=0.02, chunk=True, chunk_width=200, smooth=True, window_width=200, smooth_thresh=0.05):
    """
    Generate a boolean mask to identify inconsistent regions between observed and synthetic spectra.

    This function first computes the difference between observed and synthetic fluxes, and applies the given line_mask
    to select the spectral region of interest. For each chunk of the masked wavelength array (with chunk size in Angstroms),
    it identifies pixels where the absolute difference exceeds sigma_thresh times the standard deviation within the chunk,
    and extends the mask to the left and right until the difference drops below valley_thresh or a wavelength gap is detected
    (i.e., np.abs(wave[j] - wave[j-1]) > 3 * del_lambda, where del_lambda is the median wavelength step).

    Optionally, a rolling median smoothing can be applied to the full difference array, and regions where the smoothed
    difference exceeds smooth_thresh are also masked.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array (in Angstroms).
    flux_obs : np.ndarray
        Observed flux array.
    flux_syn : np.ndarray
        Synthetic flux array.
    line_mask : np.ndarray of bool
        Boolean mask indicating the spectral region to process.
    sigma_thresh : float, optional
        Threshold in units of standard deviation to identify significant deviations (default: 2).
    valley_thresh : float, optional
        Threshold for extending the mask from a peak (default: 0.02).
    chunk : bool, optional
        Whether to process in wavelength chunks (default: True). If False, process the entire region as one chunk.
    chunk_width : float, optional
        Width of each chunk in Angstroms (default: 200).
    smooth : bool, optional
        Whether to apply rolling median smoothing to the difference array and mask regions with large smoothed residuals (default: True).
    window_width : int, optional
        Window size (in pixels) for rolling median smoothing (default: 200).
    smooth_thresh : float, optional
        Threshold for the absolute value of the smoothed difference to be masked (default: 0.05).

    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask of the same length as the input arrays, where True indicates inconsistent regions.
    smooth_arr : np.ndarray, optional
        The rolling median smoothed difference array (only returned if smooth=True).
    """
    diff = flux_obs - flux_syn
    wave_line = wave[line_mask]
    diff_line = diff[line_mask]
    mask_line = np.zeros_like(diff_line, dtype=bool)
    del_lambda = np.nanmedian(np.diff(wave_line))

    if chunk:
        wave_min = wave_line[0]
        wave_max = wave_line[-1]
        n_chunk = int(np.ceil((wave_max - wave_min) / chunk_width))
        for i in range(n_chunk):
            chunk_left = wave_min + i * chunk_width
            chunk_right = chunk_left + chunk_width
            indices = np.where((wave_line >= chunk_left) & (wave_line < chunk_right))[0]
            if len(indices) == 0:
                continue
            diff_chunk = diff_line[indices]
            wave_chunk = wave_line[indices]
            std_diff = np.nanstd(diff_chunk)
            big_peaks = np.where(np.abs(diff_chunk - np.mean(diff_chunk)) > sigma_thresh * std_diff)[0]
            for idx in big_peaks:
                # Extend to the left
                j = idx
                while j > 0 and np.abs(diff_chunk[j]) >= valley_thresh:
                    if np.abs(wave_chunk[j] - wave_chunk[j-1]) > 3 * del_lambda:
                        break
                    mask_line[indices[j]] = True
                    j -= 1
                mask_line[indices[j]] = True  # 包含起点
                # Extend to the right
                j = idx + 1
                while j < len(diff_chunk) and np.abs(diff_chunk[j]) >= valley_thresh:
                    if np.abs(wave_chunk[j] - wave_chunk[j-1]) > 3 * del_lambda:
                        break
                    mask_line[indices[j]] = True
                    j += 1
    else:
        std_diff = np.std(diff_line)
        big_peaks = np.where(np.abs(diff_line - np.mean(diff_line)) > sigma_thresh * std_diff)[0]
        for idx in big_peaks:
            # Extend to the left
            j = idx
            while j > 0 and np.abs(diff_line[j]) >= valley_thresh:
                if np.abs(wave_line[j] - wave_line[j-1]) > 3 * del_lambda:
                    break
                mask_line[j] = True
                j -= 1
            mask_line[j] = True  # 包含起点
            # Extend to the right
            j = idx + 1
            while j < len(diff_line) and np.abs(diff_line[j]) >= valley_thresh:
                if np.abs(wave_line[j] - wave_line[j-1]) > 3 * del_lambda:
                    break
                mask_line[j] = True
                j += 1

    mask = np.zeros_like(diff, dtype=bool)
    mask[np.where(line_mask)[0]] = ~mask_line
    
    # 平滑处理
    if smooth:
        smooth_arr = pd.Series(diff).rolling(window=window_width, center=True).median().to_numpy()
        smooth_mask = np.abs(smooth_arr) <= smooth_thresh
        smooth_mask[np.isnan(smooth_arr)] = False
        mask &= smooth_mask
    
        return mask, smooth_arr
    else:
        return mask

def get_false_regions_wavelengths(wavelengths, mask):
    """
    根据布尔 mask 提取所有 mask 为 False 的波长段范围。
    每段的起止点由 True/False 转换点的波长平均值得到。
    
    返回：[[start1, end1], [start2, end2], ...]
    """
    wavelengths = np.asarray(wavelengths)
    mask = np.asarray(mask)

    # 检测边界转换点：True→False 为起点，False→True 为终点
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == -1)[0]  # True → False
    ends = np.where(diff == 1)[0]     # False → True

    # 边界特殊情况：开头或结尾是 False 的情况
    if not mask[0]:
        starts = np.insert(starts, 0, -1)
    if not mask[-1]:
        ends = np.append(ends, len(mask) - 1)

    # 计算平均波长作为边界
    regions = []
    for s, e in zip(starts, ends):
        wl_start = (wavelengths[s] + wavelengths[s + 1]) / 2 if s >= 0 else wavelengths[0]
        wl_end   = (wavelengths[e] + wavelengths[e + 1]) / 2 if e + 1 < len(wavelengths) else wavelengths[-1]
        regions.append([wl_start, wl_end])

    return regions

# 修改后的函数：line_to_range 和 range_to_line
def line_to_range(line_list, vmic, vmac, vsini, R, extend_ratio=2):
    """
    Convert a list of line center wavelengths to merged lambda ranges with broadening.
    """
    v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5 / R)**2)  # km/s
    delta_lambda = [l * v_broad / 3e5 for l in line_list]
    raw_ranges = [(l - extend_ratio*dl, l + extend_ratio*dl) for l, dl in zip(line_list, delta_lambda)]

    # 合并重叠或相邻的区间
    if not raw_ranges:
        return []
    raw_ranges.sort()
    merged_ranges = [raw_ranges[0]]
    for start, end in raw_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        if start <= last_end:  # overlap
            merged_ranges[-1] = (last_start, max(last_end, end))  # merge
        else:
            merged_ranges.append((start, end))
    return merged_ranges

def range_to_line(lambda_ranges, line_candidates):
    """
    Select line candidates that fall within any of the provided lambda ranges.
    """
    selected = []
    for l0 in line_candidates:
        for l_start, l_end in lambda_ranges:
            if l_start <= l0 <= l_end:
                selected.append(l0)
                break
    return selected

def range_to_pixel(lambda_ranges, lambda_pix):
    mask = np.full(len(lambda_pix), False)
    for start, end in lambda_ranges:
        mask |= (lambda_pix >= start) & (lambda_pix <= end)
    return mask

# def pixel_to_range(lambda_pix, pixel_mask, lambda_pix, vbroad=0):
#     ranges = []
#     in_range = False
#     for i, use in enumerate(pixel_mask):
#         if use and not in_range:
#             in_range = True
#             start = lambda_pix[i]
#         elif not use and in_range:
#             end = lambda_pix[i - 1]
#             ranges.append((start, end))
#             in_range = False
#     if in_range:
#         ranges.append((start, lambda_pix[-1]))
#     return ranges

def pixel_to_range(lambda_pix, pixel_mask, vmic, vmac, vsini, R):
    """
    Convert pixel mask to lambda ranges, removing those narrower than 2 * delta_lambda.
    Also returns the updated pixel mask with short ranges removed.
    """
    ranges = []
    new_mask = pixel_mask.copy()
    in_range = False

    for i, use in enumerate(pixel_mask):
        if use and not in_range:
            in_range = True
            start_idx = i
            start = lambda_pix[i]
        elif not use and in_range:
            end_idx = i - 1
            end = lambda_pix[end_idx]
            center = (start + end) / 2
            v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5 / R)**2)
            delta_lambda = center * v_broad / 3e5
            if (end - start) >= 2 * delta_lambda:
                ranges.append((start, end))
            else:
                new_mask[start_idx:end_idx + 1] = False
            in_range = False

    if in_range:
        end_idx = len(lambda_pix) - 1
        end = lambda_pix[end_idx]
        center = (start + end) / 2
        v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5 / R)**2)
        delta_lambda = center * v_broad / 3e5
        if (end - start) >= 2 * delta_lambda:
            ranges.append((start, end))
        else:
            new_mask[start_idx:end_idx + 1] = False

    return ranges, new_mask

def chunk_array(arr, chunk_number):
    """
    Split a 1D numpy array into approximately `chunk_number` chunks.
    If the last chunk is shorter than or equal to the average chunk length,
    it is merged into the previous chunk.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array to split.
    chunk_number : int
        Desired number of chunks.

    Returns
    -------
    list of np.ndarray
        A list of arrays split from the input. Last one may be merged if too short.
    """
    arr = np.asarray(arr)
    n = len(arr)
    avg_chunk_size = int(np.ceil(n / chunk_number))

    # Initial split
    chunks = [arr[i:i + avg_chunk_size] for i in range(0, n, avg_chunk_size)]

    # If last chunk is too short, merge it
    if len(chunks) > 1 and len(chunks[-1]) <= avg_chunk_size:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()

    return chunks

def spectral_segments(wave, factor=3.0):
    '''
    Get the range of chunked spectra according to the gaps in wavelength array.
    '''

    wave = np.asarray(wave)
    d = np.diff(wave)
    # 以中位步长的若干倍作为“断点”阈值（避免噪点影响）
    thr = factor * np.median(d[np.isfinite(d)])
    # 断点位置索引（gap 大于阈值）
    cuts = np.where(d > thr)[0]
    # 把区间端点拼起来
    starts = np.r_[0, cuts + 1]
    ends   = np.r_[cuts, len(wave) - 1]
    return [(wave[s], wave[e]) for s, e in zip(starts, ends)]

from collections.abc import Iterable

def agg_ratio(val):
    """blending_ratio 的聚合：list -> 1/sum(1/r)；标量 -> float"""
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        arr = np.array(list(val), dtype=float)
        arr = np.where(arr <= 0, np.nan, arr)        # 非法或0用 NaN
        inv_sum = np.nansum(1.0 / arr)
        return float(1.0 / max(inv_sum, 1e-12))       # 防止除0
    return float(val)

def filter_lines_in_spectrum(df, wave, wl_col='wlcent', gap_factor=5.0):
    """
    Remove lines whose central wavelength is not covered by
    the given spectrum wavelength array (with possible gaps).

    Parameters
    ----------
    df : pandas.DataFrame
        Table of lines. Must contain a wavelength column (default 'wlcent').
    wave : array-like
        1D wavelength array of the spectrum. Can contain gaps and be unsorted.
    wl_col : str, default 'wlcent'
        Column name in df for the line central wavelength.
    gap_factor : float, default 5.0
        A gap is defined where the spacing between two neighbouring pixels
        is larger than gap_factor * median(delta_lambda). Increase this if
        your sampling is very irregular.

    Returns
    -------
    df_sel : pandas.DataFrame
        Subset of df with lines whose wlcent falls inside any covered region.
    """

    # ---- 1. Prepare wavelength array & identify gaps ----
    wave = np.asarray(wave, dtype=float)
    wave = np.sort(wave)               # ensure monotonic

    if wave.size < 2:
        # trivial case: only one pixel -> only lines exactly at this wavelength
        mask = (df[wl_col].values == wave[0])
        return df.loc[mask].copy()

    diffs = np.diff(wave)
    median_step = np.median(diffs)

    # define gaps where spacing is much larger than the typical step
    gap_mask = diffs > gap_factor * median_step
    # indices *after* which a new segment starts
    gap_indices = np.where(gap_mask)[0]

    # ---- 2. Build coverage segments [λ_start, λ_end] ----
    segments = []
    start_idx = 0
    for gi in gap_indices:
        end_idx = gi  # inclusive
        segments.append((wave[start_idx], wave[end_idx]))
        start_idx = gi + 1
    # last segment
    segments.append((wave[start_idx], wave[-1]))

    # ---- 3. Filter lines by checking whether wlcent is in any segment ----
    wl_lines = df[wl_col].values.astype(float)
    line_mask = np.zeros(df.shape[0], dtype=bool)

    for (lam_min, lam_max) in segments:
        line_mask |= (wl_lines >= lam_min) & (wl_lines <= lam_max)

    return df.loc[line_mask].copy()