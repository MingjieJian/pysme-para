import numpy as np
import pandas as pd

from pysme.synthesize import synthesize_spectrum

from pqdm.processes import pqdm
from multiprocessing import cpu_count
from copy import deepcopy

import warnings

from tqdm import tqdm

from contextlib import redirect_stdout

from joblib import Parallel, delayed

warnings.simplefilter("ignore")

def find_strong_lines(line_list, strong_line_element=['H', 'Mg', 'Ca', 'Na']):
    '''
    Find strong lines from the elements. This is not recommeded to use since it is limited to fixed elements.
    '''
    
    strong_indices = line_list['wlcent'] < 0
    species = line_list._lines['species'].apply(lambda x: x.split(' ')[0])
    for ele in strong_line_element:
        strong_indices = strong_indices | (species.values == ele)
            
    return line_list[~strong_indices], line_list[strong_indices]

def get_cdepth_range(sme, line_list, N_line_chunk=2000, parallel=False, n_jobs=5, pysme_out=False):
    '''
    Get the central depth and wavelength range of a line list.
    '''
    
    # Remove dupliate lines from the line list
    line_list._lines = line_list._lines.drop_duplicates(subset=['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw'])

    # Decide how many chunks to be divided
    N_chunk = int(np.ceil(len(line_list) / N_line_chunk))

    # Divide the line list to sub line lists
    sub_line_list = [line_list[N_line_chunk*i:N_line_chunk*(i+1)] for i in range(N_chunk)]

    if sum(len(item) for item in sub_line_list) != len(line_list):
        raise ValueError
    
    sub_sme = []
    for i in tqdm(range(N_chunk)):
        sub_sme.append(deepcopy(sme))
        sub_sme[i].linelist = sub_line_list[i]
        sub_sme[i].wave = np.arange(5000, 5010, 1)
        if not parallel:
            if pysme_out:
                sub_sme[i] = synthesize_spectrum(sub_sme[i])
            else:
                with redirect_stdout(open(f"/dev/null", 'w')):
                    sub_sme[i] = synthesize_spectrum(sub_sme[i])
    # return sub_sme
    if parallel:
        if pysme_out:
            sub_sme = pqdm(sub_sme, synthesize_spectrum, n_jobs=n_jobs)
        else:
            with redirect_stdout(open(f"/dev/null", 'w')):
                sub_sme = pqdm(sub_sme, synthesize_spectrum, n_jobs=n_jobs)

    # Get the central depth
    for i in range(N_chunk):
        sub_line_list[i] = sub_sme[i].linelist

    stack_linelist = deepcopy(sub_line_list[0])
    stack_linelist._lines = pd.concat([ele._lines for ele in sub_line_list])

    # Remove
    if len(stack_linelist) != len(line_list):
        raise ValueError
    for column in ['central_depth', 'line_range_s', 'line_range_e']:
        if column in line_list.columns:
            line_list._lines = line_list._lines.drop(column, axis=1)
    line_list._lines = pd.merge(line_list._lines, stack_linelist._lines[['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw', 'central_depth', 'line_range_s', 'line_range_e']], on=['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw'], how='left')

    # Manually change the depth of all H 1 lines to 1, to include them back.
    line_list._lines.loc[line_list['species'] == 'H 1', 'central_depth'] = 1 

    # Manually change the 2000 line_range to 0.01.
    indices = np.abs(line_list['line_range_e'] - line_list['line_range_s']-2000) < 0.1
    line_list._lines.loc[indices, 'line_range_s'] = line_list._lines.loc[indices, 'wlcent']-0.005
    line_list._lines.loc[indices, 'line_range_e'] = line_list._lines.loc[indices, 'wlcent']+0.005

    # Write the stellar parameters used here to the line list
    line_list.cdepth_range_paras = [sme.teff, sme.logg, sme.monh, sme.vmic]

    return line_list


def _batch_synth_line(sme, line_list, strong_list=None, strong_line_element=['H', 'Mg', 'Ca', 'Na'], wav_range=2, N_line_chunk=2000, line_margin=2, strong_line_margin=100, parallel=False, n_jobs=5, pysme_out=False):
    '''
    The function to synthize the spectra using pysme in batch, according to the number of lines. This would work faster than doing the whole spectra at once.
    For the synthesize accuracy, the code will separate the strong lines from the line list if strong_line is None, and
    always include t

    batcvh_mode : str, defaule 'line'
        The mode for performing batch synthesis. If 'line', then the spectra will be divided according to the number of lines. If 'wave' then will be divided according to wavelength range. 
   
    Example for the pre-process:
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = 5026, 3.62, 0.2, 1.17, 0, 7.55
    sme.iptype = 'gauss'
    sme.ipres = 50000
    sme.wave = np.arange(wav_s, wav_end, 0.05)
    '''

    wav_s, wav_end = np.min(sme.wave), np.max(sme.wave)
    
    # Remove dupliate lines from the line list
    line_list._lines = line_list._lines.drop_duplicates(subset=['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw'])

    # Find the strong lines if not provided
    if strong_list is None:
        line_list, strong_list = find_strong_lines(line_list, strong_line_element=strong_line_element)
    
    # Crop the line_list to target wavelength
    indices = (line_list['wlcent'] >= wav_s - line_margin) & (line_list['wlcent'] <= wav_end + line_margin)
    line_list = line_list[indices]

    # Decide how many chunks to be divided
    N_chunk = int(np.ceil(len(line_list) / N_line_chunk))

    # Divide the line list to sub line lists
    sub_line_list = [line_list[N_line_chunk*i:N_line_chunk*(i+1)] for i in range(N_chunk)]
    sub_wave_range = [[line_list_single.wlcent[0], line_list_single.wlcent[-1]] for line_list_single in sub_line_list]
    sub_wave_range = [ele for ele in sub_wave_range if not((ele[1] <= wav_s) or (ele[0] >= wav_end))]
    for i in range(len(sub_wave_range)-1):
        sub_wave_range[i+1][0] = sub_wave_range[i][1]
    sub_wave_range[0][0] = sme.wave[0][0] - 1e-3
    sub_wave_range[-1][-1] = sme.wave[0][-1] + 1e-3

    # Remove the wave ranges with no sme.wave inside 
    sub_wave_range = [ele for ele in sub_wave_range if len(sme.wave[0][(sme.wave[0] >= ele[0]) & (sme.wave[0] < ele[1])]) > 0]
    
    # Combine the sub_wave_range of length <=2 to the previous one.
    i = len(sub_wave_range) - 2  # 开始检查的索引
    while i >= 0:
        if len(sme.wave[0][(sme.wave >= sub_wave_range[i+1][0]) & (sme.wave < sub_wave_range[i+1][1])]) <= 2:  # 检查下一个元素的长度
            # 合并当前元素和下一个元素
            sub_wave_range[i] = np.array([sub_wave_range[i][0], sub_wave_range[i+1][1]])
            # 从列表中移除下一个元素
            del sub_wave_range[i+1]
        i -= 1  

    sub_line_list = []
    for (wav_start, wav_end) in sub_wave_range:
        line_list_sub = line_list[(line_list['wlcent'] >= wav_start-line_margin) & (line_list['wlcent'] <= wav_end+line_margin)]
        strong_list_sub = strong_list[(strong_list['wlcent'] >= wav_start-strong_line_margin) & (strong_list['wlcent'] <= wav_end+strong_line_margin)]
        full_list_sub = deepcopy(strong_list_sub)
        full_list_sub._lines = pd.concat([strong_list_sub._lines, line_list_sub._lines]).sort_values('wlcent').reset_index(drop=True)
        sub_line_list += [full_list_sub]
    
    N_chunk = len(sub_wave_range)

    sub_sme = []
    for i in tqdm(range(N_chunk)):
        sub_sme.append(deepcopy(sme))
        sub_sme[i].linelist = sub_line_list[i]
        sub_sme[i].wave = sme.wave[(sme.wave >= sub_wave_range[i][0]) & (sme.wave < sub_wave_range[i][1])]
        if not parallel:
            if pysme_out:
                sub_sme[i] = synthesize_spectrum(sub_sme[i])
            else:
                with redirect_stdout(open(f"/dev/null", 'w')):
                    sub_sme[i] = synthesize_spectrum(sub_sme[i])

    if parallel:
        if pysme_out:
            sub_sme = pqdm(sub_sme, synthesize_spectrum, n_jobs=n_jobs)
        else:
            with redirect_stdout(open(f"/dev/null", 'w')):
                sub_sme = pqdm(sub_sme, synthesize_spectrum, n_jobs=n_jobs)

    # Merge the spectra
    wav, flux = np.concatenate([sub_sme[i].wave[0] for i in range(N_chunk)]), np.concatenate([sub_sme[i].synth[0] for i in range(N_chunk)])

    # Get the central depth
    for i in range(N_chunk):
        sub_line_list[i] = sub_sme[i].linelist

    stack_linelist = deepcopy(sub_line_list[0])
    stack_linelist._lines = pd.concat([ele._lines for ele in sub_line_list]).drop_duplicates(subset=['species', 'wlcent', 'gflog', 'excit']).sort_values('wlcent').reset_index(drop=True)

    # Remove
    out_list = deepcopy(strong_list)
    out_list._lines = pd.concat([out_list._lines, line_list._lines]).sort_values('wlcent').reset_index(drop=True)
    if len(stack_linelist) != len(out_list):
        raise ValueError
    if 'central_depth' in out_list.columns:
        out_list._lines = out_list._lines.drop('central_depth', axis=1)
    out_list._lines = pd.merge(out_list._lines, stack_linelist._lines[['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw', 'central_depth', 'line_range_s', 'line_range_e']], on=['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw'], how='left')
    
    if np.all(wav != sme.wave[0]):
        raise ValueError
    
    return wav, flux, out_list, sub_line_list, sub_wave_range

def batch_synth(sme, line_list, N_line_chunk=2000, line_margin=2, parallel=False, n_jobs=5, pysme_out=False):
    '''
    The function to synthize the spectra using pysme in batch, according to the line_range of each line. This would work faster than doing the whole spectra at once.
    For the synthesize accuracy, the code will separate the strong lines from the line list if strong_line is None, and always include t

    batch_mode : str, defaule 'line'
        The mode for performing batch synthesis. If 'line', then the spectra will be divided according to the number of lines. If 'wave' then will be divided according to wavelength range. 
   
    Example for the pre-process:
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = 5026, 3.62, 0.2, 1.17, 0, 7.55
    sme.iptype = 'gauss'
    sme.ipres = 50000
    sme.wave = np.arange(wav_s, wav_end, 0.05)
    '''

    wav_s, wav_end = np.min(sme.wave), np.max(sme.wave)
    
    # Remove dupliate lines from the line list
    line_list._lines = line_list._lines.drop_duplicates(subset=['species', 'wlcent', 'gflog', 'excit', 'j_lo', 'e_upp', 'j_up', 'lande_lower', 'lande_upper', 'lande', 'gamrad', 'gamqst', 'gamvw'])

    # Decide how many chunks to be divided
    N_chunk = int(np.ceil(len(line_list) / N_line_chunk))

    # Divide the line list to sub line lists
    sub_wave_range = [line_list[N_line_chunk*i]['wlcent'][0] for i in range(N_chunk)]
    sub_wave_range.append(line_list[len(line_list)-1]['wlcent'][0])
    sub_wave_range = [sub_wave_range[i:i+2] for i in range(len(sub_wave_range) - 1)]
    for i in range(len(sub_wave_range)-1):
        sub_wave_range[i+1][0] = sub_wave_range[i][1]
    if sub_wave_range[0][0] > sme.wave[0][0]:
        sub_wave_range[0][0] = sme.wave[0][0]
    if sub_wave_range[-1][-1] < sme.wave[0][-1]:
        sub_wave_range[-1][-1] = sme.wave[0][-1]
    sub_wave_range[0][0] -= 1e-3
    sub_wave_range[-1][-1] += 1e-3

    # Remove the wave ranges with no sme.wave inside 
    sub_wave_range = [ele for ele in sub_wave_range if len(sme.wave[0][(sme.wave[0] >= ele[0]) & (sme.wave[0] < ele[1])]) > 0]
    
    # Combine the sub_wave_range of length <=2 to the previous one.
    if len(sme.wave[0][(sme.wave >= sub_wave_range[0][0]) & (sme.wave < sub_wave_range[0][1])]) <= 2:
        sub_wave_range[0] = np.array([sub_wave_range[0][0], sub_wave_range[1][1]])
        # 从列表中移除下一个元素
        del sub_wave_range[1]

    i = len(sub_wave_range) - 2  # 开始检查的索引
    while i >= 0:
        if len(sme.wave[0][(sme.wave >= sub_wave_range[i+1][0]) & (sme.wave < sub_wave_range[i+1][1])]) <= 2:  # 检查下一个元素的长度
            # 合并当前元素和下一个元素
            sub_wave_range[i] = np.array([sub_wave_range[i][0], sub_wave_range[i+1][1]])
            # 从列表中移除下一个元素
            del sub_wave_range[i+1]
        i -= 1  

    N_chunk = len(sub_wave_range)

    sub_sme_all = []
    args = []
    for i in tqdm(range(N_chunk)):
        line_wav_start, line_wav_end = sub_wave_range[i]
        wav_start, wav_end = sub_wave_range[i][0], sub_wave_range[i][1]
        # print(wav_start, wav_end)
        args.append([sme, line_list, line_wav_start, line_wav_end, wav_start, wav_end, line_margin, pysme_out])
        if not parallel:
            sub_sme = deepcopy(sme)
            sub_sme.linelist = line_list[~((line_list['line_range_e'] < line_wav_start-line_margin) | (line_list['line_range_s'] > line_wav_end+line_margin))]
            # Define the wavelength. Here we extend the wavelength array to include the vsini effect.
            sub_sme.wave = sme.wave[(sme.wave >= wav_start - sub_sme.vsini/3e5*wav_start) & (sme.wave < wav_end + sub_sme.vsini/3e5*wav_start)]
            # sub_sme.wave = sme.wave[(sme.wave >= wav_start) & (sme.wave < wav_end)]
            if pysme_out:
                sub_sme = synthesize_spectrum(sub_sme)
            else:
                with redirect_stdout(open(f"/dev/null", 'w')):
                    sub_sme = synthesize_spectrum(sub_sme)
            wav_indices = (sub_sme.wave >= wav_start) & (sub_sme.wave < wav_end)
            if i == 0:
                wav, flux = sub_sme.wave[0][wav_indices], sub_sme.synth[0][wav_indices]
            else:
                wav, flux = np.concatenate([wav, sub_sme.wave[0][wav_indices]]), np.concatenate([flux, sub_sme.synth[0][wav_indices]])
            sub_sme_all.append(sub_sme)

    if parallel:
        results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(synthesize_spectrum_pqdm)(*ele) for ele in tqdm(args))
        # results = pqdm(args, synthesize_spectrum_pqdm, n_jobs=n_jobs)
        wav = [item[0] for item in results]
        wav = np.concatenate(wav)
        flux = [item[1] for item in results]
        flux = np.concatenate(flux)
        sub_sme_all = [item[2] for item in results]

    # Merge the spectra
    if np.all(wav != sme.wave[0]):  
        raise ValueError
    
    return wav, flux 

def synthesize_spectrum_pqdm(sme, line_list, line_wav_start, line_wav_end, wav_start, wav_end, line_margin, pysme_out):
    sub_sme = deepcopy(sme)
    sub_sme.linelist = line_list[~((line_list['line_range_e'] < line_wav_start-line_margin) | (line_list['line_range_s'] > line_wav_end+line_margin))]
    sub_sme.wave = sme.wave[(sme.wave >= wav_start) & (sme.wave < wav_end)]
    # Define the wavelength. Here we extend the wavelength array to include the vsini effect.
    sub_sme.wave = sme.wave[(sme.wave >= wav_start - sub_sme.vsini/3e5*wav_start) & (sme.wave < wav_end + sub_sme.vsini/3e5*wav_start)]
    if pysme_out:
        sub_sme = synthesize_spectrum(sub_sme)
    else:
        with redirect_stdout(open(f"/dev/null", 'w')):
            sub_sme = synthesize_spectrum(sub_sme)

    wav_indices = (sub_sme.wave >= wav_start) & (sub_sme.wave < wav_end)
    wav, flux = sub_sme.wave[0][wav_indices], sub_sme.synth[0][wav_indices]
    # del sub_sme

    return wav, flux, sub_sme