from pysme.sme import SME_Structure
from pysme.abund import Abund
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve

from . import pysme_synth, util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, pickle
from contextlib import redirect_stdout
from tqdm import tqdm
from copy import copy

def has_overlap(range1, range2):
    """检查两个波长范围是否重叠"""
    return not (range1[1] < range2[0] or range1[0] > range2[1])

def find_line_groups(wave, ele_fit, ion_fit, line_list, v_broad, margin_ratio=2, loggf_cut=None, line_mask_remove=None):

    # Get the line range for fitting
    fit_line_group = {}
    if loggf_cut is not None:
        use_list = line_list[line_list['gflog'] > loggf_cut]
    else:
        use_list = line_list

    for ele in ele_fit:
        fit_line_group[ele] = {}
        for ion in ion_fit: 
            fit_line_group_single = []
            for i in use_list[use_list['species'] == f'{ele} {ion}'].index:
                line_wav = use_list._lines.loc[i, 'wlcent']
                line_cdepth = use_list._lines.loc[i, 'central_depth']
                fit_line_group_single.append([line_wav, line_cdepth, line_wav * (1-margin_ratio*v_broad / 3e5), line_wav * (1+margin_ratio*v_broad / 3e5)])
            
            # Remove the line group outside the observed spectra
            fit_line_group_single = [ele for ele in fit_line_group_single if ele[2] >= np.min(wave) and ele[3] <= np.max(wave) and len(wave[(wave>=ele[2]) & (wave<=ele[3])]) > 1]

            # 如果提供了line_mask_remove，过滤掉与mask重叠的谱线
            if line_mask_remove is not None:
                fit_line_group_single = [ele for ele in fit_line_group_single 
                                       if not any(has_overlap([ele[2], ele[3]], mask) 
                                                for mask in line_mask_remove)]

            # Merge the line_group if they are connected
            fit_line_group_single.sort(key=lambda x: x[0])
            merged_ranges = []
            for current_range in fit_line_group_single:
                if not merged_ranges:
                    merged_ranges.append(current_range)
                else:
                    last_range = merged_ranges[-1]
                    # 检查当前范围的起始值是否小于等于前一个范围的结束值
                    # print(current_range[2])
                    if current_range[2] <= last_range[3]:
                        # 更新已合并范围的结束值为当前范围和前一个范围的结束值的最大者
                        last_range[3] = max(last_range[3], current_range[3])
                        # Append the line wavelength and central depth
                        if type(last_range[0]) != list:
                            last_range[0] = [last_range[0]]
                        last_range[0].append(current_range[0])
                        if type(last_range[1]) != list:
                            last_range[1] = [last_range[1]]
                        last_range[1].append(current_range[1])
                    else:
                        merged_ranges.append(current_range)
            fit_line_group[ele][ion] = [{'wlcent':ele[0], 'central_depth':ele[1], 'wav_s':ele[2], 'wav_e':ele[3]} for ele in merged_ranges]
    return fit_line_group

def get_sensitive_synth(wave, R, teff, logg, m_h, vmic, vmac, vsini, line_list, abund, ele_fit, ion_fit, fit_line_group, nlte_ele=[], include_moleculer=False, varied_para=None, cdepth_thres=0):
    '''
    varied_para : default None. if specified, will only change the specified parameter.
    '''

    moleculer_line_dict = {
        'C': ['CN 1', 'CH 1', 'C2 1', 'CO 1'],
        'N': ['CN 1', 'NH 1', 'N2 1'],
        'O': ['OH 1', 'CO 1']
    }

    param_variations = {
        'teff': 100,
        'logg': 0.2,
        'monh': 0.1,
        'vmic': 0.5,
        'vmac': 1.0,
        'vsini': 1.0
    }

    # Generate the synthetic spectra using current parameters
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    sme.abund = abund
    sme.wave = wave
    for nlte_ele_single in nlte_ele:
        sme.nlte.set_nlte(nlte_ele_single)
    # if cdepth_thres > 0:
    cdepth_indices = line_list['central_depth'] > cdepth_thres
    spec_syn_all = pysme_synth.batch_synth(sme, line_list[cdepth_indices], parallel=True, n_jobs=10)

    # Calculate the sensitive spectra for the fitting elements.
    spec_syn = {}
    i = 0
    for ele in tqdm(ele_fit, desc='Calcluating sensitive spectra for all fitting elements'):
        print(f'get_sensitive_synth for element {ele}.')
        wave_indices = wave < 0
        for ion in ion_fit:
            for wav_range in fit_line_group[ele][ion]:
                wave_indices |= ((wave >= wav_range['wav_s']) & (wave <= wav_range['wav_e']))

        flux_syn_minus, flux_syn_plus = np.ones_like(wave), np.ones_like(wave)

        sme.abund = abund
        if varied_para is None:
            sme.abund[ele] += 0.1
        else:
            # 根据varied_para修改对应参数
            if varied_para == 'teff':
                sme.teff += param_variations[varied_para]
            elif varied_para == 'logg':
                sme.logg += param_variations[varied_para]
            elif varied_para == 'monh':
                sme.monh += param_variations[varied_para]
            elif varied_para == 'vmic':
                sme.vmic += param_variations[varied_para]
            elif varied_para == 'vmac':
                sme.vmac += param_variations[varied_para]
            elif varied_para == 'vsini':
                sme.vsini += param_variations[varied_para]
        sme.wave[0] = wave

        if len(sme.wave[0]) > 2:
            spec_syn_plus = pysme_synth.batch_synth(sme, line_list[cdepth_indices], parallel=True, n_jobs=10)
            flux_syn_plus = spec_syn_plus[1]
        sme.abund = abund
        if varied_para is None:
            sme.abund[ele] -= 0.2
        else:
            # 根据varied_para修改对应参数
            if varied_para == 'teff':
                sme.teff -= 2*param_variations[varied_para]
            elif varied_para == 'logg':
                sme.logg -= 2*param_variations[varied_para]
            elif varied_para == 'monh':
                sme.monh -= 2*param_variations[varied_para]
            elif varied_para == 'vmic':
                sme.vmic -= 2*param_variations[varied_para]
            elif varied_para == 'vmac':
                sme.vmac -= 2*param_variations[varied_para]
            elif varied_para == 'vsini':
                sme.vsini -= 2*param_variations[varied_para]
        if len(sme.wave[0]) > 2:
            spec_syn_minus = pysme_synth.batch_synth(sme, line_list[cdepth_indices], parallel=True, n_jobs=10)
            flux_syn_minus = spec_syn_minus[1]
        
        # Back to the input value
        if varied_para is None:
            sme.abund[ele] += 0.1
        else:
            # 根据varied_para修改对应参数
            if varied_para == 'teff':
                sme.teff += param_variations[varied_para]
            elif varied_para == 'logg':
                sme.logg += param_variations[varied_para]
            elif varied_para == 'monh':
                sme.monh += param_variations[varied_para]
            elif varied_para == 'vmic':
                sme.vmic += param_variations[varied_para]
            elif varied_para == 'vmac':
                sme.vmac += param_variations[varied_para]
            elif varied_para == 'vsini':
                sme.vsini += param_variations[varied_para]

        flux_syn_ion = {}
        for ion in ion_fit:
            indices = line_list['species'] == f'{ele} {ion}'
            flux_syn_ion[ion] = np.ones_like(wave)
            if not include_moleculer or ele not in moleculer_line_dict.keys():
                print(f'Running for {ele} {ion}. Not including molecular lines.')
            else:
                print(f'Running for {ele} {ion}. Including molecular lines of {moleculer_line_dict[ele]}.')
                for mole_species in moleculer_line_dict[ele]:
                    indices |= line_list['species'] == f'{mole_species}'
            try:
                spec_syn_ion = pysme_synth.batch_synth(sme, line_list[indices & cdepth_indices], parallel=True, n_jobs=10)
                flux_syn_ion[ion] = spec_syn_ion[1]
            except:
                pass

        spec_syn[ele] = {'minus':flux_syn_minus,
                            'plus':flux_syn_plus,
                            'partial_derivative':(flux_syn_plus-flux_syn_minus)/2, 
                            'ele_only':flux_syn_ion}
        if i == 0:
            total_partial_derivative = (flux_syn_plus-flux_syn_minus)/2
        else:
            total_partial_derivative += (flux_syn_plus-flux_syn_minus)/2
        
        i += 1

    spec_syn['total'] = {'wave':spec_syn_all[0],
                            'flux':spec_syn_all[1],
                            'minus':np.array([]),
                            'plus':np.array([]),
                            'partial_derivative':total_partial_derivative}
    
    return spec_syn

def select_lines(fit_line_group, spec_syn, ele_fit, ion_fit, sensitivity_dominance_thres=0.5, line_dominance_thres=0.5, max_line_num=None, output_all=False):

    '''
    Select lines
    '''

    # Calculate the line range parameters
    for ele in ele_fit:
        for ion in ion_fit:
            for i in range(len(fit_line_group[ele][ion])):
                wav_range = [fit_line_group[ele][ion][i]['wav_s'], fit_line_group[ele][ion][i]['wav_e']]
                
                indices = (spec_syn['total']['wave'] >= wav_range[0]) & (spec_syn['total']['wave'] <= wav_range[1]) 
                max_sensitivity = np.max(spec_syn[ele]['partial_derivative'][indices])
                sensitivity_dominance = np.sum(spec_syn[ele]['partial_derivative'][indices]) / np.sum(spec_syn['total']['partial_derivative'][indices]) 
                line_dominance = np.sum(1 - spec_syn[ele]['ele_only'][ion][indices]) / np.sum(1 - spec_syn['total']['flux'][indices])
                line_max_depth =  np.max(1 - spec_syn[ele]['ele_only'][ion][indices])
                fit_line_group[ele][ion][i]['max_sensitivity'] = max_sensitivity
                fit_line_group[ele][ion][i]['sensitivity_dominance'] = sensitivity_dominance
                fit_line_group[ele][ion][i]['line_dominance'] = line_dominance
                fit_line_group[ele][ion][i]['line_max_depth'] = line_max_depth
            if len(fit_line_group[ele][ion]) > 0:
                fit_line_group[ele][ion] = pd.DataFrame(fit_line_group[ele][ion])
            else:
                fit_line_group[ele][ion] = pd.DataFrame(pd.DataFrame(columns=['wav_s', 'wav_e', 'max_sensitivity', 'sensitivity_dominance', 'line_dominance', 'line_max_depth']))

    if output_all:
        return fit_line_group
    
    # Select the lines
    for ele in ele_fit:
        for ion in ion_fit:
            indices = (fit_line_group[ele][ion]['sensitivity_dominance'] >= sensitivity_dominance_thres) & (fit_line_group[ele][ion]['line_dominance'] >= line_dominance_thres) & (fit_line_group[ele][ion]['line_dominance'] < 1) & (fit_line_group[ele][ion]['line_max_depth'] > 0.003*2)
            fit_line_group[ele][ion] = fit_line_group[ele][ion][indices].sort_values('max_sensitivity', ascending=False).reset_index(drop=True)
            if max_line_num is not None:
                fit_line_group[ele][ion] = fit_line_group[ele][ion][:max_line_num]

    return fit_line_group

def is_within_ranges(line_wav, line_mask_remove):
    """
    判断 line_wav 是否在 line_mask_remove 的任意区间内，
    支持 line_wav 为标量或 NumPy 数组。
    Used in abund_fit.

    参数:
    - line_wav: float 或 np.ndarray，待检查的波长值（可以是单个值或数组）。
    - line_mask_remove: list of [a, b]，包含要排除的波长区间。

    返回:
    - bool 或 np.ndarray: 若 line_wav 是标量，返回 bool；若是数组，返回 bool 数组。
    """
    line_wav = np.atleast_1d(line_wav)  # 确保 line_wav 是数组
    mask = np.zeros_like(line_wav, dtype=bool)  # 初始化结果（默认 False）

    for a, b in line_mask_remove:
        mask |= (a <= line_wav) & (line_wav <= b)  # 逐个区间检查
    
    return np.all(mask)  # 保持输入输出一致性

def abund_fit(ele, ion, wav, flux, flux_uncs, line_wav, fit_range, R, teff, logg, m_h, vmic, vmac, vsini, abund, use_list, 
              spec_syn, synth_margin=5,
              ele_blend=[], ele_blend_fit=[],
              save_path=None, plot=False, atmo=None, nlte=False, fit_rv=False, telluric_spec=None, max_telluric_depth_thres=0.1,
              synth_cont_level=0.025, cscale_flag='constant', mu=None,
              blending_line_plot=[], line_mask_remove=None, star_name=None):

    '''
    Fit the abundance of a single line.
    '''

    # Crop the spectra 
    indices = (wav >= fit_range[0]-synth_margin) & (wav <= fit_range[1]+synth_margin)
    wav = wav[indices]
    flux = flux[indices]
    flux_uncs = flux_uncs[indices]
    if telluric_spec is not None:
        telluric_spec = telluric_spec[indices]
    # Crop the line list
    use_list = use_list[(use_list['line_range_e'] > fit_range[0]-synth_margin) & (use_list['line_range_s'] < fit_range[1]+synth_margin)]
    
    sme_fit = SME_Structure()
    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R
    sme_fit.abund = copy(abund)
    sme_fit.linelist = use_list
    sme_fit.wave = wav
    sme_fit = synthesize_spectrum(sme_fit)
    sme_fit.spec = flux
    sme_fit.uncs = flux_uncs
    if mu is not None:
        sme_fit.mu = mu

    # Add blending abundance to fit, if specified.
    #  Here we put the whole spectra into fitting
    print(f'ele_blend_fit is {ele_blend_fit}')
    for ele_blend_single in ele_blend_fit:
        sme_fit = solve(sme_fit, [f'abund {ele_blend_single}'], bounds=sme_fit.abund[ele_blend_single]+np.array([-2, 2]))
        print(f'Fitting blending element: {ele_blend_single}, fitting abundace is: {sme_fit.fitresults["values"]}.')

    # Define masks
    mask = np.zeros_like(sme_fit.wave[0], dtype=int)
    indices_con = (np.abs(sme_fit.synth[0] - 1) < synth_cont_level)
    if telluric_spec is not None:
        indices_con &= telluric_spec > 1-0.1
    return_mask = util.sigma_clip((flux[indices_con] / sme_fit.synth[0][indices_con]), sigma=2.0, return_mask=True)
    true_indices = np.where(indices_con)[0]
    indices_con[true_indices] = return_mask

    if cscale_flag not in ['none', 'fix']:
        mask[indices_con] = 2
    indices = (wav >= fit_range[0]) & (wav <= fit_range[1])
    mask[indices] = 1
    sme_fit.cscale_flag = cscale_flag
    sme_fit.mask = mask

    if telluric_spec is not None:
        max_telluric_depth = 1 - np.min(telluric_spec[indices])

    if nlte:
        sme_fit.nlte.set_nlte(ele)

    if atmo is not None:
        sme_fit.atmo = atmo
        sme_fit.atmo.method = 'embedded'

    if np.all(np.isnan(sme_fit.spec[0][indices])):
        sme_fit.fitresults['values'] = [np.nan]
        sme_fit.fitresults['fit_uncertainties'] = [np.nan]
        sme_fit.synth = [[np.nan]*len(sme_fit.wave[0])]
    else:
        if fit_rv:
            sme_fit = solve(sme_fit, [f'abund {ele}', 'vrad'])
        else:
            sme_fit = solve(sme_fit, [f'abund {ele}'])
        fit_flag = 'normal'
    
    # Calculate the EW
    indices = (sme_fit.wave[0] >= fit_range[0]) & (sme_fit.wave[0] <= fit_range[1])
    EW_all = np.trapz(1-sme_fit.synth[0][indices]/sme_fit.cscale[0][0], sme_fit.wave[0][indices]) * 1000
    best_fit_synth = sme_fit.synth[0].copy()

    sme_fit.linelist = use_list
    if sme_fit.fitresults['fit_uncertainties'][0] < 8:
        sme_fit.abund[ele] += sme_fit.fitresults['fit_uncertainties'][0] - sme_fit.monh
        sme_fit = synthesize_spectrum(sme_fit)
        plus_fit_synth = sme_fit.synth[0].copy()
        sme_fit.abund[ele] -= 2*sme_fit.fitresults['fit_uncertainties'][0] + sme_fit.monh
        sme_fit = synthesize_spectrum(sme_fit)
        minus_fit_synth = sme_fit.synth[0].copy()
        sigma_EW = (np.trapz(1-plus_fit_synth[indices], sme_fit.wave[0][indices]) - np.trapz(1-minus_fit_synth[indices], sme_fit.wave[0][indices])) / 2 * 1000
        # diff_EW = np.mean(flux_uncs[(wav >= fit_range[0]) & (wav <= fit_range[1])])/2 * 1000
        if EW_all <= 3*sigma_EW:
            fit_flag = 'upper_limit'
    else:
        sigma_EW = np.nan
        fit_flag = 'error'

    if line_mask_remove is not None:
        if is_within_ranges(line_wav, line_mask_remove):
            fit_flag = 'removed'

    if telluric_spec is not None and max_telluric_depth > max_telluric_depth_thres:
        fit_flag = 'telluric_blended'

    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        indices = (spec_syn['total']['wave'] >= fit_range[0]-2) & (spec_syn['total']['wave'] <= fit_range[1]+2)
        plt.fill_between(spec_syn['total']['wave'][indices], spec_syn[ele]['minus'][indices], spec_syn[ele]['plus'][indices], label=f"Synthetic spectra with [{ele}/Fe]$\pm$0.1", alpha=0.5)
        color_i = 2
        for ele_blend_single in ele_blend:
            plt.fill_between(spec_syn['total']['wave'][indices], spec_syn[ele_blend_single]['minus'][indices], spec_syn[ele_blend_single]['plus'][indices], label=f"Synthetic spectra with [{ele_blend_single}/Fe]$\pm$0.1", alpha=0.3, color=f'C{color_i}')
            color_i += 1
        plt.plot(spec_syn['total']['wave'][indices], spec_syn[ele]['ele_only'][ion][indices], c='C0', label=f"Synthetic spectra with {ele} {ion} line only")
        # plt.axvline(fit_range[0], color='C1', alpha=0.8, ls='-.')
        # plt.axvline(fit_range[1], color='C1', alpha=0.8, ls='-.')
        plt.axvspan(*fit_range, color='C1', alpha=0.1)
        if type(line_wav) == list:
            for line_wav_single in line_wav:
                plt.axvline(line_wav_single, c='C1', ls=':', label='', alpha=0.7)
        else:
            plt.axvline(line_wav, c='C1', ls=':', label='', alpha=0.7)
        plt.legend()

        if star_name is not None:   
            plt.title(f'{star_name}, {ele} {ion} ({line_wav} $\mathrm{{\AA}}$)')
        else:
            plt.title(f'{ele} {ion} ({line_wav} $\mathrm{{\AA}}$)')
        plt.ylabel('Normalized flux')

        ax1 = plt.subplot(212)
        plt.errorbar(sme_fit.wave[0], sme_fit.spec[0], yerr=sme_fit.uncs[0], fmt='.', label='Observed spectrum', zorder=0)
        plt.plot(sme_fit.wave[0], best_fit_synth, label='Synthesized spectrum', zorder=2)
        if type(line_wav) == list:
            for line_wav_single in line_wav:
                plt.axvline(line_wav_single, c='C1', ls=':', label='', alpha=0.7)
        else:
            plt.axvline(line_wav, c='C1', ls=':', label='', alpha=0.7)
        ylim = plt.ylim()
        plt.ylim(ylim)
        if telluric_spec is not None:
            plt.plot(wav, telluric_spec, label='Telluric spectrum', c='gray', alpha=0.5)
        if sme_fit.fitresults['fit_uncertainties'][0] < 8:
            plt.plot(sme_fit.wave[0], plus_fit_synth, label='', c='C1', ls='--')
            plt.plot(sme_fit.wave[0], minus_fit_synth, label='', c='C1', ls='--')
        
        plt.axvspan(*fit_range, color='C1', alpha=0.1)

        bl_count = 2
        for species in blending_line_plot:
            t_count = 0
            indices = (sme_fit.linelist['species'] == species) & (sme_fit.linelist['wlcent'] > wav[0]) & (sme_fit.linelist['wlcent'] < wav[-1])
            for line_wave in sme_fit.linelist[indices]['wlcent']:
                if t_count == 0:
                    plt.plot([line_wave, line_wave], [ylim[0]+(ylim[1]-ylim[0])*6/7, ylim[1]], ls=':', alpha=0.3, label=f'{species} line', c=f'C{bl_count}')
                else:
                    plt.plot([line_wave, line_wave], [ylim[0]+(ylim[1]-ylim[0])*6/7, ylim[1]], ls=':', alpha=0.3, c=f'C{bl_count}')
                t_count += 1
            bl_count += 1

        if sme_fit.fitresults['fit_uncertainties'][0] < 8:
            plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.2f}$\pm${sme_fit.fitresults['fit_uncertainties'][0]:.2f}, $\mathrm{{EW_{{synth, all}}}}$={EW_all:.2f}$\pm${sigma_EW:.2f} m$\mathrm{{\AA}}$, {fit_flag}")
        else:
            plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.2f}$\pm${sme_fit.fitresults['fit_uncertainties'][0]:.2f}, bad fitting")
        plt.ylabel('Normalized flux')

        ax2 = plt.twinx()
        plt.plot(sme_fit.wave[0], sme_fit.mask[0], ls='--', c='gray', zorder=-10, alpha=0.3)
        plt.ylabel('mask (dashed)')
        plt.gca().set_yticks([0, 1, 2])
        plt.gca().set_yticklabels(['bad', 'line', 'cont'])

        handles1, labels1 = ax1.get_legend_handles_labels()
        leg = ax2.legend(handles1, labels1, loc=4)  # 重新绘制 legend
        # leg.set_zorder(10)

        ax1.set_xlabel('Wavelength ($\mathrm{\AA}$)')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(f"{save_path}_{fit_range[0]:.3f}-{fit_range[1]:.3f}_line_fit.pdf")
            plt.close()
    
    fitresults = copy(sme_fit.fitresults)
    del sme_fit
    return (fitresults, EW_all, sigma_EW, fit_flag)

def plot_average_abun(ele, fit_line_group_ele, ion_fit, result_folder, standard_value=None, standard_label=None, star_name=None):

    plt.figure(figsize=(10*1.2, 3*1.2), dpi=150)
    color_i = 0
    for ion in ion_fit:
        indices = (fit_line_group_ele['fit_result']['ioni_state'] == ion) & (fit_line_group_ele['fit_result']['flag'] == 'normal')
        if len(fit_line_group_ele['fit_result'].index[indices]) > 0:
            plt.scatter(fit_line_group_ele['fit_result'].index[indices], fit_line_group_ele['fit_result'].loc[indices, f'A({ele})'], zorder=2, label=f'{ele} {ion} line', c=f'C{color_i}')
            plt.errorbar(fit_line_group_ele['fit_result'].index[indices], fit_line_group_ele['fit_result'].loc[indices, f'A({ele})'], 
                    yerr=fit_line_group_ele['fit_result'].loc[indices, f'err_A({ele})'], fmt='.', zorder=1, c=f'C{color_i}', alpha=1)
            
            indices = (fit_line_group_ele['fit_result']['ioni_state'] == ion) & (fit_line_group_ele['fit_result']['flag'] != 'normal') & (fit_line_group_ele['fit_result']['flag'] != 'upper_limit')
            plt.scatter(fit_line_group_ele['fit_result'].index[indices], fit_line_group_ele['fit_result'].loc[indices, f'A({ele})'], zorder=3, label=f'', c='red', marker='x')

            indices = (fit_line_group_ele['fit_result']['ioni_state'] == ion) & (fit_line_group_ele['fit_result']['flag'] == 'upper_limit')
            plt.errorbar(fit_line_group_ele['fit_result'].index[indices], fit_line_group_ele['fit_result'].loc[indices, f'A({ele})']+fit_line_group_ele['fit_result'].loc[indices, f'err_A({ele})'],
                        yerr=fit_line_group_ele['fit_result'].loc[indices, f'err_A({ele})'],
                        uplims=fit_line_group_ele['fit_result'].loc[indices, f'err_A({ele})'],
                        marker='_', markersize=10, ls='none')
                        
            color_i += 1
    plt.ylim(plt.ylim())
    
    if standard_value is not None:
        if standard_label is not None:
            plt.axhline(standard_value, c='C3', label=f'{standard_label}: {standard_value:.2f}', ls='--')
        else:
            plt.axhline(standard_value, c='C3', label=f'Standard value: {standard_value:.2f}', ls='--')
    plt.axhline(fit_line_group_ele['average_abundance'], label='Fitted value', ls='--')
    plt.axhspan(fit_line_group_ele['average_abundance']-fit_line_group_ele['average_abundance_err'], fit_line_group_ele['average_abundance']+fit_line_group_ele['average_abundance_err'], alpha=0.2, label='Fitted std')
    
    plt.xticks(fit_line_group_ele['fit_result'].index, ['{:.3f}-\n{:.3f}$\mathrm{{\AA}}$'.format(fit_line_group_ele['fit_result'].loc[i, 'wav_s'], fit_line_group_ele['fit_result'].loc[i, 'wav_e']) for i in fit_line_group_ele['fit_result'].index], rotation=90);
    plt.legend(fontsize=7)
    plt.ylabel(f'A({ele})')
    if star_name is not None:
        plt.title(f"{star_name}, A({ele})={fit_line_group_ele['average_abundance']:.2f}$\pm${fit_line_group_ele['average_abundance_err']:.2f}")
    else:
        plt.title(f"A({ele})={fit_line_group_ele['average_abundance']:.2f}$\pm${fit_line_group_ele['average_abundance_err']:.2f}")
    plt.tight_layout()
    plt.grid(zorder=0)
    
    plt.savefig(f'{result_folder}/{ele}/{ele}-fit.pdf')
    plt.close()

def pysme_abund(wave, flux, flux_err, R, teff, logg, m_h, vmic, vmac, vsini, line_list, ele_fit, 
                ele_blend=[], ion_fit=[1, 2], nlte_ele=[], result_folder=None, line_mask_remove=None, abund=None, plot=False, standard_values=None, standard_label=None, abund_record=None, 
                save=False, overwrite=False, central_depth_thres=0.01, cal_central_depth=True, sensitivity_dominance_thres=0.3, line_dominance_thres=0.3, max_line_num=10, 
                fit_rv=False, telluric_spec=None, max_telluric_depth_thres=None, line_select_save=False, fit_line_group=None, sensitive_synth=None, 
                blending_line_plot=[], cscale_flag='constant', include_moleculer=False, mu=None, ele_blend_fit=[], star_name=None):
    '''
    The main function for determining abundances using pysme.
    Input: observed wavelength, normalized flux, teff, logg, [M/H], vmic, vmac, vsini, line_list, pysme initial abundance list, line mask of wavelength to be removed.
    ele_fit have to be either list or string.
    Output: the fitted abundances and reports on the abundances. Can be more than one element, but we do not do parallal computing, and the elements should be fit in sequence.

    Parameters
    ----------
    wave : 

    Returns
    -------
    abund_record : 
        
    abund : 
    '''

    if abund is None:
        abund = Abund.solar() 
        abund.monh = m_h

    if abund_record is None:
        abund_record = {}

    if type(ele_fit) == str:
        ele_fit = [ele_fit]
    elif type(ele_fit) != list:
        raise TypeError('ele_fit have to be either list or string.')

    if type(ele_blend) == str:
        ele_blend = [ele_blend]
    elif type(ele_blend) != list:
        raise TypeError('ele_blend have to be either list or string.')

    if plot or save:
        # Create sub-folders for the star.
        os.makedirs(f"{result_folder}/", exist_ok=True)
    if result_folder is None:
        log_file = '/dev/null'
    else:
        log_file = f"{result_folder}/pysme-abun.log"

    time_select_line_s = time.time()

    v_broad = np.sqrt(vmic**2 + vsini**2 + (3e5/R)**2)

    if 'central_depth' not in line_list.columns or 'line_range_s' not in line_list.columns or 'line_range_e' not in line_list.columns or cal_central_depth:
        # Calculate the central_depth and line_range, if required or no such column
        sme = SME_Structure()
        sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, m_h, vmic, vmac, vsini
        line_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
    line_list = line_list[(line_list['central_depth'] > central_depth_thres) | (line_list['species'] == 'Li 1')]

    # Generate synthetic and sensitive spectra using current parameters
    if fit_line_group is not None and sensitive_synth is not None:
        print('Using the provided line selection and sensitive spectra.')
        pass
    else:
        fit_line_group = find_line_groups(wave, ele_fit+ele_blend, ion_fit, line_list, v_broad)
        sensitive_synth = get_sensitive_synth(wave, R, teff, logg, m_h, vmic, vmac, vsini, line_list, abund, ele_fit+ele_blend, ion_fit, fit_line_group, nlte_ele=nlte_ele, include_moleculer=include_moleculer)
        fit_line_group = select_lines(fit_line_group, sensitive_synth, ele_fit+ele_blend, ion_fit, sensitivity_dominance_thres=sensitivity_dominance_thres, line_dominance_thres=line_dominance_thres, max_line_num=max_line_num, output_all=False)
        if line_select_save:
            pickle.dump(fit_line_group, open(f'{result_folder}/line_selection.pkl', 'wb'))
            pickle.dump(sensitive_synth, open(f'{result_folder}/sensitive_synth.pkl', 'wb'))

    time_select_line_e = time.time()

    with redirect_stdout(open(log_file, 'w')):
        # Iterate for all the elements
        ele_fit_count = 0
        for ele in ele_fit:
            nlte_flag = ele in nlte_ele
            time_chi2_s = time.time()
            if plot:
                # Create sub-folders for each element.
                os.makedirs(f"{result_folder}/{ele}/", exist_ok=True)
                if overwrite:
                    # Remove all the files in each element folder.
                    files = os.listdir(f"{result_folder}/{ele}/")
                    for file in files:
                        file_path = os.path.join(f"{result_folder}/{ele}/", file)
                        
                        # 检查是否是文件（排除子文件夹）
                        if os.path.isfile(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                pass
            for ion in ion_fit:
                fit_result = []
                if len(fit_line_group[ele][ion]) == 0:
                    fit_line_group[ele][ion] = pd.DataFrame(pd.DataFrame(columns=list(fit_line_group[ele][ion].columns) + [f'A({ele})', f'err_A({ele})', 'EW', 'diff_EW', 'flag']))
                for i in fit_line_group[ele][ion].index:
                    fit_range = [fit_line_group[ele][ion].loc[i, 'wav_s'],  fit_line_group[ele][ion].loc[i, 'wav_e']]
                    line_wav = fit_line_group[ele][ion].loc[i, 'wlcent']

                    fitresults, EW, diff_EW, fit_flag = abund_fit(ele, ion, wave, flux, flux_err, line_wav, fit_range, R, teff, logg, m_h, vmic, vmac, vsini, abund, line_list, sensitive_synth,
                    nlte=nlte_flag,
                    ele_blend=ele_blend,
                    save_path=f"{result_folder}/{ele}/{ele}_{ion}", atmo=None, plot=plot, fit_rv=fit_rv, telluric_spec=telluric_spec,
                    max_telluric_depth_thres=max_telluric_depth_thres,
                    blending_line_plot=blending_line_plot,line_mask_remove=line_mask_remove,
                    cscale_flag=cscale_flag, mu=mu,
                    ele_blend_fit=ele_blend_fit)

                    fit_result.append({f'A({ele})':fitresults.values[0], f'err_A({ele})':fitresults.fit_uncertainties[0], 'EW':EW, 'diff_EW':diff_EW, 'flag':fit_flag})

                fit_line_group[ele][ion] = pd.concat([fit_line_group[ele][ion], pd.DataFrame(fit_result)], axis=1)

            abun_all = np.concatenate([fit_line_group[ele][i].loc[fit_line_group[ele][i]['flag'] == 'normal', f'A({ele})'].values for i in ion_fit])
            abun_err_all = np.concatenate([fit_line_group[ele][i].loc[fit_line_group[ele][i]['flag'] == 'normal', f'err_A({ele})'].values for i in ion_fit])

            # Get final abundances
            if len(abun_all) > 0:
                weights = 1 / abun_err_all**2
                average_values = np.average(abun_all, weights=weights/np.sum(weights))
                average_error = np.average((abun_all-average_values)**2, weights=weights/np.sum(weights))
                average_error = np.sqrt(average_error + 1 / np.sum(weights))
                fit_line_group[ele]['average_abundance'] = average_values
                fit_line_group[ele]['average_abundance_err'] = average_error
            else:
                fit_line_group[ele]['average_abundance'] = np.nan
                fit_line_group[ele]['average_abundance_err'] = np.nan

            i = 0
            for ion in ion_fit:
                if i == 0:
                    fit_result_df = fit_line_group[ele][ion]
                    fit_result_df['element'] = ele
                    fit_result_df['ioni_state'] = ion
                else:
                    temp = fit_line_group[ele][ion]
                    temp['element'] = ele
                    temp['ioni_state'] = ion
                    fit_result_df = pd.concat([fit_result_df, temp])
                del fit_line_group[ele][ion]
                i += 1
            fit_line_group[ele]['fit_result'] = fit_result_df.sort_values('wav_s').reset_index(drop=True)

            if plot:
                if standard_values is not None:
                    plot_average_abun(ele, fit_line_group[ele], ion_fit, result_folder, standard_value=standard_values[0][ele_fit_count], standard_label=standard_label)
                else:
                    plot_average_abun(ele, fit_line_group[ele], ion_fit, result_folder, standard_label=standard_label)
            ele_fit_count += 1

            time_chi2_e = time.time()
            fit_line_group[ele]['line_selection_time'] = time_select_line_e - time_select_line_s
            fit_line_group[ele]['chi2_time'] = time_chi2_e - time_chi2_s

    if plot:
        # Plot the final abundance and comparison
        plt.figure(figsize=(10, 3))
        plot_x = []
        label_func1 = lambda x: 'standard abunds' if x == 0 else ''
        label_func2 = lambda x: 'pysme abunds' if x == 0 else ''
        
        if standard_values is not None:
            plt.scatter(range(len(ele_fit)), standard_values[0])
        plt.scatter(range(len(ele_fit)), [fit_line_group[ele]['average_abundance'] for ele in ele_fit])
        plt.ylim(plt.ylim())

        j = 0
        for ele in ele_fit:
            plot_x.append(j)
            if standard_values is not None:
                plt.errorbar(j, standard_values[0][j], yerr=standard_values[1][j], fmt='.', alpha=0.7, label=label_func1(j), color='C0')
            plt.errorbar(j, fit_line_group[ele]['average_abundance'], yerr=fit_line_group[ele]['average_abundance_err'], fmt='.', alpha=0.7, label=label_func2(j), color='C1')
            j += 1
        plt.xticks(plot_x, ele_fit)
        plt.legend()
        plt.ylabel('A(X)')
        plt.grid()
        plt.tight_layout()
        if star_name is not None:
            plt.title(f'{star_name}') 
        plt.savefig(f'{result_folder}/abund-result.pdf')
        plt.close()

        plt.figure(figsize=(14, 3))
        plot_x = []
        label_func1 = lambda x: 'standard abunds error' if x == 0 else ''
        label_func2 = lambda x: 'pysme abunds error' if x == 0 else ''

        if standard_values is not None:
            plt.scatter(range(len(ele_fit)), np.array([fit_line_group[ele]['average_abundance'] for ele in ele_fit]) - np.array(standard_values[0]), zorder=3)
            plt.ylim(plt.ylim())
            
            j = 0
            for ele in ele_fit:
                plot_x.append(j)
                if standard_values is not None:
                    plt.errorbar(range(len(ele_fit)), np.array([fit_line_group[ele]['average_abundance'] for ele in ele_fit]) - np.array(standard_values[0]),
                                yerr=standard_values[1], fmt='.', alpha=0.7, label=label_func1(j), color='C0', zorder=1)
                plt.errorbar(range(len(ele_fit)), np.array([fit_line_group[ele]['average_abundance'] for ele in ele_fit]) - np.array(standard_values[0]), 
                            yerr=[fit_line_group[ele]['average_abundance_err'] for ele in ele_fit], fmt='.', alpha=0.7, label=label_func2(j), color='C1', zorder=2)
                j += 1
            plt.axhline(0, ls='--', color='brown')
            plt.xticks(plot_x, ele_fit)
            plt.ylabel('A(X)$_\mathrm{measure}$ - A(X)$_\mathrm{standard}$')
            plt.tight_layout()
            plt.grid(zorder=0)
        if star_name is not None:
            plt.title(f'{star_name}') 
        plt.savefig(f'{result_folder}/diff-result.pdf')
        plt.close()

    if save:
        pickle.dump(fit_line_group, open(f'{result_folder}/abun_res.pkl', 'wb'))
        abun_res_df = pd.DataFrame({'element':ele_fit, 
                                    'A(X)':[fit_line_group[ele]['average_abundance'] for ele in ele_fit],
                                    'err_A(X)':[fit_line_group[ele]['average_abundance_err'] for ele in ele_fit], 
                                    'line_selection_time':[fit_line_group[ele]['line_selection_time'] for ele in ele_fit],
                                    'chi2_time':[fit_line_group[ele]['chi2_time'] for ele in ele_fit]})
        # abun_res_df.columns = ['A(X)', 'err_A(X)', 'time_line_selection' ,'time_chi2']
        abun_res_df.to_csv(f'{result_folder}/abun_fit.csv')

    return fit_line_group