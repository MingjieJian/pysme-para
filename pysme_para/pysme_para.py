import numpy as np
import pandas as pd
from scipy.signal import convolve
import matplotlib.pyplot as plt

from pysme.abund import Abund
from pysme.sme import SME_Structure
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve
from copy import copy

from astropy.table import Table
from scipy.interpolate import Akima1DInterpolator,interp1d

import sys, pkg_resources, pickle

from contextlib import redirect_stdout
from tqdm.notebook import tqdm
from copy import deepcopy
from pqdm.processes import pqdm
from scipy import stats

from pysme.atmosphere.interpolation import AtmosphereInterpolator
from pysme.atmosphere.savfile import SavFile
from pysme.large_file_storage import setup_atmo
from pysme.synthesize import Synthesizer

from . import pysme_abund, pysme_synth

from scipy.stats import linregress

def select_lines(spectra, Teff, vald, purity_crit, fwhm, SNR, verbose=False, select_mode='depth'):

    '''
    input:
    ** spectra: pandas.DataFrame containing a column 'wave' with the wavelengths, 'flux_all' with the full spectrum of a star (all the elements, all the molecules, blends etc) and 'flux_el' with the spectrum of a given element only, computed in a similar way as flux_all (in order to have the same continuum opacities)
    ** Teff: Effective Teff, used in Boltzmann equation. 
    ** vald: pandas Dataframe containing the vald line-parameters for the target element only (ll, Echi, loggf) -- (Echi in eV)
    ** purity_crit: minimum required purity to keep the line
    ** fwhm: in A. Resolution element of the spectrograph
    ** SNR : minimum SNR per resolution element (used for line detection)
    ** sampling: spectrum's sampling (in A)
    ** verbose (optional): print information while running
    
    returns: 
    one panda sdata frame, with the following columns:
    ** wlcent : central wavelength where either side of the line has a putiry higher than purity_crit
    ** Blueratio: Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0-1.5xFWHM. 
    ** Redratio : Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0+1.5xFWHM. 
    ** Fullratio: Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0+/-1.5xFWHM. 
    ** Maxratio:  max between the right and the left blend of the line.
    ** fmin: depth of the core of line (as identified by the algorithm) for the element spectrum
    ** fmin_sp: depth of the full spectrum at the position of the core of the line. 
    ** width: width in which the ratio has been computed
    ** BlueFlag: Number of pixels that have a flux>0.9 within 1.5 times the FWHM (resolution element) 
    ** RedFlag: Number of pixels that have a flux>0.9 within 1.5 times the FWHM (resolution element)
    
    
    Steps: 
    1) Identifies the centers of the lines by computing the first two derivatives of the element spectrum.  
    
    2) Does a x-match with the vald linelist. 
    When several VALD lines are within +/- 1 pixel from the derived line core, 
    the line that has: 
      a) the highest line centeral_depth (select_mode is set to 'depth'), or
      b) the highest value of the boltzmann equation  (select_mode is set to 'boltz'),
        is selected.
        ==>log(Boltzmann): log(N)=log(A) -(E_chi/kT)+log(gf)
            log(A) is a constant we can neglect
            loggf is in vald
            T is the temperature of the star
            E_chi is the excitation potential 
        
        ==> Caution: By using Boltzmann equation to select the lines,we assume that for a given element, 
            all of the lines correspond to the same ionisation level. If this is not the case, 
            we need to involve Saha's equation too. This is not implemented yet. 
        ==> Additional Caution: when there is hyperfine structure, then the lambda of Vald that we will 
            find is not necessarily the center of the line we will be seeing

    3) Estimates the depth of the line and compares it to Careyl's formula. sigma_fmin = 1.5/SNR_resol 
    If the depth of the line is large enough to be seen at a given SNR, then the line is selected. 
    
    
    4) We estimate the width of the line as the pixel in which the flux of the element itself is close enough to the continuum. 
    
    Once the line is selected, we compute the ratio between the element spectrum and the full spectrum. 
    Note: we require that if ratio<0.8 then we must have at least two pixels of the total spectrum with flux>0.9 within 1.5 FWHM,
    
    History: 
    04 Oct. 2024: modify the code to support line selection by line central_depth. - MJ
    10 Jun. 2024: modify the code to support pysme VALD linelist format input (not compatible to pandas.DataFrame). - MJ
    20 Apr. 2023: replaced np.argmin (deprecated) with idxmin, that caused code to crash for machines with updated numpy - GK
    10 Feb. 2023: Curated the Code - GK
    04 Feb. 2023: Cleaned the readme. - GK
    
    Contact: Georges Kordopatis - georges.kordopatis -at- oca.eu
             Mingjie Jian - mingjie.jian -at- astro.su.se
    '''

    def _consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    # Assign the wavelength array to ll.
    ll = spectra['wave'].values

    depth = 1.-3.*(1.5/SNR) # for a 3sigma detection. Based on Careyl's 1988 formula

    # Blindly identify the line's position based on the derivatives. 
    # Take the derivative to find the zero crossings which correspond to
    # the peaks (positive or negative)
    kernel = [1, 0, -1]
    dY = convolve(spectra['flux_el'], kernel, 'same')
    # Use sign flipping to determine direction of change
    S = np.sign(dY)
    ddS = convolve(S, kernel, 'same')
#     print('DERIVATIVES COMPUTED')

    # Find all the indices that appear to be part of a negative peak (absorption lines)
    candidates = np.where(dY < 0)[0] + (len(kernel) - 1)
    line_inds = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))

    # Now group them and find the max highest point.
    line_inds_grouped = _consecutive(line_inds, stepsize=1)

    if len(line_inds_grouped[0]) > 0:
        #absorption_inds = [np.argmin(spectra['flux_el'][inds]) for inds in line_inds_grouped]
        absorption_inds = [spectra['flux_el'][inds].idxmin() for inds in line_inds_grouped]
    else:
        absorption_inds = []
    absorption_ind = np.array(absorption_inds)
    
    # We select the lines that are deep enough to be detected
    zz0 = np.where((spectra['flux_el'].iloc[absorption_ind]<=depth)) [0]  
    zz = absorption_ind[zz0]
    
    # BOLTZMANN METHOD 
    kboltzmann = 8.61733034e-5 # in eV/K
    vald_centers_preliminary = []
    
    # contains the wavelengths (at the pixels) where the first derivative is null and the second is positive
    
    for j in range(0, len(zz)):
        search = np.abs(ll[zz[j]] - vald['wlcent'])
        myvald = np.where((vald['wlcent'] >= ll[zz[j]] - 0.5*fwhm) & (vald['wlcent'] <= ll[zz[j]] + 0.5*fwhm))[0]
        
        if len(myvald) > 1:
            if select_mode == 'boltz':
                myBoltzmann = -vald['excit'][myvald] / (kboltzmann*Teff) + vald['gflog'][myvald]
                mysel = np.where(myBoltzmann == np.max(myBoltzmann))[0]
                vald_centers_preliminary.append(vald['wlcent'][myvald[mysel[0]]])
                if verbose: print(ll[zz[j]],'len(myvald)>1', vald['wlcent'][myvald[mysel[0]]])
            elif select_mode == 'depth':
                mylist = vald._lines.loc[(vald._lines['wlcent'] >= ll[zz[j]] - 0.5*fwhm) & (vald._lines['wlcent'] <= ll[zz[j]] + 0.5*fwhm)]
                idx = mylist['central_depth'].idxmax()
                vald_centers_preliminary.append(mylist.loc[idx, 'wlcent'])
            else:
                raise ValueError("'select_mode' must be either 'depth' or 'boltz'.")
        elif len(myvald) == 1:
            if verbose: print(ll[zz[j]],'-->',len(myvald))
            myvald = np.where(search == np.min(search))[0] # Note that this allows the center of the line to be out of the sampling. 
            vald_centers_preliminary.append(vald['wlcent'][myvald[0]])
            if verbose: print(len(myvald),vald['wlcent'][myvald[0]])
        else: 
            if verbose: print(ll[zz[j]],'-->',len(myvald), ', skip.')
    vald_unique, vald_unique_index = np.unique(np.array(vald_centers_preliminary), return_index=True)

    centers_index = zz[vald_unique_index]
    centers_ll = np.array(vald_unique) 
    
    #Integration of the fluxes in the element spectrum and the full spectrum
    n_lines = len(centers_ll)
    Fratio = np.zeros(n_lines) * np.nan
    Fratio_all = np.zeros(n_lines) * np.nan
    Fratio_blue = np.zeros(n_lines) * np.nan
    Fratio_red = np.zeros(n_lines) * np.nan
    
    width_blue = np.zeros(n_lines) * np.nan
    width_red = np.zeros(n_lines) * np.nan
    
    flag_blue = np.empty(n_lines, dtype=int) * 0
    flag_red = np.empty(n_lines, dtype=int) * 0
    
    half_window_width = 1.5 * fwhm # the total window is 3 fwhm
    
    for j in range(0, n_lines):
        # two selections: blue (left) part of the line, red (right) part of the line
        window_sel_blue = np.where((ll >= centers_ll[j] - half_window_width) & (ll <= centers_ll[j]))[0]
        window_sel_red = np.where((ll <= centers_ll[j] + half_window_width) & (ll >= centers_ll[j]) )[0]
        if len(window_sel_blue) > 0:
            width_blue[j] = ll[window_sel_blue[0]] # this will be overwritten if criteria below are fulfilled.
        else:
            width_blue[j] = 0
        if len(window_sel_red) > 0:
            width_red[j] = ll[window_sel_red[-1]] # this will be overwritten if criteria below are fulfilled.
        else:
            width_red[j] = 0

        for ww in range(0,2): # loop on blue and red wing of the line
            if ww==0: mywindow=window_sel_blue #blue window
            if ww==1: mywindow=window_sel_red # red window

            cont_crit = (1 - np.min(spectra['flux_el'][mywindow])*0.02) #(We are back to the continuum levels more or less 2% of the depth of the line)
            cont_search = np.where(spectra['flux_el'][mywindow] >= cont_crit)[0]
            
            full_continumm_search=np.where(spectra['flux_all'][mywindow]>=0.9)[0] # in order to establish the flags. We want the full spectrum to have a flux >0.9. And we search in a range of +/-1.5FWHM and not the width of the line. 

            if len(cont_search)>=1:
                if ww==0:
                    width_blue[j]=np.max(ll[mywindow[cont_search]])
                    window_sel_blue=np.where((ll>=width_blue[j]) & (ll<=centers_ll[j]))[0]
                    mywindow=window_sel_blue
                if ww==1: 
                    width_red[j]=np.min(ll[mywindow[cont_search]])
                    window_sel_red=np.where((ll<=width_red[j]) & (ll>=centers_ll[j]))[0]
                    mywindow=window_sel_red
                    
            myflux_element=np.sum(1-spectra['flux_el'][mywindow])
            myflux_full_spectrum=np.sum(1-spectra['flux_all'][mywindow])
            myline_flux_ratio=myflux_element/myflux_full_spectrum
                
            if ww==0: 
                Fratio_blue[j]=np.round(myline_flux_ratio,3)
                flag_blue[j]=len(full_continumm_search)
            if ww==1: 
                Fratio_red[j]=np.round(myline_flux_ratio,3)
                flag_red[j]=len(full_continumm_search)
                

        full_window_sel=np.append(window_sel_blue,window_sel_red) # this now contains the full width of the line
        flux_element=np.sum(1-spectra['flux_el'][full_window_sel])
        flux_full_spectrum=np.sum(1-spectra['flux_all'][full_window_sel])
        line_flux_ratio=flux_element/flux_full_spectrum

        Fratio_all[j]=np.round(line_flux_ratio,3)
        Fratio[j]=max([Fratio_blue[j],Fratio_red[j]])
        #print(line_flux_ratio, line_flux_ratio1,line_flux_ratio2,Fratio[j])
            
    keep=np.where(Fratio>purity_crit)[0]
    
    myresult=pd.DataFrame()
    myresult['wlcent'] = np.round(centers_ll[keep],5)
    myresult['Bluewidth']=np.round(width_blue[keep],5)
    myresult['Redwidth']=np.round(width_red[keep],5)
    myresult['Maxratio']=Fratio[keep]
    myresult['fmin']=np.round(spectra['flux_el'][centers_index[keep]].values,3)
    myresult['fmin_sp']=np.round(spectra['flux_all'][centers_index[keep]].values,3)
    myresult['Blueratio']=Fratio_blue[keep]
    myresult['Redratio']=Fratio_red[keep]
    myresult['Fullratio']=Fratio_all[keep]
    myresult['Blueflag']=flag_blue[keep]
    myresult['Redflag']=flag_red[keep]
    
    if verbose: 
        print(centers_ll)
        print('N lines found:',len(vald_unique), ', N lines kept:', len(keep) )

    return(myresult)

def para_first_estimate(wav, flux, flux_err):
    '''
    First estimate of the stellar parameters, assuming we know nothing of the star, using cross-correlation with the templates.
    '''

    # Todo: if the template is specified, broaden them according to R and V_broad.
    
#----------------Overall fitting----------------

def measure_para(wave, flux, flux_err, teff, logg, monh, vmic, vmac, vsini, R, line_list, ele_list, ion_list, fit_para=['teff'], abund=None, nlte=False, selected_lines_save=False, selected_lines_file=None, line_mask_remove=None, max_line_num=50):
    '''
    Measure Teff from Fe 1 lines.
    '''

    v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5/R)**2)
    if abund is None:
        abund = Abund.solar()
        abund.monh = monh

    if not {'central_depth', 'line_range_s', 'line_range_e'}.issubset(line_list._lines.columns) or np.abs(line_list.cdepth_range_paras[0]-teff) >= 500 or (np.abs(line_list.cdepth_range_paras[1]-logg) >= 1) or (np.abs(line_list.cdepth_range_paras[2]-monh) >= 0.5):
        # Calculate the line depth from the pre-set stellar parameter
        sme = SME_Structure()
        sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
        sme.abund = abund
        use_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
        use_list = use_list[use_list['central_depth'] > 0.1]
    else:
        use_list = line_list[line_list['central_depth'] > 0.1]

    # Select the Fe 1 and/or Fe 2 lines according to the current stellar parameters
    fit_line_group = pysme_abund.find_line_groups(wave, ele_list, ion_list, use_list, v_broad, loggf_cut=-2, line_mask_remove=line_mask_remove)
    spec_syn = pysme_abund.get_sensitive_synth(wave, R, teff, logg, monh, vmic, vmac, vsini, use_list, abund, ele_list, ion_list, fit_line_group)
    selected_lines = pysme_abund.select_lines(fit_line_group, spec_syn, ele_list, ion_list, sensitivity_dominance_thres=0.6, line_dominance_thres=0.5, max_line_num=max_line_num)
    
    if selected_lines_save and selected_lines_file is not None:
        pickle.dump(selected_lines, open(selected_lines_file, 'wb'))

    wav_range_list = []
    for key in selected_lines.keys():
        for ion in selected_lines[key]:
            selected_lines[key][ion] = selected_lines[key][ion].sort_values('wav_s')
            wav_range_list += [[selected_lines[key][ion].loc[i, 'wav_s'], selected_lines[key][ion].loc[i, 'wav_e']] for i in selected_lines[key][ion].index]

    # Select the line region for Teff fitting.
    # wav_range_list = [[fit_line_group['Fe'][1].loc[i, 'wav_s'], fit_line_group['Fe'][1].loc[i, 'wav_e']] for i in fit_line_group['Fe'][1].index]
    
    wave_fit = []
    flux_fit = []
    flux_err_fit = []
    for wav_range in wav_range_list:
        indices = (wave >= wav_range[0]) & (wave < wav_range[1])
        wave_fit.append(wave[indices])
        flux_fit.append(flux[indices])
        flux_err_fit.append(flux_err[indices])

    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    if nlte:
        sme.nlte.set_nlte('Fe')
    sme.wave = wave_fit
    sme.spec = flux_fit
    sme.uncs = flux_err_fit
    sme.linelist = use_list
    sme = solve(sme, fit_para, linelist_mode='auto')
    
    return sme

#------------------Teff------------------

def measure_teff_fe1(wave, flux, flux_err, teff, logg, monh, vmic, vmac, vsini, R, line_list, fit_para=['teff'], abund=None):
    '''
    Measure Teff from Fe 1 lines.
    '''

    v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5/R)**2)
    if abund is None:
        abund = Abund.solar()
        abund.monh = monh

    if not {'central_depth', 'line_range_s', 'line_range_e'}.issubset(line_list._lines.columns) or np.abs(line_list.cdepth_range_paras[0]-teff) >= 500 or (np.abs(line_list.cdepth_range_paras[1]-logg) >= 1) or (np.abs(line_list.cdepth_range_paras[2]-monh) >= 0.5):
        # Calculate the line depth from the pre-set stellar parameter
        sme = SME_Structure()
        sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
        sme.abund = abund
        use_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
        use_list = use_list[use_list['central_depth'] > 0.1]
    else:
        use_list = line_list[line_list['central_depth'] > 0.1]

    # Select the Fe 1 and Fe 2 lines according to the current stellar parameters
    fit_line_group = pysme_abund.find_line_groups(wave, ['Fe'], [1], use_list, v_broad)
    spec_syn = pysme_abund.get_sensitive_synth(wave, R, teff, logg, monh, vmic, vmac, vsini, use_list, abund, ['Fe'], [1], fit_line_group)
    fe_lines = pysme_abund.select_lines(fit_line_group, spec_syn, ['Fe'], [1], sensitivity_dominance_thres=0.6, line_dominance_thres=0.5, max_line_num=50)
    fe_lines['Fe'][1] = fe_lines['Fe'][1].sort_values('wav_s')

    # Select the line region for Teff fitting.
    wav_range_list = [[fit_line_group['Fe'][1].loc[i, 'wav_s'], fit_line_group['Fe'][1].loc[i, 'wav_e']] for i in fit_line_group['Fe'][1].index]
    
    wave_fit = []
    flux_fit = []
    flux_err_fit = []
    for wav_range in wav_range_list:
        indices = (wave >= wav_range[0]) & (wave < wav_range[1])
        wave_fit.append(wave[indices])
        flux_fit.append(flux[indices])
        flux_err_fit.append(flux_err[indices])

    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    sme.wave = wave_fit
    sme.spec = flux_fit
    sme.uncs = flux_err_fit
    sme.linelist = use_list
    sme = solve(sme, fit_para, linelist_mode='auto')

    return sme

    teff = sme.fitresults.values[0]
    teff_error = sme.fitresults.uncertainties[0]
    teff_fit_error = sme.fitresults.fit_uncertainties[0]
    vmic = sme.fitresults.values[1]
    vmic_error = sme.fitresults.uncertainties[1]
    vmic_fit_error = sme.fitresults.fit_uncertainties[1]

    return teff, teff_error, teff_fit_error

#------------------logg------------------

# The following function measures photometric logg following the GALAH approach.
# From Sven Buder

def bracket(inval,grval,nn,idx=False):
    
    '''
    bracket by +/-nn values over (irregular) grid. If idx True, then indices 
    are returned instead    
    '''

    norep = np.sort(np.array(list(dict.fromkeys(list(grval)))))
    
    x1    = np.where(norep<=inval)
    x2    = np.where(norep>inval)
    
    if idx==False:
        lo = norep[x1][-nn::]
        up = norep[x2][0:nn]        
    else:
        lo = x1[0][-nn::]
        up = x2[0][0:nn]
        
    return(lo,up)

def mal(val,gridt,gridbc,dset):
    '''
    linear interpolation for 2 points, Akima for more. Returns nan if 
    not possible or if extrapolated. The MARCS grid of BC used here is ordered
    such that gridt is monotonic. If not, sorting is necessary.
    '''
    if len(dset[0])>2:
        mfun = Akima1DInterpolator(gridt[dset],gridbc[dset])
        itp  = mfun(val)
    if len(dset[0])==2:
        mfun = interp1d(gridt[dset],gridbc[dset],bounds_error=False) 
        itp  = mfun(val)        
    if len(dset[0])<2:
        itp = np.nan
    return(itp)

# read input tables of BCs for several values of E(B-V)
files = [pkg_resources.resource_filename(__name__, "data/GALAH_DR4/auxiliary_information/BC_Tables/grid/STcolors_2MASS_GaiaDR2_EDR3_Rv3.1_EBV_0.00.dat")]

gebv   = [0.0]
gri_bc = []

kk=0
for f in files:

    grid = Table.read(f,format='ascii')
    if kk==0:
        gteff, gfeh, glogg = grid['Teff'],grid['feh'],grid['logg']

    bc_g2  = grid['mbol']-grid['G2']
    bc_bp2 = grid['mbol']-grid['BP2']
    bc_rp2 = grid['mbol']-grid['RP2']

    bc_g3  = grid['mbol']-grid['G3']
    bc_bp3 = grid['mbol']-grid['BP3']
    bc_rp3 = grid['mbol']-grid['RP3']

    bc_j   = grid['mbol']-grid['J']
    bc_h   = grid['mbol']-grid['H']
    bc_k   = grid['mbol']-grid['Ks']

    tmp = np.transpose([bc_g2,bc_bp2,bc_rp2,bc_g3,bc_bp3,bc_rp3,bc_j,bc_h,bc_k])
    gri_bc.append(tmp)

    kk=kk+1

gebv   = np.array(gebv)
gri_bc = np.array(gri_bc)

parsec = Table.read(pkg_resources.resource_filename(__name__, "data/GALAH_DR4/auxiliary_information/parsec_isochrones/parsec_isochrones_logt_8p00_0p01_10p17_mh_m2p75_0p25_m0p75_mh_m0p60_0p10_0p70_GaiaEDR3_2MASS.fits"))

def bcstar(teff,logg,feh,alpha_fe):
    '''
    compute Bolometric Corrections for stars of known input parameters    
    '''
#     teff = np.min([np.max([teff,np.min(grid['Teff'])]),np.max(grid['Teff'])])
#     if teff < 3900:
#         logg = np.min([np.max([logg,np.min(grid['logg'])]),5.5])
#     else:
#         logg = np.min([np.max([logg,np.min(grid['logg'])]),5.0])
#     feh = np.min([np.max([feh,np.min(grid['feh'])]),np.max(grid['feh'])])
    
    frange = [8]
    flist = ['BC_Ks']
    rmi = [8]

    itp_bc = np.nan
    arr_bc  = np.nan

    fold      = [feh]
        
    # take +/-3 steps in [Fe/H] grid
    snip = np.concatenate(bracket(fold,gfeh,3))
    itp1 = np.zeros((len(snip)))+np.nan
    
    for k in range(len(snip)):

        x0   = np.where((gfeh==snip[k]) & (np.abs(glogg-logg)<1.1))
        lg0  = np.array(list(dict.fromkeys(list(glogg[x0]))))
        itp0 = np.zeros((len(lg0)))+np.nan

        # at given logg and feh, range of Teff to interpolate across
        for j in range(len(lg0)):
            ok      = np.where((np.abs(gteff-teff)<1000) & \
                               (gfeh==snip[k]) & (glogg==lg0[j]))

            itp0[j] = mal(teff,gteff,gri_bc[0,:,8],ok)

        # remove any nan, in case. Either of itp[?,:,:] is enough
        k0 = np.where(np.isnan(itp0[:])==False)
        # interpolate in logg at correct Teff
        itp1[k] = mal(logg,lg0,itp0[:],k0)
        
    k1  = np.where(np.isnan(itp1[:])==False)
    
    bc_ks = mal(fold,snip,itp1[:],k1)

    if np.isnan(bc_ks):
        
        bc_grid = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/GALAH_DR4/auxiliary_information/BC_Tables/grid/STcolors_2MASS_GaiaDR2_EDR3_Rv3.1_EBV_0.00.dat"), names=True)
        
        file = open(pkg_resources.resource_filename(__name__, "data/GALAH_DR4/auxiliary_information/BC_Tables/grid/bc_grid_kdtree_ebv_0.00.pickle"), 'rb')
        
        bc_kdtree = pickle.load(file)
        file.close()
        
        bc_distance_matches, bc_closest_matches = bc_kdtree.query(np.array([np.log10(teff),logg,feh,alpha_fe]).T,k=8)
        bc_ks = np.average(bc_grid['mbol'][bc_closest_matches] - bc_grid['Ks'][bc_closest_matches],weights=bc_distance_matches,axis=-1)
        
    else:
        bc_ks = bc_ks[0]
        
    return(bc_ks)

def calculate_age_mass(teff, logg, loglum, m_h, e_teff=100, e_logg=0.5, e_loglum=0.1, e_m_h=0.2, useChabrier=False, debug=False):

    e_loglum = e_loglum * loglum
    
    # Make sure that [Fe/H] stays within parsec grid limits
    unique_m_h = np.unique(parsec['m_h'])
    if m_h < unique_m_h[0]:
        m_h = unique_m_h[0] + 0.001
        print('adjust m_h input to ',m_h)
    if m_h > unique_m_h[-1]:
        m_h = unique_m_h[-1] - 0.001
        print('adjust m_h input to ',m_h)
        
    # Make sure we have at least 2 [Fe/H] dimensions to integrate over
    lower_boundary_m_h = np.argmin(np.abs(unique_m_h - (m_h - e_m_h)))
    upper_boundary_m_h = np.argmin(np.abs(unique_m_h - (m_h + e_m_h)))
    if lower_boundary_m_h == upper_boundary_m_h:
        if lower_boundary_m_h == 0:
            upper_boundary_m_h = 1
        if lower_boundary_m_h == len(unique_m_h)-1:
            lower_boundary_m_h = len(unique_m_h)-2
    
    # find all relevant isochrones points
    relevant_isochrone_points = (
        (parsec['logT'] > np.log10(teff - e_teff)) & 
        (parsec['logT'] < np.log10(teff + e_teff)) &
        (parsec['logg'] > logg - e_logg) & 
        (parsec['logg'] < logg + e_logg) &
        (parsec['logL'] > loglum - e_loglum) & 
        (parsec['logL'] < loglum + e_loglum) &
        (parsec['m_h']  >= unique_m_h[lower_boundary_m_h]) & 
        (parsec['m_h']  <= unique_m_h[upper_boundary_m_h])
    )
    # if len(parsec['logT'][relevant_isochrone_points]) < 10:
    #     print('Only '+str(len(parsec['logT'][relevant_isochrone_points]))+' isochrones points available')
    
    # 
    model_points = np.array([
        10**parsec['logT'][relevant_isochrone_points],
        parsec['logg'][relevant_isochrone_points],
        parsec['logL'][relevant_isochrone_points],
        parsec['m_h'][relevant_isochrone_points]
    ]).T
    
    # find normalising factor
    norm = np.log(np.sqrt((2.*np.pi)**4.*np.prod(np.array([e_teff, e_logg, e_loglum ,e_m_h])**2)))
    
    # sum up lnProb and weight ages/masses by 
    if useChabrier:
        lnLike = - np.sum(((model_points - [teff, logg, loglum, m_h])/[e_teff, e_logg, e_loglum, e_m_h])**2, axis=1) - norm        
        lnPrior = np.log(22.8978 * np.exp( - (716.4/parsec['mass'][relevant_isochrone_points])**0.25) * (parsec['mass'][relevant_isochrone_points])**(-3.3))   
        lnProb = lnLike + lnPrior
        
        if debug:
            f, gs = plt.subplots(1,4,figsize=(15,5))
            ax = gs[0]
            s = ax.scatter(model_points[:,0],model_points[:,1],c=lnLike,s=1)
            ax.set_xlabel('teff')
            ax.set_ylabel('logg')
            c = plt.colorbar(s,ax=ax)
            c.set_label('lnLike')
            ax = gs[1]
            s = ax.scatter(model_points[:,0],model_points[:,1],c=lnPrior,s=1)
            ax.set_xlabel('teff')
            ax.set_ylabel('logg')
            c = plt.colorbar(s,ax=ax)
            c.set_label('lnPrior')
            ax = gs[2]
            s = ax.scatter(model_points[:,0],model_points[:,1],c=lnProb,s=1)
            ax.set_xlabel('teff')
            ax.set_ylabel('logg')
            c = plt.colorbar(s,ax=ax)
            c.set_label('lnProb')
            ax = gs[3]
            s = ax.scatter(parsec['mass'][relevant_isochrone_points],model_points[:,1],c=lnPrior,s=1)
            ax.set_xlabel('mass')
            ax.set_ylabel('logg')
            c = plt.colorbar(s,ax=ax)
            c.set_label('lnPrior')
            plt.tight_layout()
            plt.show()
            plt.close()
        
    else:
        lnProb = - np.sum(((model_points - [teff, logg, loglum, m_h])/[e_teff, e_logg, e_loglum, e_m_h])**2, axis=1) - norm        
        
    age = np.sum(10**parsec['logAge'][relevant_isochrone_points] * np.exp(lnProb)/10**9)
    mass = np.sum(parsec['mass'][relevant_isochrone_points] * np.exp(lnProb))
    
    # Normalise by probability
    Prob_sum = np.sum(np.exp(lnProb))
    age /= Prob_sum
    mass /= Prob_sum

    if debug:
        plt.figure()
        plt.hist(parsec['mass'][relevant_isochrone_points],bins=30)
        print(teff, logg, loglum, m_h)
        print(e_teff, e_logg, e_loglum, e_m_h)
        print('min_max_teff',10**np.min(parsec['logT'][relevant_isochrone_points]),10**np.max(parsec['logT'][relevant_isochrone_points]))
        print('min_max_logg',np.min(parsec['logg'][relevant_isochrone_points]),np.max(parsec['logg'][relevant_isochrone_points]))
        print('min_max_m_h',np.min(parsec['m_h'][relevant_isochrone_points]),np.max(parsec['m_h'][relevant_isochrone_points]))
        print('min_max_mass',np.min(parsec['mass'][relevant_isochrone_points]),np.max(parsec['mass'][relevant_isochrone_points]))
        print('age',str(age))
        print('mass',str(mass))
        plt.show()
        plt.close()
    
    return(age, mass)

def calculate_logg_parallax(teff, logg_in, fe_h, ks_m, ks_msigcom, r_med, r_lo, r_hi, a_ks, e_teff=100, e_logg=0.25, e_m_h=0.2):
    '''
    Main function to estimate photometric logg.
    
    r_med : 
        The distance of the star in pc.
    '''
    if fe_h < -1:
        alpha_fe = 0.4
    elif fe_h > 0:
        alpha_fe = 0.0
    else:
        alpha_fe = -0.4 *fe_h
    
    m_h = fe_h + np.log10(10**alpha_fe * 0.694 + 0.306)
        
    bc_ks = bcstar(teff, logg_in, fe_h, alpha_fe)
    
    loglbol = - 0.4 * (ks_m - 5.0*np.log10(r_med/10.) + bc_ks - a_ks - 4.75)#[0]
    # Take into account uncertainties of Ks, distance, and adds uncertainties of +- 0.05 mag for A(Ks) and BC(Ks)
    loglbol_lo = - 0.4 * (ks_m + ks_msigcom - 5.0*np.log10(r_lo/10.) + (bc_ks + 0.05) - (a_ks - 0.05) - 4.75)#[0]
    loglbol_hi = - 0.4 * (ks_m - ks_msigcom - 5.0*np.log10(r_hi/10.) + (bc_ks - 0.05) - (a_ks + 0.05) - 4.75)#[0]
    
    e_loglum = 0.5*(loglbol_hi-loglbol_lo) / loglbol

    age, mass = calculate_age_mass(teff, logg_in, loglbol, m_h, e_teff, e_logg, e_loglum, e_m_h)
    if np.isnan(mass):
        if sys.argv[1] == '-f':
            print('Mass could not be estimated, trying again with 2x errors')
        age, mass = calculate_age_mass(teff, logg_in, loglbol, m_h, e_teff*2, e_logg*2, e_loglum*2, e_m_h*2)
        if np.isnan(mass):
            if sys.argv[1] == '-f':
                print('Mass could not be estimated, trying again with 3x the errors')
            age, mass = calculate_age_mass(teff, logg_in, loglbol, m_h, e_teff*3, e_logg*3, e_loglum*3, e_m_h*3)
            if np.isnan(mass):
                if sys.argv[1] == '-f':
                    print('Mass could not be estimated, assuming 1Mbol')
                mass = 1.0
                age = np.NaN
        
    return(4.438 + np.log10(mass) + 4*np.log10(teff/5772.) - loglbol, mass, age, bc_ks, 10**loglbol, loglbol_lo, loglbol_hi)

# Measure logg from ionization balance

def mv_fit(sub_sme_single, fit_paras, fit_bounds):
    with redirect_stdout(open(f"/dev/null", 'w')):
        sub_sme_single = solve(sub_sme_single, fit_paras, bounds=fit_bounds)
    return sub_sme_single
    
def measure_monh_fe12(teff, logg, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist, nlte=False):
    '''
    Measure [M/H] using Fe 1 and Fe 2 lines. For logg measurement.
    '''

    # Measure [M/H] from Fe 1 lines
    wlcent, wlcent_mean, EW, monh_measure, monh_err_measure, vmic_measure, vmic_err_measure = [], [], [], [], [], [], []
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    if nlte:
        sme.nlte.set_nlte('Fe')
    sme.cscale_flag = 'constant'
    
    sub_sme = []
    fit_paras, fit_bounds = ['monh'], [-1, 0.8]
    for i in tqdm(fe1_lines.index):
        line_wav = fe1_lines.loc[i, 'wlcent']
        if type(line_wav) != list:
            hwhm = (vmac+vsini)/3e5*line_wav
        else:
            hwhm = (vmac+vsini)/3e5*line_wav[0]
    
        sub_sme_single = deepcopy(sme)
        if type(line_wav) != list:
            indices = (wave >= line_wav-2*hwhm) & (wave <= line_wav+2*hwhm)
        else:
            indices = (wave >= line_wav[0]-2*hwhm) & (wave <= line_wav[-1]+2*hwhm)
        sub_sme_single.wave = wave[indices]
        sub_sme_single.spec = flux[indices]
        sub_sme_single.uncs = flux_err[indices]
        sub_sme_single.linelist = linelist[~((linelist['line_range_e'] < sub_sme_single.wave[0][0]) | (linelist['line_range_s'] > sub_sme_single.wave[0][-1]))]
        if len(sub_sme_single.linelist) == 0:
            print(sub_sme_single.wave[0])
            raise ValueError('Empty linelist.')
        sub_sme.append([sub_sme_single, fit_paras, fit_bounds])
        wlcent.append(line_wav)
        wlcent_mean.append(np.mean(line_wav))

    pickle.dump(sub_sme, open('temp.pkl', 'wb'))
    sub_sme = pqdm(sub_sme, mv_fit, n_jobs=10, argument_type='args')
    mv_measure_fe1 = pd.DataFrame({'wlcent':wlcent, 'wlcent_mean':wlcent_mean, 
                  'EW':[np.sum(1 - sub_sme_single.synth[0]) * np.mean(np.diff(sub_sme_single.wave[0])) for sub_sme_single in sub_sme]})
    mv_measure_fe1['log(EW/wlcent)'] = np.log10(mv_measure_fe1['EW']/mv_measure_fe1['wlcent_mean'])
    mv_measure_fe1['monh'] = [sub_sme_single.fitresults['values'][0] for sub_sme_single in sub_sme]
    mv_measure_fe1['monh_err'] = [sub_sme_single.fitresults['fit_uncertainties'][0] for sub_sme_single in sub_sme]
    
    # Measure [M/H] from Fe 2 lines
    wlcent, wlcent_mean, EW, monh_measure, monh_err_measure, vmic_measure, vmic_err_measure = [], [], [], [], [], [], []
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    if nlte:
        sme.nlte.set_nlte('Fe')
    sme.cscale_flag = 'constant'
    
    sub_sme = []
    fit_paras, fit_bounds = ['monh'], [-1, 0.8]
    for i in tqdm(fe2_lines.index):
        line_wav = fe2_lines.loc[i, 'wlcent']
        if type(line_wav) != list:
            hwhm = (vmac+vsini)/3e5*line_wav
        else:
            hwhm = (vmac+vsini)/3e5*line_wav[0]
    
        sub_sme_single = deepcopy(sme)
        if type(line_wav) != list:
            indices = (wave >= line_wav-2*hwhm) & (wave <= line_wav+2*hwhm)
        else:
            indices = (wave >= line_wav[0]-2*hwhm) & (wave <= line_wav[-1]+2*hwhm)
        sub_sme_single.wave = wave[indices]
        sub_sme_single.spec = flux[indices]
        sub_sme_single.uncs = flux_err[indices]
        sub_sme_single.linelist = linelist[~((linelist['line_range_e'] < sub_sme_single.wave[0][0]) | (linelist['line_range_s'] > sub_sme_single.wave[0][-1]))]
        sub_sme.append([sub_sme_single, fit_paras, fit_bounds])
        wlcent.append(line_wav)
        wlcent_mean.append(np.mean(line_wav))
    
    sub_sme = pqdm(sub_sme, mv_fit, n_jobs=10, argument_type='args')
    mv_measure_fe2 = pd.DataFrame({'wlcent':wlcent, 'wlcent_mean':wlcent_mean, 
                  'EW':[np.sum(1 - sub_sme_single.synth[0]) * np.mean(np.diff(sub_sme_single.wave[0])) for sub_sme_single in sub_sme]})
    mv_measure_fe2['log(EW/wlcent)'] = np.log10(mv_measure_fe2['EW']/mv_measure_fe2['wlcent_mean'])
    mv_measure_fe2['monh'] = [sub_sme_single.fitresults['values'][0] for sub_sme_single in sub_sme]
    mv_measure_fe2['monh_err'] = [sub_sme_single.fitresults['fit_uncertainties'][0] for sub_sme_single in sub_sme]
    
    # Get the monh and precision from Fe1 and 2 lines
    indices_fe1 = np.abs(mv_measure_fe1['monh'] - np.median(mv_measure_fe1['monh'])) <= 2*np.std(mv_measure_fe1['monh'])
    monh_measure_fe1 = np.mean(mv_measure_fe1.loc[indices_fe1, 'monh'])
    scatter_fe1 = np.std(mv_measure_fe1.loc[indices_fe1, 'monh'])
    
    indices_fe2 = np.abs(mv_measure_fe2['monh'] - np.median(mv_measure_fe2['monh'])) <= 2*np.std(mv_measure_fe2['monh'])
    monh_measure_fe2 = np.mean(mv_measure_fe2.loc[indices_fe2, 'monh'])
    scatter_fe2 = np.std(mv_measure_fe2.loc[indices_fe2, 'monh'])

    p_value = stats.ttest_ind(mv_measure_fe1.loc[indices_fe1, 'monh'], mv_measure_fe2.loc[indices_fe2, 'monh'], equal_var=False).pvalue
    
    return([monh_measure_fe1, scatter_fe1, monh_measure_fe2, scatter_fe2, p_value, mv_measure_fe1, mv_measure_fe2])

def measure_logg_fe12(teff, logg_array, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist, p_threshold=0.05, logg_diff=0.05):

    monh_fe12_res_all = []
    #Run measure_monh_fe12 for the logg_array
    for logg in tqdm(logg_array):
        monh_fe12_res = measure_monh_fe12(teff, logg, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist)
        monh_fe12_res_all.append(monh_fe12_res)

    # Refine sampling： Insert grid points between p_value threshold
    p_values = np.array([ele[4] for ele in monh_fe12_res_all])
    monh1_values = np.array([ele[0] for ele in monh_fe12_res_all])
    monh2_values = np.array([ele[2] for ele in monh_fe12_res_all])
    logg_interp = np.linspace(np.min(logg_array), np.max(logg_array), 500)
    del_monh_interp = np.interp(logg_interp, logg_array, monh1_values-monh2_values)
    logg_1st_est = logg_interp[np.argmin(np.abs(del_monh_interp))]
    print('Best logg: ', logg_1st_est)
    monh_fe12_res = measure_monh_fe12(teff, logg_1st_est, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist)
    logg_array = np.concatenate([logg_array, [logg_1st_est]])
    monh_fe12_res_all.append(monh_fe12_res)
    p_values = np.concatenate([p_values, [monh_fe12_res[4]]])

    unique_indices = np.unique(logg_array, return_index=True)[1]
    # 根据索引重新排列 x 和 y
    logg_array = logg_array[unique_indices]
    p_values = p_values[unique_indices]
    monh_fe12_res_all = [monh_fe12_res_all[i] for i in unique_indices]
    
    # Measure logg from monh_fe12_res_all
    mask = p_values > p_threshold
    start_index, end_index = -99, -99
    if np.any(mask):
        start_index = np.where(mask)[0][0]  # 第一个超过阈值的位置
        if start_index != 0:
            start_index -= 1
        end_index = np.where(mask)[0][-1]  # 最后一个超过阈值的位置
        if end_index != len(logg_array)-1:
            end_index += 1
    # elif ~np.any(mask) and np.max(p_values) <= p_threshold:
    else:
        max_index = np.argmax(p_values)
        p_values_exclude_max = np.delete(p_values, max_index)
        # 找到新数组中最大值的索引
        second_max_index_in_reduced = np.argmax(p_values_exclude_max)
        # 转换回原数组中的索引
        second_max_index = np.arange(len(p_values))[np.arange(len(p_values)) != max_index][second_max_index_in_reduced]
        if max_index > second_max_index:
            start_index, end_index = second_max_index, max_index
        else:
            start_index, end_index = max_index, second_max_index
        
    if start_index != -99 and end_index != -99:
        # 加密采样区间
        fine_logg = np.arange(logg_array[start_index], logg_array[end_index] + logg_diff, logg_diff)
        print('Find logg created:', fine_logg)
        fine_monh_fe12 = [measure_monh_fe12(teff, logg, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist) for logg in tqdm(fine_logg)]
        # return monh_fe12_res_all, fine_monh_fe12
        # 合并采样点
        logg_array = np.concatenate([logg_array, fine_logg])
        monh_fe12_res_all += fine_monh_fe12

        # 去重并按顺序排列
        # 找到唯一元素的索引
        unique_indices = np.unique(logg_array, return_index=True)[1]
        
        # 根据索引重新排列 x 和 y
        logg_array = logg_array[unique_indices]
        monh_fe12_res_all = [monh_fe12_res_all[i] for i in unique_indices]

        sorted_indices = np.argsort(logg_array)
        logg_array = logg_array[sorted_indices]
        monh_fe12_res_all = [monh_fe12_res_all[i] for i in sorted_indices]
        
    return logg_array, monh_fe12_res_all

def measure_logg_from_p(logg_array, p_array, threshold=0.5):
    crossing_indices = np.where((p_array[:-1] < threshold) & (p_array[1:] > threshold) |  (p_array[:-1] > threshold) & (p_array[1:] < threshold))[0]
    
    # 线性插值计算精确的 x 值
    logg_bound = []
    for i in crossing_indices:
        x1, x2 = logg_array[i], logg_array[i+1]
        y1, y2 = p_array[i], p_array[i+1]
        
        # 线性插值公式
        interpolated_x = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
        logg_bound.append(interpolated_x)

    logg = logg_array[np.argmax(p_array)]

    return logg, np.array(logg_bound)

def measure_logg_fe12_main(wave, flux, flux_err, teff, logg, monh, vmic, vmac, vsini, R, line_list,
                           logg_array=np.arange(0, 4.5+0.1, 0.25), 
                           abund=None, 
                           sensitivity_dominance_thres=0.6,
                           line_dominance_thres=0.5,
                           max_line_num=50,
                           plot=False, save_plot=None):
    '''
    This is the main entrypoint for measuring logg using ionization balance.
    '''

    if abund is None:
        abund = Abund.solar()
        abund.monh = monh
    
    if not {'central_depth', 'line_range_s', 'line_range_e'}.issubset(line_list._lines.columns) or np.abs(line_list.cdepth_range_paras[0]-teff) >= 500 or (np.abs(line_list.cdepth_range_paras[1]-logg) >= 1) or (np.abs(line_list.cdepth_range_paras[2]-monh) >= 0.5):
        # Calculate the line depth from the pre-set stellar parameter
        sme = SME_Structure()
        sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
        sme.abund = abund
        use_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
        use_list = use_list[use_list['central_depth'] > 0.1]
    else:
        use_list = line_list[line_list['central_depth'] > 0.1]

    v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5/R)**2)
    
    # Select the Fe 1 and Fe 2 lines for ionization balance.
    fit_line_group = pysme_abund.find_line_groups(wave, ['Fe'], [1, 2], use_list, v_broad)
    spec_syn = pysme_abund.get_sensitive_synth(wave, R, teff, logg, monh, vmic, vmac, vsini, line_list, abund, ['Fe'], [1, 2], fit_line_group)
    fe_lines = pysme_abund.select_lines(fit_line_group, spec_syn, ['Fe'], [1, 2], 
                                                   sensitivity_dominance_thres=sensitivity_dominance_thres, 
                                                   line_dominance_thres=line_dominance_thres, 
                                                   max_line_num=max_line_num)
    fe_lines['Fe'][1] = fe_lines['Fe'][1].sort_values('wav_s')
    fe_lines['Fe'][2] = fe_lines['Fe'][2].sort_values('wav_s')

    # Search for the logg which gives ionization balance.
    # Find the logg range
    s = Synthesizer()
    sme = SME_Structure()
    a = AtmosphereInterpolator(lfs_atmo=s.lfs_atmo)
    atmo_file = a.lfs_atmo.get(sme.atmo.source)
    atmo_grid = SavFile(atmo_file, source=sme.atmo.source, lfs=setup_atmo())
    indices = (np.abs(atmo_grid.teff-teff) < 200) & (np.abs(atmo_grid.monh-monh) < 0.2) 
    logg_array = np.linspace(min(atmo_grid.logg[indices]), max(atmo_grid.logg[indices]), 7)
    logg_array, monh_fe12_res_all = measure_logg_fe12(teff, logg_array, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe_lines['Fe'][1], fe_lines['Fe'][2], 
                                                 use_list)
    logg, logg_bound = measure_logg_from_p(logg_array, np.array([ele[4] for ele in monh_fe12_res_all]))

    if plot:
        plt.figure(dpi=150)
        plt.plot(logg_array, [ele[0] for ele in monh_fe12_res_all], 'o-', label='Fe 1 lines')
        plt.fill_between(logg_array, [ele[0]-ele[1] for ele in monh_fe12_res_all], [ele[0]+ele[1] for ele in monh_fe12_res_all], alpha=0.1)
        plt.plot(logg_array, [ele[2] for ele in monh_fe12_res_all], 'o-', label='Fe 2 lines')
        plt.fill_between(logg_array, [ele[2]-ele[3] for ele in monh_fe12_res_all], [ele[2]+ele[3] for ele in monh_fe12_res_all], alpha=0.1)
        plt.legend(loc=2)
        plt.xlabel('$\log{g}$')
        plt.ylabel('[M/H] (solid)')
        
        plt.twinx()
        plt.plot(logg_array, np.array([ele[4] for ele in monh_fe12_res_all]), 'o--', c='C2') 
        plt.axvline(logg, c='C2')
        plt.axvspan(*logg_bound, facecolor='C2', alpha=0.2)
        plt.ylabel('T-test $p$ (dashed)')
        plt.title(f'$\log{{g}}$={logg:.2f}$^{{+{logg_bound[1] - logg:.2f}}}_{{{logg_bound[0] - logg:.2f}}}$')
        if save_plot is not None:
            plt.savefig(f'{save_plot}/logg_fe12_measure.png')
    return logg, logg_bound

#---------------------Vmic+monh-----------------------------

def find_range(arr, threshold=0.05):
    """
    从最大值位置开始，向左右扩展，寻找值下降到 threshold 处的范围。

    参数:
    - arr: np.ndarray, 目标数组
    - threshold: float, 设定的下降阈值（默认 0.05）

    返回:
    - left_idx: 下降到 threshold 的左边界索引
    - right_idx: 下降到 threshold 的右边界索引
    """
    # 找到最大值的索引
    max_idx = np.argmax(arr)
    
    # 初始化左右边界索引
    left_idx, right_idx = max_idx, max_idx
    
    # 向左搜索，找到下降到 threshold 的位置
    while left_idx > 0 and arr[left_idx] > threshold:
        left_idx -= 1
    
    # 向右搜索，找到下降到 threshold 的位置
    while right_idx < len(arr) - 1 and arr[right_idx] > threshold:
        right_idx += 1
    
    return left_idx, right_idx

def measure_monh_vmic_fe12(teff, logg, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist, nlte=False, margin=2):
    wlcent, wlcent_mean, EW, monh_measure, monh_err_measure, vmic_measure, vmic_err_measure = [], [], [], [], [], [], []
    margin = 2
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = 50000
    sme.cscale_flag = 'constant'
    
    sub_sme = []
    fit_paras, fit_bounds = ['monh', 'vmic'], [[-1, 0], [0.8, 4]]
    for i in tqdm(fe1_lines.index):
        line_wav = fe1_lines.loc[i, 'wlcent']
        if type(line_wav) != list:
            hwhm = (vmac+vsini)/3e5*line_wav
        else:
            hwhm = (vmac+vsini)/3e5*line_wav[0]
    
        sub_sme_single = deepcopy(sme)
        if type(line_wav) != list:
            indices = (wave >= line_wav-2*hwhm) & (wave <= line_wav+2*hwhm)
        else:
            indices = (wave >= line_wav[0]-2*hwhm) & (wave <= line_wav[-1]+2*hwhm)
        sub_sme_single.wave = wave[indices]
        sub_sme_single.spec = flux[indices]
        sub_sme_single.uncs = flux_err[indices]
        sub_sme_single.linelist = linelist[~((linelist['line_range_e'] < sub_sme_single.wave[0][0]) | (linelist['line_range_s'] > sub_sme_single.wave[0][-1]))]
        sub_sme.append([sub_sme_single, fit_paras, fit_bounds])
        wlcent.append(line_wav)
        wlcent_mean.append(np.mean(line_wav))
    
    for i in tqdm(fe2_lines.index):
        line_wav = fe2_lines.loc[i, 'wlcent']
        if type(line_wav) != list:
            hwhm = (vmac+vsini)/3e5*line_wav
        else:
            hwhm = (vmac+vsini)/3e5*line_wav[0]
    
        sub_sme_single = deepcopy(sme)
        if type(line_wav) != list:
            indices = (wave >= line_wav-2*hwhm) & (wave <= line_wav+2*hwhm)
        else:
            indices = (wave >= line_wav[0]-2*hwhm) & (wave <= line_wav[-1]+2*hwhm)
        sub_sme_single.wave = wave[indices]
        sub_sme_single.spec = flux[indices]
        sub_sme_single.uncs = flux_err[indices]
        sub_sme_single.linelist = linelist[~((linelist['line_range_e'] < sub_sme_single.wave[0][0]) | (linelist['line_range_s'] > sub_sme_single.wave[0][-1]))]
        sub_sme.append([sub_sme_single, fit_paras, fit_bounds])
        wlcent.append(line_wav)
        wlcent_mean.append(np.mean(line_wav))

    sub_sme = pqdm(sub_sme, mv_fit, n_jobs=10, argument_type='args')
    mv_measure = pd.DataFrame({'wlcent':wlcent, 'wlcent_mean':wlcent_mean, 
                  'EW':[np.sum(1 - sub_sme_single.synth[0]) * np.mean(np.diff(sub_sme_single.wave[0])) for sub_sme_single in sub_sme], 
                  'monh':[sub_sme_single.fitresults['values'][0] for sub_sme_single in sub_sme], 
                  'monh_err':[sub_sme_single.fitresults['fit_uncertainties'][0] for sub_sme_single in sub_sme],
                  'vmic':[sub_sme_single.fitresults['values'][1] for sub_sme_single in sub_sme], 
                  'vmic_err':[sub_sme_single.fitresults['fit_uncertainties'][0] for sub_sme_single in sub_sme]})
    return mv_measure

def measure_vmic(teff, logg, monh, vmic_array, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist, p_threshold=0.05, vmic_diff=0.15, EW_thres=0):

    slope_all, p_values, monh_all, monh_err_all = [], [], [], []
    #Run measure_monh_fe12 for the logg_array
    for vmic in tqdm(vmic_array):
        monh_fe12_res = pd.concat(measure_monh_fe12(teff, logg, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist)[5:]).reset_index(drop=True)
        indices = (monh_fe12_res['monh_err'] < np.median(monh_fe12_res['monh_err'])+np.std(monh_fe12_res['monh_err'])) & (monh_fe12_res['EW'] > EW_thres/1000)
        # Perform weighted linear fitting
        x, y = monh_fe12_res.loc[indices, 'log(EW/wlcent)'].values, monh_fe12_res.loc[indices, 'monh'].values
        yerr = monh_fe12_res.loc[indices, 'monh_err'].values    
        fit_res = linregress(x, y)
        weights = 1/yerr**2
        monh_mean = np.average(y, weights=weights)
        monh_mean_err = np.std(y)
        monh_all.append(monh_mean)
        monh_err_all.append(monh_mean_err)
        slope_all.append(fit_res[0])
        p_values.append(fit_res[3])

    # if np.all(np.array(p_values) < 0.05):
    #     # Find the slope 0 using linear interpolation
    #     print("Didn't find vmic with p-value > 0.05, find it from 0 slope.")
    #     interp_x = np.linspace(0, 4, 100)
    #     interp_y = np.abs(np.interp(interp_x, vmic_array, slope_all))
    #     min_slope_vmic = interp_x[np.argmin(interp_y)]
    #     monh_fe12_res = pd.concat(measure_monh_fe12(teff, logg, monh, min_slope_vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist)[5:]).reset_index(drop=True)
    #     indices = (monh_fe12_res['monh_err'] < np.median(monh_fe12_res['monh_err'])+np.std(monh_fe12_res['monh_err'])) & (monh_fe12_res['EW'] > 10/1000)
    #     # Perform weighted linear fitting
    #     x, y = monh_fe12_res.loc[indices, 'log(EW/wlcent)'].values, monh_fe12_res.loc[indices, 'monh'].values
    #     yerr = monh_fe12_res.loc[indices, 'monh_err'].values    
    #     fit_res = linregress(x, y)
    #     weights = 1/yerr**2
    #     monh_mean = np.average(y, weights=weights)
    #     monh_mean_err = np.std(y)
        
    #     print(min_slope_vmic)
    #     vmic_array = np.concatenate([vmic_array, [min_slope_vmic]])
    #     monh_all.append(monh_mean)
    #     monh_err_all.append(monh_mean_err)
    #     slope_all.append(fit_res[0])
    #     p_values.append(fit_res[3])

    # Refine sampling： Insert grid points between p_value threshold
    vmic_interp = np.linspace(np.min(vmic_array), np.max(vmic_array), 500)
    slope_interp = np.interp(vmic_interp, vmic_array, slope_all)
    vmic_1st_est = vmic_interp[np.argmin(np.abs(slope_interp))]
    print('Best vmic: ', vmic_1st_est)
    monh_fe12_res = pd.concat(measure_monh_fe12(teff, logg, monh, vmic_1st_est, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist)[5:]).reset_index(drop=True)
    indices = (monh_fe12_res['monh_err'] < np.median(monh_fe12_res['monh_err'])+np.std(monh_fe12_res['monh_err'])) & (monh_fe12_res['EW'] > 10/1000)
    # Perform weighted linear fitting
    x, y = monh_fe12_res.loc[indices, 'log(EW/wlcent)'].values, monh_fe12_res.loc[indices, 'monh'].values
    yerr = monh_fe12_res.loc[indices, 'monh_err'].values
    fit_res = linregress(x, y)
    weights = 1/yerr**2
    monh_mean = np.average(y, weights=weights)
    monh_mean_err = np.std(y)
    vmic_array = np.concatenate([vmic_array, [vmic_1st_est]])
    monh_all.append(monh_mean)
    monh_err_all.append(monh_mean_err)
    slope_all.append(fit_res[0])
    p_values.append(fit_res[3])

    unique_indices = np.unique(vmic_array, return_index=True)[1]
    # 根据索引重新排列 x 和 y
    vmic_array = vmic_array[unique_indices]
    p_values = np.array(p_values)[unique_indices]
    slope_all = np.array(slope_all)[unique_indices]
    monh_all = np.array(monh_all)[unique_indices]
    monh_err_all = np.array(monh_err_all)[unique_indices]

    sorted_indices = np.argsort(vmic_array)
    vmic_array = vmic_array[sorted_indices]
    monh_all = list(np.array(monh_all)[sorted_indices])
    monh_err_all = list(np.array(monh_err_all)[sorted_indices])
    slope_all = list(np.array(slope_all)[sorted_indices])
    p_values = list(np.array(p_values)[sorted_indices])
    # pickle.dump([vmic_array, monh_all, monh_err_all, slope_all, p_values], open('temp.pkl', 'wb'))
    # return vmic_array, monh_all, monh_err_all, slope_all, p_values
    # print(vmic_array)
    # print(p_values)
    
    print(vmic_array)
    print(slope_all)
    print(p_values)
    # Refine sampling： Insert grid points between p_value threshold
    mask = np.array(p_values) > p_threshold
    start_index, end_index = -99, -99
    if np.any(mask):
        start_index = np.where(mask)[0][0]  # 第一个超过阈值的位置
        if start_index != 0:
            start_index -= 1
        end_index = np.where(mask)[0][-1]  # 最后一个超过阈值的位置
        if end_index != len(vmic_array)-1:
            end_index += 1
    elif ~np.any(mask) and np.max(p_values) <= p_threshold:
        max_index = np.argmax(p_values)
        p_values_exclude_max = np.delete(p_values, max_index)
        # 找到新数组中最大值的索引
        second_max_index_in_reduced = np.argmax(p_values_exclude_max)
        # 转换回原数组中的索引
        second_max_index = np.arange(len(p_values))[np.arange(len(p_values)) != max_index][second_max_index_in_reduced]
        if max_index > second_max_index:
            start_index, end_index = second_max_index, max_index
        else:
            start_index, end_index = max_index, second_max_index
            
    if start_index != -99 and end_index != -99:
        # 加密采样区间
        fine_vmic = np.arange(vmic_array[start_index], vmic_array[end_index] + vmic_diff, vmic_diff)
        print(start_index, end_index)
        print(vmic_array[start_index], vmic_array[end_index])
        print(fine_vmic)
        fine_monh_fe12 = [measure_monh_fe12(teff, logg, monh, vmic, vmac, vsini, R, wave, flux, flux_err, fe1_lines, fe2_lines, linelist)[5:] for vmic in tqdm(fine_vmic)]
        for fine_monh_fe12_single in fine_monh_fe12:
            monh_fe12_res = pd.concat(fine_monh_fe12_single).reset_index(drop=True)
            indices = (monh_fe12_res['monh_err'] < np.median(monh_fe12_res['monh_err'])+np.std(monh_fe12_res['monh_err'])) & (monh_fe12_res['EW'] > 10/1000)
            # Perform weighted linear fitting
            x, y = monh_fe12_res.loc[indices, 'log(EW/wlcent)'].values, monh_fe12_res.loc[indices, 'monh'].values
            yerr = monh_fe12_res.loc[indices, 'monh_err'].values    
            fit_res = linregress(x, y)
            weights = 1/yerr**2
            monh_mean = np.average(y, weights=weights)
            monh_mean_err = np.std(y)
            monh_all.append(monh_mean)
            monh_err_all.append(monh_mean_err)
            slope_all.append(fit_res[0])
            p_values.append(fit_res[3])
        # 合并采样点
        vmic_array = np.concatenate([vmic_array, fine_vmic])
        
    vmic_df = pd.DataFrame({'vmic':vmic_array, 'slope':slope_all, 'p_value':p_values, 
                        'monh':monh_all, 'monh_err':monh_err_all})
    vmic_df = vmic_df.sort_values('vmic').drop_duplicates('vmic')

    return vmic_df

def measure_vmic_main(wave, flux, flux_err, teff, logg, monh, vmic, vmac, vsini, R, line_list,
                      vmic_array=np.linspace(0, 4, 6),
                           abund=None, 
                           sensitivity_dominance_thres=0.6,
                           line_dominance_thres=0.5,
                           max_line_num=50,
                           plot=False, save_plot=None):

    if abund is None:
        abund = Abund.solar()
        abund.monh = monh
        
    if not {'central_depth', 'line_range_s', 'line_range_e'}.issubset(line_list._lines.columns) or np.abs(line_list.cdepth_range_paras[0]-teff) >= 500 or (np.abs(line_list.cdepth_range_paras[1]-logg) >= 1) or (np.abs(line_list.cdepth_range_paras[2]-monh) >= 0.5):
        # Calculate the line depth from the pre-set stellar parameter
        sme = SME_Structure()
        sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
        sme.abund = abund
        use_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
        use_list = use_list[use_list['central_depth'] > 0.1]
    else:
        use_list = line_list[line_list['central_depth'] > 0.1]

    v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5/R)**2)

    # Determine logg using ionization balance.
    fit_line_group = pysme_abund.find_line_groups(wave, ['Fe'], [1, 2], line_list, v_broad)
    spec_syn = pysme_abund.get_sensitive_synth(wave, R, teff, logg, monh, vmic, vmac, vsini, line_list, abund, ['Fe'], [1, 2], fit_line_group)
    fe_lines = pysme_abund.select_lines(fit_line_group, spec_syn, ['Fe'], [1, 2], 
                                                   sensitivity_dominance_thres=sensitivity_dominance_thres, 
                                                   line_dominance_thres=line_dominance_thres, 
                                                   max_line_num=max_line_num)
    fe_lines['Fe'][1] = fe_lines['Fe'][1].sort_values('wav_s')
    fe_lines['Fe'][2] = fe_lines['Fe'][2].sort_values('wav_s')

    vmic_df = measure_vmic(teff, logg, monh, vmic_array, vmac, vsini, R, wave, flux, flux_err, fe_lines['Fe'][1], fe_lines['Fe'][2], line_list)

    # Get the vmic from 
    vmic_measure = vmic_df.loc[vmic_df['p_value'].idxmax(), 'vmic']
    vmic_inerp = np.linspace(np.min(vmic_df['vmic']), np.max(vmic_df['vmic']), 500)
    p_value_interp = np.interp(vmic_inerp, vmic_df['vmic'], vmic_df['p_value'])
    vmic_bound = [vmic_inerp[ele] for ele in find_range(p_value_interp)]
    monh = vmic_df.loc[vmic_df['p_value'].idxmax(), 'monh']
    monh_err = vmic_df.loc[vmic_df['p_value'].idxmax(), 'monh_err']
    vmic = vmic_measure

    if plot:
        plt.figure()
        plt.plot(vmic_df['vmic'], vmic_df['slope'], 'o-')
        plt.twinx()
        plt.plot(vmic_df['vmic'], vmic_df['p_value'], 'o-', c='C1')
        
        plt.axvline(vmic_measure)
        plt.axvspan(*vmic_bound, alpha=0.2)
        plt.title(f'$v_\mathrm{{mic}}$={vmic_measure:.2f}$^{{+{vmic_bound[1] - vmic_measure:.2f}}}_{{{vmic_bound[0] - vmic_measure:.2f}}}$')
        if save_plot is not None:
            plt.savefig(f'{save_plot}/logg_fe12_measure.png')

    return vmic, vmic_bound, monh, monh_err

#---------------------vmac/vsini-----------------------------

def measure_vsini_main(wave, flux, flux_err, teff, logg, monh, vmic, vmac, vsini, R, line_list,
                           abund=None, 
                           sensitivity_dominance_thres=0.6,
                           line_dominance_thres=0.5,
                           max_line_num=50,
                           plot=False, save_plot=None):

    if abund is None:
        abund = Abund.solar()
        abund.monh = monh
        
    if not {'central_depth', 'line_range_s', 'line_range_e'}.issubset(line_list._lines.columns) or np.abs(line_list.cdepth_range_paras[0]-teff) >= 500 or (np.abs(line_list.cdepth_range_paras[1]-logg) >= 1) or (np.abs(line_list.cdepth_range_paras[2]-monh) >= 0.5):
        # Calculate the line depth from the pre-set stellar parameter
        sme = SME_Structure()
        sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
        sme.abund = abund
        use_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
        use_list = use_list[use_list['central_depth'] > 0.1]
    else:
        use_list = line_list[line_list['central_depth'] > 0.1]

    v_broad = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (3e5/R)**2)

    # Find all the Fe 1 and Fe 2 lines
    fit_line_group = pysme_abund.find_line_groups(wave, ['Fe'], [1, 2], line_list, v_broad)
    spec_syn = pysme_abund.get_sensitive_synth(wave, R, teff, logg, monh, vmic, vmac, vsini, line_list, abund, ['Fe'], [1, 2], fit_line_group)
    fe_lines = pysme_abund.select_lines(fit_line_group, spec_syn, ['Fe'], [1, 2], 
                                                   sensitivity_dominance_thres=sensitivity_dominance_thres, 
                                                   line_dominance_thres=line_dominance_thres, 
                                                   max_line_num=max_line_num)
    fe_lines['Fe'][1] = fe_lines['Fe'][1].sort_values('wav_s').reset_index(drop=True)
    fe_lines['Fe'][2] = fe_lines['Fe'][2].sort_values('wav_s').reset_index(drop=True)

    wlcent, EW, vmac_measure, vmac_err_measure, vsini_measure, vsini_err_measure = [], [], [], [], [], []

    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R

    sub_sme = []
    fit_paras, fit_bounds = ['vsini'], [0, 50]
    for i in tqdm(fe_lines['Fe'][1].index):
        line_wav = fe_lines['Fe'][1].loc[i, 'wlcent']
        if type(line_wav) != list:
            hwhm = (vmac+vsini)/3e5*line_wav
        else:
            hwhm = (vmac+vsini)/3e5*line_wav[0]
    
        sub_sme_single = deepcopy(sme)
        if type(line_wav) != list:
            indices = (wave >= line_wav-2*hwhm) & (wave <= line_wav+2*hwhm)
        else:
            indices = (wave >= line_wav[0]-2*hwhm) & (wave <= line_wav[-1]+2*hwhm)
        sub_sme_single.wave = wave[indices]
        sub_sme_single.spec = flux[indices]
        sub_sme_single.uncs = flux_err[indices]
        sub_sme_single.linelist = use_list[~((use_list['line_range_e'] < sub_sme_single.wave[0][0]) | (use_list['line_range_s'] > sub_sme_single.wave[0][-1]))]
        sub_sme.append([sub_sme_single, fit_paras, fit_bounds])
        wlcent.append(line_wav)
        # wlcent_mean.append(np.mean(line_wav))
    
    for i in tqdm(fe_lines['Fe'][2].index):
        line_wav = fe_lines['Fe'][2].loc[i, 'wlcent']
        if type(line_wav) != list:
            hwhm = (vmac+vsini)/3e5*line_wav
        else:
            hwhm = (vmac+vsini)/3e5*line_wav[0]
    
        sub_sme_single = deepcopy(sme)
        if type(line_wav) != list:
            indices = (wave >= line_wav-2*hwhm) & (wave <= line_wav+2*hwhm)
        else:
            indices = (wave >= line_wav[0]-2*hwhm) & (wave <= line_wav[-1]+2*hwhm)
        sub_sme_single.wave = wave[indices]
        sub_sme_single.spec = flux[indices]
        sub_sme_single.uncs = flux_err[indices]
        sub_sme_single.linelist = use_list[~((use_list['line_range_e'] < sub_sme_single.wave[0][0]) | (use_list['line_range_s'] > sub_sme_single.wave[0][-1]))]
        sub_sme.append([sub_sme_single, fit_paras, fit_bounds])
        wlcent.append(line_wav)
        # wlcent_mean.append(np.mean(line_wav))

    sub_sme = pqdm(sub_sme, mv_fit, n_jobs=10, argument_type='args')
    vsini_measure = pd.DataFrame({'wlcent':wlcent, 
                'EW':[np.sum(1 - sub_sme_single.synth[0]) * np.mean(np.diff(sub_sme_single.wave[0])) for sub_sme_single in sub_sme], 
                # 'vmac':[sub_sme_single.fitresults['values'][0] for sub_sme_single in sub_sme], 
                # 'vmac_err':[sub_sme_single.fitresults['fit_uncertainties'][0] for sub_sme_single in sub_sme],
                'vsini':[sub_sme_single.fitresults['values'][0] for sub_sme_single in sub_sme], 
                'vsini_err':[sub_sme_single.fitresults['fit_uncertainties'][0] for sub_sme_single in sub_sme]})
    
    linear_fit = np.polyfit(vsini_measure['EW'], vsini_measure['vsini'], 1)
    scatter = np.std(vsini_measure['vsini'] - np.polyval(linear_fit, vsini_measure['EW']))
    indices = (np.abs(vsini_measure['vsini'] - np.polyval(linear_fit, vsini_measure['EW'])) <= 2*scatter)
    vsini_mean = np.mean(vsini_measure.loc[indices, 'vsini'])
    scatter = np.std(vsini_measure.loc[indices, 'vsini'])

    if plot:
        plt.figure(figsize=(13/2, 3), dpi=150)
        
        plt.scatter(vsini_measure['EW'], vsini_measure['vsini'], s=5)
        plt.scatter(vsini_measure.loc[~indices, 'EW'], vsini_measure.loc[~indices, 'vsini'], c='red', marker='x', s=10)

        plt.axhline(vsini_mean, ls='--', c='C1')
        plt.xlabel('EW (AA)')
        plt.ylabel('Vsini (km/s)')
        plt.title(f'Vsini={vsini_mean:.2f}$\pm${scatter:.2f} km/s')

        if save_plot is not None:
            plt.savefig(f'{save_plot}/vsini_measure.png')

    return vsini_mean, scatter, vsini_measure

def pysme_para_main(wav_obs, flux_obs, flux_err_obs, R, s_n, line_list, teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init, ion_list=['Fe 1', 'Fe 2'], spec_margin=0.2, linelist_margin=2):
    
    # Find all isolated Fe I and Fe II lines
    wav_start, wav_end = np.min(wav_obs), np.max(wav_obs)
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init
    sme.iptype = 'gauss'
    sme.ipres = R
    wave_synth_array = np.arange(wav_start, wav_end, 0.05)
    sme.wave = wave_synth_array

    indices = (line_list['wlcent'] >= wav_start) & (line_list['wlcent'] <= wav_end)

    sme.linelist = line_list[indices]
    sme = synthesize_spectrum(sme)
    wav_all, flux_all = copy(sme.wave[0]), copy(sme.synth[0])

    indices_use_paras = wav_obs < 0
    indices_linelist_use_paras = line_list['wlcent'] < 0

    for ion in ion_list:
        fe1_indices = (line_list['species'] == ion)
        sme.linelist = line_list[indices & fe1_indices]
        sme = synthesize_spectrum(sme)
        wav_fe, flux_fe = copy(sme.wave[0]), copy(sme.synth[0])
        spectra = pd.DataFrame({'wave':wav_all, 'flux_all':flux_all, 'flux_el':flux_fe})
        selected_lines_fe1 = select_lines(spectra, teff_init, 
                                    line_list[indices & fe1_indices], 
                                    0.7, 0.2, s_n)
    
    # Select the Fe1 and Fe2 spectral regions, also the sub line list
    indices_use_paras = wav_obs < 0
    indices_linelist_use_paras = line_list['wlcent'] < 0
    for i in selected_lines_fe1.index:
        wav_chunk_start, wav_chunk_end = selected_lines_fe1.loc[i, ['Bluewidth', 'Redwidth']].values
        indices_use_paras = indices_use_paras | ((wav_obs >= wav_chunk_start-spec_margin) & (wav_obs <= wav_chunk_end+spec_margin))
        indices_linelist_use_paras = indices_linelist_use_paras | (((line_list['wlcent'] >= wav_chunk_start-linelist_margin) & (line_list['wlcent'] <= wav_chunk_end+linelist_margin)))
    for i in selected_lines_fe2.index:
        wav_chunk_start, wav_chunk_end = selected_lines_fe2.loc[i, ['Bluewidth', 'Redwidth']].values
        indices_use_paras = indices_use_paras | ((wav_obs >= wav_chunk_start-spec_margin) & (wav_obs <= wav_chunk_end+spec_margin))
        indices_linelist_use_paras = indices_linelist_use_paras | (((line_list['wlcent'] >= wav_chunk_start-linelist_margin) & (line_list['wlcent'] <= wav_chunk_end+linelist_margin)))

    wav_obs_use_paras, flux_obs_use_paras = wav_obs[indices_use_paras], flux_obs[indices_use_paras]
    line_list_use_paras = line_list[indices_linelist_use_paras]
    
    plt.figure(figsize=(14, 3), dpi=150)
    plt.plot(wav_obs, flux_obs)
    
    for i in selected_lines_fe1.index:
        plt.axvspan(*selected_lines_fe1.loc[i, ['Bluewidth', 'Redwidth']].values, alpha=0.5, color='C1')
    for i in selected_lines_fe2.index:
        plt.axvspan(*selected_lines_fe2.loc[i, ['Bluewidth', 'Redwidth']].values, alpha=0.5, color='C2')

    plt.scatter(wav_obs_use_paras, flux_obs_use_paras, s=5, c='red', zorder=5)

    sme_fit = SME_Structure()
    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init

    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R

    sme_fit.linelist = line_list_use_paras
    sme_fit.wave = wav_obs_use_paras
    sme_fit.spec = flux_obs_use_paras
    sme_fit.uncs = flux_err_obs

    sme_fit = solve(sme_fit, ['teff', 'logg', 'monh', 'vmic', 'vsini'])

    if sme_fit.vsini > 15:
        print('Large Vsini, second minimization.')
        sme_fit_2 = SME_Structure()
        sme_fit_2.teff, sme_fit_2.logg, sme_fit_2.monh, sme_fit_2.vmic, sme_fit_2.vsini = sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vsini
        sme_fit_2.vmac = 0
        sme_fit_2.iptype = 'gauss'
        sme_fit_2.ipres = R
        sme_fit_2.abund = copy(sme_fit.abund)

        sme_fit_2.linelist = sme_fit.linelist
        sme_fit_2.wave = sme_fit.wave
        sme_fit_2.spec = sme_fit.spec
        sme_fit_2.uncs = sme_fit.uncs

        sme_fit_2.accft, sme_fit_2.accgt, sme_fit_2.accxt = 0.1*sme_fit.accft, 0.1*sme_fit.accgt, 0.1*sme_fit.accxt

        sme_fit_2 = solve(sme_fit_2, ['teff', 'logg', 'monh', 'vmic'])
        
        return sme_fit_2
    
    plt.plot(sme_fit.wave[0], sme_fit.synth[0])
    
    return sme_fit