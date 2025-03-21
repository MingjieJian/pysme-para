import numpy as np
import pandas as pd
from scipy.signal import convolve
import matplotlib.pyplot as plt

from pysme.sme import SME_Structure
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve
from copy import copy

from astropy.table import Table
from scipy.interpolate import Akima1DInterpolator,interp1d

import sys

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
    
#------------------Teff------------------

def measure_teff_fe(wav, flux, flux_err, R, s_n, line_list, teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init, ion_list=['Fe 1', 'Fe 2'], spec_margin=0.2, linelist_margin=2):
    '''
    Measure the effective temperature of a star using the Fe I lines.
    '''
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff-200, logg, monh, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = 50000
    sme.wave = wave_fit
    sme.spec = flux_fit[wave_fit < 5700]
    sme.uncs = flux_err_fit[wave_fit < 5700]
    sme.linelist = line_list
    sme = solve(sme, ['teff'], linelist_mode='auto')


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
files = ['/home/mingjie/software/GALAH_DR4/auxiliary_information/BC_Tables/grid/STcolors_2MASS_GaiaDR2_EDR3_Rv3.1_EBV_0.00.dat']
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

parsec = Table.read('/home/mingjie/software/GALAH_DR4/auxiliary_information/parsec_isochrones/parsec_isochrones_logt_8p00_0p01_10p17_mh_m2p75_0p25_m0p75_mh_m0p60_0p10_0p70_GaiaEDR3_2MASS.fits')

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
        
        bc_grid = np.genfromtxt('../../../../software/GALAH_DR4//auxiliary_information/BC_Tables/grid/STcolors_2MASS_GaiaDR2_EDR3_Rv3.1_EBV_0.00.dat',names=True)
        file = open('../../../../software/GALAH_DR4/auxiliary_information/BC_Tables/grid/bc_grid_kdtree_ebv_0.00.pickle','rb')
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