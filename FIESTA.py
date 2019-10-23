# FIESTA #

import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from functions import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

freq_HL 	= 0.0293 									# arbitrary for now
freq_HN 	= 0.1563									# higher limit of frequency range
X 			= (np.arange(401)-200)/10					# CCF Velocity grid
idx 		= (abs(X) <= 10)
x 			= X[idx]

##############
# Line shift #
##############

N 			= 21
RV_gauss 	= np.zeros(N)								# RV derived from a Gaussian fit
RV 			= np.zeros(N)								# FT-derived RV
RV_L 		= np.zeros(N)								# FT-derived RV over the lower-freq range
RV_H 		= np.zeros(N)								# FT-derived RV over the higher-freq range
SHIFT 		= np.linspace(-10, 10, num=N, endpoint=True)/1000

wspace 		= 0.3   # the amount of width reserved for blank space between subplots
plt.rcParams.update({'font.size': 12})
fig, axes 	= plt.subplots(figsize=(12, 12))

ccf_tpl 	= gaussian(x, 1, 0, 2.5, 0)
for n in np.arange(N):
	shift 		= SHIFT[n]
	ccf 		= gaussian(x-shift, 1, 0, 2.5, 0)
	popt, pcov 	= curve_fit(gaussian, x, ccf)
	RV_gauss[n] = popt[1]
	power, phase, freq = FT(ccf, 0.1)
	power_tpl, phase_tpl, freq_tpl = FT(ccf_tpl, 0.1)
	RV[n], RV_L[n], RV_H[n] = plot_overview(x, ccf, power, phase, phase_tpl, freq, freq_HL, freq_HN)
plt.savefig('./outputs/Shift_overview.png')

# ---------------------------- #
# Radial velocity correlations #
# ---------------------------- #
RV_gauss 	= (RV_gauss - RV_gauss[0]) * 1000 			# all radial velocities are relative to the first ccf
fig, axes 	= plt.subplots(figsize=(15, 5))
plt.subplots_adjust(wspace=wspace)
plot_correlation(RV_gauss, RV, RV_L, RV_H)
plt.savefig('./outputs/Shift_RV_FT.png')
plt.close('all')


####################
# Line deformation #
####################

FILE 		= sorted(glob.glob('./fits/*.fits'))
N_file 		= len(FILE)
RV_gauss 	= np.zeros(N_file)							# RV derived from a Gaussian fit
RV 			= np.zeros(N_file)							# FT-derived RV
RV_L 		= np.zeros(N_file)							# FT-derived RV over the lower-freq range
RV_H 		= np.zeros(N_file)							# FT-derived RV over the higher-freq range

plt.rcParams.update({'font.size': 12})
fig, axes 	= plt.subplots(figsize=(12, 12))

for n in range(N_file):
	hdulist     = fits.open(FILE[n])
	CCF         = 1 - hdulist[0].data 					# flip the line profile
	ccf 		= CCF[idx]
	popt, pcov 	= curve_fit(gaussian, x, ccf)
	RV_gauss[n] = popt[1]
	if n == 0: 
		ccf_tpl = ccf 									# choose the first file as a template
		power_tpl, phase_tpl, freq_tpl = FT(ccf_tpl, 0.1)
	power, phase, freq = FT(ccf, 0.1)
	RV[n], RV_L[n], RV_H[n] = plot_overview(x, ccf, power, phase, phase_tpl, freq, freq_HL, freq_HN)

plt.savefig('./outputs/Overview.png')

# ---------------------------- #
# Radial velocity correlations #
# ---------------------------- #
RV_gauss 	= (RV_gauss - RV_gauss[0]) * 1000 			# all radial velocities are relative to the first ccf
fig, axes 	= plt.subplots(figsize=(15, 5))
plt.subplots_adjust(wspace=wspace)
plot_correlation(RV_gauss, RV, RV_L, RV_H)
plt.savefig('./outputs/RV_FT.png')
plt.close('all')
