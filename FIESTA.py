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
# freq_HN 	= 0.145										# higher limit of frequency range
freq_HN 	= 0.1563
X 			= (np.arange(401)-200)/10					# CCF Velocity grid
idx 		= (abs(X) <= 10)
x 			= X[idx]

##############
# Line shift #
##############
if 1:
	N 			= 101
	RV_gauss 	= np.zeros(N)								# RV derived from a Gaussian fit
	RV 			= np.zeros(N)								# FT-derived RV
	RV_L 		= np.zeros(N)								# FT-derived RV over the lower-freq range
	RV_H 		= np.zeros(N)								# FT-derived RV over the higher-freq range
	SHIFT 		= np.linspace(-10, 10, num=N, endpoint=True)/1000

	# amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c
	ccf_tpl 	= gaussian(x, 1, 0, 1)

	ccf1 		= gaussian(X, 1, 0, 0.1)
	[gamma, beta, alpha] = [1000, 1, 1]
	delta_ccf 	= gaussian(X, gamma, beta, alpha)
	ccf2 		= ccf1 + delta_ccf
	# plt.plot(X, ccf1)
	# plt.plot(X, delta_ccf)
	# plt.show()

	power, phase, freq = FT(ccf2, 0.1)
	power_tpl, phase_tpl, freq_tpl = FT(ccf1, 0.1)
	plot_overview(X, ccf2, ccf1, power, power_tpl, phase, phase_tpl, freq, 0.14, 2)
	plt.savefig('./simulation.png')
	plt.close('all')
	# plt.show()

	xi = (np.arange(501))/1000
	RE = gamma * np.pi**0.5 * np.exp(-np.pi**2 * xi**2) + (np.pi/alpha)**0.5 * np.exp(-np.pi**2 * xi**2 / alpha) * np.cos(2*np.pi*beta*xi)
	IM = - (np.pi/alpha)**0.5 * np.exp(-np.pi**2 * xi**2 / alpha) * np.sin(2*np.pi*beta*xi)
	angle = np.arctan(IM/RE)
	# RE = np.pi**0.5 * np.exp(-np.pi**2 * xi**2) + 1/gamma * (np.pi/alpha)**0.5 * np.exp(-np.pi**2 * xi**2 / alpha) * np.cos(2*np.pi*beta*xi)
	# IM = -1/gamma * (np.pi/alpha)**0.5 * np.exp(-np.pi**2 * xi**2 / alpha) * np.sin(2*np.pi*beta*xi)
	# angle = np.arctan(IM/RE)
	plt.plot(xi, angle)
	# plt.savefig('./analytical.png')
	plt.show()


	for n in np.arange(N):
		shift 		= SHIFT[n]
		ccf 		= gaussian(x-shift, 1, 0, 1)
		popt, pcov 	= curve_fit(gaussian, x, ccf)
		RV_gauss[n] = popt[1]
		power, phase, freq = FT(ccf, 0.1)
		power_tpl, phase_tpl, freq_tpl = FT(ccf_tpl, 0.1)
		RV[n], RV_L[n], RV_H[n] = plot_overview(x, ccf, ccf_tpl, power, power_tpl, phase, phase_tpl, freq, freq_HL, freq_HN)
		figure_name = './outputs/Shift_snapshot' + str(n) + '.png'
		plt.savefig(figure_name)
		plt.close()
	# plt.savefig('./outputs/Shift_overview.png')

	# ---------------------------- #
	# Radial velocity correlations #
	# ---------------------------- #
	RV_gauss 	= (RV_gauss) * 1000 			# all radial velocities are relative to the first ccf
	fig, axes 	= plt.subplots(figsize=(15, 5))
	plt.subplots_adjust(wspace=0.3)
	plot_correlation(RV_gauss, RV, RV_L, RV_H)
	plt.savefig('./outputs/Shift_RV_FT.png')
	plt.close('all')

####################
# Line deformation #
####################
if 1:
	FILE 		= sorted(glob.glob('./fits/*.fits'))
	N_file 		= len(FILE)
	RV_gauss 	= np.zeros(N_file)							# RV derived from a Gaussian fit
	RV 			= np.zeros(N_file)							# FT-derived RV
	RV_L 		= np.zeros(N_file)							# FT-derived RV over the lower-freq range
	RV_H 		= np.zeros(N_file)							# FT-derived RV over the higher-freq range

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
		RV[n], RV_L[n], RV_H[n] = plot_overview(x, ccf, ccf_tpl, power, power_tpl, phase, phase_tpl, freq, freq_HL, freq_HN)
		figure_name = './outputs/deformation_snapshot' + str(n) + '.png'
		plt.savefig(figure_name)
		plt.close()
	 
	# plt.savefig('./outputs/Overview.png')

	# ---------------------------- #
	# Radial velocity correlations #
	# ---------------------------- #
	RV_gauss 	= (RV_gauss - RV_gauss[0]) * 1000 			# all radial velocities are relative to the first ccf
	# RV_gauss 	= RV_gauss * 1000
	fig, axes 	= plt.subplots(figsize=(15, 5))
	plt.subplots_adjust(wspace=0.3)
	plot_correlation(RV_gauss, RV, RV_L, RV_H)
	plt.savefig('./outputs/RV_FT.png')
	plt.close('all')



##################################

for SN in [2000, 10000, 50000]:
	freq_HL 	= 0.0293 									# arbitrary for now
	freq_HN 	= 0.1563
	X 			= (np.arange(401)-200)/10					# CCF Velocity grid
	idx 		= (abs(X) <= 10)
	x 			= X[idx]

	# ADD_NOISE = False
	ADD_NOISE = True
	FILE 		= sorted(glob.glob('./fits/*.fits'))

	hdulist     = fits.open(FILE[0])
	CCF0         = 1 - hdulist[0].data 					# flip the line profile
	ccf0 		= CCF0[idx]
	if ADD_NOISE == True:
		ccf0 = np.random.normal(ccf0, (1-ccf0)**0.5/2000)
	popt, pcov 	= curve_fit(gaussian, x, ccf0)
	RV0  		= popt[1]

	hdulist     = fits.open(FILE[40])
	CCF40       = 1 - hdulist[0].data 					# flip the line profile
	ccf40 		= CCF40[idx]
	if ADD_NOISE == True:
		ccf40 = np.random.normal(ccf40, (1-ccf40)**0.5/2000)
	popt, pcov 	= curve_fit(gaussian, x, ccf40)
	RV40  		= popt[1]
	print ((RV40 - RV0)*1000)

	f = interp1d(X, CCF0, kind='cubic')
	ccf_shift = f(x-5.4/1000)   # use interpolation function returned by `interp1d`
	if ADD_NOISE == True:
		ccf_shift = np.random.normal(ccf_shift, (1-ccf_shift)**0.5/2000)
	popt, pcov 	= curve_fit(gaussian, x, ccf_shift)
	RV_shift  	= popt[1]
	print ((RV_shift-RV0)*1000) # = 5.40

	power_tpl, phase_tpl, freq_tpl = FT(ccf0, 0.1)
	power, phase, freq = FT(ccf40, 0.1)
	power_shift, phase_shift, freq_shift = FT(ccf_shift, 0.1)

	if 0:
		plt.rcParams.update({'font.size': 12})
		fig, axes 	= plt.subplots(figsize=(18, 11))
		plt.subplots_adjust(hspace=0.3, wspace=0.3) # the amount of width and height reserved for blank space between subplots

		idx 	= (freq <= freq_HN)

		Singal # 
		plt.subplot(231)
		plt.plot(x, ccf40, 'r', alpha=alpha)
		plt.plot(x, ccf_shift, 'b', alpha=alpha)
		plt.title('Signal (CCF)')
		plt.xlabel('Velocity [km/s]')
		plt.ylabel('Normalized intensity')
		plt.grid(True)

		# Singal deformation # 
		plt.subplot(234)
		plt.plot(x, ccf40 - ccf0, 'r', alpha=alpha)
		plt.plot(x, ccf_shift - ccf0, 'b', alpha=alpha)
		plt.title('Signal deformation')
		plt.xlabel('Velocity [km/s]')
		plt.ylabel('Normalized intensity')
		plt.grid(True)

		# power spectrum # 
		plt.subplot(232)
		plt.plot(freq[idx], power[idx], 'r', alpha=alpha)
		plt.plot(freq_shift[idx], power_shift[idx], 'b', alpha=alpha)
		plt.title('Power spectrum')
		plt.xlabel(r'$\xi$ [s/km]')
		plt.ylabel('Power')
		plt.grid(True)

		# differential phase spectrum 
		plt.subplot(235)
		diff_phase = np.unwrap(phase)-np.unwrap(phase_tpl) # Necessary! Don't use # diff_phase = phase - phase_tpl
		diff_phase_shift = np.unwrap(phase_shift)-np.unwrap(phase_tpl)
		plt.plot(freq[idx], diff_phase[idx], 'r', alpha=alpha)
		plt.plot(freq_shift[idx], diff_phase_shift[idx], 'b', alpha=alpha)
		plt.title('Differential phase spectrum')
		plt.xlabel(r'$\xi$ [s/km]')
		plt.ylabel(r'$\Delta \phi$ [radian]')
		plt.grid(True)

		# shift spectrum # 
		plt.subplot(236)

		rv = np.zeros(len(diff_phase))
		rv[1:] = - diff_phase[1:] / (2*np.pi*freq[1:])
		rv[0] = rv[1]

		rv_shift = np.zeros(len(diff_phase_shift))
		rv_shift[1:] = - diff_phase_shift[1:] / (2*np.pi*freq[1:])
		rv_shift[0] = rv_shift[1]

		plt.plot(freq[idx], rv[idx] * 1000, 'r', alpha=alpha)
		plt.plot(freq_shift[idx], rv_shift[idx] * 1000, 'b', alpha=alpha)
		plt.title('Shift spectrum')
		plt.xlabel(r'$\xi$ [s/km]')	
		plt.ylabel('RV [m/s]')
		plt.grid(True)
		if ADD_NOISE == False:
			plt.savefig('comparison_noise-free.png')
		else:
			plt.savefig('comparison_noise-added.png')
		plt.show()


	if 1: 
		idx = (freq < freq_HN) & (freq > 0)
		diff_phase = np.unwrap(phase)-np.unwrap(phase_tpl) # Necessary! Don't use # diff_phase = phase - phase_tpl
		diff_phase_shift = np.unwrap(phase_shift)-np.unwrap(phase_tpl)
		rv = np.zeros(len(diff_phase))
		rv[1:] = - diff_phase[1:] / (2*np.pi*freq[1:])
		rv[0] = rv[1]

		rv_shift = np.zeros(len(diff_phase_shift))
		rv_shift[1:] = - diff_phase_shift[1:] / (2*np.pi*freq[1:])
		rv_shift[0] = rv_shift[1]





	idx = (freq < freq_HN) & (freq > 0)
	freq = freq[idx]
	shift_spectrum = np.zeros((100, len(freq)))









	for k in range(100):
		ccf0 = np.random.normal(ccf0, (1-ccf0)**0.5/SN)
		ccf_shift = np.random.normal(ccf_shift, (1-ccf_shift)**0.5/SN)
		_, phase, _ = FT(ccf_shift, 0.1)
		_, phase_tpl, _ = FT(ccf0, 0.1)
		phase = phase[idx]
		phase_tpl = phase_tpl[idx]
		diff_phase = np.unwrap(phase)-np.unwrap(phase_tpl)
		shift_spectrum[k, :] = -diff_phase / (2*np.pi*freq)

	if 0:
		plt.plot(freq, np.transpose(shift_spectrum)*1000)
		plt.title('Shift spectrum due to noise')
		plt.xlabel(r'$\xi$ [s/km]')	
		plt.ylabel('RV [m/s]')
		plt.grid(True)	
		# plt.savefig('ShiftSpectrum_100_Simulations.png')			
		plt.show()


	plt.plot(freq, np.std(shift_spectrum, axis=0)*1000, '--')
	plt.title('Shift spectrum error estimation')
	plt.xlabel(r'$\xi$ [s/km]')	
	plt.ylabel('RV [m/s]')
	plt.yscale('log')
	plt.grid(True)	
	# plt.savefig('ShiftSpectrum_100_Simulations_error.png')

plt.plot(freq, abs(rv[idx] - (RV40 - RV0)) * 1000, 'r', alpha=alpha)
plt.plot(freq_shift[idx], abs(rv_shift[idx] - (RV_shift-RV0)) * 1000, 'b', alpha=alpha)
plt.savefig('sn=2000.png')
plt.show()
plt.close('all')
	# pure_plot(x, ccf40, ccf0, power, phase, phase_tpl, freq, freq_HN)
	# plt.savefig('deformation-40.png')
	# plt.show()

	 # = 5.4 m/s


	# pure_plot(x, ccf_shift, ccf0, power, phase, phase_tpl, freq, freq_HN)
	# plt.savefig('shift5.4.png')
	# plt.show()








