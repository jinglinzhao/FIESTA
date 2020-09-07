# Fourier transform using numpy.fft.rfft # 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

alpha 	=	0.5

####################################################################

def FT(signal, spacing):
	oversample 	= 8 										# oversample folds; to be experiemented further
	n 			= 2**(int(np.log(signal.size)/np.log(2))+1 + oversample)
	fourier 	= np.fft.rfft(signal, n, norm='ortho')
	freq 		= np.fft.rfftfreq(n, d=spacing)
	power 		= np.abs(fourier)
	phase 		= np.angle(fourier)

	return [power, phase, freq]

####################################################################

def gaussian(x, gamma, beta, alpha):
	return 1/gamma * np.exp(-alpha * np.power(x - beta, 2.)) 

####################################################################

def plot_overview(x, ccf, ccf_tpl, power, power_tpl, phase, phase_tpl, freq, freq_HL, freq_HN):

	plt.rcParams.update({'font.size': 15})
	fig, axes 	= plt.subplots(figsize=(18, 11))
	plt.subplots_adjust(hspace=0.3, wspace=0.3) # the amount of width and height reserved for blank space between subplots

	idx 	= (freq <= freq_HN)
	idx_L 	= (freq < freq_HL)
	idx_H 	= (freq >= freq_HL) & (freq < freq_HN)

	# Singal # 
	plt.subplot(231)
	plt.plot(x, ccf, 'k', alpha=alpha)
	plt.xlim([-10,10])
	plt.title('Signal (CCF)')
	plt.xlabel('Velocity [km/s]')
	plt.ylabel('Normalized intensity')
	plt.grid(True)

	# Singal deformation # 
	plt.subplot(234)
	plt.plot(x, ccf - ccf_tpl, 'k', alpha=alpha)
	plt.xlim([-10,10])
	# plt.ylim([-0.003, 0.003])
	plt.title('Signal deformation')
	plt.xlabel('Velocity [km/s]')
	plt.ylabel('Normalized intensity')
	plt.grid(True)

	# power spectrum # 
	plt.subplot(232)
	plt.plot(freq[idx], power[idx], 'k', alpha=alpha)
	plt.title('Power spectrum')
	plt.xlabel(r'$\xi$ [s/km]')
	plt.ylabel('Power')
	plt.grid(True)

	# differential power spectrum # 
	# plt.subplot(235)
	# plt.plot(freq[idx], power[idx]-power_tpl[idx], 'k', alpha=alpha)
	# plt.title('Differential power spectrum')
	# plt.xlabel(r'$\xi$ [s/km]')
	# plt.ylabel('Power')
	# plt.grid(True)

	# differential phase spectrum 
	plt.subplot(233)
	diff_phase = np.unwrap(phase)-np.unwrap(phase_tpl) # Necessary! Don't use # diff_phase = phase - phase_tpl
	plt.plot(freq[idx], diff_phase[idx], 'k', alpha=alpha)
	# plt.ylim([-0.01, 0.01])
	plt.title('Differential phase spectrum')
	plt.xlabel(r'$\xi$ [s/km]')
	plt.ylabel(r'$\Delta \phi$ [radian]')
	plt.grid(True)

	# shift spectrum # 
	plt.subplot(236)
	# rv = -np.gradient(diff_phase, np.mean(np.diff(freq))) / (2*np.pi)
	# rv = - diff_phase / (2*np.pi*freq)
	rv = np.zeros(len(diff_phase))
	rv[1:] = - diff_phase[1:] / (2*np.pi*freq[1:])
	rv[0] = rv[1]	
	plt.plot(freq[idx], rv[idx] * 1000, 'k', alpha=alpha)
	# if mode == 1: 							# for a line shift 
	# 	plt.ylim([-11, 11])
	# else:									# for a line deformation
	# 	plt.ylim([-10, 10])	
	plt.title('Shift spectrum')
	plt.xlabel(r'$\xi$ [s/km]')	
	plt.ylabel('RV [m/s]')
	plt.grid(True)

	# calculate the "averaged" radial veflocity shift in Fourier space
	freq_full 		= np.concatenate((-freq[idx][:0:-1], freq[idx]), axis=None) 
	diff_phase_full = np.concatenate((-diff_phase[idx][:0:-1], diff_phase[idx]), axis=None)
	power_full		= np.concatenate((power[idx][:0:-1], power[idx]), axis=None)  
	coeff 			= np.polyfit(freq_full, diff_phase_full, 1, w=power_full**0.5)
	RV 				= -coeff[0] / (2*np.pi) * 1000
	coeff 			= np.polyfit(freq[idx_L], diff_phase[idx_L], 1, w=power[idx_L]**0.5)
	RV_L 			= -coeff[0] / (2*np.pi) * 1000
	coeff 			= np.polyfit(freq[idx_H], diff_phase[idx_H], 1, w=power[idx_H]**0.5)
	RV_H 			= -coeff[0] / (2*np.pi) * 1000    

	if 0: # experimented version
		# shift spectrum # 
		plt.subplot(236)
		rv = -diff_phase[1:] / freq[1:] / (2*np.pi)
		rv = np.hstack((rv[0],rv))
		plt.plot(freq[idx], rv[idx] * 1000, 'k', alpha=alpha)
		if mode == 1: 							# for a line shift 
			plt.ylim([-11, 11])
		else:									# for a line deformation
			plt.ylim([-10, 10])	
		plt.title('Shift spectrum')
		plt.xlabel(r'$\xi$ [s/km]')	
		plt.ylabel('RV [m/s]')
		plt.grid(True)	

		# calculate the "averaged" radial veflocity shift in Fourier space
		freq_full 		= np.concatenate((-freq[idx][:0:-1], freq[idx]), axis=None) 
		diff_phase_full = np.concatenate((-diff_phase[idx][:0:-1], diff_phase[idx]), axis=None)
		power_full		= np.concatenate((power[idx][:0:-1], power[idx]), axis=None)  
		coeff 			= np.polyfit(freq_full, diff_phase_full, 1, w=power_full**0.5)
		RV 				= -coeff[0] / (2*np.pi) * 1000
		# RV_full 		= np.average(rv[idx], weights=power[idx]) * 1000
		RV_L 			= np.average(rv[idx_L], weights=power[idx_L]) * 1000
		RV_H 			= np.average(rv[idx_H], weights=power[idx_H]) * 1000

	return RV, RV_L, RV_H

####################################################################

def plot_correlation(RV_gauss, RV, RV_L, RV_H):

	plt.subplot(131)
	plt.plot(RV_gauss, RV, 'k.', alpha=alpha)
	b0, b1 	= np.polyfit(RV_gauss, RV, 1)
	r, p 	= stats.pearsonr(RV_gauss, RV)
	plt.title(r'$k$ = %.2f, $\rho$ = %.2f'%(b0, r))
	plt.xlabel(r'$RV_{Gaussian}$ [m/s]')
	plt.ylabel(r'$RV_{FT}$ [m/s]')
	plt.grid(True)

	plt.subplot(132)
	plt.plot(RV_gauss, RV_L, 'k.', alpha=alpha)
	b0, b1 	= np.polyfit(RV_gauss, RV_L, 1)
	r, p 	= stats.pearsonr(RV_gauss, RV_L)
	plt.title(r'$k$ = %.2f, $\rho$ = %.2f'%(b0, r))
	plt.xlabel(r'$RV_{Gaussian}$ [m/s]')
	plt.ylabel(r'$RV_{FT,L}$ [m/s]')
	plt.grid(True)

	plt.subplot(133)
	plt.plot(RV_gauss, RV_H, 'k.', alpha=alpha)
	b0, b1 	= np.polyfit(RV_gauss, RV_H, 1)
	r, p 	= stats.pearsonr(RV_gauss, RV_H)
	plt.title(r'$k$ = %.2f, $\rho$ = %.2f'%(b0, r))
	plt.xlabel(r'$RV_{Gaussian}$ [m/s]')
	plt.ylabel(r'$RV_{FT,H}$ [m/s]')	
	plt.grid(True)

# if 0:
# 	In [2]: import progressbar 
#    ...: from time import sleep 
#    ...: bar = progressbar.ProgressBar(maxval=20, \ 
#    ...:     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percenta
#    ...: ge()]) 
#    ...: bar.start() 
#    ...: for i in range(20): 
#    ...:     bar.update(i+1) 
#    ...:     sleep(0.1) 
#    ...: bar.finish()      



def pure_plot(x, ccf, ccf_tpl, power, phase, phase_tpl, freq, freq_HN):

	plt.rcParams.update({'font.size': 12})
	fig, axes 	= plt.subplots(figsize=(18, 11))
	plt.subplots_adjust(hspace=0.3, wspace=0.3) # the amount of width and height reserved for blank space between subplots

	idx 	= (freq <= freq_HN)

	# Singal # 
	plt.subplot(231)
	plt.plot(x, ccf, 'k', alpha=alpha)
	plt.title('Signal (CCF)')
	plt.xlabel('Velocity [km/s]')
	plt.ylabel('Normalized intensity')
	plt.grid(True)

	# Singal deformation # 
	plt.subplot(234)
	plt.plot(x, ccf - ccf_tpl, 'k', alpha=alpha)
	plt.title('Signal deformation')
	plt.xlabel('Velocity [km/s]')
	plt.ylabel('Normalized intensity')
	plt.grid(True)


	# power spectrum # 
	plt.subplot(232)
	plt.plot(freq[idx], power[idx], 'k', alpha=alpha)
	plt.title('Power spectrum')
	plt.xlabel(r'$\xi$ [s/km]')
	plt.ylabel('Power')
	plt.grid(True)

	# differential phase spectrum 
	plt.subplot(235)
	diff_phase = np.unwrap(phase)-np.unwrap(phase_tpl) # Necessary! Don't use # diff_phase = phase - phase_tpl
	plt.plot(freq[idx], diff_phase[idx], 'k', alpha=alpha)
	plt.title('Differential phase spectrum')
	plt.xlabel(r'$\xi$ [s/km]')
	plt.ylabel(r'$\Delta \phi$ [radian]')
	plt.grid(True)

	# shift spectrum # 
	plt.subplot(236)

	rv = np.zeros(len(diff_phase))
	rv[1:] = - diff_phase[1:] / (2*np.pi*freq[1:])
	rv[0] = rv[1]

	plt.plot(freq[idx], rv[idx] * 1000, 'k', alpha=alpha)
	plt.title('Shift spectrum')
	plt.xlabel(r'$\xi$ [s/km]')	
	plt.ylabel('RV [m/s]')
	plt.grid(True)
