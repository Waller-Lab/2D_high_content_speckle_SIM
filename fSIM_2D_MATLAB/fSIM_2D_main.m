%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fSIM_2D_main takes 2D unknown speckle structured illuminated     %
% fluorescent and coherent images to reconstruct super-resolution  %
% fluorescence image with resolution of lambda_f/2(NA_obj + NAs),  %
% where lambda_f is the fluorescence wavelength, NA_obj is the     %
% objective NA, and NAs is the speckle NA.                         %
%                                                                  %
%          Copyright (C) Li-Hao Yeh 2019                           %
% Please cite:                                                     %
%  L.-H. Yeh, S. Chowdhury, and L. Waller, "Computational          %
%  structured illumination for high-content fluorescence and phase %
%  microscopy," Biomed. Opt. Express 10, 1978-1998 (2019)          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;
set(0,'DefaultFigureWindowStyle','docked');

F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));
gpuDevice(3);

%% Load data

load ../calibrated_coherent_data.mat;
load ../calibrated_fluorescent_data.mat;

global lambda ps N_bound_pad N M Nc Mc Np Mp ...
    xshift_max yshift_max Nimg

addpath('fSIM_2D_func');


%% Experimental setup parameters

lambda            = 0.605;      % wavelength (micons) 
mag               = 5;          % magnification
pscrop            = 5.5/mag;    % Pixels size (microns)
NA_obj            = 0.1;        % detection NA
NAs               = 0.3;        % speckle NA  
upsampling_factor = 4;          % image upsampling factor
bg                = 0;         % image background noise
N_bound_pad       = 80;         % number of pixels padded for better boundary condition


%% pre-calibration


[Ncrop,Mcrop,Nimg] = size(I_image);
N  = Ncrop*upsampling_factor; 
M  = Mcrop*upsampling_factor; 
ps = pscrop/upsampling_factor;

I_image_up      = image_upsampling(I_image, upsampling_factor, bg);    % image upsampling 
[xshift,yshift] = image_registration(Ic_image(:,:,:,1), 100, upsampling_factor); % image registration

%% initialization

[fxx_c, fyy_c, fxxp, fyyp] = ...
    spatial_freq_gen(xshift, yshift);                                 % spatial frequency generation

I_obj_init =...
    gpuArray(padarray(mean(I_image_up,3),[N_bound_pad,N_bound_pad])); % object initialization

I_p_whole_init =...
    gpuArray(ones(Np,Mp));                                            % pattern initialization

[T_incoherent, speckle_f_support, obj_f_support, OTF_support] = ...
    kernel_gen(fxx_c, fyy_c, fxxp, fyyp, NA_obj, NAs);                % kernel generation

itr = 20;                                                              % iteration number 


%% Iterative algorithm

[I_obj, I_p_whole, T_incoherent, xshift_new, yshift_new, err] = ...
    fSIM_2D_iter_alg(I_image_up, I_obj_init, I_p_whole_init, T_incoherent, ...
    fxxp, fyyp, speckle_f_support, obj_f_support, OTF_support, xshift, yshift, itr);


save(['recon'],'I_obj','I_p_whole', 'T_incoherent','xshift_new','yshift_new','err')

