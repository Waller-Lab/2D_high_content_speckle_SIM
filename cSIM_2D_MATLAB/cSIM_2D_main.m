%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fSIM_2D_main takes 2D unknown speckle structured illuminated     %
% coherent images to reconstruct super-resolution phase image with %
% resolution of lambda_c/(NA_obj + NAs), where lambda_c is the     %
% coherent wavelength, NA_obj is the objective NA, and NAs is the  %
% speckle NA.                                                      %
%                                                                  %
%          Copyright (C) Li-Hao Yeh 2019                           %
% Please cite:                                                     %
%  L.-H. Yeh, S. Chowdhury, and L. Waller, "Computational          %
%  structured illumination for high-content fluorescence and phase %
%  microscopy," Biomed. Opt. Express 10, 1978-1998 (2019)          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;
set(0,'DefaultFigureWindowStyle','docked');
gpuDevice(3);

%% Load data

load ../calibrated_coherent_data.mat;

global lambda ps z_camera N_bound_pad N M Nc Mc Np Mp ...
    xshift_max yshift_max Nimg N_defocus

addpath('cSIM_2D_func');

%% Experimental parameters

lambda            = 0.532;      % wavelength (micons) 
mag               = 5;          % magnification
pscrop            = 5.5/mag;    % Pixels size (microns)
NA_obj            = 0.1;        % detection NA
NAs               = 0.3;        % speckle NA  
z_camera          = [0;-32];    % camera defocus
upsampling_factor = 4;          % image upsampling factor
bg                = 96;         % image background noise
N_bound_pad       = 80;         % number of pixels padded for better boundary condition


%% pre-calibration

[Ncrop,Mcrop,Nimg,N_defocus] = size(Ic_image);
N  = Ncrop*upsampling_factor; 
M  = Mcrop*upsampling_factor; 
ps = pscrop/upsampling_factor;

I_image_up      = image_upsampling(Ic_image, upsampling_factor, bg);    % image upsampling 
[xshift,yshift] = image_registration(Ic_image, 100, upsampling_factor); % image registration

%% initialization


[fxx_c, fyy_c, fxxp, fyyp] = ...
    spatial_freq_gen(xshift, yshift);                     % spatial frequency generation

[obj_init, field_p_whole_init] = ...
    cSIM_2D_init(I_image_up, xshift, yshift, fxxp, fyyp); % initialization

[Hz_det, Pupil_obj, Gaussian, Pupil_NAs] = ...
    kernel_gen(fxx_c, fyy_c, fxxp, fyyp, NA_obj, NAs);    % kernel generation


% Zernike polynomial

n = [0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6];
m = [0 -1 1 -2 0 2 -3 -1 1 3 -4 -2 0 2 4 -5 -3 -1 1 3 5 -6 -4 -2 0 2 4 6];
zerpoly = zernike_gen(n, m, fxx_c, fyy_c, NA_obj);


itr = 20; % iteration number 

%% Iterative algorithm


[obj, field_p_whole, Pupil_obj, xshift_new, yshift_new, err] = ...
    cSIM_2D_iter_alg(I_image_up, obj_init, field_p_whole_init, Pupil_obj, ...
    fxxp, fyyp, Hz_det, Gaussian, Pupil_NAs, zerpoly, xshift, yshift, itr);


save(['recon'],'obj','field_p_whole', 'Pupil_obj','xshift_new','yshift_new','err')




