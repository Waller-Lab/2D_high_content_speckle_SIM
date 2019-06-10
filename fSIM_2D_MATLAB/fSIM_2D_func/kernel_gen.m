
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel_gen generates the required filter for the algorithm, including the OTF  % 
% and the Fourier support of the object, speckle and the OTF                     %                          
%                                                                                %
% Inputs:                                                                        %
%       fxx_c             : x spatial frequency with the size of the object      %
%       fyy_c             : y spatial frequency with the size of the object      %
%       fxxp              : x spatial frequency with the size of the speckle     %
%       fyyp              : y spatial frequency with the size of the speckle     %
%       NA_obj            : objective NA                                         %
%       NAs               : speckle NA                                           %
% Outputs:                                                                       %
%       T_incoherent      : OTF of the image system                              %
%       speckle_f_support : Fourier support of the speckle                       %
%       obj_f_support     : Fourier support of the object                        %
%       OTF_f_support     : Fourier support of the OTF                           %
%                                                                                %
%                                                                                %
%          Copyright (C) Li-Hao Yeh 2019                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T_incoherent, speckle_f_support, obj_f_support, OTF_support] = ...
    kernel_gen(fxx_c, fyy_c, fxxp, fyyp, NA_obj, NAs)

global lambda Nc Mc Np Mp 

F = @(x) fft2(x);
iF = @(x) ifft2(x);

Pupil_obj = zeros(Nc,Mc);
r_obj=(fxx_c.^2+fyy_c.^2).^(1/2);
Pupil_obj(find(r_obj<NA_obj/lambda))=1;
T_incoherent = abs(F(abs(iF(Pupil_obj)).^2));
T_incoherent = gpuArray(T_incoherent/max(T_incoherent(:)));

% support constraint for pattern update

speckle_f_support = zeros(Np,Mp,'gpuArray');
speckle_f_support(sqrt(fxxp.^2+fyyp.^2)<2*NAs/lambda) = 1;

obj_f_support = zeros(Nc,Mc);
obj_f_support(sqrt(fxx_c.^2+fyy_c.^2)<2*(NAs+NA_obj)/lambda) = 1;
obj_f_support = gpuArray(obj_f_support);

OTF_support = zeros(Nc,Mc);
OTF_support(r_obj<2*NA_obj/lambda) = 1;
OTF_support = gpuArray(OTF_support);

end