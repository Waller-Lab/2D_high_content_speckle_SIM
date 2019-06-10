%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% spatial_freq_gen generates spatial frequency arrays for later use %                                 
%                                                                   %
% Inputs:                                                           %
%       xshift : x-dimension shift of scanning positions            %
%       yshift : y-dimension shift of scanning positions            %
% Outputs:                                                          %
%       fxx_c  : x spatial frequency with the size of the object    %
%       fyy_c  : y spatial frequency with the size of the object    %
%       fxxp   : x spatial frequency with the size of the speckle   %
%       fyyp   : y spatial frequency with the size of the speckle   %
%                                                                   %
%                                                                   %
%          Copyright (C) Li-Hao Yeh 2019                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fxx_c, fyy_c, fxxp, fyyp] = spatial_freq_gen(xshift,yshift)

global ps N_bound_pad N M Nc Mc Np Mp xshift_max yshift_max

Nc = N + 2*N_bound_pad;
Mc = M + 2*N_bound_pad;


fx_c = (-Mc/2:(Mc/2-1))./(ps*Mc); fy_c = (-Nc/2:(Nc/2-1))./(ps*Nc);
[fxx_c,fyy_c] = meshgrid(fx_c,fy_c);

fxx_c = ifftshift(fxx_c);
fyy_c = ifftshift(fyy_c);


yshift_max = round(max(abs(yshift(:))));
xshift_max = round(max(abs(xshift(:))));


Np = Nc + 2*yshift_max;
Mp = Mc + 2*xshift_max;

fxp = (-Mp/2:(Mp/2-1))./(ps*Mp); fyp = (-Np/2:(Np/2-1))./(ps*Np);
[fxxp,fyyp] = meshgrid(fxp,fyp);

fxxp = gpuArray(ifftshift(fxxp));
fyyp = gpuArray(ifftshift(fyyp));
end