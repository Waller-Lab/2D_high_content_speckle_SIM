
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel_gen generates the required filter for the algorithm, including  %
% the propagation kernel, pupil function of the imaging system, Fourier  %
% supports of the object and speckle pattern.                            %                          
%                                                                        %
% Inputs:                                                                %
%       fxx_c     : x spatial frequency with the size of the object      %
%       fyy_c     : y spatial frequency with the size of the object      %
%       fxxp      : x spatial frequency with the size of the speckle     %
%       fyyp      : y spatial frequency with the size of the speckle     %
%       NA_obj    : objective NA                                         %
%       NAs       : speckle NA                                           %
% Outputs:                                                               %
%       Hz_det    : propagation kernel to different camera planes        %
%       Pupil_obj : pupil function of the imaging system                 %
%       Gaussian  : Gaussian Fourier support of the object               %
%       Pupil_NAs : Fourier support of the speckle pattern               %
%                                                                        %
%                                                                        %
%          Copyright (C) Li-Hao Yeh 2019                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Hz_det, Pupil_obj, Gaussian, Pupil_NAs] = kernel_gen(fxx_c, fyy_c, fxxp, fyyp, NA_obj, NAs)

global lambda z_camera Nc Mc Np Mp N_defocus

r_obj=(fxx_c.^2+fyy_c.^2).^(1/2);
Pupil_support = zeros(Nc,Mc);
Pupil_support(r_obj<NA_obj/lambda) = 1;
Pupil_support = gpuArray(Pupil_support);

Hz_det = zeros(Nc,Mc,N_defocus,'gpuArray');
Pupil_obj = Pupil_support;

for i = 1:N_defocus
    Hz_det(:,:,i) = Pupil_support.*gpuArray(exp(-1j*pi*lambda*z_camera(i)*(fxx_c.^2+fyy_c.^2)));
end

Gaussian = exp(-r_obj.^2/(2*((NA_obj+NAs)*0.95/lambda)^2));
Gaussian = gpuArray(Gaussian/max(Gaussian(:)));

Pupil_NAs = zeros(Np,Mp);
r_s=(fxxp.^2+fyyp.^2).^(1/2);
Pupil_NAs(find(r_s<NAs/lambda))=1;
Pupil_NAs = gpuArray(Pupil_NAs);

end