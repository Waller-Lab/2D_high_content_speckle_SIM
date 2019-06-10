
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% image_upsampling performs upsampling on the image stack, the     %
% upsampling factor should be large enough to sample the expected  %
% resolution with Nyquist rate.                                    %
%                                                                  %
% Inputs:                                                          %
%       I_image          : the input image stack                   %
%       upsamplng_factor : the upsampling factor                   %
%       bg               : background noise                        %
% Outputs:                                                         %
%       I_image_up       : the upsampled image stack               %
%                                                                  %
%                                                                  %
%          Copyright (C) Li-Hao Yeh 2019                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function I_image_up = image_upsampling(I_image, upsampling_factor, bg)

F = @(x) fftshift(fft2(ifftshift(x)));
iF = @(x) fftshift(ifft2(ifftshift(x)));

[Ncrop,Mcrop,Nimg] = size(I_image);


N = Ncrop*upsampling_factor; M = Mcrop*upsampling_factor; 


I_image_up = zeros(N,M,Nimg);
for i = 1:Nimg
    I_image_up(:,:,i) = abs(iF(padarray(F(max(0,I_image(:,:,i)-bg)),[(N-Ncrop)/2,(M-Mcrop)/2])));
end

end