%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% image_registration performs image registration on the coherent image    %
% stack to find the scanning trajectory. The registration                 %
% code is from [1].                                                       %
%                                                                         %
% Inputs:                                                                 %
%       img_stack        : the coherent image stack                       %
%       usfac            : the subpixel accuracy (100 means 1/100 pixels) %                
%       upsamplng_factor : the image upsampling factor                    %
% Outputs:                                                                %
%       xshift       : x-dimension shift of scanning positions            %
%       yshift       : y-dimension shift of scanning positions            %
%                                                                         %
%          Copyright (C) Li-Hao Yeh 2019                                  %
%                                                                         %
% [1] M.Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, "Efficient      %
% subpixel registration algorithms," Opt.Lett. 33, 156-158, (2008)        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xshift,yshift] = image_registration(img_stack, usfac, upsampling_factor)

global Nimg

yshift = zeros(Nimg,1);
xshift = zeros(Nimg,1);


for j = 1:Nimg
    if j == 1
        yshift(j) = 0;
        xshift(j) = 0;
    else
        [output, ~] = dftregistration(fft2(img_stack(:,:,1)),fft2(img_stack(:,:,j)),usfac);
        yshift(j) = (output(3))*upsampling_factor; 
        xshift(j) = (output(4))*upsampling_factor;
    end
end

end