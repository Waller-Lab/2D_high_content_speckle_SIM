
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fSIM_2D_iter_alg iteratively solves the fluorescence speckle structured         % 
% illumination optimization problem for super-resolved fluorescence image. At the %
% same time, the algorithm jointly estimates the unknown speckle intensity,       % 
% aberrated OTF, and refines the scan positions.                                  %                      
%                                                                                 %
% Inputs:                                                                         %
%       I_image_up        : measurement image stack                               %
%       I_obj             : initial guess of the fluorescence sample              %
%       I_p_whole         : initial guess of the speckle                          %
%       T_incoherent      : OTF of the image system                               %
%       fxxp              : x spatial frequency with the size of the speckle      %
%       fyyp              : y spatial frequency with the size of the speckle      %
%       speckle_f_support : Fourier support of the speckle                        %
%       obj_f_support     : Fourier support of the object                         %
%       OTF_f_support     : Fourier support of the OTF                            %
%       xshift            : x-dimension shift of scanning positions               %
%       yshift            : y-dimension shift of scanning positions               %
%       itr               : max number of iterations                              %
% Outputs:                                                                        %
%       I_obj             : super-resolved fluorescence image                     %
%       I_p_whole         : estimated speckle pattern                             %
%       T_incoherent      : estimated OTF                                         %
%       xshift            : refined x-dimension shift of scanning positions       %
%       yshift            : refined y-dimension shift of scanning positions       %
%       err               : modified cost function to evaluate convergence        % 
%                                                                                 %
%                                                                                 %
%          Copyright (C) Li-Hao Yeh 2019                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [I_obj, I_p_whole, T_incoherent, xshift, yshift, err] = ...
    fSIM_2D_iter_alg(I_image_up, I_obj, I_p_whole, T_incoherent, ...
    fxxp, fyyp, speckle_f_support, obj_f_support, OTF_support, xshift, yshift, itr)


F = @(x) fft2(x);
iF = @(x) ifft2(x);

global ps N_bound_pad N M Nc Mc xshift_max yshift_max Nimg

% cost function
err = zeros(1,itr+1);

tic;
fprintf('| Iter  |   error    | Elapsed time (sec) |\n');
for i = 1:itr
    
    % Sequential update
    
    for j = 1:Nimg
        
        Ip_shift = max(0,real(iF(F(I_p_whole).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))))));
        I_p_gpu = Ip_shift(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);
        
        I_image_current = gpuArray(I_image_up(:,:,j));

        I_multi_f = F(I_p_gpu.*I_obj);
        I_est =  iF(T_incoherent.*I_multi_f);        
        I_diff = I_image_current - I_est(N_bound_pad+1:N_bound_pad+N,N_bound_pad+1:N_bound_pad+M);
        
        I_temp = iF(T_incoherent.*F(padarray(I_diff,[N_bound_pad,N_bound_pad])));

        grad_Iobj = -real(I_p_gpu.*I_temp);        
        grad_Ip = -real(iF(F(padarray(I_obj.*I_temp/max(I_obj(:))^2,[yshift_max,xshift_max],0)).*exp(-1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j)))));            
        grad_OTF = -conj(I_multi_f).*F(I_temp).*OTF_support;
        
        
        I_obj = real(iF(F(I_obj - real(grad_Iobj/max(I_p_gpu(:))^2)).*obj_f_support));
        I_p_whole = real(iF(F(I_p_whole -grad_Ip).*speckle_f_support));
        T_incoherent = T_incoherent - grad_OTF/max(abs(I_multi_f(:))).*abs(I_multi_f)./(abs(I_multi_f).^2 + 1e-3)/12;

        
        % shift estimate
        Ip_shift_fx = iF(F(I_p_whole).*(1j*2*pi*fxxp).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))));
        Ip_shift_fy = iF(F(I_p_whole).*(1j*2*pi*fyyp).*exp(1j*2*pi*ps*(fxxp.*xshift(j) + fyyp.*yshift(j))));
        
        Ip_shift_fx = Ip_shift_fx(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);
        Ip_shift_fy = Ip_shift_fy(1+yshift_max:Nc+yshift_max,1+xshift_max:Mc+xshift_max);

        grad_xshift = -real(sum(sum(conj(I_temp).*I_obj.*Ip_shift_fx)));
        grad_yshift = -real(sum(sum(conj(I_temp).*I_obj.*Ip_shift_fy)));

        xshift(j) = xshift(j) - gather(grad_xshift/N/M/max(I_obj(:))^2*1e3);
        yshift(j) = yshift(j) - gather(grad_yshift/N/M/max(I_obj(:))^2*1e3);
        
        err(i+1) = err(i+1) + gather(sum(sum(abs(I_diff).^2)));

        
    end
    
    
    temp = I_obj;
    temp_Ip = I_p_whole;
    if i == 1           
        t = 1;
        
        I_obj = temp;
        tempp = temp;
        
        I_p_whole = temp_Ip;
        tempp_Ip = temp_Ip;
    else
        if (err(i) >= err(i-1))
            t = 1;
        
            I_obj = temp;
            tempp = temp;

            I_p_whole = temp_Ip;
            tempp_Ip = temp_Ip;
        else
            tp = t;
            t = (1+sqrt(1+4*tp^2))/2;

            I_obj = temp + (tp-1)*(temp - tempp)/t;
            tempp = temp;

            I_p_whole = temp_Ip + (tp-1)*(temp_Ip - tempp_Ip)/t;
            tempp_Ip = temp_Ip;
        end
    end
    
    
    if mod(i,1) == 0
        fprintf('|  %2d   |  %.2e  |        %.2f        |\n', i, err(i+1),toc);
        figure(31);
        subplot(1,2,1),imagesc(I_obj,[0 max(I_obj(:))]);colormap gray;axis square;
        subplot(1,2,2),imagesc(I_p_whole,[0 max(I_p_whole(:))]);colormap gray;axis square;
        figure(32);
        subplot(1,2,1),plot(xshift,yshift,'bo');axis square;
        subplot(1,2,2),imagesc(abs(fftshift(T_incoherent)));colormap jet;axis image;
        pause(0.001);
    end

end


end