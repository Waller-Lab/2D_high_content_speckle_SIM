
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% zernike_gen generates Zernike basis based on [1] for aberration estimation        %                                                      %                         
%                                                                                   %
% Inputs:                                                                           %
%       n, m    : indices of Zernike bases                                          %
%       fxx_c   : x spatial frequency with the size of the object                   %
%       fyy_c   : y spatial frequency with the size of the object                   %
%       NA_obj  : objective NA                                                      %
% Outputs:                                                                          %
%       zerpoly : the corresponding Zernike bases from indices n, m                 %
%                                                                                   %
% [1] https://www.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials %
%                                                                                   %
%          Copyright (C) Li-Hao Yeh 2019                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function zerpoly = zernike_gen(n, m, fxx_c, fyy_c, NA_obj)

global Nc Mc lambda
N_idx = length(n);
zerpoly = zeros(Nc,Mc,N_idx);
for i =1:N_idx
    [theta,rr] = cart2pol(fxx_c/NA_obj*lambda,fyy_c/NA_obj*lambda);
    idx = rr<=1;
    z = zeros(size(fxx_c));
    z(idx) = zernfun(n(i),m(i),rr(idx),theta(idx));
    z = z/max(z(:));
    zerpoly(:,:,i) = z;
end
zerpoly = gpuArray(zerpoly);

end