import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from IPython import display
from fSIM_2D_func.dftregistration import dftregistration


def image_upsampling(I_image, upsamp_factor = 1, bg = 0):
    
    '''
    image_upsampling performs upsampling on the image stack, the 
    upsampling factor should be large enough to sample the expected 
    resolution with Nyquist rate.   
    
    Inputs:                                                         
            I_image          : the input image stack                   
            upsamplng_factor : the upsampling factor                   
            bg               : background noise                     
    
    Outputs:                                                         
            I_image_up      : the upsampled image stack
    '''
    
    F = lambda x: ifftshift(fft2(fftshift(x)))
    iF = lambda x: ifftshift(ifft2(fftshift(x)))
    
    Nimg, Ncrop, Mcrop = I_image.shape

    N = Ncrop*upsamp_factor
    M = Mcrop*upsamp_factor

    I_image_up = np.zeros((Nimg,N,M))
    
    for i in range(0,Nimg):
        I_image_up[i] = abs(iF(np.pad(F(np.maximum(0,I_image[i]-bg)),\
                                  (((N-Ncrop)//2,),((M-Mcrop)//2,)),mode='constant')))
        
    return I_image_up


def display_image_movie(image_stack, frame_num, size, pause_time=0.0001):
    
    '''
    display_image_movie displays an image stack as a movie in the notebook.
    
    Inputs:
            image_stack : image stack to be displayed with size of (Nframe, N, M)
            frame_num   : number of frames to be displayed
            size        : figure size
            pause_time  : pausing time between frames
    
    '''
    
    f1,ax = plt.subplots(1,1,figsize=size)
    max_val = np.max(image_stack)

    for i in range(0,frame_num):
        if i != 1:
            ax.cla()
        ax.imshow(image_stack[i],cmap='gray',vmin=0,vmax=max_val)
        display.display(f1)
        display.clear_output(wait=True)
        time.sleep(pause_time)
        
        
        
def image_registration(img_stack,usfac, img_up):
    
    '''
    
    image_registration performs image registration on the coherent image    
    stack to find the scanning trajectory. The registration code is 
    translated to python [1] from MATLAB published in [2].
     
    Inputs:                                                                 
        img_stack : the coherent image stack                       
        usfac     : the subpixel accuracy (100 means 1/100 pixels) 
        img_up    : the image upsampling factor                    
    
    Outputs:                                                              
        xshift    : x-dimension shift of scanning positions            
        yshift    : y-dimension shift of scanning positions            
                                                                          
    [1] https://github.com/keflavich/image_registration
    [2] M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, "Efficient
    subpixel registration algorithms," Opt.Lett. 33, 156-158, (2008)
    
    '''
    
    Nimg,_,_ = img_stack.shape
    xshift = np.zeros(Nimg)
    yshift = np.zeros(Nimg)

    for i in range(0,Nimg):
        if i == 0:
            yshift[i] == 0
            xshift[i] == 0
        else:
            output = dftregistration(fft2(img_stack[0]),fft2(img_stack[i]),usfac)
            yshift[i] = output[0] * img_up
            xshift[i] = output[1] * img_up
            
    return xshift, yshift


class fSIM_2D_solver:
    
    def __init__(self, I_image_up, xshift, yshift, N_bound_pad, lambda_f, pscrop, upsamp_factor, NA_obj, NAs, itr):
        
        
        '''
        
        initialize the system parameters and optimization parameters to get ready for the iterative algorithm processing
        
        Inputs:
            I_image_up    : measurement image stack                    
            xshift        : x-dimension shift of scanning positions      
            yshift        : y-dimension shift of scanning positions 
            N_bound_pad   : number of pixels padded for better boundary condition
            lambda_f      : wavelength of fluorescent light
            pscrop        : effective pixel size on camera plane
            upsamp_factor : upsampling factor of the measurement
            NA_obj        : objective NA
            NAs           : speckle NA
            itr           : number of iterations for the algorithm
        
        '''
        
        # Basic parameter 
        self.Nimg, self.N, self.M = I_image_up.shape
        self.N_bound_pad = N_bound_pad
        self.Nc = self.N + 2*N_bound_pad
        self.Mc = self.M + 2*N_bound_pad
        self.ps = pscrop/upsamp_factor
        self.itr = itr
        self.err = np.zeros(self.itr+1)
        
        # Shift variable
        self.xshift = xshift
        self.yshift = yshift        
        self.xshift_max = np.int(np.round(np.max(abs(xshift))))
        self.yshift_max = np.int(np.round(np.max(abs(yshift))))
        
        self.spatial_freq_gen()
        self.initialization(I_image_up)
        self.kernel_gen(lambda_f, NA_obj, NAs)
        
    def spatial_freq_gen(self):
        
        '''
        
        spatial_freq_gen generates spatial frequency arrays for later use
        
        '''
        
        # Frequency grid definition to create TF
        fx_c = np.r_[-self.Mc/2:self.Mc/2]/self.ps/self.Mc
        fy_c = np.r_[-self.Nc/2:self.Nc/2]/self.ps/self.Nc

        fxx_c, fyy_c = np.meshgrid(fx_c,fy_c)

        self.fxx_c = ifftshift(fxx_c)
        self.fyy_c = ifftshift(fyy_c)
        
        self.Npp = self.Nc + 2*self.yshift_max
        self.Mpp = self.Mc + 2*self.xshift_max


        fxp = np.r_[-self.Mpp/2:self.Mpp/2]/self.ps/self.Mpp
        fyp = np.r_[-self.Npp/2:self.Npp/2]/self.ps/self.Npp

        fxxp, fyyp = np.meshgrid(fxp,fyp)

        self.fxxp = ifftshift(fxxp)
        self.fyyp = ifftshift(fyyp)
        

    def initialization(self, I_image_up):
        
        '''
        
        initialization initializes the fluorescence image and the speckle intensity.
        
        '''
        
        # Initialization of object and pattern
        self.I_obj = np.pad(np.mean(I_image_up,axis=0),(self.N_bound_pad,),mode='constant')
        self.I_p_whole = np.ones((self.Npp, self.Mpp))
    
    def kernel_gen(self, lambda_f, NA_obj, NAs):
        
        '''
        
        kernel_gen generates the required filter for the algorithm, including the OTF 
        and the Fourier support of the object, speckle and the OTF.
        
        '''
        
        # Compute transfer function
        Pupil_obj = np.zeros((self.Nc,self.Mc))
        frc = (self.fxx_c**2 + self.fyy_c**2)**(1/2)
        Pupil_obj[frc<NA_obj/lambda_f] = 1
        T_incoherent = abs(fft2(abs(ifft2(Pupil_obj))**2))
        self.T_incoherent = T_incoherent/np.max(T_incoherent)
        
        # Compute support function
        self.Pattern_support = np.zeros((self.Npp,self.Mpp))
        frp = (self.fxxp**2 + self.fyyp**2)**(1/2)
        self.Pattern_support[frp<2*NAs/lambda_f] = 1

        self.Object_support = np.zeros((self.Nc,self.Mc))
        self.Object_support[frc<2*(NA_obj+NAs)/lambda_f] = 1

        self.OTF_support = np.zeros((self.Nc,self.Mc))
        self.OTF_support[frc<2*NA_obj/lambda_f] = 1
        
        
    
    def iterative_algorithm(self, I_image_up, update_shift=1, shift_alpha=1, update_OTF=0, OTF_alpha=1, figsize=(15,5)):
        
        
        '''
        
        fSIM_2D_iter_alg iteratively solves the fluorescence speckle structured 
        illumination optimization problem for super-resolved fluorescence image. 
        At the same time, the algorithm jointly estimates the unknown speckle 
        intensity, aberrated OTF, and refines the scan positions.

        '''
        
        f1,ax = plt.subplots(1,3,figsize=figsize)

        tic_time = time.time()
        print('|  Iter  |  error  |  Elapsed time (sec)  |')

        for i in range(0,self.itr):

            # sequential update
            for j in range(0,self.Nimg):
                
                Ip_shift = np.maximum(0, np.real(ifft2(fft2(self.I_p_whole) * \
                                      np.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] + self.fyyp * self.yshift[j])))))
                I_p = Ip_shift[self.yshift_max:self.Nc+self.yshift_max, self.xshift_max:self.Mc+self.xshift_max]
                I_image_current = I_image_up[j]
                I_multi_f = fft2(I_p * self.I_obj)
                I_est = ifft2(self.T_incoherent * I_multi_f)
                I_diff = I_image_current - I_est[self.N_bound_pad:self.N_bound_pad+self.N, self.N_bound_pad:self.N_bound_pad+self.M]

                I_temp = ifft2(self.T_incoherent * fft2(np.pad(I_diff,(self.N_bound_pad,),mode='constant')))

                # gradient computation
                
                grad_Iobj = -np.real(I_p * I_temp)
                grad_Ip = -np.real(ifft2(fft2(np.pad(self.I_obj * I_temp,((self.yshift_max,),(self.xshift_max,)), mode='constant'))\
                                         * np.exp(-1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] + self.fyyp * self.yshift[j]))))
                if update_OTF ==1:
                    grad_OTF = -np.conj(I_multi_f) * fft2(I_temp) 

                # updating equation
                self.I_obj = np.real(ifft2(fft2(self.I_obj - grad_Iobj/(np.max(I_p)**2)) * self.Object_support))
                self.I_p_whole = np.real(ifft2(fft2(self.I_p_whole - grad_Ip/(np.max(self.I_obj)**2)) * self.Pattern_support))
                
                if update_OTF ==1:
                    self.T_incoherent = (self.T_incoherent - grad_OTF/np.max(abs(I_multi_f)) * \
                         abs(I_multi_f) / (abs(I_multi_f)**2 + 1e-3) * OTF_alpha * self.OTF_support).copy()

                # shift estimate
                if update_shift ==1:
                    Ip_shift_fx = ifft2(fft2(self.I_p_whole) * (1j*2*np.pi*self.fxxp) * \
                                       np.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] + self.fyyp * self.yshift[j])))
                    Ip_shift_fy = ifft2(fft2(self.I_p_whole) * (1j*2*np.pi*self.fyyp) * \
                                       np.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] + self.fyyp * self.yshift[j])))
                    Ip_shift_fx = Ip_shift_fx[self.yshift_max:self.yshift_max+self.Nc,self.xshift_max:self.xshift_max+self.Mc]
                    Ip_shift_fy = Ip_shift_fy[self.yshift_max:self.yshift_max+self.Nc,self.xshift_max:self.xshift_max+self.Mc]

                    grad_xshift = -np.real(np.sum(np.conj(I_temp) * self.I_obj * Ip_shift_fx))
                    grad_yshift = -np.real(np.sum(np.conj(I_temp) * self.I_obj * Ip_shift_fy))

                    self.xshift[j] = self.xshift[j] - grad_xshift/self.N/self.M/(np.max(self.I_obj)**2) * shift_alpha
                    self.yshift[j] = self.yshift[j] - grad_yshift/self.N/self.M/(np.max(self.I_obj)**2) * shift_alpha

                self.err[i+1] += np.sum(abs(I_diff)**2)

            # Nesterov acceleration
            temp = self.I_obj.copy()
            temp_Ip = self.I_p_whole.copy()
            if i == 0:
                t = 1

                self.I_obj = temp.copy()
                tempp = temp.copy()

                self.I_p_whole = temp_Ip.copy()
                tempp_Ip = temp_Ip.copy()
            else:
                if self.err[i] >= self.err[i-1]:
                    t = 1

                    self.I_obj = temp.copy()
                    tempp = temp.copy()

                    self.I_p_whole = temp_Ip.copy()
                    tempp_Ip = temp_Ip.copy()
                else:
                    tp = t
                    t = (1 + (1 + 4 * tp**2)**(1/2))/2

                    self.I_obj = temp + (tp - 1) * (temp - tempp) / t
                    tempp = temp.copy()

                    self.I_p_whole = temp_Ip + (tp - 1) * (temp_Ip - tempp_Ip) / t
                    tempp_Ip = temp_Ip.copy()

            if np.mod(i,1) == 0:
                print('|  %d  |  %.2e  |   %.2f   |'%(i+1,self.err[i+1],time.time()-tic_time))
                if i != 0:
                    ax[0].cla()
                    ax[1].cla()
                    ax[2].cla()
                ax[0].imshow(np.maximum(0,self.I_obj),cmap='gray');
                ax[1].imshow(np.maximum(0,self.I_p_whole),cmap='gray')
                ax[2].plot(self.xshift,self.yshift,'w')
                display.display(f1)
                display.clear_output(wait=True)
                time.sleep(0.0001)
                if i == self.itr-1:
                    print('|  %d  |  %.2e  |   %.2f   |'%(i+1,self.err[i+1],time.time()-tic_time))


