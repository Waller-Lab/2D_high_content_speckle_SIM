# 2D_high_content_speckle_SIM

These are MATLAB/Python3 implementations of coherent/fluorescent structured illumination microscopy algorithm with random speckle illumination. With the speckle NA much larger than the objective NA, we are able to achieve 4x resolution gain with this technique. For detailed explanation, please see our paper [1]. To use this code, please also cite [1]. <br/>

```cSIM_2D_MATLAB/Python3``` contains the coherent part of the processing code. It takes the coherent speckle illuminated measurements and reconstruct super-resolution phase image. <br/>

```fSIM_2D_MATLAB/Python3```: contains the fluorescent part of the processing code. It takes the coherent speckle illuminated measurements and register to get the scanning path. Then, it takes the fluorescent speckle illuminated measurements to reconstruct super-resolution fluorescence image. <br/>

[1] L.-H. Yeh, S. Chowdhury, and L. Waller, "Computational structured illumination for high-content fluorescence and phase microscopy," Biomed. Opt. Express 10, 1978-1998 (2019)

## Python Environment requirement
Python 3.6, ArrayFire <br/>

1. Follow http://arrayfire.org/docs/index.htm for ArrayFire installation
2. Follow https://github.com/arrayfire/arrayfire-python to install ArrayFire Python and set the path to the libraries

## Data download
You can find sample experiment data (2 um fluorescence beads imaged with 4x 0.1 NA objective) from here: <br/>
1. Coherent image stack: https://drive.google.com/file/d/1GETISfw-NZxPmIvGWJfxIlyClB0Eebt2/view?usp=sharing <br/>
2. Fluorescence image stack: https://drive.google.com/file/d/1Fn_i4kqypjx_0_zD9ehHAfYp7Jrf49Tp/view?usp=sharing <br/>

Please make sure your dataset is in the same path as in the one assigned in the code. <br/>
