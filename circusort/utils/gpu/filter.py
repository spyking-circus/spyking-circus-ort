# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import sosfilt

import pyopencl
mf = pyopencl.mem_flags

"""
Note: Filter logic from pyacq project at https://github.com/pyacq/pyacq
"""

class GpuFilterBase:
    """
    Base class for OpenCL implementation
    """    
    def __init__(self, context, coefficients, nb_channels, dtype, chunksize):
        self.dtype = np.dtype(dtype)
        assert self.dtype == np.dtype('float32')
        
        self.nb_channels = nb_channels
        self.chunksize = chunksize
        assert self.chunksize is not None, 'chunksize for opencl must be fixed'
        
        self.coefficients = coefficients.astype(self.dtype)
        if self.coefficients.ndim == 2: #(nb_sections, 6) to (nb_channels, nb_sections, 6)
            self.coefficients = np.tile(self.coefficients[None,:,:], (nb_channels, 1,1))
        if not self.coefficients.flags['C_CONTIGUOUS']:
            self.coefficients = self.coefficients.copy()      
        assert self.coefficients.shape[0] == self.nb_channels, 'wrong coefficients.shape'
        assert self.coefficients.shape[2] == 6, 'wrong coefficients.shape'

        self.nb_sections = self.coefficients.shape[1]
        self.zi = np.zeros((nb_channels, self.nb_sections, 2), dtype = self.dtype)

        #self.ctx = pyopencl.create_some_context()
        self.ctx = context
        self.queue = pyopencl.CommandQueue(self.ctx)
                
        # gpu buffers
        nbytes = self.chunksize * self.nb_channels * self.dtype.itemsize
        self.coefficients_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.coefficients)
        self.zi_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.zi)
        self.input_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size = nbytes)
        self.output_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size = nbytes)
        
        kernel = self.kernel%dict(chunksize = self.chunksize, nb_sections = self.nb_sections, nb_channels = self.nb_channels)
        prg = pyopencl.Program(self.ctx, kernel)
        self.opencl_prg = prg.build(options = '-cl-mad-enable')

class GpuFilter(GpuFilterBase):
    """
    Implementation with OpenCL: this version scales nb_channels
    """
    def __init__(self, context, coefficients, nb_channels, dtype, chunksize):
        GpuFilterBase.__init__(self, context, coefficients, nb_channels, dtype, chunksize)
        # global work-group size
        self.global_size = (self.nb_channels, )
        # local work-group size
        self.local_size = (self.nb_channels, )
        
        self.output = np.zeros((self.chunksize, self.nb_channels), dtype = self.dtype)
        self.kernel_func_name = 'sos_filter'
    
    def compute_one_chunk(self, chunk):
        assert chunk.dtype == self.dtype
        assert chunk.shape == (self.chunksize, self.nb_channels), 'wrong shape'
        
        if not chunk.flags['C_CONTIGUOUS']:
            chunk = chunk.copy()
        pyopencl.enqueue_copy(self.queue, self.input_cl, chunk)

        kern_call = getattr(self.opencl_prg, self.kernel_func_name)
        event = kern_call(self.queue, self.global_size, self.local_size,
                          self.input_cl, self.output_cl, self.coefficients_cl, self.zi_cl)
        event.wait()
        
        pyopencl.enqueue_copy(self.queue, self.output, self.output_cl)
        chunk_filtered = self.output
        return chunk_filtered
    
    kernel = """
    #define chunksize %(chunksize)d
    #define nb_sections %(nb_sections)d
    #define nb_channels %(nb_channels)d

    __kernel void sos_filter(__global float *input, __global float *output, 
                             __constant float *coefficients, __global float *zi) {
    
        // implement as cascade of second-order section filters with Direct Form II structure    
        int chan = get_global_id(0); // channel index
        
        int offset_filt2; // offset channel within section
        int offset_zi = chan*nb_sections*2;
        
        int idx;
    
        float w0, w1, w2; // state of the filter
        float res;
    
        for (int section=0; section<nb_sections; section++){        
                // DF-II difference equations
                // w[n] = input[n] - a1*w[n-1] - a2*w[n-2]
                // output[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
            
                // sos matrix
                // (b0, b1, b2, 1, a1, a2) (1st section)
                // ...
                
            offset_filt2 = chan*nb_sections*6 + section*6;
            
            w1 = zi[offset_zi + section*2 + 0];
            w2 = zi[offset_zi + section*2 + 1];
            
            for (int s=0; s<chunksize; s++){    
                idx = s*nb_channels + chan;
                if (section == 0)  
                    w0 = input[idx];
                else 
                    w0 = output[idx];
                
                w0 -= coefficients[offset_filt2 + 4] * w1;
                w0 -= coefficients[offset_filt2 + 5] * w2;
                res = coefficients[offset_filt2 + 0] * w0 + coefficients[offset_filt2 + 1] * w1 + coefficients[offset_filt2 + 2] * w2;
                w2 = w1; w1 = w0;
                
                output[idx] = res;
            }
            
            zi[offset_zi + section*2 + 0] = w1;
            zi[offset_zi + section*2 + 1] = w2;
        }
    }    
    """