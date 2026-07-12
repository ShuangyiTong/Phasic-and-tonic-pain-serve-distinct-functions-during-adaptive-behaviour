/*
 * Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
 * Licensed under the MIT License.
 */

#ifndef CUDA_DEFS_H
#define CUDA_DEFS_H

#ifdef CUDA_ENABLED
    #include <cuda_runtime.h>
    #include <assert.h>

    #define THREADSPERBLOCK 512
    #define MAX_LAUNCH_SIZE 0x8000000u // (4G / 16)
#endif

#ifdef CUDA_ENABLED
    #define CUDA_CHECKS(code) do { \
    if (code != cudaSuccess) \
        { \
            std::cerr << "CUDA call error: " << cudaGetErrorString(code) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0);

    #define CUDA_DEVICE_ALLOC_AND_COPY(__device_ptr, __host_ptr, __size, __size_unit, ret) do { \
        ret = cudaMalloc(&__device_ptr, __size * __size_unit); \
        CUDA_CHECKS(ret) \
        ret = cudaMemcpy(__device_ptr, __host_ptr, __size * __size_unit, cudaMemcpyHostToDevice); \
        CUDA_CHECKS(ret) \
    } while (0);

    #define CUDA_DEVICE_ALLOC_AND_COPY_NESTED_ARRAY(DATA_TYPE, __device_ptr, __host_ptr_to_dim2_ptr, __host_ptr, __dim1_size, __dim1_size_unit, __dim2_size, __dim2_size_unit, ret) do { \
        __host_ptr_to_dim2_ptr = new DATA_TYPE*[__dim1_size]; \
        for (int __dim1_iter = 0; __dim1_iter < __dim1_size; __dim1_iter++) \
        { \
            CUDA_DEVICE_ALLOC_AND_COPY(__host_ptr_to_dim2_ptr[__dim1_iter], __host_ptr[__dim1_iter], __dim2_size[__dim1_iter], __dim2_size_unit, ret) \
        } \
        CUDA_DEVICE_ALLOC_AND_COPY(__device_ptr, __host_ptr_to_dim2_ptr, __dim1_size, __dim1_size_unit, ret) \
    } while (0);

    #define CUDA_FREE_WITH_CHECK(__devptr, ret) do { \
        ret = cudaFree(__devptr); \
        CUDA_CHECKS(ret) \
    } while (0);

    #define CUDA_FREE_WITH_CHECK_NESTED_ARRAY(__devptr, __host_ptr_to_dim2_ptr, __size, ret) do { \
        for (int __ptr_iter = 0; __ptr_iter < __size; __ptr_iter++) \
        { \
            ret = cudaFree(__host_ptr_to_dim2_ptr[__ptr_iter]); \
            CUDA_CHECKS(ret) \
        } \
        CUDA_FREE_WITH_CHECK(__devptr, ret); \
    } while (0);
#endif

#endif // CUDA_DEFS_H