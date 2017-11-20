#include <stdio.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>



/********************************
 *
 * 测试gpu的个数
 ********************************/
int main() {
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;
    printf("%d,%d\n",gpuDeviceCount,deviceCount);

    for(int device = 0; device< deviceCount; ++device)
    {
        cudaGetDeviceProperties(&properties,device);
        if(properties.major!=9999)
        {
            ++gpuDeviceCount;
        }
    }
    printf("%d,%d\n",gpuDeviceCount,deviceCount);
    if (gpuDeviceCount > 0)
        return 0; /* success */
    else
        return 1; /* failure */
} 