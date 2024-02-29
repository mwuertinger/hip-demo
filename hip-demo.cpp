#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>

__global__ void vector_add(const float *a, const float *b, float *c, int size) {
  int index = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
  if (index < size) {
    c[index] += (a[index] + b[index]);
  }
}

// Function to populate an array with random float numbers
void random_floats(float *array, int size) {
  for (int i = 0; i < size; i++) {
    array[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

int main() {
  constexpr int size = 1024 * 1024 * 1024;

  float *a = (float *)malloc(size * sizeof(float));
  float *b = (float *)malloc(size * sizeof(float));
  float *c = (float *)malloc(size * sizeof(float));

  random_floats(a, size);
  random_floats(b, size);

  float *device_a, *device_b, *device_c;
  hipMalloc((void **)&device_a, size * sizeof(float));
  hipMalloc((void **)&device_b, size * sizeof(float));
  hipMalloc((void **)&device_c, size * sizeof(float));

  auto transfer_start = std::chrono::high_resolution_clock::now();
  hipMemcpy(device_a, a, size * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(device_b, b, size * sizeof(float), hipMemcpyHostToDevice);
  auto transfer_end = std::chrono::high_resolution_clock::now();

  auto kernel_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1024; i++) {
    hipLaunchKernelGGL(vector_add,
                       dim3((size + 255) / 256), // Blocks
                       dim3(256),                // Threads per block
                       0, 0, device_a, device_b, device_c, size);
  }
  auto kernel_end = std::chrono::high_resolution_clock::now();

  auto result_start = std::chrono::high_resolution_clock::now();
  hipMemcpy(c, device_c, size * sizeof(float), hipMemcpyDeviceToHost);
  auto result_end = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 100; i++) {
    printf("%5.3f + %5.3f = %5.3f\n", a[i], b[i], c[i]);
  }

  auto transfer_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(transfer_end -
                                                            transfer_start);
  auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      kernel_end - kernel_start);
  auto result_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      result_end - result_start);

  std::cout << "Data transfer time: " << transfer_duration.count()
            << " microseconds" << std::endl;
  std::cout << "Kernel execution time: " << kernel_duration.count()
            << " microseconds" << std::endl;
  std::cout << "Result transfer time: " << result_duration.count()
            << " microseconds" << std::endl;

  hipFree(device_a);
  hipFree(device_b);
  hipFree(device_c);
  return 0;
}
