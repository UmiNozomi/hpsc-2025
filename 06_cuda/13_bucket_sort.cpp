#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void bucket_sort(int *key, int *bucket, int n) {
  int tid = threadIdx.x;
  if (tid < n) {
    atomicAdd(&bucket[key[tid]], 1);
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range, 0);
  
  int *d_key, *d_bucket;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));
  
  cudaMemset(d_bucket, 0, range * sizeof(int));
  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  
  bucket_sort<<<1, 256>>>(d_key, d_bucket, n);
  
  cudaDeviceSynchronize();
  cudaMemcpy(bucket.data(), d_bucket, range * sizeof(int), cudaMemcpyDeviceToHost);
  
  int j = 0;
  for (int i = 0; i < range; i++) {
    while (bucket[i] > 0) {
      key[j++] = i;
      bucket[i]--;
    }
  }

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");

  cudaFree(d_key);
  cudaFree(d_bucket);
  return 0;
}
