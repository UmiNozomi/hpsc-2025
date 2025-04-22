#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];

  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j+=16) {
      __m512 xi = _mm512_set1_ps(x[i]);
      __m512 yi = _mm512_set1_ps(y[i]);
      
      __m512 xj = _mm512_loadu_ps(&x[j]);
      __m512 yj = _mm512_loadu_ps(&y[j]);
      __m512 mj = _mm512_loadu_ps(&m[j]);
      
      __m512 rx = _mm512_sub_ps(xi, xj);
      __m512 ry = _mm512_sub_ps(yi, yj);
      
      __m512 rx2 = _mm512_mul_ps(rx, rx);
      __m512 ry2 = _mm512_mul_ps(ry, ry);
      __m512 r2 = _mm512_add_ps(rx2, ry2);
      
      __m512 invr = _mm512_rsqrt14_ps(r2);
      __m512 invr3 = _mm512_mul_ps(invr, _mm512_mul_ps(invr, invr));
      
      __mmask16 mask = _mm512_cmp_epi32_mask(
        _mm512_set1_epi32(i),
        _mm512_set_epi32(j+15, j+14, j+13, j+12, j+11, j+10, j+9, j+8,
                          j+7, j+6, j+5, j+4, j+3, j+2, j+1, j),
        _MM_CMPINT_NE
      );
      
      __m512 fx_vec = _mm512_maskz_mul_ps(mask, _mm512_mul_ps(rx, mj), invr3);
      __m512 fy_vec = _mm512_maskz_mul_ps(mask, _mm512_mul_ps(ry, mj), invr3);
      
      alignas(64) float fx_arr[16], fy_arr[16];
      _mm512_store_ps(fx_arr, fx_vec);
      _mm512_store_ps(fy_arr, fy_vec);
      
      for(int k=0; k<16 && j+k<N; k++) {
        fx[i] -= fx_arr[k];
        fy[i] -= fy_arr[k];
      }
    }
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
