#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

#define M 8
#define N 32
#define K 16

template <typename T>
__global__ void set_abc(T *a, T *b, T *c) {
  int idx = threadIdx.x;
  if (idx == 0) {
    a[0] = T(1.1);
    b[0] = T(2.2);
    c[0] = T(3.3);
  }
}

template <typename T>
__global__ void print(T *a) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx == 0) {
    float v = __half2float(a[0]);

    printf("v = %f\n", v);
  }
}

template <int kMmaM, int kMmaN, int kMmaK>
struct FragmentA {
  static_assert((kMmaM == 16 && kMmaN == 8 && kMmaK == 16),
                "not implemented FragmentA");
};

template <>
struct FragmentA<16, 8, 16> {
  union {
    int32_t data[4];
    half hdata[8];
  };

  __device__ __forceinline__ void clear() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
  }
};

template <int kMmaM, int kMmaN, int kMmaK>
struct FragmentB {
  static_assert((kMmaM == 16 && kMmaN == 8 && kMmaK == 16),
                "not implemented FragmentA");
};

template <>
struct FragmentB<16, 8, 16> {
  union {
    int32_t data[2];
    half hdata[4];
  };

  __device__ __forceinline__ void clear() {
    data[0] = 0;
    data[1] = 0;
  }
};

template <int kMmaM, int kMmaN, int kMmaK>
struct FragmentC {
  static_assert((kMmaM == 16 && kMmaN == 8 && kMmaK == 16),
                "not implemented FragmentA");
};

template <>
struct FragmentC<16, 8, 16> {
  union {
    int32_t data[2];
    half hdata[4];
  };

  __device__ __forceinline__ void clear() {
    data[0] = 0;
    data[1] = 0;
  }

  __device__ void print() {
    printf("data[0] = %d, data[1] = %d\n", data[0], data[1]);
    printf("hdata[0] = %f, hdata[1] = %f, hdata[2] = %f, hdata[3] = %f,\n",
           __half2float(hdata[0]), __half2float(hdata[1]),
           __half2float(hdata[2]), __half2float(hdata[3]));
  }
};

template <typename FragmentTA, typename FragmentTB, typename FragmentTC>
__device__ __forceinline__ void Mma16816(FragmentTC &d, const FragmentTA &a,
                                         const FragmentTB &b,
                                         const FragmentTC &c) {
  uint32_t *D = (uint32_t *)(&d);
  uint32_t *A = (uint32_t *)(&a);
  uint32_t *B = (uint32_t *)(&b);
  uint32_t *C = (uint32_t *)(&c);

  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9}\n;"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]));
  return;
}

template <typename T>
__global__ void mma(const T *a, const T *b, T *c) {
  FragmentA<16, 8, 16> A;
  FragmentB<16, 8, 16> B;
  FragmentC<16, 8, 16> C;

  A.clear();
  B.clear();
  C.clear();

  if (threadIdx.x == 0) {
    A.hdata[0] = a[0];
    B.hdata[0] = b[0];
    C.hdata[0] = c[0];
  }

  Mma16816(C, A, B, C);

  // C.print();

  if (threadIdx.x == 0) {
    c[0] = C.hdata[0];
  }
}

int main() {
  int n = M * N * K;
  using T = half;

  T *a, *b, *c;
  cudaMalloc(&a, sizeof(T) * n);
  cudaMalloc(&b, sizeof(T) * n);
  cudaMalloc(&c, sizeof(T) * n);

  printf("M = %d, N = %d, K = %d\n", M, N, K);

  set_abc<<<1, 1>>>(a, b, c);
  mma<<<1, 32>>>(a, b, c);
  cudaDeviceSynchronize();

  print<<<1, 1>>>(c);
  cudaDeviceSynchronize();
}
