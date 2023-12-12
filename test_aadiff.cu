#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cub/cub.cuh"
#include <vector>
#include <memory>
#include <string>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <thread>
#include <mutex>
#include "stdio.h"
#include <set>
#include <map>
#include <sstream>
#include <random>
#include <iostream>


#define M 8192
#define K 19456
#define N 14336
#define ITERATION 10 


#define DTOR_THROW noexcept(false) 


#define CUDA_CHECK(__cond)                              \
  do {                                                  \
    auto __err = (__cond);                              \
    if (__err != cudaSuccess) {                         \
      auto __msg = cudaGetErrorString(__err);           \
      throw std::runtime_error(std::string(__FILE__)    \
        + ":" + std::to_string(__LINE__) + ": "         \
        + #__cond + " failed with message : " + __msg   \
        + " , code : " + std::to_string(__err));        \
    }                                                   \
  } while (0)


#define CUBLAS_CHECK(__cond)                            \
  do {                                                  \
    auto __err = (__cond);                              \
    if (__err != CUBLAS_STATUS_SUCCESS) {               \
      throw std::runtime_error(std::string(__FILE__)    \
        + ":" + std::to_string(__LINE__) + ": "         \
        + #__cond + " failed with code : "              \
        + std::to_string(__err));                       \
    }                                                   \
  } while (0)


#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete
  

struct DeviceGuard {
  DISABLE_COPY_AND_ASSIGN(DeviceGuard);

 public:
  explicit DeviceGuard(int dev_id) {
    CUDA_CHECK(cudaGetDevice(&dev_id_));
    CUDA_CHECK(cudaSetDevice(dev_id));
  }

  ~DeviceGuard() DTOR_THROW {
    CUDA_CHECK(cudaSetDevice(dev_id_));
  }
  
 private:
  int dev_id_;
}; 

struct CUDAResource {
  DISABLE_COPY_AND_ASSIGN(CUDAResource);

 public:
  int dev_id;
  cudaStream_t stream;
  cublasHandle_t handle;

  explicit CUDAResource(int dev_id) : dev_id(dev_id) {
    DeviceGuard guard(dev_id);
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&handle)); 
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }

  ~CUDAResource() DTOR_THROW {
    SyncStream();
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SyncStream() const {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};


class Matrix {
  DISABLE_COPY_AND_ASSIGN(Matrix); 
  
 public:
  Matrix(const CUDAResource &resource, const float *cpu_data, int height, int width) {
    dev_id_ = resource.dev_id;
    height_ = height;
    width_ = width;

    DeviceGuard guard(dev_id_); 
    size_t nbytes = sizeof(data_[0]) * height_ * width_;
    resource.SyncStream();
    CUDA_CHECK(cudaMalloc(&data_, nbytes));
    if (cpu_data != nullptr) {
      CUDA_CHECK(cudaMemcpyAsync(data_, cpu_data, nbytes, cudaMemcpyHostToDevice, resource.stream)); 
    }
    resource.SyncStream();
  }

  ~Matrix() DTOR_THROW {
    DeviceGuard guard(dev_id_);
    CUDA_CHECK(cudaFree(data_));
  }

  void Matmul(const CUDAResource &resource, const Matrix &other, Matrix *z) const {
    DeviceGuard guard(dev_id_);
    if (width_ != other.height_) {
      throw std::runtime_error("Invalid Argument: " + std::to_string(width_) + " vs " + std::to_string(other.height_));
    }

    if (height_ != z->height_) {
      throw std::runtime_error("Invalid Argument: " + std::to_string(height_) + " vs " + std::to_string(z->height_));
    }

    if (other.width_ != z->width_) {
      throw std::runtime_error("Invalid Argument: " + std::to_string(other.width_) + " vs " + std::to_string(z->width_));
    }

    const auto *x_data = data_;
    const auto *y_data = other.data_; 
    int m = height_; 
    int k = width_;
    int n = other.width_; 
  
    int lda = k;
    int ldb = n;
    int ldc = n;

    using DType = typename std::remove_pointer<decltype(x_data)>::type;
    DType alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasSgemmEx(
        resource.handle, 
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        y_data,
        CUDA_R_32F,
        ldb,
        x_data,
        CUDA_R_32F,
        lda,
        &beta,
        z->data_,
        CUDA_R_32F,
        ldc));

    resource.SyncStream();
  } 

  std::vector<float> ToCPU(const CUDAResource &resource) const {
    DeviceGuard guard(dev_id_);
    size_t nbytes = height_ * width_ * sizeof(data_[0]);
    std::vector<float> out(height_ * width_);
    resource.SyncStream();
    CUDA_CHECK(cudaMemcpyAsync(out.data(), data_, nbytes, cudaMemcpyDeviceToHost, resource.stream)); 
    return out;
  }

 private:
  float *data_;
  int height_;
  int width_;
  int dev_id_;  
}; 


template <typename T>
bool IsEqual(const std::vector<T> &x, const std::vector<T> &y) {
  if (x.size() != y.size()) return false;
  bool is_equal = (std::memcmp(x.data(), y.data(), sizeof(T) * x.size()) == 0);
  return is_equal;
}


void ThreadMain(std::reference_wrapper<const CUDAResource> resource, const float *x_data, const float *y_data, 
                int m, int k, int n, int iteration, std::vector<float> *out, int *has_aadiff) {
  *has_aadiff = 0;
  auto &res = resource.get();
  Matrix x(res, x_data, m, k);
  Matrix y(res, y_data, k, n); 
  Matrix z(res, nullptr, m, n);

  auto compute = [&] {
    x.Matmul(res, y, &z); 
    return z.ToCPU(res);
  };

  *out = compute(); 
  for (int i = 1; i < iteration; ++i) {
    auto tmp_out = compute();
    if (!IsEqual(*out, tmp_out)) {
      *has_aadiff = 1;
    }
  }
}


template <typename T>
std::vector<std::vector<int>> FindEqualGroup(const std::vector<std::vector<T>> &data) {
  int n = static_cast<int>(data.size());
  if (n == 0) return {}; 

  std::map<int, std::set<int>> result;
  std::set<int> left;
  for (int i = 0; i < n; ++i) {
    left.insert(i);
  }

  while (!left.empty()) { 
    auto beg = left.begin();
    auto first_value = *beg;
    result[first_value].insert(first_value); 
    left.erase(beg);

    for (auto iter = left.begin(); iter != left.end(); ) {
      bool is_equal = IsEqual(data[first_value], data[*iter]);
      if (is_equal) {
        result[first_value].insert(*iter); 
        iter = left.erase(iter);
      }
    } 
  } 
  std::vector<std::vector<int>> group;
  for (const auto &pair : result) {
    group.emplace_back();
    group.back().assign(pair.second.begin(), pair.second.end());
  } 
  return group;
} 


template <typename T>
std::string VectorToString(const std::vector<T> &data) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < data.size(); ++i) {
    if (i > 0) {
      ss << ", "; 
    }
    ss << data[i];
  }
  ss << "]";
  return ss.str();
}


static unsigned int GetSeed() {
  std::random_device rd;
  return rd();
}

template <typename T, typename Generator>  
void GenerateData(std::vector<T> *data, Generator generator) {
  for (auto iter = data->begin(); iter != data->end(); ++iter) {
    *iter = static_cast<T>(generator());
  }
}

template <typename T>
std::vector<T> CPUMatmul(const T *x, const T *y, int m, int k, int n) {
  std::vector<T> z(m * n, static_cast<T>(0));
  for (int z_i = 0; z_i < m; ++z_i) {
    for (int z_j = 0; z_j < n; ++z_j) {
      auto &z_data = z[z_i * n + z_j]; 
      for (int x_k = 0; x_k < k; ++x_k) {
         auto &x_data = x[z_i * k + x_k];
         auto &y_data = y[x_k * n + z_j]; 
         z_data += (x_data * y_data);
      }
    }
  } 
  return z;
}

void TestMain() {
  int dev_cnt = -1;
  int rt_ver = -1, driver_ver = -1;
  CUDA_CHECK(cudaGetDeviceCount(&dev_cnt));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));
  CUDA_CHECK(cudaDriverGetVersion(&driver_ver));
  printf("Device Number : %d , Runtime Version : %d , Driver Version : %d\n", dev_cnt, rt_ver, driver_ver);

  CUDA_CHECK(cudaSetDevice(0));
  std::vector<std::unique_ptr<CUDAResource>> resources;
  std::vector<std::vector<float>> outputs(dev_cnt);
  for (int i = 0; i < dev_cnt; ++i) {
    resources.emplace_back(new CUDAResource(i));
  }

  int m = M;
  int k = K;
  int n = N;
  int iteration = ITERATION; 

  std::vector<float> x(m * k);
  std::vector<float> y(k * n);

  auto seed = GetSeed();
  std::default_random_engine engine(seed);
  
  std::normal_distribution<float> dist(0.0, 0.01); 

  auto generator = [&engine, &dist] { return dist(engine); };
  GenerateData(&x, generator);
  GenerateData(&y, generator);

  std::vector<std::thread> threads; 
  std::vector<int> has_aadiff(dev_cnt, 0);
  for (int i = 0; i < dev_cnt; ++i) {
    auto &resource = *(resources[i]);
    threads.emplace_back(ThreadMain, std::cref(resource), x.data(), y.data(), m, k, n, iteration, &(outputs[i]), &(has_aadiff[i]));
  }

  for (auto &th : threads) {
    th.join();
  }

  std::vector<int> aadiff_devs;  
  for (int i = 0; i < dev_cnt; ++i) {
    if (has_aadiff[i]) {
      aadiff_devs.push_back(i);
    }
  }
  
  auto group = FindEqualGroup(outputs);
  std::string group_str;
  size_t i = 0;
  for (const auto &g : group) {
    if (i + 1 != group.size()) {
      group_str += " | ";
    } 
    group_str += "Group " + std::to_string(i) + " : ";
    group_str += VectorToString(g); 
    ++i;
  } 
  if (group.size() <= 1 && aadiff_devs.empty()) {
    group_str = "NoAADiff : " + group_str;
  } else {
    group_str = "HasAADiff : " + VectorToString(aadiff_devs) + " || " + group_str;
  }
  printf("%s\n", group_str.c_str());
}

int main() {
  TestMain();
  return 0;
}
