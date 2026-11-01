#pragma once
#include "kernel.cuh"
#include <pybind11/functional.h>

// csrc/apis/config.hpp

// csrc/jit_kernels/heuristics/runtime.hpp

// include/deep_jit/utils/lazy.hpp
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <dlfcn.h>

// include/deep_jit/utils/exception.hpp
#include <exception>
#include <format>
#include <string>
namespace deep_jit::exception {
class Exception final : public std::exception {
    std::string message;
public:
    Exception(const char* name, const char* file, int line, const std::string& error)
        : message(std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error) {}
    const char* what() const noexcept override { return message.c_str(); }
};
}
#ifndef DJ_STATIC_ASSERT
#define DJ_STATIC_ASSERT(cond, ...) static_assert(cond, "" __VA_ARGS__)
#endif

#ifndef DJ_HOST_ASSERT
#define DJ_HOST_ASSERT(cond, ...) \
do { \
    if (not (cond)) { \
        throw deep_jit::exception::Exception("Assertion", __FILE__, __LINE__, std::string(#cond) __VA_OPT__(+ ": " + std::format(__VA_ARGS__))); \
    } \
} while (0)
#endif

#ifndef DJ_PANIC
#define DJ_PANIC(...) \
do { \
    throw deep_jit::exception::Exception("Panic", __FILE__, __LINE__, std::format(__VA_ARGS__)); \
} while (0)
#endif

#define DJ_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace deep_jit {

template <typename T>
class LazyInit {
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;

public:
    explicit LazyInit(std::nullptr_t) {}

    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}

    T* operator->() {
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
        if (ptr == nullptr)
            ptr = factory();
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
        return ptr.get();
    }

    std::shared_ptr<T> get() {
        (void)operator->();
        return ptr;
    }
};

}  // namespace deep_jit

#define DJ_DECL_LAZY_DL_HANDLE(handle_func_name, lib)                                         \
    inline void* handle_func_name() {                                                         \
        static void* handle = [] {                                                            \
            ::dlerror();                                                                      \
            void* value = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);                                \
            if (value == nullptr) {                                                           \
                const char* error = ::dlerror();                                              \
                DJ_PANIC("failed to load {}: {}", lib, error == nullptr ? "unknown" : error); \
            }                                                                                 \
            return value;                                                                     \
        }();                                                                                  \
        return handle;                                                                        \
    }

#define DJ_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                                                   \
    template <typename... Args>                                                                                            \
    static auto lazy_##name(Args&&... args) {                                                                              \
        static const auto func = []() {                                                                                    \
            void* symbol = ::dlsym(handle_func_name(), #name);                                                             \
            if (symbol == nullptr) {                                                                                       \
                const char* error = ::dlerror();                                                                           \
                DJ_PANIC("failed to load {} from {}: {}", #name, #handle_func_name, error == nullptr ? "unknown" : error); \
            }                                                                                                              \
            return reinterpret_cast<decltype(&name)>(symbol);                                                              \
        }();                                                                                                               \
        return func(std::forward<Args>(args)...);                                                                          \
    }

// csrc/runtime/jit.hpp
#include <memory>
#include <string>

// include/deep_jit/backend/cuda/device.hpp
#include <cstdint>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace deep_jit::cuda {

inline void check_cuda_runtime(const cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        DJ_PANIC("{} failed with CUDA error {} ({}): {}",
                 expression,
                 static_cast<int>(error),
                 cudaGetErrorName(error),
                 cudaGetErrorString(error));
    }
}

#ifndef DJ_CUDA_RUNTIME_CHECK
#define DJ_CUDA_RUNTIME_CHECK(expr) ::deep_jit::cuda::check_cuda_runtime((expr), #expr)
#endif

class Device {
    cudaDeviceProp prop{};
    int64_t clock_rate = 0;
    bool initialized = false;

public:
    const cudaDeviceProp& get_prop() {
        if (not initialized) {
            // `cudaFree(nullptr)` is to ensure the current CUDA context exists before later driver API calls
            int device_index = 0;
            DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
            DJ_CUDA_RUNTIME_CHECK(cudaFree(nullptr));
            DJ_CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_index));
            initialized = true;
        }
        return prop;
    }

    int get_num_sms() { return get_prop().multiProcessorCount; }

    int get_num_l2_cache_bytes() { return get_prop().l2CacheSize; }

    int get_num_smem_bytes() { return static_cast<int>(get_prop().sharedMemPerBlockOptin); }

    int64_t get_clock_rate() {
        if (clock_rate == 0) {
            int device_index = 0;
            int rate = 0;
            DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
            DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, device_index));
            clock_rate = static_cast<int64_t>(rate) * 1000;
        }
        return clock_rate;
    }

    int get_arch_major() { return get_prop().major; }

    int get_arch_minor() { return get_prop().minor; }

    std::pair<int, int> get_arch_pair() { return {get_arch_major(), get_arch_minor()}; }

    std::string get_arch(const bool use_arch_family = true, const bool number_only = false) {
        const auto [major, minor] = get_arch_pair();
        const auto arch = std::to_string(major * 10 + minor);

        // E.g., sm_80, sm_90, sm_100
        if (number_only or major < 9)
            return arch;

        // E.g., sm_90a, sm_100f, sm_100a
        if (major > 9)
            return arch + (use_arch_family ? "f" : "a");
        return arch + "a";
    }
};

}  // namespace deep_jit::cuda

// include/deep_jit/backend/cuda/kernel.hpp
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

// include/deep_jit/backend/cuda/driver.hpp
#include <cuda.h>

DJ_STATIC_ASSERT(CUDA_VERSION >= 12040, "DeepJIT requires CUDA 12.4 or newer");

namespace deep_jit::cuda::driver {

DJ_DECL_LAZY_DL_HANDLE(get_cuda_handle, "libcuda.so.1");

DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuGetErrorName);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuGetErrorString);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuFuncSetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLaunchKernelEx);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuTensorMapEncodeTiled);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryLoadFromFile)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryUnload)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryGetKernelCount)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryEnumerateKernels)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuKernelGetFunction)

inline void check_cuda_driver(const CUresult error, const char* expression) {
    if (error == CUDA_SUCCESS)
        return;

    const char* error_name = "unknown";
    const char* error_description = "unknown";
    lazy_cuGetErrorName(error, &error_name);
    lazy_cuGetErrorString(error, &error_description);
    DJ_PANIC("{} failed with CUDA error {} ({}): {}",
             expression,
             static_cast<int>(error),
             error_name,
             error_description);
}

#ifndef DJ_CUDA_DRIVER_CHECK
#define DJ_CUDA_DRIVER_CHECK(expr) ::deep_jit::cuda::driver::check_cuda_driver((expr), #expr)
#endif

}  // namespace deep_jit::cuda::driver

// include/deep_jit/backend/cuda/options.hpp
#include <optional>
#include <cuda.h>
#include <cuda_runtime.h>

// include/deep_jit/utils/env.hpp
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <locale>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace deep_jit {

// Parse a whole string as an integer; malformed or partial values throw.
template <typename T>
inline T parse_integer(const char* raw, const std::string& name) {
    T value;
    const char* end = raw + std::strlen(raw);
    const auto [position, error] = std::from_chars(raw, end, value);
    if (error != std::errc{} or position != end)
        DJ_PANIC("invalid value for {}: {}", name, raw);
    return value;
}

template <typename T>
inline T get_env(const std::string& name, const T& default_value = T()) {
    const char* raw = std::getenv(name.c_str());
    if (raw == nullptr)
        return default_value;
    if constexpr (std::is_same_v<T, std::string>) {
        return std::string(raw);
    } else if constexpr (std::is_same_v<T, bool>) {
        // Accept true/false, yes/no, and integer (non-zero is true)
        // notations, case-insensitively; anything else is an error.
        std::string value(raw);
        for (char& c : value)
            c = std::tolower(c, std::locale::classic());
        if (value == "true" or value == "yes")
            return true;
        if (value == "false" or value == "no")
            return false;
        return parse_integer<long long>(raw, name) != 0;
    } else if constexpr (std::is_integral_v<T>) {
        return parse_integer<T>(raw, name);
    } else {
        DJ_STATIC_ASSERT((std::is_same_v<T, void>), "unsupported type for get_env");
        return T();
    }
}

class Env {
public:
    std::string prefix;

    explicit Env(std::string prefix) : prefix(std::move(prefix)) {
        DJ_HOST_ASSERT(not this->prefix.empty(), "environment variable prefix must not be empty");
        DJ_HOST_ASSERT(this->prefix != "DJ", "DJ is reserved for global environment variables");
    }

    template <typename T>
    std::optional<T> get(const std::string_view suffix) const {
        if (const auto value = get_with_prefix<T>(prefix, suffix))
            return value;
        return get_with_prefix<T>("DJ", suffix);
    }

    template <typename T>
    T get(const std::string_view suffix, const T& default_value) const {
        return get<T>(suffix).value_or(default_value);
    }

    template <typename T>
    static std::optional<T> get_with_prefix(const std::string_view prefix, const std::string_view suffix) {
        const auto name = std::string(prefix) + "_" + std::string(suffix);
        if (std::getenv(name.c_str()) == nullptr)
            return std::nullopt;
        return get_env<T>(name);
    }
};

}  // namespace deep_jit

namespace deep_jit::cuda {
// Per-launch CUDA options. Unset fields inherit from runtime defaults.
struct LaunchOptions {
    // Use current stream if unset
    std::optional<CUstream> stream;

    // Dynamic shared memory size
    std::optional<int> num_smem_bytes;

    // Grid
    std::optional<dim3> grid_dim;
    std::optional<dim3> block_dim;
    std::optional<dim3> cluster_dim;

    // Cooperative launch
    std::optional<bool> cooperative;

    // Dependent kernel launch
    std::optional<bool> enable_pdl;

    // Others
    std::optional<bool> nonportable_cluster_size_allowed;

    static LaunchOptions default_options(const Env&) {
        return {
            .stream = std::nullopt,
            .num_smem_bytes = 0,
            .grid_dim = std::nullopt,
            .block_dim = std::nullopt,
            .cluster_dim = dim3(1, 1, 1),
            .cooperative = false,
            .enable_pdl = false,
            .nonportable_cluster_size_allowed = false,
        };
    }

    [[nodiscard]] LaunchOptions override_with(const LaunchOptions& overrides) const {
        LaunchOptions result = *this;
        if (overrides.stream)
            result.stream = overrides.stream;
        if (overrides.num_smem_bytes)
            result.num_smem_bytes = overrides.num_smem_bytes;
        if (overrides.grid_dim)
            result.grid_dim = overrides.grid_dim;
        if (overrides.block_dim)
            result.block_dim = overrides.block_dim;
        if (overrides.cluster_dim)
            result.cluster_dim = overrides.cluster_dim;
        if (overrides.cooperative)
            result.cooperative = overrides.cooperative;
        if (overrides.enable_pdl)
            result.enable_pdl = overrides.enable_pdl;
        if (overrides.nonportable_cluster_size_allowed)
            result.nonportable_cluster_size_allowed = overrides.nonportable_cluster_size_allowed;
        return result;
    }
};

}  // namespace deep_jit::cuda

// include/deep_jit/utils/gil.hpp
#include <optional>

#include <pybind11/pybind11.h>

namespace deep_jit {

class GilScopedRelease {
public:
    std::optional<pybind11::gil_scoped_release> release;

    GilScopedRelease() {
        if (Py_IsInitialized() and PyGILState_Check())
            release.emplace();
    }

    GilScopedRelease(const GilScopedRelease&) = delete;
    GilScopedRelease& operator=(const GilScopedRelease&) = delete;
    GilScopedRelease(GilScopedRelease&&) = delete;
    GilScopedRelease& operator=(GilScopedRelease&&) = delete;
};

}  // namespace deep_jit

// include/deep_jit/utils/no_ref_ptr.hpp
namespace deep_jit {

// The pointer itself is the kernel argument storage; do not take its address.
struct NoRefPtr {
    void* ptr = nullptr;
};

}  // namespace deep_jit

namespace deep_jit::cuda {

template <typename T>
inline void* kernel_arg_pointer(const T& value) {
    if constexpr (std::is_base_of_v<NoRefPtr, std::decay_t<T>>) {
        return value.ptr;
    } else {
        return const_cast<void*>(static_cast<const void*>(&value));
    }
}

// Immutable CUDA kernel handles with shared ownership. Driver resources are
// unloaded when the last shared owner is destroyed.
class Kernel {
public:
    CUlibrary library_handle{};
    CUfunction kernel_handle{};

    Kernel(const CUlibrary& library_handle, const CUfunction& kernel_handle)
        : library_handle(library_handle), kernel_handle(kernel_handle) {}

    ~Kernel() { unload(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Check existence
        const auto cubin_path = dir / "kernel.cubin";
        if (not std::filesystem::is_regular_file(cubin_path))
            DJ_PANIC("missing CUDA CUBIN: {}", cubin_path.string());

        // Record start time
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
        if (debug)
            std::fputs(std::format("Loading CUBIN: {}\n", cubin_path.string()).c_str(), stdout);
        const auto start_time = std::chrono::steady_clock::now();

        // Load kernel
        CUlibrary library_handle{};
        CUfunction kernel_handle{};
        DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryLoadFromFile(
            &library_handle, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        try {
            unsigned int num_kernels = 0;
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryGetKernelCount(&num_kernels, library_handle));
            if (num_kernels != 1)
                DJ_PANIC("expected exactly one kernel in {}", cubin_path.string());

            CUkernel library_kernel_handle{};
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryEnumerateKernels(&library_kernel_handle, 1, library_handle));
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuKernelGetFunction(&kernel_handle, library_kernel_handle));
        } catch (...) {
            driver::lazy_cuLibraryUnload(library_handle);
            throw;
        }

        // Print and return
        if (print_load_time) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start_time;
            std::fputs(std::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(), stdout);
        }
        return std::make_shared<Kernel>(library_handle, kernel_handle);
    }

    template <typename... Args>
    void launch(const LaunchOptions& launch_options, const Args&... args) const {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Checks
        DJ_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
        DJ_HOST_ASSERT(launch_options.num_smem_bytes.has_value(), "CUDA dynamic shared-memory size must be specified");
        DJ_HOST_ASSERT(launch_options.grid_dim.has_value(), "CUDA grid dimension must be specified");
        DJ_HOST_ASSERT(launch_options.block_dim.has_value(), "CUDA block dimension must be specified");
        DJ_HOST_ASSERT(launch_options.cluster_dim.has_value(), "CUDA cluster dimension must be specified");
        DJ_HOST_ASSERT(launch_options.cooperative.has_value(), "CUDA cooperative option must be specified");
        DJ_HOST_ASSERT(launch_options.enable_pdl.has_value(), "CUDA PDL option must be specified");
        DJ_HOST_ASSERT(launch_options.nonportable_cluster_size_allowed.has_value(),
                       "CUDA non-portable cluster option must be specified");
        DJ_HOST_ASSERT(*launch_options.num_smem_bytes >= 0, "CUDA dynamic shared-memory size must not be negative");
        DJ_HOST_ASSERT(launch_options.grid_dim->x > 0 and launch_options.grid_dim->y > 0 and launch_options.grid_dim->z > 0,
                       "CUDA grid dimensions must be positive");
        DJ_HOST_ASSERT(launch_options.block_dim->x > 0 and launch_options.block_dim->y > 0 and launch_options.block_dim->z > 0,
                       "CUDA block dimensions must be positive");
        DJ_HOST_ASSERT(launch_options.cluster_dim->x > 0, "CUDA cluster dimension must be positive");
        DJ_HOST_ASSERT(launch_options.cluster_dim->y == 1 and launch_options.cluster_dim->z == 1,
                       "only one-dimensional CUDA clusters are supported");

        // Set **maximum** dynamic shared memory
        if (*launch_options.num_smem_bytes > 0) {
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
                kernel_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                *launch_options.num_smem_bytes));
        }

        // Set non-portable cluster size
        // NOTES: you can not write `if (nonportable_cluster_size_allowed) ...`,
        // this may lead to a bug, e.g., launch with `= true` then `= false`,
        // in such case, the second launch will still use the first attribute
        // But, we still write this, because the launch overhead is important
        // And for `= true`'s side-effects for `= false` can be ignored
        if (*launch_options.nonportable_cluster_size_allowed) {
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
                kernel_handle, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
        }

        // NOTES: please enlarge the array size if you want more attributes
        std::array<CUlaunchAttribute, 3> attributes{};
        unsigned int num_attributes = 0;
        if (*launch_options.cooperative) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
            attribute.value.cooperative = 1;
        }
        if (launch_options.cluster_dim->x > 1) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            attribute.value.clusterDim.x = launch_options.cluster_dim->x;
            attribute.value.clusterDim.y = 1;
            attribute.value.clusterDim.z = 1;
        }
        if (*launch_options.enable_pdl) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attribute.value.programmaticStreamSerializationAllowed = 1;
        }

        // Set launch config
        void* kernel_arg_ptrs[sizeof...(Args) + 1] = {kernel_arg_pointer(args)..., nullptr};
        CUlaunchConfig config{};
        config.gridDimX = launch_options.grid_dim->x;
        config.gridDimY = launch_options.grid_dim->y;
        config.gridDimZ = launch_options.grid_dim->z;
        config.blockDimX = launch_options.block_dim->x;
        config.blockDimY = launch_options.block_dim->y;
        config.blockDimZ = launch_options.block_dim->z;
        config.sharedMemBytes = *launch_options.num_smem_bytes;
        config.hStream = launch_options.stream
            ? *launch_options.stream
            : at::cuda::getCurrentCUDAStream().stream();
        config.attrs = num_attributes == 0 ? nullptr : attributes.data();
        config.numAttrs = num_attributes;
        DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLaunchKernelEx(
            &config, kernel_handle, sizeof...(Args) == 0 ? nullptr : kernel_arg_ptrs, nullptr));
    }

    void unload() noexcept {
        if (library_handle == nullptr)
            return;

        try {
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryUnload(library_handle));
        } catch (...) {
        }
        library_handle = nullptr;
        kernel_handle = nullptr;
    }
};

}  // namespace deep_jit::cuda

namespace deep_gemm {

class Launcher {
public:
    using Kernel = deep_jit::cuda::Kernel;
    using LaunchOptions = deep_jit::cuda::LaunchOptions;

    deep_jit::Env env;
    deep_jit::cuda::Device device;
    LaunchOptions default_launch_options;

    explicit Launcher(const std::string& env_prefix)
        : env(env_prefix),
          default_launch_options(LaunchOptions::default_options(env)) {}

    template <typename... Args>
    void launch(const std::shared_ptr<Kernel>& kernel, const LaunchOptions& override_options, const Args&... args) {
        DJ_HOST_ASSERT(kernel != nullptr, "kernel must not be null");
        kernel->launch(default_launch_options.override_with(override_options), args...);
    }

};

inline deep_jit::LazyInit<Launcher> jit(nullptr);

// Kernel acquisition belongs to precompiled::Store; init_jit only
// restores the original environment, device and launch defaults.
inline void init_jit() {
    jit = deep_jit::LazyInit<Launcher>([] {
        return std::make_shared<Launcher>("DG");
    });
}

}  // namespace deep_gemm

// csrc/utils/exception.hpp
#include <exception>
#include <sstream>
#include <string>

// csrc/utils/compatibility.hpp
#include <torch/version.h>
#include <cuda.h>
#include <cuda_runtime.h>

DG_STATIC_ASSERT(TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3),
                 "DeepGEMM requires PyTorch 2.3 or newer");
DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer");

namespace deep_gemm {

class DGException final : public std::exception {
    std::string message = {};

public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw DGException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_HOST_UNREACHABLE
#define DG_HOST_UNREACHABLE(reason) (throw DGException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef DG_CUDA_RUNTIME_CHECK
#define DG_CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != cudaSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw DGException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

} // namespace deep_gemm

namespace deep_gemm {

class HeuristicsRuntime {
public:
    static constexpr int kLegacyMKAlignmentForContiguousLayout = 128;

    bool ignore_compile_dims = false;
    bool deterministic_algorithms = false;
    int block_m_multiple_of = 1;
    int block_n_multiple_of = 1;
    int mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;

    void use_deterministic_algorithms(const bool enabled) {
        deterministic_algorithms = enabled;
    }

    bool get_deterministic_algorithms() const {
        return deterministic_algorithms;
    }

    void set_ignore_compile_dims(const bool& new_value) {
        ignore_compile_dims = new_value;
    }

    bool get_ignore_compile_dims() const {
        return ignore_compile_dims;
    }

    void set_block_size_multiple_of(const int& new_block_m_multiple_of, const int& new_block_n_multiple_of) {
        block_m_multiple_of = new_block_m_multiple_of;
        block_n_multiple_of = new_block_n_multiple_of;
    }

    int get_block_m_multiple_of() const {
        return block_m_multiple_of;
    }

    int get_block_n_multiple_of() const {
        return block_n_multiple_of;
    }

    void set_mk_alignment_for_contiguous_layout(const int& new_value) {
        mk_alignment_for_contiguous_layout = new_value;
    }

    int get_mk_alignment_for_contiguous_layout() const {
        return mk_alignment_for_contiguous_layout;
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
        if (jit->device.get_arch_major() != 10)
            return kLegacyMKAlignmentForContiguousLayout;

        // The newly supported UMMA_N=256 allows a fixed alignment of 256, which is friendlier to MoE cast, M-grouped, and K-grouped operators.
        // NOTES: `expected_m` is ignored, so small values may incur performance loss; use Mega MoE directly for such workloads.
        return 256;
    }
};

inline auto heuristics_runtime = deep_jit::LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm

// csrc/runtime/runtime.hpp
#include <cstddef>
#include <memory>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>

namespace deep_gemm {

class Runtime {
public:
    int num_sms = 0, tc_util = 0;

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 < new_num_sms and new_num_sms <= jit->device.get_num_sms());
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = jit->device.get_num_sms();
        return num_sms;
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

};

inline auto runtime = deep_jit::LazyInit<Runtime>([](){ return std::make_shared<Runtime>(); });

}  // namespace deep_gemm

// csrc/runtime/precompiled.hpp
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <functional>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <filesystem>

// csrc/runtime/spec_config.hpp
#include <cstdint>
#include <cstddef>
#include <format>
#include <string>
#include <vector>

namespace deep_gemm::precompiled {

/// One token per template parameter of the impl function, in declaration order.
using SpecConfig = std::vector<std::string>;

/// Build a configuration from the same expressions the delivered call sites formatted into
/// a stub.  Each argument is spelled with `std::format("{}", x)`, i.e. exactly as it
/// appeared between the `<` and the `>` of the stub's template instantiation, so a token
/// here and the recorded `args` entry for that specialization are the same string.
template <typename... Args>
[[nodiscard]] inline SpecConfig spec_config(Args&&... args) {
    return SpecConfig{std::format("{}", std::forward<Args>(args))...};
}

/// Key separator.  `\x1f` (unit separator) cannot appear in a C++ token spelling, so two
/// distinct configurations can never produce the same key by concatenation.
inline constexpr char kConfigSeparator = '\x1f';

[[nodiscard]] inline std::string config_key(const std::string& launch_tag,
                                            const SpecConfig& config) {
    auto key = launch_tag;
    for (const auto& token: config) {
        key += kConfigSeparator;
        key += token;
    }
    return key;
}

/// Same key from a generated table entry's C array, so the table and the call site are
/// keyed by one function rather than by two implementations that have to agree.
[[nodiscard]] inline std::string config_key(const std::string& launch_tag,
                                            const char* const* args, uint32_t n_args) {
    SpecConfig config;
    config.reserve(n_args);
    for (uint32_t i = 0; i < n_args; ++i)
        config.emplace_back(args[i]);
    return config_key(launch_tag, config);
}

/// Readable form for the miss message: the requested configuration, value by value.
[[nodiscard]] inline std::string describe(const SpecConfig& config) {
    std::string text;
    for (std::size_t i = 0; i < config.size(); ++i) {
        if (i)
            text += ", ";
        text += config[i];
    }
    return text;
}

}  // namespace deep_gemm::precompiled

namespace deep_jit::cuda::driver {
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuCtxGetCurrent);
}  // namespace deep_jit::cuda::driver

namespace deep_gemm::precompiled {

using Kernel = deep_jit::cuda::Kernel;

// csrc/runtime/spec_table.hpp
namespace embedded {
struct Specialization {
    const char* dir_name;
    const char* launch_tag;
    uint32_t n_args;
    const char* const* args;
    const void* kernel_symbol;
};
static const char* const config_0[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "7168", "2048", "128", "128", "64", "32", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguousWithPsumLayout", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_1[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "3072", "4096", "128", "128", "64", "16", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguousWithPsumLayout", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_2[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "4096", "1536", "128", "128", "64", "32", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguous", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_3[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "7168", "2048", "128", "128", "64", "16", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguousWithPsumLayout", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_4[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "4096", "7168", "128", "128", "64", "16", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguous", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_5[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "7168", "2048", "128", "128", "64", "16", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguous", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_6[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "1536", "2048", "128", "128", "64", "8", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguous", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_7[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "4096", "7168", "128", "128", "64", "8", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguous", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const char* const config_8[] = {"dg::UMMA::Major::K", "dg::UMMA::Major::K", "0", "4096", "1536", "128", "128", "64", "64", "128", "128", "128", "9", "128", "128", "2", "true", "148", "128", "1", "true", "GemmType::MGroupedContiguousWithPsumLayout", "false", "dg::bfloat16_t", "epilogue::transform::EpilogueIdentity", "100"};
static const Specialization kSpecializations[] = {
    {"config_0", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_0, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 7168, 2048, 128, 128, 64, 32, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguousWithPsumLayout, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_1", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_1, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 3072, 4096, 128, 128, 64, 16, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguousWithPsumLayout, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_2", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_2, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 4096, 1536, 128, 128, 64, 32, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguous, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_3", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_3, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 7168, 2048, 128, 128, 64, 16, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguousWithPsumLayout, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_4", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_4, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 4096, 7168, 128, 128, 64, 16, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguous, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_5", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_5, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 7168, 2048, 128, 128, 64, 16, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguous, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_6", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_6, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 1536, 2048, 128, 128, 64, 8, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguous, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_7", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_7, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 4096, 7168, 128, 128, 64, 8, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguous, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
    {"config_8", "sm100_bf16_m_grouped_gemm_contiguous", 26, config_8, reinterpret_cast<const void*>(&deep_gemm::sm100_bf16_gemm_impl<dg::UMMA::Major::K, dg::UMMA::Major::K, 0, 4096, 1536, 128, 128, 64, 64, 128, 128, 128, 9, 128, 128, 2, true, 148, 128, 1, true, GemmType::MGroupedContiguousWithPsumLayout, false, dg::bfloat16_t, epilogue::transform::EpilogueIdentity, 100>)},
};
}

class Store {
public:
    [[nodiscard]] uint64_t generation() const {
        return generation_.load(std::memory_order_acquire);
    }
    using Resolver = std::function<std::string(const std::string&, const SpecConfig&)>;
    void set_resolver(Resolver resolver) {
        std::unique_lock lock(mutex_, std::defer_lock);
        { deep_jit::GilScopedRelease release; lock.lock(); }
        // One process-lifetime callback per extension. Deliberately avoid destruction
        // after Python finalization; replacing it happens with the binding's GIL held.
        if (resolver_ == nullptr) resolver_ = new Resolver(std::move(resolver));
        else *resolver_ = std::move(resolver);
        generation_.fetch_add(1, std::memory_order_release);
    }

    void init(const std::filesystem::path& library_root) {
        std::scoped_lock lock(mutex_);
        // STAGE C: there is no directory to scan.  `library_root` is still resolved and
        // kept so every DJ_PANIC below names the same path the pre-compiled store used to
        // live at; nothing is stat'ed, opened or read.
        root_ = std::filesystem::absolute(library_root) / "precompiled";
        generation_.fetch_add(1, std::memory_order_release);
        by_config_.clear();
        loaded_.clear();
        tags_.clear();
        for (const auto& entry: embedded::kSpecializations) {
            const auto key = config_key(entry.launch_tag, entry.args, entry.n_args);
            DJ_HOST_ASSERT(by_config_.find(key) == by_config_.end(),
                           "two specializations share one explicit configuration: {} and {}",
                           by_config_.find(key)->second->dir_name, entry.dir_name);
            by_config_.emplace(key, &entry);
            tags_.push_back(entry.dir_name);
        }
        entries_ = by_config_.size();
        DJ_HOST_ASSERT(entries_ > 0, "embedded specialization store for {} is empty", root_.string());
    }

    [[nodiscard]] std::shared_ptr<Kernel> load(const std::string& name,
                                               const SpecConfig& config) const {
        std::unique_lock lock(mutex_, std::defer_lock);
        { deep_jit::GilScopedRelease release; lock.lock(); }
        DJ_HOST_ASSERT(not config.empty(),
                       "launch tag '{}' was given an empty configuration", name);
        const auto found = by_config_.find(config_key(name, config));
        if (found == by_config_.end()) {
            const auto key = config_key(name, config);
            if (const auto hit = loaded_.find(key); hit != loaded_.end()) return hit->second;
            DJ_HOST_ASSERT(resolver_ != nullptr, "missing dynamic configuration resolver");
            const auto directory = (*resolver_)(name, config);
            auto kernel = Kernel::load(directory, deep_jit::Env("DG"));
            loaded_.emplace(key, kernel);
            return kernel;
        }
        const auto* entry = found->second;
        const auto key = std::string(entry->dir_name);
        const auto cached = loaded_.find(key);
        if (cached != loaded_.end())
            return cached->second;
        CUfunction kernel_handle{};
        const auto status = cudaGetFuncBySymbol(&kernel_handle, entry->kernel_symbol);
        DJ_HOST_ASSERT(status == cudaSuccess,
                       "cudaGetFuncBySymbol failed for embedded specialization {} (cudaError {})",
                       key, static_cast<int>(status));
        auto kernel = std::make_shared<Kernel>(CUlibrary{}, kernel_handle);
        loaded_.emplace(key, kernel);
        return kernel;
    }

    [[nodiscard]] size_t size() const {
        std::scoped_lock lock(mutex_);
        return entries_;
    }

    [[nodiscard]] std::vector<std::string> tags() const {
        std::scoped_lock lock(mutex_);
        auto sorted = tags_;
        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }

    [[nodiscard]] std::string root() const {
        std::scoped_lock lock(mutex_);
        return root_.string();
    }

private:
    // `mutable` because `load` memoizes loaded kernels and is called from const contexts.
    std::atomic<uint64_t> generation_{0};
    Resolver* resolver_ = nullptr;
    mutable std::mutex mutex_;
    std::filesystem::path root_;
    std::unordered_map<std::string, const embedded::Specialization*> by_config_;
    mutable std::unordered_map<std::string, std::shared_ptr<Kernel>> loaded_;
    std::vector<std::string> tags_;
    size_t entries_ = 0;
};

inline Store& store() {
    static Store instance;
    return instance;
}

// Called from `config.hpp`'s `init`, next to `init_jit`, with the same library root.
inline void init_precompiled(const std::string& library_root_path) {
    store().init(library_root_path);
}

// The substitution itself: `jit->compile(tag, formatted_stub)` becomes
// `precompiled::load(tag, spec_config(<the same argument expressions>))`.
inline std::shared_ptr<Kernel> load(const std::string& name, const SpecConfig& config) {
    return store().load(name, config);
}

// Only the immediately preceding *configuration* is memoized per host thread
// and argument signature. Compare every original parameter by value, plus the
// launch tag, CUDA context and store generation. Tensor pointers/descriptors,
// launch options and data still flow through the unchanged launch on every call.
template <typename T>
inline auto config_value_view(const T& value) {
    if constexpr (std::is_convertible_v<const T&, std::string_view>)
        return std::string_view(value);
    else
        return value;
}

template <typename T>
inline auto own_config_value(const T& value) {
    if constexpr (std::is_convertible_v<const T&, std::string_view>)
        return std::string(value);
    else
        return value;
}

template <typename... Args>
[[nodiscard]] inline std::shared_ptr<Kernel> load_cached(const std::string& name,
                                                       const Args&... args) {
    using Values = decltype(std::make_tuple(own_config_value(args)...));
    struct LastConfiguration {
        std::string name;
        std::optional<Values> values;
        CUcontext context{};
        uint64_t generation{};
        std::shared_ptr<Kernel> kernel;
    };
    thread_local LastConfiguration previous;
    CUcontext context{};
    const auto status = deep_jit::cuda::driver::lazy_cuCtxGetCurrent(&context);
    DJ_HOST_ASSERT(status == CUDA_SUCCESS, "cannot identify configuration CUDA context");
    const auto generation = store().generation();
    const auto values = std::make_tuple(config_value_view(args)...);
    if (previous.kernel && previous.generation == generation &&
        previous.context == context && previous.name == name &&
        previous.values && *previous.values == values)
        return previous.kernel;

    // A miss uses the original complete configuration and resolver. Own string
    // contents only after a successful load; no views outlive this call.
    auto kernel = load(name, spec_config(args...));
    previous.name = name;
    previous.values = std::make_tuple(own_config_value(args)...);
    previous.context = context;
    previous.generation = generation;
    previous.kernel = kernel;
    return kernel;
}

}  // namespace deep_gemm::precompiled

// Every rewritten launch site spells the substitution as
// `precompiled::load(tag, spec_config(<the same argument expressions>))`: `load` is
// qualified, `spec_config` is not.  The call sites all live in `namespace deep_gemm`
// (csrc/jit_kernels/impls/*.hpp, csrc/apis/*.hpp), so the name has to be visible there --
// one using-declaration, next to the header that defines it, instead of editing 29 call
// sites across 14 rows.  This adds no capability: it is name lookup for the same function.
namespace deep_gemm {
using precompiled::SpecConfig;
using precompiled::spec_config;
}  // namespace deep_gemm

namespace deep_gemm::config {

static void register_apis(pybind11::module_& m) {
    m.def("init", [&](const std::string& library_root_path) {
        init_jit();
        precompiled::init_precompiled(library_root_path);
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return runtime->get_tc_util();
    });
    m.def("set_pdl", [](const bool& new_enable_pdl) {
        jit->default_launch_options.enable_pdl = new_enable_pdl;
    });
    m.def("get_pdl", []() {
        return *jit->default_launch_options.enable_pdl;
    });
    m.def("use_deterministic_algorithms", [&](const bool enabled) {
        heuristics_runtime->use_deterministic_algorithms(enabled);
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
}

} // namespace deep_gemm::config
