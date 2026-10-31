#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <climits>

namespace {

// Each element maps a group of state values z to scale * z + values
// The grouped values share one scale to avoid repeating the same work
template <typename T, int Width>
struct Affine {
    T scale;
    T values[Width];
};

template <typename T, int Width>
struct Compose {
    __host__ __device__ Affine<T, Width> operator()(Affine<T, Width> left, Affine<T, Width> right) const {
        // Apply the earlier transform on the left, then the later one on the right
        // Composition is associative, but swapping the arguments changes the result
        left.scale *= right.scale;
        #pragma unroll
        for (int j = 0; j < Width; ++j)
            left.values[j] = right.scale * left.values[j] + right.values[j];
        return left;
    }
};

struct SegmentKey {
    int length;

    __host__ __device__ int operator()(int i) const {
        // Every group has N consecutive events; a new key starts an independent scan
        return i / length;
    }
};

template <typename T, int Width>
struct ReadAffine {
    const T* input;
    int length, components, groups;

    __device__ Affine<T, Width> operator()(int i) const {
        // Read one event from a kernel/component group in the original [K, N, M+1] tensor
        const int segment = i / length;
        const int event = i % length;
        const int64_t kernel = segment / groups;
        const int component = segment % groups * Width;
        const int64_t offset = (kernel * length + event) * (components + 1);
        Affine<T, Width> result;
        result.scale = input[offset];
        // Pad the last group with zeros when M is not a multiple of Width
        #pragma unroll
        for (int j = 0; j < Width; ++j)
            result.values[j] = component + j < components ? input[offset + component + j + 1] : T{0};
        return result;
    }
};

template <typename T, int Width>
__global__ void unpack(const Affine<T, Width>* scanned, T* output,
                       int64_t count, int length, int components, int groups) {
    // Restore the grouped scan output to the original [K, N, M+1] layout
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
         i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const int component = i % components;
        const int event = (i / components) % length;
        const int64_t kernel = i / (components * length);
        const auto& state = scanned[(kernel * groups + component / Width) * length + event];
        const int64_t offset = (kernel * length + event) * (components + 1);
        output[offset + component + 1] = state.values[component % Width];
        // All groups have the same scale prefix, so write it only once per event
        if (component == 0) {
            output[offset] = state.scale;
        }
    }
}

template <typename T, int Width>
void scan(const at::Tensor& input, at::Tensor& output, cudaStream_t stream) {
    const int length = input.size(1), components = input.size(2) - 1;
    const int groups = (components + Width - 1) / Width;
    const int count = input.size(0) * int64_t(length) * groups;
    // Generate indices and segment keys as CUB reads them, without allocating arrays
    auto indices = thrust::make_counting_iterator(0);
    auto keys = thrust::make_transform_iterator(indices, SegmentKey{length});
    auto launch = [&](auto values, auto scanned) {
        size_t bytes = 0;
        auto run = [&](void* scratch) {
            // A single segment needs no keys or reset checks
            if (input.size(0) * groups == 1) {
                return cub::DeviceScan::InclusiveScan(
                    scratch, bytes, values, scanned, Compose<T, Width>{}, count, stream);
            }
            // A key change resets the state between kernel/component groups
            return cub::DeviceScan::InclusiveScanByKey(
                scratch, bytes, keys, values, scanned, Compose<T, Width>{}, count, cub::Equality{}, stream);
        };
        // First ask CUB for its workspace size, then allocate it through PyTorch
        C10_CUDA_CHECK(run(nullptr));
        auto scratch = at::empty({static_cast<int64_t>(bytes)}, input.options().dtype(at::kByte));
        C10_CUDA_CHECK(run(scratch.data_ptr()));
    };

    // With M=1 or M=4, the tensor already has the layout CUB needs
    // Read and write it directly to avoid a temporary buffer and an unpack kernel
    if (components == Width) {
        launch(reinterpret_cast<const Affine<T, Width>*>(input.data_ptr<T>()),
               reinterpret_cast<Affine<T, Width>*>(output.data_ptr<T>()));
        return;
    }
    // Assemble groups on demand instead of copying the input into a packed tensor
    auto values = thrust::make_transform_iterator(
        indices, ReadAffine<T, Width>{input.data_ptr<T>(), length, components, groups});
    // CUB writes complete groups here before unpack restores the tensor layout
    auto packed = at::empty({count, Width + 1}, input.options());
    auto* scanned = reinterpret_cast<Affine<T, Width>*>(packed.data_ptr<T>());
    launch(values, scanned);

    const int64_t output_count = input.size(0) * int64_t(length) * components;
    const int blocks = static_cast<int>(std::min<int64_t>((output_count + 255) / 256, 65535));
    unpack<<<blocks, 256, 0, stream>>>(scanned, output.data_ptr<T>(), output_count, length, components, groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor affine_scan(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "CUB scan requires a CUDA tensor.");
    TORCH_CHECK(input.is_contiguous(), "CUB scan requires contiguous input.");
    TORCH_CHECK(input.dim() == 3 && input.size(2) >= 2, "Expected [K, N, M+1] affine states.");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble,
                "CUB scan supports float32 and float64.");
    // Select the tensor's device even if another GPU is currently active
    const c10::cuda::CUDAGuard guard(input.device());
    // Empty and single-event sequences need no scan, but still return a new tensor
    if (input.numel() == 0 || input.size(1) == 1) {
        return input.clone();
    }
    // CUB scan indices use 32 bits, while tensor address calculations use 64 bits
    TORCH_CHECK(input.size(0) * input.size(1) * (input.size(2) - 1) <= INT_MAX,
                "CUB scan has too many elements; reduce the event batch size.");
    auto output = at::empty_like(input);
    // Follow PyTorch's current stream without synchronizing the CPU
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hawkes_affine_scan", [&] {
        // Groups of four share scale work; smaller states avoid unnecessary padding
        if (input.size(2) - 1 >= 3) scan<scalar_t, 4>(input, output, stream);
        else scan<scalar_t, 1>(input, output, stream);
    });
    return output;
}

}  // namespace

// Register the operation so Python can call it through torch.ops.hawkes_cub.scan
TORCH_LIBRARY(hawkes_cub, m) {
    m.def("scan(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(hawkes_cub, CUDA, m) {
    m.impl("scan", affine_scan);
}
