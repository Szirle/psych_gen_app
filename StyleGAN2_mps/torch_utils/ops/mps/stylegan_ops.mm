#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include "stylegan_ops.h"

namespace {
id<MTLLibrary> library = nil;
std::map<std::string, id<MTLComputePipelineState>> pipelines;
std::mutex pipeline_mutex;

void initialize(const std::string& source) {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    if (library) return;
    @autoreleasepool {
        auto device = at::mps::getCurrentMPSStream()->device();
        auto options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion2_3;
        options.fastMathEnabled = YES;
        NSError* error = nil;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                      options:options error:&error];
        [options release];
        TORCH_CHECK(library, "StyleGAN Metal compilation failed: ", [[error description] UTF8String]);
    }
}

id<MTLComputePipelineState> pipeline(const std::string& name, int up=0, int down=0, bool cl=false) {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    std::string key = name + ':' + std::to_string(up) + ':' + std::to_string(down) + ':' + std::to_string(cl);
    auto found = pipelines.find(key);
    if (found != pipelines.end()) return found->second;
    TORCH_CHECK(library, "StyleGAN Metal library is not initialized");
    @autoreleasepool {
        NSError* error = nil;
        id<MTLFunction> function;
        NSString* label = [NSString stringWithUTF8String:name.c_str()];
        if (up) {
            auto constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&up type:MTLDataTypeInt atIndex:0];
            [constants setConstantValue:&down type:MTLDataTypeInt atIndex:1];
            [constants setConstantValue:&cl type:MTLDataTypeBool atIndex:2];
            function = [library newFunctionWithName:label constantValues:constants error:&error];
            [constants release];
        } else function = [library newFunctionWithName:label];
        TORCH_CHECK(function, "Cannot create Metal function ", name, ": ", [[error description] UTF8String]);
        auto pso = [at::mps::getCurrentMPSStream()->device() newComputePipelineStateWithFunction:function error:&error];
        [function release];
        TORCH_CHECK(pso, "Cannot create Metal pipeline: ", [[error description] UTF8String]);
        pipelines.emplace(key, pso);
        return pso;
    }
}

void bind(id<MTLComputeCommandEncoder> encoder, const torch::Tensor& t, int index) {
    auto buffer = __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
    [encoder setBuffer:buffer offset:t.storage_offset()*t.element_size() atIndex:index];
}

void dispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pso, int count) {
    NSUInteger width = pso.threadExecutionWidth;
    NSUInteger group = std::min(NSUInteger(256), pso.maxTotalThreadsPerThreadgroup);
    group = std::max(width, group / width * width);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

void check_tensor(const torch::Tensor& t) {
    TORCH_CHECK(t.is_mps(), "input must reside on MPS");
    TORCH_CHECK(t.scalar_type()==at::kFloat || t.scalar_type()==at::kHalf, "Metal ops support float32 and float16");
    TORCH_CHECK(t.numel()<=INT_MAX, "tensor is too large");
    int64_t footprint=0;
    for (int d=0; d<t.dim(); ++d) {
        TORCH_CHECK(t.stride(d)>=0 && t.stride(d)<=INT_MAX && t.size(d)<=INT_MAX, "invalid or oversized stride/size");
        footprint += std::max(int64_t(0), t.size(d)-1)*t.stride(d);
        TORCH_CHECK(footprint<=INT_MAX, "tensor memory footprint is too large");
    }
}

bool same_layout(const torch::Tensor& a, const torch::Tensor& b) {
    if (a.sizes()!=b.sizes()) return false;
    for (int d=0; d<a.dim(); ++d)
        if (a.size(d)>1 && a.stride(d)!=b.stride(d)) return false;
    return true;
}

torch::Tensor bias_act(torch::Tensor x, torch::Tensor b, torch::Tensor xr, torch::Tensor yr,
                       torch::Tensor dy, int grad, int dim, int act, float alpha, float gain, float clamp) {
    check_tensor(x);
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");
    TORCH_CHECK(grad>=0 && grad<=2 && act>=1 && act<=9, "invalid activation or derivative order");
    TORCH_CHECK(b.dim()==1 && b.is_contiguous(), "bias must be a contiguous vector");
    if (b.numel()) {
        TORCH_CHECK(b.device()==x.device() && b.scalar_type()==x.scalar_type(), "bias dtype/device must match x");
        TORCH_CHECK(dim>=0 && dim<x.dim() && b.numel()==x.size(dim), "bias dimension/size mismatch");
    }
    for (const auto& ref : {xr, yr, dy}) {
        TORCH_CHECK(!ref.numel() || (ref.device()==x.device() && ref.scalar_type()==x.scalar_type() && same_layout(ref,x)),
                    "reference shape, dtype, device, and layout must match x");
    }
    auto y = torch::empty_like(x);
    TORCH_CHECK(same_layout(x,y), "output must preserve dense layout");
    if (!x.numel()) return y;
    BiasParams p{int(x.numel()), int(b.numel()), b.numel()?int(x.stride(dim)):1, grad, act,
                 int(b.numel()!=0), int(xr.numel()!=0), int(yr.numel()!=0), int(dy.numel()!=0), alpha, gain, clamp};
    auto pso = pipeline(x.scalar_type()==at::kHalf ? "bias_f16" : "bias_f32");
    auto stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            auto encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            bind(encoder,x,0); bind(encoder,b.numel()?b:x,1);
            bind(encoder,xr.numel()?xr:x,2); bind(encoder,yr.numel()?yr:x,3);
            bind(encoder,dy.numel()?dy:x,4); bind(encoder,y,5);
            [encoder setBytes:&p length:sizeof(p) atIndex:6];
            dispatch(encoder,pso,p.size);
        }
    });
    return y;
}

torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int ux, int uy, int dx, int dy,
                       int px0, int px1, int py0, int py1, bool flip, float gain) {
    check_tensor(x); check_tensor(f);
    TORCH_CHECK(x.dim()==4 && f.dim()==2 && x.numel()>0 && f.numel()>0, "expected nonempty rank-4 input and rank-2 filter");
    TORCH_CHECK(f.scalar_type()==at::kFloat && f.device()==x.device(), "filter must be float32 on the input device");
    TORCH_CHECK(!f.requires_grad(), "filter gradients are not supported");
    TORCH_CHECK(ux>=1 && uy>=1 && dx>=1 && dy>=1, "scaling factors must be positive");
    int64_t extent_x=x.size(3)*ux+int64_t(px0)+px1;
    int64_t extent_y=x.size(2)*uy+int64_t(py0)+py1;
    TORCH_CHECK(extent_x>=f.size(1) && extent_y>=f.size(0), "output must be at least 1x1");
    int64_t ow=(extent_x-f.size(1))/dx+1, oh=(extent_y-f.size(0))/dy+1;
    TORCH_CHECK(ow<=INT_MAX && oh<=INT_MAX && x.size(0)*x.size(1)<=INT_MAX/ow/oh, "output is too large");
    // Bound signed shader coordinate intermediates as well as storage indices.
    TORCH_CHECK((ow-1)*dx+ux-1-int64_t(px0)+f.size(1)<=INT_MAX &&
                (oh-1)*dy+uy-1-int64_t(py0)+f.size(0)<=INT_MAX &&
                int64_t(ux)-1-px0>=INT_MIN && int64_t(uy)-1-py0>=INT_MIN &&
                (x.size(3)+1)*ux<=INT_MAX && (x.size(2)+1)*uy<=INT_MAX,
                "resampling coordinates exceed signed 32-bit range");
    auto y=torch::empty({x.size(0),x.size(1),oh,ow},x.options(),x.suggest_memory_format());
    check_tensor(y);
    FirParams p{ux,uy,dx,dy,px0,py0,int(x.size(3)),int(x.size(2)),int(x.size(1)),int(x.size(0)),
        int(x.stride(3)),int(x.stride(2)),int(x.stride(1)),int(x.stride(0)),
        int(f.size(1)),int(f.size(0)),int(f.stride(1)),int(f.stride(0)),
        int(ow),int(oh),int(y.stride(3)),int(y.stride(2)),int(y.stride(1)),int(y.stride(0)),
        int(y.numel()),int(y.stride(1)==1),int(flip),gain};
    const char* env=std::getenv("STYLEGAN_MPS_FIR_KERNEL");
    std::string mode=env?env:"tiled";
    TORCH_CHECK(mode=="generic" || mode=="direct" || mode=="tiled", "STYLEGAN_MPS_FIR_KERNEL must be generic, direct, or tiled");
    bool hot=p.fw==4 && p.fh==4 && ux==uy && dx==dy &&
             ((ux==1 && (dx==1 || dx==2)) || (ux==2 && dx==1)) &&
             (x.is_contiguous() || x.is_contiguous(at::MemoryFormat::ChannelsLast)) && mode!="generic";
    bool tiled=hot && mode=="tiled";
    int tw=p.channels_last?8:16, th=p.channels_last?4:8, tc=p.channels_last?4:1;
    NSUInteger scratch=hot ? ((((tw-1)*dx+3)/ux+1)*(((th-1)*dy+3)/uy+1)*tc*sizeof(float)+15)/16*16 : 0;
    auto stream=at::mps::getCurrentMPSStream();
    std::string suffix=x.scalar_type()==at::kHalf?"_f16":"_f32";
    auto pso=pipeline((hot?(tiled?"tiled":"direct"):"fir")+suffix,hot?ux:0,hot?dx:0,p.channels_last);
    if (tiled && (scratch>stream->device().maxThreadgroupMemoryLength || pso.maxTotalThreadsPerThreadgroup<128 || 128%pso.threadExecutionWidth)) {
        tiled=false;
        pso=pipeline("direct"+suffix,ux,dx,p.channels_last);
    }
    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            auto encoder=stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            bind(encoder,x,0); bind(encoder,f,1); bind(encoder,y,2);
            [encoder setBytes:&p length:sizeof(p) atIndex:3];
            if (tiled) {
                [encoder setThreadgroupMemoryLength:scratch atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake((p.ow+tw-1)/tw,(p.oh+th-1)/th,p.batch*((p.channels+tc-1)/tc))
                        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            } else dispatch(encoder,pso,p.size);
        }
    });
    return y;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", &initialize);
    m.def("bias_act", &bias_act);
    m.def("upfirdn2d", &upfirdn2d);
}
