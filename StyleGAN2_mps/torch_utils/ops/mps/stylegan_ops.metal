// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <metal_stdlib>
using namespace metal;

template <typename T>
void bias_impl(device const T* input, device const T* b_buf,
               device const T* xref_buf, device const T* yref_buf,
               device const T* dy_buf, device T* output,
               constant BiasParams& p, uint gid) {
    int xi = int(gid);
    if (xi >= p.size) return;
    int G = p.grad, A = p.act;
    float alpha = p.alpha, gain = p.gain, clamp = p.clamp;
    float one = 1, two = 2, expRange = 80, halfExpRange = 40;
    float seluScale = 1.0507009873554804934f;
    float seluAlpha = 1.6732632423543772848f;
    // Load.
    float x = (float)input[xi];
    float b = (p.has_b) ? (float)b_buf[(xi / p.step_b) % p.size_b] : 0;
    float xref = (p.has_xref) ? (float)xref_buf[xi] : 0;
    float yref = (p.has_yref) ? (float)yref_buf[xi] : 0;
    float dy = (p.has_dy) ? (float)dy_buf[xi] : one;
    float yy = (gain != 0) ? yref / gain : 0;
    float y = 0;

    // Apply bias.
    if (G == 0) x += b; else xref += b;

    // linear
    if (A == 1)
    {
        if (G == 0) y = x;
        if (G == 1) y = x;
    }

    // relu
    if (A == 2)
    {
        if (G == 0) y = (x > 0) ? x : 0;
        if (G == 1) y = (yy > 0) ? x : 0;
    }

    // lrelu
    if (A == 3)
    {
        if (G == 0) y = (x > 0) ? x : x * alpha;
        if (G == 1) y = (yy > 0) ? x : x * alpha;
    }

    // tanh
    if (A == 4)
    {
        if (G == 0) { float c = exp(x); float d = one / c; y = (x < -expRange) ? -one : (x > expRange) ? one : (c - d) / (c + d); }
        if (G == 1) y = x * (one - yy * yy);
        if (G == 2) y = x * (one - yy * yy) * (-two * yy);
    }

    // sigmoid
    if (A == 5)
    {
        if (G == 0) y = (x < -expRange) ? 0 : one / (exp(-x) + one);
        if (G == 1) y = x * yy * (one - yy);
        if (G == 2) y = x * yy * (one - yy) * (one - two * yy);
    }

    // elu
    if (A == 6)
    {
        if (G == 0) y = (x >= 0) ? x : exp(x) - one;
        if (G == 1) y = (yy >= 0) ? x : x * (yy + one);
        if (G == 2) y = (yy >= 0) ? 0 : x * (yy + one);
    }

    // selu
    if (A == 7)
    {
        if (G == 0) y = (x >= 0) ? seluScale * x : (seluScale * seluAlpha) * (exp(x) - one);
        if (G == 1) y = (yy >= 0) ? x * seluScale : x * (yy + seluScale * seluAlpha);
        if (G == 2) y = (yy >= 0) ? 0 : x * (yy + seluScale * seluAlpha);
    }

    // softplus
    if (A == 8)
    {
        if (G == 0) y = (x > expRange) ? x : log(exp(x) + one);
        if (G == 1) y = x * (one - exp(-yy));
        if (G == 2) { float c = exp(-yy); y = x * c * (one - c); }
    }

    // swish
    if (A == 9)
    {
        if (G == 0)
            y = (x < -expRange) ? 0 : x / (exp(-x) + one);
        else
        {
            // Algebraically equivalent derivatives without exp(xref)^3,
            // which overflows float for otherwise finite xref near 30.
            float c = exp(-abs(xref));
            float d = one + c;
            float sigmoid = (xref >= 0) ? one / d : c / d;
            float deriv = c / (d * d);
            if (G == 1)
                y = (xref > halfExpRange) ? x : x * (sigmoid + xref * deriv);
            else
                y = (xref > halfExpRange) ? 0 : x * deriv * (two + xref * (one - two * sigmoid));
            yref = (xref < -expRange) ? 0 : xref / (exp(-xref) + one) * gain;
        }
    }

    // Apply gain.
    y *= gain * dy;

    // Clamp.
    if (clamp >= 0)
    {
        if (G == 0)
            y = (y > -clamp & y < clamp) ? y : (y >= 0) ? clamp : -clamp;
        else
            y = (yref > -clamp & yref < clamp) ? y : 0;
    }

    output[xi] = T(y);
}
#define BIAS_KERNEL(T, NAME) \
kernel void NAME(device const T* x [[buffer(0)]], device const T* b [[buffer(1)]], \
    device const T* xr [[buffer(2)]], device const T* yr [[buffer(3)]], \
    device const T* dy [[buffer(4)]], device T* y [[buffer(5)]], \
    constant BiasParams& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) { \
    bias_impl(x, b, xr, yr, dy, y, p, gid); \
}
BIAS_KERNEL(float, bias_f32)
BIAS_KERNEL(half, bias_f16)

int floor_div(int a, int b) { return a / b - int(a % b < 0); }

void output_coords(int i, constant FirParams& p, thread int& ox,
                   thread int& oy, thread int& c, thread int& n) {
    if (p.channels_last) {
        c = i % p.channels; i /= p.channels;
        ox = i % p.ow; i /= p.ow;
        oy = i % p.oh; n = i / p.oh;
    } else {
        ox = i % p.ow; i /= p.ow;
        oy = i % p.oh; i /= p.oh;
        c = i % p.channels; n = i / p.channels;
    }
}

template <typename T>
void fir_generic(device const T* x, device const float* f, device T* y,
                 constant FirParams& p, uint gid) {
    int i = int(gid);
    if (i >= p.size) return;
    int ox, oy, c, n;
    output_coords(i, p, ox, oy, c, n);
    int mx = ox * p.down_x + p.up_x - 1 - p.pad_x;
    int my = oy * p.down_y + p.up_y - 1 - p.pad_y;
    int ix = metal::clamp(floor_div(mx, p.up_x), 0, p.iw);
    int iy = metal::clamp(floor_div(my, p.up_y), 0, p.ih);
    int w = metal::clamp(floor_div(mx + p.fw, p.up_x), 0, p.iw) - ix;
    int h = metal::clamp(floor_div(my + p.fh, p.up_y), 0, p.ih) - iy;
    int fx = mx + p.fw - (ix + 1) * p.up_x;
    int fy = my + p.fh - (iy + 1) * p.up_y;
    if (p.flip) { fx = p.fw - 1 - fx; fy = p.fh - 1 - fy; }
    int dx = p.flip ? p.up_x : -p.up_x;
    int dy = p.flip ? p.up_y : -p.up_y;
    float v = 0;
    for (int ky = 0; ky < h; ++ky)
        for (int kx = 0; kx < w; ++kx)
            v += float(x[n*p.sn + c*p.sc + (iy+ky)*p.sy + (ix+kx)*p.sx]) *
                 f[(fy+ky*dy)*p.fsy + (fx+kx*dx)*p.fsx];
    y[n*p.osn + c*p.osc + oy*p.osy + ox*p.osx] = T(v*p.gain);
}
#define FIR_KERNEL(T, NAME) \
kernel void NAME(device const T* x [[buffer(0)]], device const float* f [[buffer(1)]], \
    device T* y [[buffer(2)]], constant FirParams& p [[buffer(3)]], \
    uint gid [[thread_position_in_grid]]) { fir_generic(x, f, y, p, gid); }
FIR_KERNEL(float, fir_f32)
FIR_KERNEL(half, fir_f16)

// Specialize only the common square 4x4 filters. Both variants use the same
// tap order and arithmetic, making direct vs staged benchmarks meaningful.
constant int hot_up [[function_constant(0)]];
constant int hot_down [[function_constant(1)]];
constant bool hot_cl [[function_constant(2)]];

template <typename T>
void fir_direct(device const T* x, device const float* f, device T* y,
                constant FirParams& p, uint gid) {
    int i = int(gid);
    if (i >= p.size) return;
    int ox, oy, c, n;
    output_coords(i, p, ox, oy, c, n);
    int mx = ox*hot_down + hot_up - 1 - p.pad_x;
    int my = oy*hot_down + hot_up - 1 - p.pad_y;
    int ix = floor_div(mx, hot_up), iy = floor_div(my, hot_up);
    int fx = mx + 4 - (ix+1)*hot_up;
    int fy = my + 4 - (iy+1)*hot_up;
    float v = 0;
    #pragma unroll
    for (int ky = 0; ky < 4/hot_up; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 4/hot_up; ++kx) {
            int xx = ix+kx, yy = iy+ky;
            int tx = fx-kx*hot_up, ty = fy-ky*hot_up;
            if (p.flip) { tx=3-tx; ty=3-ty; }
            if (xx>=0 && xx<p.iw && yy>=0 && yy<p.ih)
                v += float(x[n*p.sn+c*p.sc+yy*p.sy+xx*p.sx])*f[ty*p.fsy+tx*p.fsx];
        }
    }
    y[n*p.osn+c*p.osc+oy*p.osy+ox*p.osx] = T(v*p.gain);
}
#define DIRECT_KERNEL(T, NAME) \
kernel void NAME(device const T* x [[buffer(0)]], device const float* f [[buffer(1)]], \
    device T* y [[buffer(2)]], constant FirParams& p [[buffer(3)]], \
    uint gid [[thread_position_in_grid]]) { fir_direct(x, f, y, p, gid); }
DIRECT_KERNEL(float, direct_f32)
DIRECT_KERNEL(half, direct_f16)

template <typename T>
void fir_tiled(device const T* x, device const float* f, device T* y,
               constant FirParams& p, threadgroup float* tile, uint tid, uint3 group) {
    int tw = hot_cl ? 8 : 16, th = hot_cl ? 4 : 8, tc = hot_cl ? 4 : 1;
    int iw = ((tw-1)*hot_down+3)/hot_up+1;
    int ih = ((th-1)*hot_down+3)/hot_up+1;
    int groups_c = (p.channels+tc-1)/tc;
    int n = int(group.z)/groups_c, cb = (int(group.z)%groups_c)*tc;
    int ox0 = int(group.x)*tw, oy0 = int(group.y)*th;
    int ix0 = floor_div(ox0*hot_down+hot_up-1-p.pad_x, hot_up);
    int iy0 = floor_div(oy0*hot_down+hot_up-1-p.pad_y, hot_up);
    for (int j=int(tid); j<iw*ih*tc; j+=128) {
        int c=cb+j%tc, xx=ix0+(j/tc)%iw, yy=iy0+j/(tc*iw);
        tile[j] = (c<p.channels && xx>=0 && xx<p.iw && yy>=0 && yy<p.ih)
            ? float(x[n*p.sn+c*p.sc+yy*p.sy+xx*p.sx]) : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int lc=int(tid)%tc, ox=ox0+(int(tid)/tc)%tw, oy=oy0+int(tid)/(tc*tw), c=cb+lc;
    if (ox>=p.ow || oy>=p.oh || c>=p.channels) return; // after uniform barrier
    int mx=ox*hot_down+hot_up-1-p.pad_x, my=oy*hot_down+hot_up-1-p.pad_y;
    int ix=floor_div(mx, hot_up), iy=floor_div(my, hot_up);
    int fx=mx+4-(ix+1)*hot_up, fy=my+4-(iy+1)*hot_up;
    float v=0;
    #pragma unroll
    for (int ky=0; ky<4/hot_up; ++ky) {
        #pragma unroll
        for (int kx=0; kx<4/hot_up; ++kx) {
            int tx=fx-kx*hot_up, ty=fy-ky*hot_up;
            if (p.flip) { tx=3-tx; ty=3-ty; }
            v += tile[((iy+ky-iy0)*iw+ix+kx-ix0)*tc+lc]*f[ty*p.fsy+tx*p.fsx];
        }
    }
    y[n*p.osn+c*p.osc+oy*p.osy+ox*p.osx] = T(v*p.gain);
}
#define TILED_KERNEL(T, NAME) \
kernel void NAME(device const T* x [[buffer(0)]], device const float* f [[buffer(1)]], \
    device T* y [[buffer(2)]], constant FirParams& p [[buffer(3)]], \
    threadgroup float* tile [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]], \
    uint3 group [[threadgroup_position_in_grid]]) { fir_tiled(x, f, y, p, tile, tid, group); }
TILED_KERNEL(float, tiled_f32)
TILED_KERNEL(half, tiled_f16)
