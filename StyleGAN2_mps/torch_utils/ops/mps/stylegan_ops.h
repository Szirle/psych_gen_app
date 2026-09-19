#pragma once
// Shared host/MSL ABI: only 32-bit scalars, no vector or bool alignment rules.
struct BiasParams {
    int size, size_b, step_b, grad, act;
    int has_b, has_xref, has_yref, has_dy;
    float alpha, gain, clamp;
};
struct FirParams {
    int up_x, up_y, down_x, down_y, pad_x, pad_y;
    int iw, ih, channels, batch;
    int sx, sy, sc, sn;
    int fw, fh, fsx, fsy;
    int ow, oh, osx, osy, osc, osn;
    int size, channels_last, flip;
    float gain;
};
#ifndef __METAL_VERSION__
static_assert(sizeof(int) == 4 && sizeof(BiasParams) == 48);
static_assert(sizeof(FirParams) == 112);
#endif
