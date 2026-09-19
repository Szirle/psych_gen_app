# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""OpenGL 3.3-core helpers for the visualizer.

The original NVIDIA code used the fixed-function pipeline (glMatrixMode,
glPushAttrib, GL_LUMINANCE, glDrawPixels, ...). Those APIs are unavailable
on macOS, which only exposes a Core profile. This module keeps the same
Texture / drawing interface but renders with shaders.
"""

import os
import ctypes
import functools
import contextlib
import numpy as np
import OpenGL.GL as gl
import dnnlib

#----------------------------------------------------------------------------

_VERTEX_SRC = """
#version 330 core
uniform mat4 ProjMtx;
in vec2 Position;
in vec2 UV;
out vec2 Frag_UV;
void main() {
    Frag_UV = UV;
    gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
}
"""

_FRAGMENT_SRC = """
#version 330 core
uniform sampler2D Texture;
uniform vec4 Color;
uniform int Mode;
in vec2 Frag_UV;
out vec4 Out_Color;
void main() {
    vec4 tex = (Mode == 0) ? vec4(1.0) : texture(Texture, Frag_UV);
    Out_Color = Color * tex;
}
"""

#----------------------------------------------------------------------------

def init_egl():
    assert os.environ['PYOPENGL_PLATFORM'] == 'egl' # Must be set before importing OpenGL.
    import OpenGL.EGL as egl

    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    assert display != egl.EGL_NO_DISPLAY
    major = ctypes.c_int32()
    minor = ctypes.c_int32()
    ok = egl.eglInitialize(display, major, minor)
    assert ok
    assert major.value * 10 + minor.value >= 14

    config_attribs = [
        egl.EGL_RENDERABLE_TYPE,    egl.EGL_OPENGL_BIT,
        egl.EGL_SURFACE_TYPE,       egl.EGL_PBUFFER_BIT,
        egl.EGL_NONE
    ]
    configs = (ctypes.c_int32 * 1)()
    num_configs = ctypes.c_int32()
    ok = egl.eglChooseConfig(display, config_attribs, configs, 1, num_configs)
    assert ok
    assert num_configs.value == 1
    config = configs[0]

    surface_attribs = [
        egl.EGL_WIDTH,  1,
        egl.EGL_HEIGHT, 1,
        egl.EGL_NONE
    ]
    surface = egl.eglCreatePbufferSurface(display, config, surface_attribs)
    assert surface != egl.EGL_NO_SURFACE

    ok = egl.eglBindAPI(egl.EGL_OPENGL_API)
    assert ok
    context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, None)
    assert context != egl.EGL_NO_CONTEXT
    ok = egl.eglMakeCurrent(display, surface, surface, context)
    assert ok

#----------------------------------------------------------------------------

def _as_gl_id(value):
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(np.asarray(value).reshape(-1)[0])

def _compile_shader(src, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, src)
    gl.glCompileShader(shader)
    status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not status:
        log = gl.glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode('utf-8', 'replace')
        raise RuntimeError(f'GLSL compile error:\n{log}')
    return shader

def _link_program(vs_src, fs_src):
    program = gl.glCreateProgram()
    vs = _compile_shader(vs_src, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if not status:
        log = gl.glGetProgramInfoLog(program)
        if isinstance(log, bytes):
            log = log.decode('utf-8', 'replace')
        raise RuntimeError(f'GLSL link error:\n{log}')
    return program

#----------------------------------------------------------------------------

class _GLProgram:
    def __init__(self):
        self.program = _link_program(_VERTEX_SRC, _FRAGMENT_SRC)
        self.attrib_pos = gl.glGetAttribLocation(self.program, 'Position')
        self.attrib_uv = gl.glGetAttribLocation(self.program, 'UV')
        self.uni_proj = gl.glGetUniformLocation(self.program, 'ProjMtx')
        self.uni_tex = gl.glGetUniformLocation(self.program, 'Texture')
        self.uni_color = gl.glGetUniformLocation(self.program, 'Color')
        self.uni_mode = gl.glGetUniformLocation(self.program, 'Mode')
        self.vao = _as_gl_id(gl.glGenVertexArrays(1))
        self.vbo = _as_gl_id(gl.glGenBuffers(1))
        self.proj = (ctypes.c_float * 16)(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        )

    def set_ortho(self, width, height):
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        self.proj = (ctypes.c_float * 16)(
             2.0 / width,  0.0,            0.0, 0.0,
             0.0,         -2.0 / height,   0.0, 0.0,
             0.0,          0.0,           -1.0, 0.0,
            -1.0,          1.0,            0.0, 1.0,
        )

_program = None

def _get_program():
    global _program
    if _program is None:
        _program = _GLProgram()
    return _program

def set_ortho(width, height):
    _get_program().set_ortho(width, height)

def begin_frame(display_width, display_height, fb_width, fb_height):
    gl.glViewport(0, 0, int(fb_width), int(fb_height))
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquation(gl.GL_FUNC_ADD)
    gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    set_ortho(display_width, display_height)

#----------------------------------------------------------------------------

_texture_formats = {
    ('uint8',   1): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RED,  internalformat=gl.GL_R8,    align=1),
    ('uint8',   2): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RG,   internalformat=gl.GL_RG8,   align=2),
    ('uint8',   3): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGB,  internalformat=gl.GL_RGB8,  align=1),
    ('uint8',   4): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGBA, internalformat=gl.GL_RGBA8, align=4),
    ('float32', 1): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RED,  internalformat=gl.GL_R32F,  align=4),
    ('float32', 2): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RG,   internalformat=gl.GL_RG32F, align=8),
    ('float32', 3): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RGB,  internalformat=gl.GL_RGB32F, align=4),
    ('float32', 4): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RGBA, internalformat=gl.GL_RGBA32F, align=4),
}

def get_texture_format(dtype, channels):
    return _texture_formats[(np.dtype(dtype).name, int(channels))]

#----------------------------------------------------------------------------

def prepare_texture_data(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.dtype.name == 'float64':
        image = image.astype('float32')
    return np.ascontiguousarray(image)

def _to_rgba(image):
    image = prepare_texture_data(image)
    height, width, channels = image.shape
    if channels == 4:
        return image
    if image.dtype == np.uint8:
        out = np.zeros((height, width, 4), dtype=np.uint8)
        if channels == 1:
            out[:, :, 0:3] = image
            out[:, :, 3] = 255
        elif channels == 2:
            rgb = image[:, :, 0:1]
            alpha = image[:, :, 1:2]
            # Premultiply so GL_ONE / GL_ONE_MINUS_SRC_ALPHA matches the old path.
            out[:, :, 0:3] = (rgb.astype(np.uint16) * alpha.astype(np.uint16) // 255).astype(np.uint8)
            out[:, :, 3:4] = alpha
        else:
            out[:, :, 0:3] = image
            out[:, :, 3] = 255
        return out
    out = np.zeros((height, width, 4), dtype=np.float32)
    if channels == 1:
        out[:, :, 0:3] = image
        out[:, :, 3] = 1.0
    elif channels == 2:
        rgb = image[:, :, 0:1]
        alpha = image[:, :, 1:2]
        out[:, :, 0:3] = rgb * alpha
        out[:, :, 3:4] = alpha
    else:
        out[:, :, 0:3] = image
        out[:, :, 3] = 1.0
    return out

#----------------------------------------------------------------------------

def _draw_arrays(vertices, uvs, *, mode=gl.GL_TRIANGLE_FAN, color=(1, 1, 1, 1), texture_id=None):
    prog = _get_program()
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)
    interleaved = np.empty((vertices.shape[0], 4), dtype=np.float32)
    interleaved[:, 0:2] = vertices
    interleaved[:, 2:4] = uvs

    prev_program = _as_gl_id(gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM))
    prev_vao = _as_gl_id(gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING))
    prev_array = _as_gl_id(gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING))
    prev_tex = _as_gl_id(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
    prev_active = _as_gl_id(gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE))

    gl.glUseProgram(prog.program)
    gl.glUniformMatrix4fv(prog.uni_proj, 1, gl.GL_FALSE, prog.proj)
    gl.glUniform4f(prog.uni_color, float(color[0]), float(color[1]), float(color[2]), float(color[3]))
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glUniform1i(prog.uni_tex, 0)
    if texture_id is None:
        gl.glUniform1i(prog.uni_mode, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    else:
        gl.glUniform1i(prog.uni_mode, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, _as_gl_id(texture_id))

    gl.glBindVertexArray(prog.vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, prog.vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STREAM_DRAW)
    stride = 4 * 4
    gl.glEnableVertexAttribArray(prog.attrib_pos)
    gl.glEnableVertexAttribArray(prog.attrib_uv)
    gl.glVertexAttribPointer(prog.attrib_pos, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
    gl.glVertexAttribPointer(prog.attrib_uv, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
    gl.glDrawArrays(mode, 0, vertices.shape[0])

    gl.glBindTexture(gl.GL_TEXTURE_2D, prev_tex)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, prev_array)
    gl.glBindVertexArray(prev_vao)
    gl.glActiveTexture(prev_active)
    gl.glUseProgram(prev_program)

#----------------------------------------------------------------------------

def draw_pixels(image, *, pos=0, zoom=1, align=0, rint=True):
    tex = Texture(image=image, bilinear=False, mipmap=False)
    try:
        tex.draw(pos=pos, zoom=zoom, align=align, rint=rint)
    finally:
        tex.delete()

#----------------------------------------------------------------------------

def read_pixels(width, height, *, pos=0, dtype='uint8', channels=3):
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    dtype = np.dtype(dtype)
    fmt = get_texture_format(dtype, channels)
    image = np.empty([height, width, channels], dtype=dtype)
    prev_pack = gl.glGetIntegerv(gl.GL_PACK_ALIGNMENT)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadPixels(int(np.round(pos[0])), int(np.round(pos[1])), width, height, fmt.format, fmt.type, image)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, int(prev_pack))
    return np.flipud(image)

#----------------------------------------------------------------------------

class Texture:
    def __init__(self, *, image=None, width=None, height=None, channels=None, dtype=None, bilinear=True, mipmap=True):
        self.gl_id = None
        self.bilinear = bilinear
        self.mipmap = mipmap
        self._rgba_dtype = None

        if image is not None:
            image = prepare_texture_data(image)
            self.height, self.width, self.channels = image.shape
            self.dtype = image.dtype
        else:
            assert width is not None and height is not None
            self.width = width
            self.height = height
            self.channels = channels if channels is not None else 3
            self.dtype = np.dtype(dtype) if dtype is not None else np.uint8

        assert isinstance(self.width, int) and self.width >= 0
        assert isinstance(self.height, int) and self.height >= 0
        assert isinstance(self.channels, int) and self.channels >= 1
        assert self.is_compatible(width=width, height=height, channels=channels, dtype=dtype)

        self.gl_id = _as_gl_id(gl.glGenTextures(1))
        with self.bind():
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            mag = gl.GL_LINEAR if self.bilinear else gl.GL_NEAREST
            if self.mipmap:
                minf = gl.GL_LINEAR_MIPMAP_LINEAR if self.bilinear else gl.GL_NEAREST_MIPMAP_NEAREST
            else:
                minf = mag
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, minf)
        self.update(image)

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteTextures([self.gl_id])
            self.gl_id = None

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    @contextlib.contextmanager
    def bind(self):
        prev_id = _as_gl_id(gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_id)
        yield
        gl.glBindTexture(gl.GL_TEXTURE_2D, prev_id)

    def update(self, image):
        if image is not None:
            image = prepare_texture_data(image)
            assert self.is_compatible(image=image)
            rgba = _to_rgba(image)
        else:
            rgba = None
        if rgba is not None:
            self._rgba_dtype = rgba.dtype
        upload_dtype = self._rgba_dtype if self._rgba_dtype is not None else np.dtype('uint8')
        fmt = get_texture_format(upload_dtype, 4)
        with self.bind():
            prev_align = gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt.internalformat, self.width, self.height, 0, fmt.format, fmt.type, rgba)
            if self.mipmap:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, int(prev_align))

    def draw(self, *, pos=0, zoom=1, align=0, rint=False, color=1, alpha=1, rounding=0):
        zoom = np.broadcast_to(np.asarray(zoom, dtype='float32'), [2])
        size = zoom * np.array([self.width, self.height], dtype=np.float32)
        color = np.broadcast_to(np.asarray(color, dtype='float32'), [3])
        alpha = float(np.clip(alpha, 0, 1))
        draw_rect(pos=pos, size=size, align=align, rint=rint, color=color, alpha=alpha, rounding=rounding, texture_id=self.gl_id)

    def is_compatible(self, *, image=None, width=None, height=None, channels=None, dtype=None): # pylint: disable=too-many-return-statements
        if image is not None:
            if image.ndim != 3:
                return False
            ih, iw, ic = image.shape
            if not self.is_compatible(width=iw, height=ih, channels=ic, dtype=image.dtype):
                return False
        if width is not None and self.width != width:
            return False
        if height is not None and self.height != height:
            return False
        if channels is not None and self.channels != channels:
            return False
        if dtype is not None and self.dtype != dtype:
            return False
        return True

#----------------------------------------------------------------------------

class Framebuffer:
    def __init__(self, *, texture=None, width=None, height=None, channels=None, dtype=None, msaa=0):
        self.texture = texture
        self.gl_id = None
        self.gl_color = None
        self.gl_depth_stencil = None
        self.msaa = msaa

        if texture is not None:
            assert isinstance(self.texture, Texture)
            self.width = texture.width
            self.height = texture.height
            self.channels = texture.channels
            self.dtype = texture.dtype
        else:
            assert width is not None and height is not None
            self.width = width
            self.height = height
            self.channels = channels if channels is not None else 4
            self.dtype = np.dtype(dtype) if dtype is not None else np.float32

        assert isinstance(self.width, int) and self.width >= 0
        assert isinstance(self.height, int) and self.height >= 0
        assert isinstance(self.channels, int) and self.channels >= 1
        assert width is None or width == self.width
        assert height is None or height == self.height
        assert channels is None or channels == self.channels
        assert dtype is None or dtype == self.dtype

        self.gl_id = _as_gl_id(gl.glGenFramebuffers(1))
        with self.bind():
            if self.texture is not None:
                assert self.msaa == 0
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture.gl_id, 0)
            else:
                fmt = get_texture_format(self.dtype, self.channels)
                self.gl_color = _as_gl_id(gl.glGenRenderbuffers(1))
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.gl_color)
                gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.msaa, fmt.internalformat, self.width, self.height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, self.gl_color)

            self.gl_depth_stencil = _as_gl_id(gl.glGenRenderbuffers(1))
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.gl_depth_stencil)
            gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.msaa, gl.GL_DEPTH24_STENCIL8, self.width, self.height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self.gl_depth_stencil)

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteFramebuffers([self.gl_id])
            self.gl_id = None
        if self.gl_color is not None:
            gl.glDeleteRenderbuffers(1, [self.gl_color])
            self.gl_color = None
        if self.gl_depth_stencil is not None:
            gl.glDeleteRenderbuffers(1, [self.gl_depth_stencil])
            self.gl_depth_stencil = None

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    @contextlib.contextmanager
    def bind(self):
        prev_fbo = _as_gl_id(gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING))
        prev_rbo = _as_gl_id(gl.glGetInteger(gl.GL_RENDERBUFFER_BINDING))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.gl_id)
        if self.width is not None and self.height is not None:
            gl.glViewport(0, 0, self.width, self.height)
        yield
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, prev_fbo)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, prev_rbo)

    def blit(self, dst=None):
        assert dst is None or isinstance(dst, Framebuffer)
        with self.bind():
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0 if dst is None else dst.gl_id)
            gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

#----------------------------------------------------------------------------

def draw_shape(vertices, *, mode=gl.GL_TRIANGLE_FAN, pos=0, size=1, color=1, alpha=1, texture_id=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 2
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    size = np.broadcast_to(np.asarray(size, dtype='float32'), [2])
    color = np.broadcast_to(np.asarray(color, dtype='float32'), [3])
    alpha = float(np.clip(np.broadcast_to(np.asarray(alpha, dtype='float32'), []), 0, 1))
    verts = vertices * size + pos
    uvs = vertices.astype(np.float32)
    _draw_arrays(verts, uvs, mode=mode, color=(color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha), texture_id=texture_id)

#----------------------------------------------------------------------------

def draw_rect(*, pos=0, pos2=None, size=None, align=0, rint=False, color=1, alpha=1, rounding=0, texture_id=None):
    assert pos2 is None or size is None
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    pos2 = np.broadcast_to(np.asarray(pos2, dtype='float32'), [2]) if pos2 is not None else None
    size = np.broadcast_to(np.asarray(size, dtype='float32'), [2]) if size is not None else None
    size = size if size is not None else pos2 - pos if pos2 is not None else np.array([1, 1], dtype='float32')
    pos = pos - size * align
    if rint:
        pos = np.rint(pos)
    rounding = np.broadcast_to(np.asarray(rounding, dtype='float32'), [2])
    rounding = np.minimum(np.abs(rounding) / np.maximum(np.abs(size), 1e-8), 0.5)
    if np.min(rounding) == 0:
        rounding *= 0
    vertices = _setup_rect(float(rounding[0]), float(rounding[1]))
    draw_shape(vertices, mode=gl.GL_TRIANGLE_FAN, pos=pos, size=size, color=color, alpha=alpha, texture_id=texture_id)

@functools.lru_cache(maxsize=10000)
def _setup_rect(rx, ry):
    t = np.linspace(0, np.pi / 2, 1 if max(rx, ry) == 0 else 64)
    s = 1 - np.sin(t); c = 1 - np.cos(t)
    x = [c * rx, 1 - s * rx, 1 - c * rx, s * rx]
    y = [s * ry, c * ry, 1 - s * ry, 1 - c * ry]
    v = np.stack([x, y], axis=-1).reshape(-1, 2)
    return v.astype('float32')

#----------------------------------------------------------------------------

def draw_circle(*, center=0, radius=100, hole=0, color=1, alpha=1):
    hole = np.broadcast_to(np.asarray(hole, dtype='float32'), [])
    vertices = _setup_circle(float(hole))
    draw_shape(vertices, mode=gl.GL_TRIANGLE_STRIP, pos=center, size=radius, color=color, alpha=alpha)

@functools.lru_cache(maxsize=10000)
def _setup_circle(hole):
    t = np.linspace(0, np.pi * 2, 128)
    s = np.sin(t); c = np.cos(t)
    v = np.stack([c, s, c * hole, s * hole], axis=-1).reshape(-1, 2)
    return v.astype('float32')

#----------------------------------------------------------------------------
