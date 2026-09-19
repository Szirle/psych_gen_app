# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import imgui
import imgui.integrations.glfw

from . import glfw_window
from . import imgui_utils
from . import text_utils

#----------------------------------------------------------------------------

_CORE_VERTEX_SHADER = """
#version 330 core

uniform mat4 ProjMtx;
in vec2 Position;
in vec2 UV;
in vec4 Color;
out vec2 Frag_UV;
out vec4 Frag_Color;

void main() {
    Frag_UV = UV;
    Frag_Color = Color;
    gl_Position = ProjMtx * vec4(Position.xy, 0, 1);
}
"""

_CORE_FRAGMENT_SHADER = """
#version 330 core

uniform sampler2D Texture;
in vec2 Frag_UV;
in vec4 Frag_Color;
out vec4 Out_Color;

void main() {
    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
}
"""

#----------------------------------------------------------------------------

class ImguiWindow(glfw_window.GlfwWindow):
    def __init__(self, *, title='ImguiWindow', font=None, font_sizes=range(14,24), **glfw_kwargs):
        if font is None:
            font = text_utils.get_default_font()
        font_sizes = {int(size) for size in font_sizes}
        super().__init__(title=title, **glfw_kwargs)

        self._imgui_context  = None
        self._imgui_renderer = None
        self._imgui_fonts    = None
        self._cur_font_size  = max(font_sizes)

        if os.path.isfile('imgui.ini'):
            os.remove('imgui.ini')

        self._imgui_context = imgui.create_context()
        self._imgui_renderer = _GlfwRenderer(self._glfw_window)
        self._attach_glfw_callbacks()
        io = imgui.get_io()
        io.ini_saving_rate = 0
        io.mouse_drag_threshold = 0
        try:
            io.ini_file_name = None
        except Exception:
            pass

        fb_scale = self.framebuffer_width / max(self.content_width, 1)
        self._imgui_fonts = {
            size: io.fonts.add_font_from_file_ttf(font, max(size * fb_scale, 1))
            for size in font_sizes
        }
        if fb_scale != 1.0:
            io.font_global_scale = 1.0 / fb_scale
        self._imgui_renderer.refresh_font_texture()

    def close(self):
        self.make_context_current()
        self._imgui_fonts = None
        if self._imgui_renderer is not None:
            self._imgui_renderer.shutdown()
            self._imgui_renderer = None
        if self._imgui_context is not None:
            self._imgui_context = None
        super().close()

    def _glfw_key_callback(self, *args):
        super()._glfw_key_callback(*args)
        self._imgui_renderer.keyboard_callback(*args)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.4)

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def begin_frame(self):
        super().begin_frame()

        self._imgui_renderer.mouse_wheel_multiplier = self._cur_font_size / 10
        if self.content_width > 0 and self.content_height > 0:
            self._imgui_renderer.process_inputs()

        imgui.new_frame()
        imgui.push_font(self._imgui_fonts[self._cur_font_size])
        imgui_utils.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)

    def end_frame(self):
        imgui.pop_font()
        imgui.render()
        self._imgui_renderer.render(imgui.get_draw_data())
        super().end_frame()

#----------------------------------------------------------------------------

class _GlfwRenderer(imgui.integrations.glfw.GlfwRenderer):
    """GLFW renderer with macOS-safe GLSL 330 core shaders and wheel scaling."""

    VERTEX_SHADER_SRC = _CORE_VERTEX_SHADER
    FRAGMENT_SHADER_SRC = _CORE_FRAGMENT_SHADER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_wheel_multiplier = 1

    def scroll_callback(self, window, x_offset, y_offset):
        self.io.mouse_wheel += y_offset * self.mouse_wheel_multiplier

#----------------------------------------------------------------------------
