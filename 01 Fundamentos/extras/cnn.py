# cnn_convolution_slider_layout.py
# Interactivo con TkAgg: sliders y botones bien distribuidos y compactos

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Backend interactivo
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider, RadioButtons

# ----------- Imagen base -----------
def make_synthetic_image(H=32, W=32):
    img = np.zeros((H, W), dtype=float)
    img[6:14, 6:14] = 4.0
    cy, cx, r = 22, 22, 5
    y, x = np.ogrid[:H, :W]
    mask = (y - cy)**2 + (x - cx)**2 <= r**2
    img[mask] = 8.0
    img += (np.linspace(0, 1, W)[None, :] + np.linspace(0, 1, H)[:, None]) * 1.5
    return img

# ----------- Kernels -----------
KERNELS = {
    "Identity": np.array([[0,0,0],[0,1,0],[0,0,0]], float),
    "Blur": (1/9.0)*np.ones((3,3), float),
    "Gaussian": (1/16.0)*np.array([[1,2,1],[2,4,2],[1,2,1]], float),
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], float),
    "Edge": np.array([[0,1,0],[1,-4,1],[0,1,0]], float),
    "Sobel X": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float),
    "Sobel Y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float),
}

# ----------- Convolución -----------
def conv2d_valid_stride(x, k, stride=1, padding=0):
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode="constant")
    H, W = x.shape
    kh, kw = k.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    windows = []
    for i_out in range(out_h):
        for j_out in range(out_w):
            i = i_out * stride
            j = j_out * stride
            windows.append((i, j))
            out[i_out, j_out] = np.sum(x[i:i+kh, j:j+kw] * k)
    return out, x, windows

class ConvInteractive:
    def __init__(self):
        self.img0 = make_synthetic_image()
        self.kernel_name = "Sobel X"
        self.kernel = KERNELS[self.kernel_name]
        self.stride = 1
        self.padding = 0

        self.out_full, self.img_eff, self.windows = conv2d_valid_stride(
            self.img0, self.kernel, self.stride, self.padding
        )
        self.kh, self.kw = self.kernel.shape
        self.out_partial = np.zeros_like(self.out_full)
        self.current_steps = 0

        # ---------- FIGURA ----------
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.0, 0.25], wspace=0.3, hspace=0.3)

        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_kernel = self.fig.add_subplot(gs[0, 1])
        self.ax_out = self.fig.add_subplot(gs[0, 2])
        self.ax_controls = self.fig.add_subplot(gs[1, :]); self.ax_controls.axis("off")

        self.im_in = self.ax_img.imshow(self.img_eff, interpolation="nearest")
        self.ax_img.set_title("Entrada (con padding)")
        self.rect = Rectangle((0, 0), self.kw, self.kh, fill=False, color='red', lw=2)
        self.ax_img.add_patch(self.rect)

        self.im_k = self.ax_kernel.imshow(self.kernel, interpolation="nearest")
        self.ax_kernel.set_title("Kernel")
        self._annotate_kernel()

        self.im_out = self.ax_out.imshow(self.out_partial, interpolation="nearest")
        self.ax_out.set_title("Salida (parcial)")

        self.txt = self.fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)

        # ---------- WIDGETS ----------
        # RadioButtons - kernel
        ax_radio = self.fig.add_axes([0.02, 0.35, 0.12, 0.55])
        self.radiok = RadioButtons(ax_radio, list(KERNELS.keys()), active=list(KERNELS.keys()).index(self.kernel_name))
        self.radiok.on_clicked(self.on_kernel)

        # Sliders compactos
        ax_stride = self.fig.add_axes([0.16, 0.15, 0.18, 0.05])
        ax_pad = self.fig.add_axes([0.16, 0.08, 0.18, 0.05])
        ax_step = self.fig.add_axes([0.37, 0.08, 0.35, 0.05])

        self.s_stride = Slider(ax_stride, "Stride", 1, 4, valinit=self.stride, valstep=1)
        self.s_stride.on_changed(self.on_stride)

        self.s_pad = Slider(ax_pad, "Padding", 0, 5, valinit=self.padding, valstep=1)
        self.s_pad.on_changed(self.on_padding)

        self.step_slider = Slider(ax_step, "Step (ventana)", 0, len(self.windows), valinit=0, valstep=1)
        self.step_slider.on_changed(self.on_step_slider)

        # Botones pequeños
        ax_full = self.fig.add_axes([0.74, 0.08, 0.1, 0.05])
        ax_reset = self.fig.add_axes([0.85, 0.08, 0.1, 0.05])
        self.btn_full = Button(ax_full, "Full", color="lightgray", hovercolor="lightblue")
        self.btn_full.on_clicked(self.on_full)
        self.btn_reset = Button(ax_reset, "Reset", color="lightgray", hovercolor="lightblue")
        self.btn_reset.on_clicked(self.on_reset)

        self.update_view(force_text=True)

    # ---------- FUNCIONES ----------
    def _annotate_kernel(self):
        for t in list(self.ax_kernel.texts):
            t.remove()
        for (ii, jj), val in np.ndenumerate(self.kernel):
            self.ax_kernel.text(jj, ii, f"{val:.0f}", ha="center", va="center")

    def recompute_all(self):
        self.out_full, self.img_eff, self.windows = conv2d_valid_stride(
            self.img0, self.kernel, self.stride, self.padding
        )
        self.out_partial = np.zeros_like(self.out_full)
        self.current_steps = 0
        self.step_slider.ax.clear()
        self.step_slider = Slider(self.step_slider.ax, "Step (ventana)", 0, len(self.windows), valinit=0, valstep=1)
        self.step_slider.on_changed(self.on_step_slider)

    def compute_until(self, steps):
        self.out_partial[...] = 0
        out_h, out_w = self.out_partial.shape
        for idx in range(min(steps, len(self.windows))):
            i, j = self.windows[idx]
            val = np.sum(self.img_eff[i:i+self.kh, j:j+self.kw] * self.kernel)
            self.out_partial[idx // out_w, idx % out_w] = val

    # ---------- CALLBACKS ----------
    def on_kernel(self, label):
        self.kernel_name = label
        self.kernel = KERNELS[self.kernel_name]
        self.im_k.set_data(self.kernel)
        self._annotate_kernel()
        self.recompute_all()
        self.update_view(True)

    def on_stride(self, val):
        self.stride = int(val)
        self.recompute_all()
        self.update_view(True)

    def on_padding(self, val):
        self.padding = int(val)
        self.recompute_all()
        self.update_view(True)

    def on_step_slider(self, val):
        self.current_steps = int(val)
        self.compute_until(self.current_steps)
        self.update_view()

    def on_full(self, _):
        self.current_steps = len(self.windows)
        self.compute_until(self.current_steps)
        self.step_slider.set_val(self.current_steps)

    def on_reset(self, _):
        self.recompute_all()
        self.update_view(True)

    # ---------- UPDATE ----------
    def update_view(self, force_text=False):
        self.im_in.set_data(self.img_eff)
        if self.windows:
            idx = max(0, min(self.current_steps, len(self.windows)) - 1)
            if self.current_steps == 0: idx = 0
            i, j = self.windows[idx]
            self.rect.set_xy((j, i))
            self.rect.set_width(self.kw)
            self.rect.set_height(self.kh)

        self.im_out.set_data(self.out_partial)
        if np.any(self.out_partial):
            vmin, vmax = np.min(self.out_partial), np.max(self.out_partial)
            if vmax == vmin: vmax = vmin + 1
            self.im_out.set_clim(vmin=vmin, vmax=vmax)

        total = len(self.windows)
        estado = "Salida (parcial)" if self.current_steps < total else "Salida (completa)"
        self.ax_out.set_title(estado)

        if total and self.current_steps:
            i, j = self.windows[self.current_steps - 1]
            val = np.sum(self.img_eff[i:i+self.kh, j:j+self.kw] * self.kernel)
            txt = f"{self.current_steps}/{total} | Kernel={self.kernel_name} | Última salida={val:.2f}"
        else:
            txt = f"Listo | Kernel={self.kernel_name}"
        if force_text or True:
            self.txt.set_text(txt)

        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()

if __name__ == "__main__":
    ConvInteractive().run()
