# -*- coding: utf-8 -*-
# Ejecuta:  python mezcla_interactiva.py
# Requiere: matplotlib>=3.6, Tk instalado (en Windows ya viene)

import matplotlib
matplotlib.use("TkAgg")  # <<< backend TkAgg activado antes de pyplot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# erf vectorial: usa SciPy si existe, si no vectoriza math.erf
try:
    from scipy.special import erf
except Exception:
    import math
    erf = np.vectorize(math.erf)

# ---------- PDFs y CDFs ----------
def phi(z): return np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
def Phi(z): return 0.5*(1.0 + erf(z/np.sqrt(2.0)))

def pdf_truncnorm(x, mu, sigma):
    Z = 1.0 - Phi((0.0 - mu)/sigma)
    out = np.zeros_like(x, dtype=float)
    m = x >= 0
    z = (x[m]-mu)/sigma
    out[m] = phi(z)/(sigma*Z)
    return out

def cdf_truncnorm(x, mu, sigma):
    Z = 1.0 - Phi((0.0 - mu)/sigma)
    out = np.zeros_like(x, dtype=float)
    m = x >= 0
    out[m] = (Phi((x[m]-mu)/sigma) - Phi((0.0 - mu)/sigma))/Z
    return out

def pdf_weibull(x, k, lam):
    out = np.zeros_like(x, dtype=float)
    m = x >= 0
    xr = x[m]/lam
    out[m] = (k/lam)*(xr**(k-1))*np.exp(-(xr**k))
    return out

def cdf_weibull(x, k, lam):
    out = np.zeros_like(x, dtype=float)
    m = x >= 0
    out[m] = 1.0 - np.exp(- (x[m]/lam)**k)
    return out

def pdf_pareto(x, alpha, xm):
    out = np.zeros_like(x, dtype=float)
    m = x >= xm
    out[m] = alpha*(xm**alpha)/(x[m]**(alpha+1))
    return out

def cdf_pareto(x, alpha, xm):
    out = np.zeros_like(x, dtype=float)
    m0 = x < xm
    out[m0] = 0.0
    m1 = x >= xm
    out[m1] = 1.0 - (xm/x[m1])**alpha
    return out

# ---------- Dominio y estado inicial ----------
xmax = 120.0
x = np.linspace(0.0, xmax, 2000)

w0, mu0, sd0 = 0.6, 18.0, 16.0
lam0, kfix = 18.0, 10.0
alpha0, xm0 = 2.5, 1.0
mode0 = "Weibull"

# ---------- Figure y ejes ----------
plt.close('all')
fig = plt.figure(figsize=(9.6, 6.8))
gs = fig.add_gridspec(2, 1, height_ratios=[1,1],
                      left=0.08, right=0.98, top=0.92, bottom=0.27, hspace=0.30)
ax_pdf = fig.add_subplot(gs[0,0])
ax_cdf = fig.add_subplot(gs[1,0])

axcolor = '0.95'
ax_w   = plt.axes([0.08, 0.20, 0.84, 0.03], facecolor=axcolor)
ax_mu  = plt.axes([0.08, 0.16, 0.84, 0.03], facecolor=axcolor)
ax_sd  = plt.axes([0.08, 0.12, 0.84, 0.03], facecolor=axcolor)
ax_lam = plt.axes([0.08, 0.08, 0.84, 0.03], facecolor=axcolor)
ax_alp = plt.axes([0.08, 0.04, 0.84, 0.03], facecolor=axcolor)
ax_xm  = plt.axes([0.08, 0.00, 0.84, 0.03], facecolor=axcolor)
ax_mode = plt.axes([0.81, 0.90, 0.11, 0.08], facecolor=axcolor)

# ---------- Widgets ----------
sl_w   = Slider(ax_w,   'w mezcla', 0.0, 1.0, valinit=w0, valstep=0.01)
sl_mu  = Slider(ax_mu,  'μ TN', 0.0, 60.0, valinit=mu0, valstep=0.1)
sl_sd  = Slider(ax_sd,  'σ TN', 0.5, 40.0, valinit=sd0, valstep=0.1)
sl_lam = Slider(ax_lam, 'λ Weibull', 0.5, 60.0, valinit=lam0, valstep=0.1)
sl_alp = Slider(ax_alp, 'α Pareto', 0.5, 6.0, valinit=alpha0, valstep=0.05)
sl_xm  = Slider(ax_xm,  'x_m Pareto', 0.1, 20.0, valinit=xm0, valstep=0.1)
rb_mode = RadioButtons(ax_mode, ('Weibull', 'Pareto'), active=0)

# ---------- Curvas iniciales ----------
pdf_tn  = pdf_truncnorm(x, mu0, sd0);  cdf_tn  = cdf_truncnorm(x, mu0, sd0)
pdf_w   = pdf_weibull(x, kfix, lam0);  cdf_w   = cdf_weibull(x, kfix, lam0)
pdf_p   = pdf_pareto(x, alpha0, xm0);  cdf_p   = cdf_pareto(x, alpha0, xm0)

pdf_mix = w0*pdf_tn + (1-w0)*pdf_w
cdf_mix = w0*cdf_tn + (1-w0)*cdf_w

l_pdf_mix, = ax_pdf.plot(x, pdf_mix, lw=2, label='PDF mezcla')
l_pdf_tn,  = ax_pdf.plot(x, pdf_tn,  lw=1, alpha=0.7, label='PDF TN trunc')
l_pdf_2,   = ax_pdf.plot(x, pdf_w,   lw=1, alpha=0.7, label='PDF 2ª comp')

l_cdf_mix, = ax_cdf.plot(x, cdf_mix, lw=2, label='CDF mezcla')
l_cdf_tn,  = ax_cdf.plot(x, cdf_tn,  lw=1, alpha=0.7, label='CDF TN trunc')
l_cdf_2,   = ax_cdf.plot(x, cdf_w,   lw=1, alpha=0.7, label='CDF 2ª comp')

ax_pdf.set_title('Densidad'); ax_pdf.set_xlim(0, xmax); ax_pdf.set_ylim(bottom=0); ax_pdf.legend(loc='upper right')
ax_cdf.set_title('Función de distribución acumulada'); ax_cdf.set_xlim(0, xmax); ax_cdf.set_ylim(0, 1.0); ax_cdf.legend(loc='lower right')

state = {"mode": mode0}

# ---------- Lógica de actualización ----------
def recompute_and_draw(_=None):
    w, mu, sd = sl_w.val, sl_mu.val, sl_sd.val
    lam, alp, xm = sl_lam.val, sl_alp.val, sl_xm.val

    pdf_tn = pdf_truncnorm(x, mu, sd); cdf_tn = cdf_truncnorm(x, mu, sd)

    if state["mode"] == "Weibull":
        pdf2 = pdf_weibull(x, kfix, lam); cdf2 = cdf_weibull(x, kfix, lam)
    else:
        xm_eff = max(xm, 1e-12)
        pdf2 = pdf_pareto(x, alp, xm_eff); cdf2 = cdf_pareto(x, alp, xm_eff)

    l_pdf_mix.set_ydata(w*pdf_tn + (1-w)*pdf2)
    l_pdf_tn.set_ydata(pdf_tn)
    l_pdf_2.set_ydata(pdf2)

    l_cdf_mix.set_ydata(w*cdf_tn + (1-w)*cdf2)
    l_cdf_tn.set_ydata(cdf_tn)
    l_cdf_2.set_ydata(cdf2)

    ymax = float(np.nanmax(l_pdf_mix.get_ydata())*1.1)
    ax_pdf.set_ylim(0, max(1e-12, ymax))
    fig.canvas.draw_idle()

def on_mode(label):
    state["mode"] = label
    # resaltar sliders relevantes
    if label == "Weibull":
        sl_lam.ax.set_alpha(1.0); sl_lam.label.set_alpha(1.0)
        sl_alp.ax.set_alpha(0.25); sl_alp.label.set_alpha(0.25)
        sl_xm.ax.set_alpha(0.25);  sl_xm.label.set_alpha(0.25)
    else:
        sl_lam.ax.set_alpha(0.25); sl_lam.label.set_alpha(0.25)
        sl_alp.ax.set_alpha(1.0);  sl_alp.label.set_alpha(1.0)
        sl_xm.ax.set_alpha(1.0);   sl_xm.label.set_alpha(1.0)
    fig.canvas.draw_idle()
    recompute_and_draw()

for sl in (sl_w, sl_mu, sl_sd, sl_lam, sl_alp, sl_xm):
    sl.on_changed(recompute_and_draw)
rb_mode.on_clicked(on_mode)

# init
on_mode(mode0)
recompute_and_draw()
plt.show()
