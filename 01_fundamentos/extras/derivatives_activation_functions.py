# activations_derivatives_interactive.py
# Ejecuta: python activations_derivatives_interactive.py

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# ---------------- Simbólico base ----------------
x = sp.symbols('x', real=True)

def sigmoid_sym(x):   return 1/(1+sp.exp(-x))
def tanh_sym(x):      return sp.tanh(x)
def softplus_sym(x):  return sp.log(1+sp.exp(x))
def softsign_sym(x):  return x/(1+sp.Abs(x))
def silu_sym(x):      return x*sigmoid_sym(x)  # Swish / SiLU
def gelu_sym(x):      return x*sp.Rational(1,2)*(1+sp.erf(x/sp.sqrt(2)))
def relu_sym(x):      return sp.Piecewise((0, x <= 0), (x, x > 0))
def leaky_relu_sym(x, a=sp.Rational(1,100)):
    return sp.Piecewise((a*x, x <= 0), (x, x > 0))
def elu_sym(x, alpha=1):
    return sp.Piecewise((alpha*(sp.exp(x)-1), x <= 0), (x, x > 0))

ACTIVATIONS = {
    "Sigmoid":      sigmoid_sym(x),
    "Tanh":         tanh_sym(x),
    "Softplus":     softplus_sym(x),
    "Softsign":     softsign_sym(x),
    "SiLU (Swish)": silu_sym(x),
    "GELU":         gelu_sym(x),
    "ReLU":         relu_sym(x),
    "LeakyReLU":    leaky_relu_sym(x),
    "ELU":          elu_sym(x),
}

# ---------------- Cache de derivadas ----------------
_cache = {}
def nth_derivative_expr(name: str, n: int) -> sp.Expr:
    key = (name, n)
    if key in _cache:
        return _cache[key]
    base = ACTIVATIONS[name]
    expr = sp.simplify(sp.diff(base, x, n))
    _cache[key] = expr
    return expr

def funcs_numeric(name: str, n_max: int):
    out = []
    for k in range(n_max+1):
        expr = nth_derivative_expr(name, k)
        f = sp.lambdify(x, expr, "numpy")
        out.append((k, expr, f))
    return out

# ---------------- Parámetros de UI y estilos ----------------
N_MAX = 25
ALPHA_OTRAS = 0.35
LW_OTRAS = 1.6
LW_N = 2.8
COLOR_OTRAS = "skyblue"   # derivadas pasadas
COLOR_N = "pink"          # derivada actual
COLOR_BASE = "black"      # función original
LW_BASE = 2.2
XMIN, XMAX = -8.0, 8.0
RES = 2000

# ---------------- Estado ----------------
current_name = "Sigmoid"
X = np.linspace(XMIN, XMAX, RES)

# ---------------- Figura y ejes ----------------
plt.figure(figsize=(12.5, 7.2), dpi=110)
ax_plot = plt.axes([0.25, 0.25, 0.70, 0.70])
ax_plot.grid(True, alpha=0.25)

# Slider grande (solo n, desde 1)
axcolor = 'lightgoldenrodyellow'
ax_n = plt.axes([0.25, 0.15, 0.50, 0.06], facecolor=axcolor)
s_n = Slider(ax_n, 'n', 1, N_MAX, valinit=1, valstep=1)

# Botones: Aplicar y Reset
ax_apply = plt.axes([0.78, 0.15, 0.08, 0.06])
b_apply  = Button(ax_apply, 'Aplicar')
ax_reset = plt.axes([0.88, 0.15, 0.07, 0.06])
b_reset  = Button(ax_reset, 'Reset')

# RadioButtons para elegir activación
ax_radio = plt.axes([0.05, 0.35, 0.16, 0.55], facecolor='whitesmoke')
radio = RadioButtons(ax_radio, list(ACTIVATIONS.keys()),
                     active=list(ACTIVATIONS.keys()).index(current_name))

# ---------------- Lógica de dibujo ----------------
def draw_activation_derivs(name: str, n: int):
    ax_plot.clear()
    ax_plot.grid(True, alpha=0.25)

    # 0) Función original en negro
    base_expr = ACTIVATIONS[name]
    f_base = sp.lambdify(x, base_expr, "numpy")
    y_base = f_base(X)
    ax_plot.plot(X, y_base, linewidth=LW_BASE, color=COLOR_BASE, label="función original")

    # 1..n-1 en skyblue
    funcs = funcs_numeric(name, n)
    for k, expr, f in funcs:
        if k == 0 or k == n:
            continue
        y = f(X)
        ax_plot.plot(X, y, linewidth=LW_OTRAS, alpha=ALPHA_OTRAS, color=COLOR_OTRAS,
                     label=f"{k}ª derivada")

    # n en rosa
    y_n = funcs[n][2](X)
    ax_plot.plot(X, y_n, linewidth=LW_N, alpha=1.0, color=COLOR_N,
                 label=f"{n}ª derivada", zorder=5)

    ax_plot.axhline(0, linewidth=0.9, linestyle="--", color='k', alpha=0.7)
    ax_plot.set_xlim(XMIN, XMAX)
    ax_plot.set_xlabel("x")
    ax_plot.set_ylabel("y")
    ax_plot.set_title(f"{name}: función y derivadas 1..{n}")
    ax_plot.legend(loc="upper right", frameon=False)
    plt.draw()

def apply_procedure(_=None):
    global current_name
    current_name = radio.value_selected
    n = int(s_n.val)
    draw_activation_derivs(current_name, n)

def reset_all(_=None):
    s_n.reset()
    radio.set_active(list(ACTIVATIONS.keys()).index("Sigmoid"))
    apply_procedure()

def on_n_change(_):
    apply_procedure()

# ---------------- Callbacks ----------------
s_n.on_changed(on_n_change)
b_apply.on_clicked(apply_procedure)
b_reset.on_clicked(reset_all)
radio.on_clicked(lambda _: None)  # se aplica al presionar "Aplicar"

# ---------------- Render inicial ----------------
apply_procedure()
plt.show()
