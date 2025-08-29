# %% perceptron_tkagg.py
# Requires: numpy, matplotlib, tkinter (tk), python -m pip install numpy matplotlib
# On some Linux distros: sudo apt-get install python3-tk

import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")  # must come before pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# -----------------------------
# Data generation: random plane
# -----------------------------
def make_random_data(n=160, seed=7, box=4.0, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-box, box, size=(n, 2))
    theta = rng.uniform(0, 2*np.pi)
    normal = np.array([np.cos(theta), np.sin(theta)])
    c = rng.uniform(-box, box)
    scores = X @ normal + c
    y = np.where(scores >= 0.0, 1.0, -1.0)
    k = int(noise * n)
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        y[idx] *= -1
    # line a x + b y + c = 0
    return X, y, (normal[0], normal[1], c)

# ---------------------------------
# Perceptron training with history
# ---------------------------------
def perceptron_train(X, y, max_epochs=40, lr=1.0, shuffle=True, seed=7):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    hist = [(w.copy(), b)]
    for _ in range(max_epochs):
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        mistakes = 0
        for i in idx:
            if y[i] * (X[i] @ w + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                hist.append((w.copy(), b))
                mistakes += 1
        if mistakes == 0:
            break
    return w, b, hist

# ---------------------
# Plotting primitives
# ---------------------
def boundary_from_wb(ax, w, b):
    xlim = ax.get_xlim()
    if abs(w[1]) < 1e-12:
        x = -b / (w[0] + 1e-12)
        return np.array([x, x]), np.array(ax.get_ylim())
    xs = np.array(xlim)
    ys = -(w[0]/(w[1]+1e-12))*xs - b/(w[1]+1e-12)
    return xs, ys

def boundary_from_abc(ax, a, b, c):
    xlim = ax.get_xlim()
    if abs(b) < 1e-12:
        x = -c / (a + 1e-12)
        return np.array([x, x]), np.array(ax.get_ylim())
    xs = np.array(xlim)
    ys = -(a/b)*xs - c/b
    return xs, ys

# ---------------------
# Tk App
# ---------------------
class PerceptronApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptron • TkAgg")
        self.geometry("980x640")

        # State
        self.X = None
        self.y = None
        self.sep = None
        self.hist = [(np.zeros(2), 0.0)]
        self.k = 0
        self.playing = False

        # Controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        def add_scale(parent, text, frm, to, init, step, fmt=None):
            lab = ttk.Label(parent, text=text)
            lab.pack(anchor="w")
            var = tk.DoubleVar(value=init)
            s = ttk.Scale(parent, from_=frm, to=to, orient=tk.HORIZONTAL, variable=var)
            s.pack(fill=tk.X, pady=4)
            # discrete step emulation
            def snap(_):
                v = var.get()
                if step > 0:
                    v = round(v / step) * step
                    var.set(v)
            s.bind("<ButtonRelease-1>", snap)
            return var

        self.n_var = add_scale(ctrl, "n (20..600)", 20, 600, 160, 10)
        self.box_var = add_scale(ctrl, "box (1..10)", 1, 10, 4.0, 0.5)
        self.noise_var = add_scale(ctrl, "noise (0..0.4)", 0.0, 0.4, 0.0, 0.02)
        self.lr_var = add_scale(ctrl, "lr (0.05..2.0)", 0.05, 2.0, 1.0, 0.05)
        self.epochs_var = add_scale(ctrl, "epochs (1..200)", 1, 200, 40, 1)
        self.seed_var = add_scale(ctrl, "seed (0..10000)", 0, 10000, 7, 1)

        self.shuffle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="shuffle", variable=self.shuffle_var).pack(anchor="w", pady=6)

        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=4)
        ttk.Button(row1, text="Generate", command=self.on_generate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row1, text="Train", command=self.on_train).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=4)
        ttk.Button(row2, text="Step +1", command=self.on_step).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row2, text="Reset", command=self.on_reset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        row3 = ttk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=4)
        ttk.Button(row3, text="Play", command=self.on_play).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row3, text="Pause", command=self.on_pause).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Update slider
        ttk.Label(ctrl, text="update index").pack(anchor="w", pady=(10,0))
        self.update_var = tk.IntVar(value=0)
        self.update_scale = ttk.Scale(ctrl, from_=0, to=0, orient=tk.HORIZONTAL,
                                      variable=self.update_var, command=self.on_update_slider)
        self.update_scale.pack(fill=tk.X, pady=4)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(ctrl, textvariable=self.status, foreground="#333").pack(anchor="w", pady=8)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7.2, 5.6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.scatter0 = None
        self.scatter1 = None
        self.true_line = None
        self.learn_line = None
        self.txt = None

        # First draw
        self.on_generate()

    # ----- UI callbacks -----
    def on_generate(self):
        n = int(round(self.n_var.get()))
        box = float(self.box_var.get())
        noise = float(self.noise_var.get())
        seed = int(round(self.seed_var.get()))
        self.status.set("Generating data...")
        self.X, self.y, self.sep = make_random_data(n=n, seed=seed, box=box, noise=noise)
        self.hist = [(np.zeros(2), 0.0)]
        self.k = 0
        self.draw_initial()
        self.draw_learn_boundary(*self.hist[0])
        self.update_metrics(*self.hist[0], k=0, K=0)
        self.update_scale.configure(from_=0, to=0)
        self.update_scale.set(0)
        self.status.set("Data ready.")

    def on_train(self):
        if self.X is None:
            self.status.set("Generate data first.")
            return
        self.status.set("Training...")
        _, _, hist = perceptron_train(
            self.X, self.y,
            max_epochs=int(round(self.epochs_var.get())),
            lr=float(self.lr_var.get()),
            shuffle=self.shuffle_var.get(),
            seed=int(round(self.seed_var.get()))
        )
        self.hist = hist
        self.k = 0
        self.update_scale.configure(from_=0, to=len(self.hist)-1)
        self.update_scale.set(0)
        self.draw_learn_boundary(*self.hist[0])
        self.update_metrics(*self.hist[0], k=0, K=len(self.hist)-1)
        self.canvas.draw_idle()
        self.status.set(f"Trained. Updates: {len(self.hist)-1}.")

    def on_step(self):
        if self.hist is None: 
            return
        self.k = min(self.k + 1, len(self.hist)-1)
        self.update_scale.set(self.k)  # triggers slider handler

    def on_reset(self):
        if self.hist is None: 
            return
        self.k = 0
        self.update_scale.set(0)

    def on_play(self):
        self.playing = True
        self._play_loop()

    def on_pause(self):
        self.playing = False

    def _play_loop(self):
        if not self.playing:
            return
        if self.hist and self.k < len(self.hist)-1:
            self.k += 1
            self.update_scale.set(self.k)
            # schedule next frame
            self.after(80, self._play_loop)  # ms per step
        else:
            self.playing = False

    def on_update_slider(self, _=None):
        if not self.hist:
            return
        k = int(round(self.update_var.get()))
        k = max(0, min(k, len(self.hist)-1))
        self.k = k
        w, b = self.hist[k]
        self.draw_learn_boundary(w, b)
        self.update_metrics(w, b, k=k, K=len(self.hist)-1)
        self.canvas.draw_idle()

    # ----- Drawing helpers -----
    def draw_initial(self):
        self.ax.clear()
        X, y = self.X, self.y
        pad = 0.3 * float(self.box_var.get())
        xmin, xmax = X[:,0].min()-pad, X[:,0].max()+pad
        ymin, ymax = X[:,1].min()-pad, X[:,1].max()+pad
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        self.scatter0 = self.ax.scatter(X[y==-1,0], X[y==-1,1], marker="o", label="y=-1")
        self.scatter1 = self.ax.scatter(X[y==+1,0], X[y==+1,1], marker="^", label="y=+1")
        self.ax.legend(loc="upper left")
        self.ax.set_title("Perceptron on random points (TkAgg)")

        # true separator
        a, b, c = self.sep
        lx, ly = boundary_from_abc(self.ax, a, b, c)
        self.true_line, = self.ax.plot(lx, ly, linestyle="--", linewidth=1)

        # learning boundary placeholder
        self.learn_line, = self.ax.plot([], [], linewidth=2)

        self.txt = self.ax.text(0.02, 0.97, "", transform=self.ax.transAxes, va="top")

    def draw_learn_boundary(self, w, b):
        lx, ly = boundary_from_wb(self.ax, w, b)
        self.learn_line.set_data(lx, ly)

    def update_metrics(self, w, b, k, K):
        margins = self.y * (self.X @ w + b)
        acc = float((margins > 0).mean())
        self.txt.set_text(f"update={k}/{K}   ||w||={np.linalg.norm(w):.3f}   acc={acc:.3f}")

if __name__ == "__main__":
    app = PerceptronApp()
    app.mainloop()

# %%
