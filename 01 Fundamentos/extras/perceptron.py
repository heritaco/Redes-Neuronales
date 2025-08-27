# Interactive Perceptron in 2D with matplotlib (click to add points, train online)
# - Left click: class +1
# - Right click: class -1
# - Buttons: Step (one online update), Epoch (full pass), Reset, Random (toy data)
# - Slider: learning rate
#
# A .py file will be saved so you can reuse it outside this notebook.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from dataclasses import dataclass, field
from typing import Tuple

# ------------------------ Core Perceptron ------------------------
@dataclass
class Perceptron2D:
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [b, w1, w2]
    lr: float = 1.0

    def predict_raw(self, X_aug: np.ndarray) -> np.ndarray:
        return X_aug @ self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = self._augment(X)
        return np.sign(self.predict_raw(X_aug))

    def step_update(self, x: np.ndarray, y: int) -> bool:
        """Perform one online update on a single example if it's misclassified.
        Returns True if an update occurred."""
        x_aug = self._augment(x.reshape(1, -1))[0]
        if y * (x_aug @ self.w) <= 0:
            self.w += self.lr * y * x_aug
            return True
        return False

    def epoch(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> int:
        """Run one pass over data; returns number of updates."""
        n = len(X)
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        updates = 0
        for i in idx:
            updates += int(self.step_update(X[i], int(y[i])))
        return updates

    @staticmethod
    def _augment(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])


# ------------------------ Interactive Plot ------------------------
class InteractivePerceptron:
    def __init__(self):
        self.X = np.empty((0, 2))
        self.y = np.empty((0,), dtype=int)
        self.model = Perceptron2D(lr=1.0)

        # Figure and axes
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title("Perceptrón Interactivo (click para añadir puntos)")
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_title("Izq: +1 | Der: -1 — Botones: Step, Epoch, Reset, Random")

        # Place UI controls
        plt.subplots_adjust(bottom=0.26)
        ax_step   = plt.axes([0.10, 0.15, 0.12, 0.06])
        ax_epoch  = plt.axes([0.24, 0.15, 0.12, 0.06])
        ax_reset  = plt.axes([0.38, 0.15, 0.12, 0.06])
        ax_random = plt.axes([0.52, 0.15, 0.12, 0.06])
        ax_lr     = plt.axes([0.10, 0.06, 0.54, 0.03])

        self.btn_step = Button(ax_step, "Step")
        self.btn_epoch = Button(ax_epoch, "Epoch")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_random = Button(ax_random, "Random")
        self.slider_lr = Slider(ax_lr, "lr", 0.01, 5.0, valinit=self.model.lr, valstep=0.01)

        # Event bindings
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.btn_step.on_clicked(self.on_step)
        self.btn_epoch.on_clicked(self.on_epoch)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_random.on_clicked(self.on_random)
        self.slider_lr.on_changed(self.on_lr_change)

        # Plot holders
        self.sc_pos = None
        self.sc_neg = None
        self.db_line = None
        self.metrics_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top")

        # Limits
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

        self.redraw()

    # --------------- UI Handlers ---------------
    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        # 1: left button -> +1, 3: right button -> -1
        label = +1 if event.button == 1 else -1 if event.button == 3 else None
        if label is None:
            return
        point = np.array([[event.xdata, event.ydata]])
        self.X = np.vstack([self.X, point])
        self.y = np.append(self.y, label)

        # Online update immediately on new point (optional: comment to disable)
        self.model.step_update(point[0], label)
        self.redraw()

    def on_step(self, _):
        if len(self.X) == 0:
            return
        # pick a random sample to try one update
        i = np.random.randint(len(self.X))
        self.model.step_update(self.X[i], int(self.y[i]))
        self.redraw()

    def on_epoch(self, _):
        if len(self.X) == 0:
            return
        updates = self.model.epoch(self.X, self.y, shuffle=True)
        self.ax.set_title(f"Epoch complete — updates: {updates}")
        self.redraw()

    def on_reset(self, _):
        self.X = np.empty((0, 2))
        self.y = np.empty((0,), dtype=int)
        self.model = Perceptron2D(lr=self.slider_lr.val)
        self.ax.set_title("Izq: +1 | Der: -1 — Botones: Step, Epoch, Reset, Random")
        self.redraw()

    def on_random(self, _):
        rng = np.random.default_rng()
        n = 40
        m1 = np.array([ -1.5,  -1.0])
        m2 = np.array([  1.5,   1.0])
        C = np.array([[0.6, 0.0],[0.0, 0.6]])
        X_pos = rng.multivariate_normal(m2, C, n//2)
        X_neg = rng.multivariate_normal(m1, C, n//2)
        y_pos = np.ones(n//2, dtype=int)
        y_neg = -np.ones(n//2, dtype=int)
        self.X = np.vstack([X_pos, X_neg])
        self.y = np.concatenate([y_pos, y_neg])
        self.model = Perceptron2D(lr=self.slider_lr.val)
        self.redraw()

    def on_lr_change(self, val):
        self.model.lr = float(val)

    # --------------- Drawing ---------------
    def redraw(self):
        self.ax.cla()
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

        # Scatter points by class (no explicit colors set)
        if len(self.X) > 0:
            X_pos = self.X[self.y == +1]
            X_neg = self.X[self.y == -1]
            if len(X_pos) > 0:
                self.sc_pos = self.ax.scatter(X_pos[:, 0], X_pos[:, 1], marker="o", label="+1")
            if len(X_neg) > 0:
                self.sc_neg = self.ax.scatter(X_neg[:, 0], X_neg[:, 1], marker="x", label="-1")

        # Decision boundary: w0 + w1 x + w2 y = 0  =>  y = -(w0 + w1 x)/w2
        w0, w1, w2 = self.model.w
        if not np.allclose([w1, w2], 0):
            xs = np.linspace(-5, 5, 200)
            if abs(w2) > 1e-9:
                ys = -(w0 + w1 * xs) / w2
                self.db_line, = self.ax.plot(xs, ys, linestyle="-", linewidth=2)
            else:
                # Vertical line x = -w0 / w1
                if abs(w1) > 1e-9:
                    x_vert = -w0 / w1
                    self.db_line = self.ax.axvline(x_vert, linestyle="-", linewidth=2)

        # Metrics
        acc, mistakes = self.metrics()
        self.metrics_text = self.ax.text(
            0.02, 0.98,
            f"w = [{w0:.2f}, {w1:.2f}, {w2:.2f}] | lr = {self.model.lr:.2f}\n"
            f"n = {len(self.X)} | acc = {acc:.2%} | mistakes (last epoch) ≈ {mistakes}",
            transform=self.ax.transAxes, va="top"
        )
        self.ax.legend(loc="lower right")
        self.fig.canvas.draw_idle()

    def metrics(self) -> Tuple[float, int]:
        if len(self.X) == 0:
            return 0.0, 0
        X_aug = Perceptron2D._augment(self.X)
        yhat = np.sign(X_aug @ self.model.w)
        # Treat zeros (on boundary) as mistakes
        yhat[yhat == 0] = -1
        acc = (yhat == self.y).mean()
        mistakes = int((yhat != self.y).sum())
        return acc, mistakes


# Instantiate the app
app = InteractivePerceptron()
plt.show()

