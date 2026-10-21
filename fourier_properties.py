import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # EE102A: Fourier Transforms and Filtering

        Explore the properties of the Fourier transform interactively.
        Frequency is measured in **Hz**, and we use the convention

        $$X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j2\pi f t}\, dt,
        \qquad
        x(t) = \int_{-\infty}^{\infty} X(f)\, e^{j2\pi f t}\, df .$$

        Throughout the notebook, the basic test signal is a **rectangular pulse** of width $T$:

        $$x(t) = \mathrm{rect}\!\left(\frac{t}{T}\right) =
        \begin{cases} 1 & |t| \le T/2 \\ 0 & \text{otherwise} \end{cases}
        \quad\Longleftrightarrow\quad
        X(f) = T\,\mathrm{sinc}(fT), \quad \mathrm{sinc}(u)=\frac{\sin(\pi u)}{\pi u}.$$

        **How to use this page:** read the definition, move the sliders, and compare
        what you see against the *"What to observe"* box at the end of each section.

        Sections: 1. Time-shift · 2. Frequency-shift · 3. Scaling · 4. Filtering · 5. 2D transform
        """
    )
    return


@app.cell
def _(np):
    def rect(t, T):
        """Rectangular pulse of width T centred at 0."""
        return (np.abs(t) <= T / 2).astype(float)

    def rect_ft(f, T):
        """Fourier transform of rect(t/T):  T * sinc(f T)."""
        return T * np.sinc(f * T)

    return rect, rect_ft


# ---------------------------------------------------------------------------
# 1. Time-shift theorem
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Time-shift theorem

        **Definition.** Delaying a signal by $t_0$ seconds does not change the *magnitude*
        of its spectrum. It only adds a phase that is linear in frequency:

        $$x(t - t_0) \;\Longleftrightarrow\; e^{-j2\pi f t_0}\, X(f)$$

        **Why?** Substitute $\tau = t - t_0$ in the transform integral:

        $$\int x(t-t_0)\,e^{-j2\pi f t}\,dt
        = e^{-j2\pi f t_0}\int x(\tau)\,e^{-j2\pi f \tau}\,d\tau
        = e^{-j2\pi f t_0}X(f).$$

        Equivalently, $|X_{\text{shifted}}(f)| = |X(f)|$ and
        $\angle X_{\text{shifted}}(f) = \angle X(f) - 2\pi f t_0$.

        **Try it:** change the pulse width and the time shift.
        """
    )
    return


@app.cell
def _(mo):
    ts_width = mo.ui.slider(0.5, 3.0, step=0.1, value=1.0, label="Pulse width T (s)", show_value=True)
    ts_shift = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.0, label="Time shift t₀ (s)", show_value=True)
    mo.hstack([ts_width, ts_shift], justify="start", gap=2)
    return ts_shift, ts_width


@app.cell
def _(np, plt, rect, rect_ft, ts_shift, ts_width):
    _T = ts_width.value
    _t0 = ts_shift.value
    _t = np.linspace(-6, 6, 3000)
    _f = np.linspace(-4, 4, 3000)

    _X = rect_ft(_f, _T)
    _Xs = _X * np.exp(-2j * np.pi * _f * _t0)
    _show = np.abs(_X) > 0.02 * _T  # only show phase where the spectrum is not ~0

    _fig, _ax = plt.subplots(1, 3, figsize=(14, 3.8))

    _ax[0].plot(_t, rect(_t, _T), color="gray", label=r"$x(t)$")
    _ax[0].plot(_t, rect(_t - _t0, _T), "--", color="C0", label=r"$x(t-t_0)$")
    _ax[0].set_title("Time domain")
    _ax[0].set_xlabel("t (s)")
    _ax[0].set_ylim(-0.1, 1.3)
    _ax[0].legend(loc="upper right")

    _ax[1].plot(_f, np.abs(_X), color="gray", linewidth=4, alpha=0.5, label=r"$|X(f)|$")
    _ax[1].plot(_f, np.abs(_Xs), "--", color="C0", label=r"$|X_{shifted}(f)|$")
    _ax[1].set_title("Magnitude spectrum")
    _ax[1].set_xlabel("f (Hz)")
    _ax[1].set_ylim(-0.1, 3.2)
    _ax[1].legend(loc="upper right")

    _ax[2].plot(_f[_show], np.angle(_X)[_show], ".", color="gray", markersize=3, label=r"$\angle X(f)$")
    _ax[2].plot(_f[_show], np.angle(_Xs)[_show], ".", color="C0", markersize=3, label=r"$\angle X_{shifted}(f)$")
    _ax[2].set_title("Phase spectrum (radians)")
    _ax[2].set_xlabel("f (Hz)")
    _ax[2].set_ylim(-3.6, 3.6)
    _ax[2].legend(loc="upper right")

    for _a in _ax:
        _a.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What to observe**

            - **Time domain:** the dashed pulse is just the gray pulse moved by $t_0$ (same shape, same width).
            - **Magnitude:** the dashed and gray curves lie *exactly on top of each other* for every $t_0$.
              A time shift never changes how much of each frequency is present.
            - **Phase:** the gray phase is only 0 or $\pm\pi$ (sign changes of the sinc). The blue phase is that
              plus a linear-in-$f$ term $-2\pi f t_0$. Increase $|t_0|$ and the phase "winds" faster, so
              the slope is proportional to the delay. Flip the sign of $t_0$ and the slope flips.
            - Set $t_0 = 0$ and the two phase curves coincide.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 2. Frequency-shift (modulation) theorem
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Frequency-shift (modulation) theorem

        **Definition.** Multiplying by a complex exponential in time shifts the spectrum in frequency
        (the dual of the time-shift theorem):

        $$x(t)\,e^{j2\pi f_0 t} \;\Longleftrightarrow\; X(f - f_0)$$

        For a real signal we usually modulate with a cosine, which gives *two* shifted copies:

        $$x(t)\cos(2\pi f_0 t) \;\Longleftrightarrow\; \tfrac12\big[X(f-f_0) + X(f+f_0)\big].$$

        Below we plot the complex signal $x(t)e^{j2\pi f_0 t}$ (real and imaginary parts) and its spectrum.
        """
    )
    return


@app.cell
def _(mo):
    fs_width = mo.ui.slider(0.5, 3.0, step=0.1, value=1.0, label="Pulse width T (s)", show_value=True)
    fs_shift = mo.ui.slider(-6.0, 6.0, step=0.25, value=3.0, label="Frequency shift f₀ (Hz)", show_value=True)
    mo.hstack([fs_width, fs_shift], justify="start", gap=2)
    return fs_shift, fs_width


@app.cell
def _(fs_shift, fs_width, np, plt, rect, rect_ft):
    _T = fs_width.value
    _f0 = fs_shift.value
    _t = np.linspace(-3, 3, 4000)
    _f = np.linspace(-10, 10, 4000)

    _x = rect(_t, _T) * np.exp(2j * np.pi * _f0 * _t)

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))

    _ax[0].plot(_t, _x.real, color="C0", linewidth=1, label=r"Re$\{x(t)e^{j2\pi f_0 t}\}$")
    _ax[0].plot(_t, _x.imag, color="C3", linewidth=1, alpha=0.7, label=r"Im$\{x(t)e^{j2\pi f_0 t}\}$")
    _ax[0].plot(_t, rect(_t, _T), "k--", linewidth=1, label="envelope")
    _ax[0].plot(_t, -rect(_t, _T), "k--", linewidth=1)
    _ax[0].set_title("Time domain")
    _ax[0].set_xlabel("t (s)")
    _ax[0].set_ylim(-1.5, 1.7)
    _ax[0].legend(loc="upper right", fontsize=8)

    _ax[1].plot(_f, np.abs(rect_ft(_f, _T)), color="gray", linewidth=4, alpha=0.5, label=r"$|X(f)|$")
    _ax[1].plot(_f, np.abs(rect_ft(_f - _f0, _T)), color="C0", label=r"$|X(f-f_0)|$")
    _ax[1].axvline(_f0, color="C0", linestyle=":", alpha=0.7)
    _ax[1].set_title("Magnitude spectrum")
    _ax[1].set_xlabel("f (Hz)")
    _ax[1].set_ylim(-0.1, 3.2)
    _ax[1].legend(loc="upper right")

    for _a in _ax:
        _a.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What to observe**

            - **Spectrum:** the blue spectrum is the gray one *picked up and moved* so that its center sits at
              $f = f_0$ (dotted line). Its shape and height do not change.
            - **Time domain:** the envelope (dashed) is unchanged. Only the oscillation *inside* the pulse changes.
              Larger $|f_0|$ gives faster oscillation: about $f_0 T$ cycles fit inside the pulse.
            - The real and imaginary parts are a cosine and a sine at $f_0$. Changing the sign of $f_0$ swaps
              which one leads, and moves the spectrum to negative frequencies.
            - Compare with Section 1: a shift in **time** gives a **phase** change, while a shift in
              **frequency** gives a **modulation** in time. The two properties are duals.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 3. Time scaling
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Scaling (pulse width) property

        **Definition.** Compressing a signal in time stretches its spectrum in frequency, and vice versa:

        $$x(at) \;\Longleftrightarrow\; \frac{1}{|a|}\,X\!\left(\frac{f}{a}\right)$$

        Here the reference pulse is $\mathrm{rect}(t)$ (width 1), whose transform is $\mathrm{sinc}(f)$.
        Choosing pulse width $T = 1/a$ gives

        $$\mathrm{rect}(t/T) \;\Longleftrightarrow\; T\,\mathrm{sinc}(fT).$$

        The first zero of the spectrum is at $f = 1/T$, and the peak value is $X(0) = T$ (the area under the pulse).
        """
    )
    return


@app.cell
def _(mo):
    sc_width = mo.ui.slider(0.25, 4.0, step=0.05, value=2.0, label="Pulse width T (s)", show_value=True)
    sc_width
    return (sc_width,)


@app.cell
def _(mo, np, plt, rect, rect_ft, sc_width):
    _T = sc_width.value
    _t = np.linspace(-4, 4, 3000)
    _f = np.linspace(-6, 6, 3000)

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))

    _ax[0].plot(_t, rect(_t, 1.0), "--", color="gray", label="reference: width 1")
    _ax[0].plot(_t, rect(_t, _T), color="C0", label=f"width T = {_T:.2f}")
    _ax[0].set_title("Time domain")
    _ax[0].set_xlabel("t (s)")
    _ax[0].set_ylim(-0.1, 1.3)
    _ax[0].legend(loc="upper right")

    _ax[1].plot(_f, rect_ft(_f, 1.0), "--", color="gray", label="reference spectrum")
    _ax[1].plot(_f, rect_ft(_f, _T), color="C0", label=r"$T\,\mathrm{sinc}(fT)$")
    _ax[1].axvline(1 / _T, color="C0", linestyle=":", alpha=0.7)
    _ax[1].axvline(-1 / _T, color="C0", linestyle=":", alpha=0.7)
    _ax[1].set_title("Spectrum X(f)  (dotted = first zeros at ±1/T)")
    _ax[1].set_xlabel("f (Hz)")
    _ax[1].set_ylim(-1.2, 4.3)
    _ax[1].legend(loc="upper right")

    for _a in _ax:
        _a.grid(alpha=0.3)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**T = {_T:.2f} s** &nbsp;→&nbsp; first zero at 1/T = **{1/_T:.2f} Hz**, "
                f"peak X(0) = T = **{_T:.2f}**, &nbsp; (T × first-zero frequency = 1)"
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What to observe**

            - **Narrow pulse → wide spectrum.** Decrease $T$: the main lobe of the sinc gets wider
              (first zero at $1/T$ moves outward) but its peak gets *lower* ($X(0)=T$).
            - **Wide pulse → narrow spectrum.** Increase $T$: the spectrum squeezes toward $f = 0$ and its peak grows.
            - The product (pulse width) × (main-lobe half-width) is always 1. You cannot make a signal
              short in time *and* narrow in frequency at the same time (the time-bandwidth tradeoff).
            - Check the dynamic readout above the plots: the peak equals the pulse area, and the zero location is $1/T$.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 4. Filtering (convolution <-> multiplication)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Filtering (convolution property)

        **Definition.** Passing a signal through an LTI filter with impulse response $h(t)$ is a
        *convolution* in time, which becomes a simple *multiplication* in frequency:

        $$y(t) = (x * h)(t) \;\Longleftrightarrow\; Y(f) = H(f)\,X(f).$$

        We use **ideal** (brick-wall) filters with cutoff $f_c$:

        $$H_{\text{LP}}(f)=\begin{cases}1 & |f|\le f_c\\ 0 & \text{else}\end{cases}\qquad
        H_{\text{HP}}(f)=\begin{cases}0 & |f|< f_c\\ 1 & \text{else}\end{cases}\qquad
        H_{\text{BP}}(f)=\begin{cases}1 & \big||f|-f_c\big|\le 3\,\text{Hz}\\ 0 & \text{else}\end{cases}$$

        **The test signal** is a 2 Hz square wave (which contains the odd harmonics 2, 6, 10, 14, … Hz)
        plus random noise (which contains *all* frequencies). The spectrum is drawn on a log scale so the
        noise floor is visible.
        """
    )
    return


@app.cell
def _(np):
    # Shared test signal for the filtering demo: 4 seconds sampled at 200 Hz.
    fs_hz = 200.0
    n_samples = 800
    t_sig = np.arange(n_samples) / fs_hz
    _rng = np.random.default_rng(1)
    x_sig = np.sign(np.sin(2 * np.pi * 2.0 * (t_sig + 0.5 / fs_hz))) + 0.5 * _rng.standard_normal(n_samples)
    return fs_hz, n_samples, t_sig, x_sig


@app.cell
def _(mo):
    fl_type = mo.ui.dropdown(["Low-pass", "High-pass", "Band-pass"], value="Low-pass", label="Filter type")
    fl_cutoff = mo.ui.slider(1.0, 50.0, step=0.5, value=10.0, label="Filter cutoff / center f_c (Hz)", show_value=True)
    mo.hstack([fl_type, fl_cutoff], justify="start", gap=2)
    return fl_cutoff, fl_type


@app.cell
def _(fl_cutoff, fl_type, fs_hz, n_samples, np, plt, t_sig, x_sig):
    _fc = fl_cutoff.value
    _kind = fl_type.value

    _X = np.fft.rfft(x_sig)
    _f = np.fft.rfftfreq(n_samples, 1 / fs_hz)

    if _kind == "Low-pass":
        _H = (_f <= _fc).astype(float)
    elif _kind == "High-pass":
        _H = (_f >= _fc).astype(float)
    else:  # Band-pass: passband is f_c +/- 3 Hz
        _H = (np.abs(_f - _fc) <= 3.0).astype(float)

    _Y = _H * _X
    _y = np.fft.irfft(_Y, n=n_samples)

    _amp_x = np.maximum(2 * np.abs(_X) / n_samples, 1e-6)
    _amp_y = np.maximum(2 * np.abs(_Y) / n_samples, 1e-6)

    _fig, _ax = plt.subplots(1, 2, figsize=(13, 4))

    _m = t_sig <= 2.0
    _ax[0].plot(t_sig[_m], x_sig[_m], color="gray", alpha=0.6, linewidth=1, label="input x(t)")
    _ax[0].plot(t_sig[_m], _y[_m], color="C0", linewidth=2, label="output y(t)")
    _ax[0].set_title(f"Time domain ({_kind}, f_c = {_fc:g} Hz)")
    _ax[0].set_xlabel("t (s)")
    _ax[0].set_ylim(-3, 3)
    _ax[0].legend(loc="upper right")

    _ax[1].semilogy(_f, _amp_x, color="gray", alpha=0.6, linewidth=1, label="|X(f)| input")
    _ax[1].semilogy(_f, _amp_y, color="C0", linewidth=1.5, label="|Y(f)| = |H(f)||X(f)| output")
    _ax[1].fill_between(_f, 1e-3, 2.5, where=_H > 0, color="C1", alpha=0.15, label="filter passband H(f)=1")
    _ax[1].set_xlim(0, 60)
    _ax[1].set_ylim(1e-3, 2.5)
    _ax[1].set_title("Spectrum (positive frequencies, log scale)")
    _ax[1].set_xlabel("f (Hz)")
    _ax[1].legend(loc="upper right", fontsize=8)

    for _a in _ax:
        _a.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What to observe**

            - **Spectrum:** the output spectrum is the input spectrum *multiplied* by the orange passband.
              Everything inside is unchanged. Everything outside is removed (it drops to the bottom of the plot).
            - **Low-pass:** with a cutoff around 10 Hz the noise disappears and the output looks like a smoothed
              square wave. Push the cutoff down to ~3 Hz and only the 2 Hz fundamental survives, so the
              output is a sine wave. Raise it to 50 Hz and the output gets closer to the original, noise included.
            - **Ringing:** near the square-wave edges you should see overshoot and ripples. This is the Gibbs phenomenon,
              caused by the sharp (ideal) cutoff.
            - **High-pass:** removes the square wave and keeps mostly noise plus the highest harmonics.
              The output has zero average, and its edges look like short spikes.
            - **Band-pass:** only harmonics within ±3 Hz of $f_c$ survive. Move $f_c$ onto 2, 6, 10, … Hz to
              "pick out" individual harmonics, and between them (e.g. 4 Hz) to see only noise.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 5. 2D Fourier transform
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. The 2D Fourier transform (separability)

        **Definition.** For an image $g(x,y)$ the 2D transform is

        $$G(f_x, f_y) = \iint g(x,y)\, e^{-j2\pi (f_x x + f_y y)}\, dx\, dy .$$

        If the image is **separable**, $g(x,y) = g_1(x)\,g_2(y)$, the 2D transform is just the product
        of two 1D transforms: $G(f_x,f_y) = G_1(f_x)\,G_2(f_y)$. For a rectangle of width $W$ (in $x$) and
        height $H$ (in $y$):

        $$\mathrm{rect}\!\left(\tfrac{x}{W}\right)\mathrm{rect}\!\left(\tfrac{y}{H}\right)
        \;\Longleftrightarrow\; WH\,\mathrm{sinc}(f_x W)\,\mathrm{sinc}(f_y H).$$

        The scaling property from Section 3 now applies **independently along each axis**.
        """
    )
    return


@app.cell
def _(mo):
    td_x = mo.ui.slider(0.5, 4.0, step=0.1, value=2.0, label="Pulse width / x-width W", show_value=True)
    td_y = mo.ui.slider(0.5, 4.0, step=0.1, value=1.0, label="2D y-width H", show_value=True)
    mo.hstack([td_x, td_y], justify="start", gap=2)
    return td_x, td_y


@app.cell
def _(np, plt, rect, td_x, td_y):
    _W = td_x.value
    _H = td_y.value

    _xs = np.linspace(-4, 4, 400)
    _X, _Y = np.meshgrid(_xs, _xs)
    _img = rect(_X, _W) * rect(_Y, _H)

    _fs = np.linspace(-3, 3, 400)
    _FX, _FY = np.meshgrid(_fs, _fs)
    _G = np.abs(np.sinc(_FX * _W) * np.sinc(_FY * _H))  # normalised by W*H

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.6))

    _ax[0].imshow(_img, extent=[-4, 4, -4, 4], origin="lower", cmap="gray", vmin=0, vmax=1)
    _ax[0].set_title(f"Image g(x, y): {_W:.1f} × {_H:.1f}")
    _ax[0].set_xlabel("x")
    _ax[0].set_ylabel("y")

    _im = _ax[1].imshow(_G**0.5, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis", vmin=0, vmax=1)
    _ax[1].set_title(r"$\sqrt{|G(f_x,f_y)|/(WH)}$  (sqrt makes sidelobes visible)")
    _ax[1].set_xlabel(r"$f_x$ (Hz)")
    _ax[1].set_ylabel(r"$f_y$ (Hz)")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What to observe**

            - **Wider in $x$ → narrower in $f_x$.** Increase the x-width and the bright pattern squeezes
              *horizontally*. Increase the y-width and it squeezes *vertically*. The two directions
              are independent.
            - A wide, short rectangle (e.g. W = 4, H = 0.5) produces a tall, thin spectrum pattern: the
              spectrum is elongated in the **opposite** direction from the image.
            - The spectrum is a **grid of lobes**: the bright center, with a cross of bright bars along the
              $f_x$ and $f_y$ axes, and a checkerboard of weaker lobes in between. This
              is the product $\mathrm{sinc}(f_xW)\cdot\mathrm{sinc}(f_yH)$.
            - The dark lines are at $f_x = k/W$ and $f_y = k/H$ (zeros of each sinc). Check that they move
              as you change each slider.
            - Set $W = H$: the image is a square and the spectrum is symmetric under swapping $f_x \leftrightarrow f_y$.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
