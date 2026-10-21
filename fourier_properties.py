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
        # EE102A: Fourier Transform Properties

        Explore the properties of the Fourier transform interactively.
        Frequency is measured in **Hz**, and we use the convention

        $$X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j2\pi f t}\, dt,
        \qquad
        x(t) = \int_{-\infty}^{\infty} X(f)\, e^{j2\pi f t}\, df .$$

        The main test signal is a **rectangular pulse** of width $T$:

        $$x(t) = \mathrm{rect}\!\left(\frac{t}{T}\right) =
        \begin{cases} 1 & |t| \le T/2 \\ 0 & \text{otherwise} \end{cases}
        \quad\Longleftrightarrow\quad
        X(f) = T\,\mathrm{sinc}(fT), \quad \mathrm{sinc}(u)=\frac{\sin(\pi u)}{\pi u}.$$

        **How to use this page:** read the definition, move the sliders, and compare
        what you see against the *"What to observe"* box at the end of each section.

        Sections: 1. Addition (linearity) · 2. Scaling · 3. Time shift · 4. Frequency shift · 5. Rayleigh's theorem
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

    def tri(t, W):
        """Triangular pulse with peak 1 and base from -W to W."""
        return np.clip(1 - np.abs(t) / W, 0, None)

    def tri_ft(f, W):
        """Fourier transform of tri(t, W):  W * sinc^2(f W)."""
        return W * np.sinc(f * W) ** 2

    def integrate(y, x):
        """Trapezoid-rule integral of y over x."""
        return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)

    return integrate, rect, rect_ft, tri, tri_ft


# ---------------------------------------------------------------------------
# 1. Addition (linearity)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Addition (linearity)

        **Definition.** The Fourier transform of a weighted sum of signals is the same weighted sum
        of their transforms:

        $$a\,x_1(t) + b\,x_2(t) \;\Longleftrightarrow\; a\,X_1(f) + b\,X_2(f)$$

        **Why?** The transform is an integral, and integrals are linear:

        $$\int \big[a x_1(t) + b x_2(t)\big]e^{-j2\pi f t}dt
        = a\!\int x_1(t)e^{-j2\pi f t}dt + b\!\int x_2(t)e^{-j2\pi f t}dt.$$

        **Signals used here:**

        - $x_1(t) = \mathrm{rect}(t)$ (width 1), with $X_1(f) = \mathrm{sinc}(f)$
        - $x_2(t) = \mathrm{tri}(t)$ (triangle, base from $-1$ to $1$, peak 1), with $X_2(f) = \mathrm{sinc}^2(f)$

        The dots in the right-hand plot are a **direct numerical** Fourier transform of the sum $y(t)$,
        computed without using the formula, so you can check the property yourself.
        """
    )
    return


@app.cell
def _(mo):
    ad_a = mo.ui.slider(-2.0, 2.0, step=0.1, value=1.0, label="Coefficient a (on rectangle)", show_value=True)
    ad_b = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.5, label="Coefficient b (on triangle)", show_value=True)
    mo.hstack([ad_a, ad_b], justify="start", gap=2)
    return ad_a, ad_b


@app.cell
def _(ad_a, ad_b, np, plt, rect, rect_ft, tri, tri_ft):
    _a = ad_a.value
    _b = ad_b.value
    _t = np.linspace(-4, 4, 4000)
    _f = np.linspace(-4, 4, 500)
    _dt = _t[1] - _t[0]

    _x1 = rect(_t, 1.0)
    _x2 = tri(_t, 1.0)
    _y = _a * _x1 + _b * _x2

    _X1 = rect_ft(_f, 1.0)
    _X2 = tri_ft(_f, 1.0)
    _Y = _a * _X1 + _b * _X2

    # Direct numerical transform of y(t) (y is even, so the transform is real: integral of y*cos)
    _Ynum = (_y[None, :] * np.cos(2 * np.pi * _f[:, None] * _t[None, :])).sum(axis=1) * _dt

    _fig, _ax = plt.subplots(1, 2, figsize=(13, 4))

    _ax[0].plot(_t, _a * _x1, color="C0", alpha=0.7, label=r"$a\,x_1(t)$")
    _ax[0].plot(_t, _b * _x2, color="C1", alpha=0.7, label=r"$b\,x_2(t)$")
    _ax[0].plot(_t, _y, color="k", linewidth=2, label=r"$y(t)=a\,x_1+b\,x_2$")
    _ax[0].set_title("Time domain")
    _ax[0].set_xlabel("t (s)")
    _ax[0].set_ylim(-4.3, 4.3)
    _ax[0].legend(loc="upper right", fontsize=8)

    _ax[1].plot(_f, _a * _X1, color="C0", alpha=0.7, label=r"$a\,X_1(f)$")
    _ax[1].plot(_f, _b * _X2, color="C1", alpha=0.7, label=r"$b\,X_2(f)$")
    _ax[1].plot(_f, _Y, color="k", linewidth=2, label=r"$a\,X_1+b\,X_2$")
    _ax[1].plot(_f[::6], _Ynum[::6], "o", color="C3", markersize=4, label="numerical FT of y(t)")
    _ax[1].set_title("Frequency domain (both transforms are real here)")
    _ax[1].set_xlabel("f (Hz)")
    _ax[1].set_ylim(-4.3, 4.3)
    _ax[1].legend(loc="upper right", fontsize=8)

    for _a_ in _ax:
        _a_.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What to observe**

            - The red dots (a direct transform of the summed signal) always land on the black curve
              ($aX_1 + bX_2$), for every choice of $a$ and $b$. This is linearity.
            - **Scaling by a constant:** doubling $a$ doubles the entire blue spectrum, but its *shape* does not change.
              A negative coefficient flips the spectrum upside down.
            - **Setting $a=0$** leaves only the triangle's spectrum, and **setting $b=0$** leaves only the rectangle's.
            - **Cancellation:** the rectangle and triangle both have area 1, so $X_1(0) = X_2(0) = 1$.
              With $a = 1$, $b = -1$ the sum has zero area, so the combined spectrum is **0 at $f = 0$**.
            - The sum's spectrum is built point by point: at each frequency, add the two heights. Notice the
              triangle's spectrum ($\mathrm{sinc}^2$) dies out faster than the rectangle's, so at high frequencies
              the sum looks like just the rectangle's contribution.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 2. Scaling
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Scaling (pulse width)

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
            - Check the readout above the plots: the peak equals the pulse area, and the zero location is $1/T$.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 3. Time shift
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Time shift

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
# 4. Frequency shift
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Frequency shift (modulation)

        **Definition.** Multiplying by a complex exponential in time shifts the spectrum in frequency
        (the dual of the time-shift property):

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
            - Compare with Section 3: a shift in **time** gives a **phase** change, while a shift in
              **frequency** gives a **modulation** in time. The two properties are duals.
            """
        ),
        kind="info",
    )
    return


# ---------------------------------------------------------------------------
# 5. Rayleigh's theorem
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Rayleigh's energy theorem

        **Definition.** The total energy of a signal can be computed in either domain, and the answer is the same:

        $$\int_{-\infty}^{\infty} |x(t)|^2\,dt \;=\; \int_{-\infty}^{\infty} |X(f)|^2\,df .$$

        $|x(t)|^2$ is the *energy density in time*, and $|X(f)|^2$ is the *energy spectral density*
        (energy per Hz). For the rectangular pulse $\mathrm{rect}(t/T)$:

        $$\text{time domain: } \int |x(t)|^2dt = T,
        \qquad
        \text{frequency domain: } \int T^2\mathrm{sinc}^2(fT)\,df = T.$$

        The theorem also lets us ask *where in frequency* the energy lives. The energy within
        $|f| \le B$ is $\int_{-B}^{B}|X(f)|^2\,df$.
        """
    )
    return


@app.cell
def _(mo):
    ry_width = mo.ui.slider(0.25, 4.0, step=0.05, value=1.0, label="Pulse width T (s)", show_value=True)
    ry_band = mo.ui.slider(0.1, 6.0, step=0.1, value=1.0, label="Band limit B (Hz)", show_value=True)
    mo.hstack([ry_width, ry_band], justify="start", gap=2)
    return ry_band, ry_width


@app.cell
def _(integrate, mo, np, plt, rect, rect_ft, ry_band, ry_width):
    _T = ry_width.value
    _B = ry_band.value

    _t = np.linspace(-5, 5, 4000)
    _f = np.linspace(-8, 8, 4000)

    # Energy computed numerically in each domain
    _tt = np.linspace(-10, 10, 200001)
    _E_time = integrate(rect(_tt, _T) ** 2, _tt)

    _ff = np.linspace(-500, 500, 400001)
    _E_freq = integrate(rect_ft(_ff, _T) ** 2, _ff)

    _fb = np.linspace(-_B, _B, 20001)
    _E_band = integrate(rect_ft(_fb, _T) ** 2, _fb)

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))

    _ax[0].fill_between(_t, rect(_t, _T) ** 2, color="C0", alpha=0.4)
    _ax[0].plot(_t, rect(_t, _T) ** 2, color="C0")
    _ax[0].set_title(rf"$|x(t)|^2$: area = {_E_time:.3f}")
    _ax[0].set_xlabel("t (s)")
    _ax[0].set_ylim(-0.1, 1.3)

    _ax[1].plot(_f, rect_ft(_f, _T) ** 2, color="C3")
    _ax[1].fill_between(_f, rect_ft(_f, _T) ** 2, where=np.abs(_f) <= _B, color="C3", alpha=0.4)
    _ax[1].set_title(rf"$|X(f)|^2$: shaded = energy in $|f|\leq B$ = {_E_band:.3f}")
    _ax[1].set_xlabel("f (Hz)")
    _ax[1].set_ylim(-0.5, 16.5)

    for _a in _ax:
        _a.grid(alpha=0.3)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**Energy from time domain:** {_E_time:.3f} &nbsp;|&nbsp; "
                f"**Energy from frequency domain (all f):** {_E_freq:.3f} &nbsp;|&nbsp; "
                f"**Fraction of energy inside |f| ≤ {_B:.1f} Hz:** {100*_E_band/_E_freq:.1f}%"
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

            - **Rayleigh's theorem:** the two energy numbers above the plots always match
              (both equal $T$), no matter how you change the pulse width. Small differences in the third decimal
              place come from numerically integrating a finite range.
            - **Scaling connection:** as $T$ gets small, the pulse's energy $T$ gets small, and the
              spectrum $|X(f)|^2$ gets *lower* but *wider*. As $T$ grows, $|X(f)|^2$ becomes a tall, narrow spike
              at $f=0$. The area under the red curve tracks the area in the time domain either way.
            - **Where the energy lives:** for $T = 1$, the main lobe ($|f| \le 1$ Hz) already contains about 90%
              of the energy. Increase $B$ and watch the percentage climb toward 100%. The sidelobes
              carry very little energy each.
            - Make the pulse narrower and the fraction inside a fixed $B$ drops: a short pulse needs a larger bandwidth
              to capture the same share of its energy.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
