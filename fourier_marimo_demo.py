"""EE102A interactive Fourier lecture demos.
Run: pip install marimo numpy matplotlib && marimo run fourier_marimo_demo.py
"""
import marimo

app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt

@app.cell
def _(mo):
    mo.md(r"""# EE102A: Fourier transforms and filtering
    Explore the properties interactively. Frequency uses **Hz**, with
    $X(f)=\int x(t)e^{-j2\pi ft}\,dt$.""")
    return

@app.cell
def _(mo):
    concept = mo.ui.dropdown(options=["Rectangle ↔ sinc", "Time-shift theorem", "Frequency-shift theorem", "Filtering", "2D separability"], value="Rectangle ↔ sinc", label="Demonstration")
    width = mo.ui.slider(0.2, 4, step=0.1, value=1, label="Pulse width / x-width")
    shift = mo.ui.slider(-3, 3, step=0.1, value=1, label="Time shift (s) / frequency shift (Hz)")
    cutoff = mo.ui.slider(0.5, 9, step=0.5, value=3, label="Filter cutoff (Hz)")
    kind = mo.ui.dropdown(options=["Low-pass", "High-pass", "Band-pass"], value="Low-pass", label="Filter type")
    ywidth = mo.ui.slider(0.2, 4, step=0.1, value=1, label="2D y-width")
    mo.vstack([concept, mo.hstack([width, shift, cutoff]), mo.hstack([kind, ywidth])])
    return concept, width, shift, cutoff, kind, ywidth

@app.cell
def _(np):
    N, dt = 4096, 0.01
    t = (np.arange(N) - N // 2) * dt
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    def ft(y):
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y))) * dt
    def ift(Y):
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Y))) / dt
    def rect(z):
        return (np.abs(z) < 0.5).astype(float)
    return t, f, ft, ift, rect

@app.cell
def _(concept, width, shift, cutoff, kind, ywidth, np, plt, mo, t, f, ft, ift, rect):
    choice = concept.value
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    a1, a2, a3, a4 = axes.flat
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    note = ""
    if choice == "Rectangle ↔ sinc":
        w = width.value
        y = rect(t / w)
        Y = ft(y)
        a1.plot(t, y); a1.set_xlim(-4, 4); a1.set_title("rect(t / width)")
        a2.plot(f, Y.real, label="Numerical"); a2.plot(f, w*np.sinc(w*f), '--', label="Analytical")
        a2.set_xlim(-8, 8); a2.legend(); a2.set_title("Signed Fourier transform")
        a3.plot(f, abs(Y)); a3.set_xlim(-8, 8); a3.set_title("Magnitude")
        a4.plot(f, np.angle(Y)); a4.set_xlim(-8, 8); a4.set_title("Wrapped phase (unstable near zeros)")
        note = r"$\mathrm{rect}(t/w)\leftrightarrow w\,\mathrm{sinc}(wf)$. Increasing pulse width narrows the spectrum."
    elif choice == "Time-shift theorem":
        d = shift.value
        original = rect(t)
        shifted = rect(t-d)
        X, Y = ft(original), ft(shifted)
        a1.plot(t, original, '--', label="Original"); a1.plot(t, shifted, label="Shifted")
        a1.set_xlim(-5, 5); a1.legend(); a1.set_title("Time signals")
        a2.plot(f, abs(X), '--', label="Original"); a2.plot(f, abs(Y), label="Shifted")
        a2.set_xlim(-5, 5); a2.legend(); a2.set_title("Magnitude unchanged")
        mask = (abs(X) > 0.08) & (abs(f) < 5)
        a3.scatter(f[mask], np.angle(Y[mask]/X[mask]), s=5); a3.set_xlim(-5, 5); a3.set_title("Measured relative phase (wrapped)")
        a4.plot(f, -2*np.pi*f*d); a4.set_xlim(-3, 3); a4.set_title("Expected unwrapped phase")
        note = r"$x(t-t_0)\leftrightarrow e^{-j2\pi ft_0}X(f)$. Time shifting changes phase, not magnitude."
    elif choice == "Frequency-shift theorem":
        d = shift.value
        original = rect(t)
        modulated = original*np.exp(2j*np.pi*d*t)
        X, Y = ft(original), ft(modulated)
        a1.plot(t, original, '--', label="Envelope"); a1.plot(t, modulated.real, label="Real part")
        a1.set_xlim(-2, 2); a1.legend(); a1.set_title("Modulated signal")
        a2.plot(f, abs(X), '--', label="Original"); a2.plot(f, abs(Y), label="Shifted")
        a2.set_xlim(-8, 8); a2.legend(); a2.set_title("Spectrum shifts by f₀")
        a3.plot(t, modulated.real, label="Real"); a3.plot(t, modulated.imag, label="Imaginary")
        a3.set_xlim(-2, 2); a3.legend(); a3.set_title("Complex modulation")
        a4.plot(f, abs(Y), label="Numerical"); a4.plot(f, np.sinc(f-d), '--', label="Predicted magnitude")
        a4.set_xlim(-8, 8); a4.legend(); a4.set_title("Comparison to X(f − f₀)")
        note = r"$x(t)e^{j2\pi f_0t}\leftrightarrow X(f-f_0)$. Multiplying by a complex sinusoid shifts the spectrum."
    elif choice == "Filtering":
        y = np.sin(2*np.pi*t) + 0.5*np.sin(2*np.pi*6*t)
        Y = ft(y)
        c = cutoff.value
        if kind.value == "Low-pass": H = (abs(f) <= c).astype(float)
        elif kind.value == "High-pass": H = (abs(f) >= c).astype(float)
        else: H = ((abs(f) >= c) & (abs(f) <= c+3)).astype(float)
        output = ift(Y*H).real
        a1.plot(t, y); a1.set_xlim(-2, 2); a1.set_title("Original: 1 Hz + 6 Hz")
        a2.plot(f, abs(Y), label="Before"); a2.plot(f, abs(Y*H), '--', label="After")
        a2.set_xlim(-10, 10); a2.legend(); a2.set_title("Magnitude spectra")
        a3.plot(f, H); a3.set_xlim(-10, 10); a3.set_ylim(-0.1, 1.1); a3.set_title(f"{kind.value} H(f)")
        a4.plot(t, output); a4.set_xlim(-2, 2); a4.set_title("Filtered signal")
        note = r"Filtering in frequency: $Y(f)=H(f)X(f)$. Compare which sinusoidal components remain."
    else:
        n = 128
        u = np.linspace(-4, 4, n, endpoint=False)
        xx, yy = np.meshgrid(u, u)
        image = rect(xx/width.value)*rect(yy/ywidth.value)
        transform = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        one_x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(rect(u/width.value))))
        one_y = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(rect(u/ywidth.value))))
        separate = np.outer(one_y, one_x)
        fx = np.fft.fftshift(np.fft.fftfreq(n, d=u[1]-u[0]))
        ext = [fx[0], fx[-1], fx[0], fx[-1]]
        a1.imshow(image, extent=[-4,4,-4,4], origin="lower", cmap="gray"); a1.set_title("Separable 2D rectangle")
        for ax, data, title in [(a2, abs(transform), "Magnitude of 2D FFT"), (a3, abs(separate), "Product of 1D FFTs"), (a4, abs(transform-separate), "Numerical difference")]:
            ax.imshow(data, extent=ext, origin="lower"); ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_title(title)
        note = r"For $g(x,y)=a(x)b(y)$, the 2D Fourier transform is $G(f_x,f_y)=A(f_x)B(f_y)$."
    for ax in axes.flat:
        ax.tick_params(labelsize=9)
    mo.vstack([mo.md(note), fig])
    plt.close(fig)
    return

if __name__ == "__main__":
    app.run()
