# Gravitational Lensing

`viz` contains a local interactive application for comparing geometric-optics and wave-optics
gravitational-lensing calculations. The Python backend uses the numerical work in `ab_initio` and
`bgo`; the browser frontend draws the returned arrays without requiring a JavaScript package install.

## Run

From the repository root:

```bash
/home/ankur/micromamba/envs/gw/bin/python -m viz.server
```

Open <http://127.0.0.1:8765>. A different port can be selected with `--port`.

The application is local-only. It does not send lens parameters or calculation results to an
external service. Scientific libraries are configured to use one thread by default.

## Scientific definitions

The dimensionless thin-lens equation and Fermat potential are

$$
\boldsymbol y = \boldsymbol x - \boldsymbol\alpha(\boldsymbol x), \qquad
\tau(\boldsymbol x,\boldsymbol y) = \frac{1}{2}|\boldsymbol x-\boldsymbol y|^2
- \psi(\boldsymbol x).
$$

The point-mass and singular isothermal sphere (SIS) potentials used in the sky view are

$$
\psi_{\rm PM}=b^2\log |\boldsymbol x|, \qquad
\psi_{\rm SIS}=b|\boldsymbol x|.
$$

External convergence and shear add

$$
\psi_{\rm ext}=\frac{\kappa}{2}(x_1^2+x_2^2)
+\frac{\gamma_1}{2}(x_1^2-x_2^2)+\gamma_2x_1x_2.
$$

The physical gravitational-wave frequency is converted to dimensionless frequency using the
redshifted lens mass,

$$
w = 2\pi f\,\frac{4GM_{Lz}}{c^3}.
$$

Planck18 supplies angular-diameter distances and the displayed equivalent point-mass Einstein
angle. The BGO solver is the stationary-phase expansion documented in `bgo/REPORT.md` and follows
[Takahashi (2004), Appendix B](https://arxiv.org/abs/astro-ph/0402165). GLoW supplies the analytic
point-mass, semianalytic radial SIS, and resolution-checked contour calculations.

The time-domain comparison removes the unlensed delta function and displays the direct positive-
frequency transform

$$
F_t(t)=\operatorname{Re}\left[\frac{1}{\pi}\int_0^\infty
\bigl(F(w)-1\bigr)e^{-iwt}\,dw\right].
$$

It is a comparison diagnostic rather than a replacement for the specialised singular-response
handling in `ab_initio/helper_functions.py`.

## Interaction

- Choose any combination of available solvers. Line colour and dash pattern identify solvers.
- Move a slider and wait for the selected automatic delay, or press **Refresh**.
- Hover over control names or sliders for simple explanations.
- Hover over **Layers** in a sky panel to choose maps, curves, images, and mapped cells.
- Use the mouse wheel over a panel to zoom, drag to pan, and press the home button to reset only
  that panel.
- Drag the thin central grid edges to change the relative row or column sizes.

The page itself is fixed to the browser viewport and has no horizontal or vertical page scrolling.
