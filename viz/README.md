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

The angular unit is calculated rather than entered as a separate profile scale.  For the
redshifted mass \(M_{Lz}\), the point-mass angular scale uses the physical mass
\(M=M_{Lz}/(1+z_l)\) and Planck18 angular-diameter distances.  Point mass, SIS, SIE, CIS, NFW,
elliptical NFW, elliptical power-law, and Hernquist profiles can then be selected in those units.
For example, the first two potentials are

$$
\psi_{\rm PM}=\log |\boldsymbol x|, \qquad
\psi_{\rm SIS}=|\boldsymbol x|.
$$

External convergence and shear add

$$
\psi_{\rm ext}=\frac{\kappa}{2}(x_1^2+x_2^2)
+\frac{\gamma}{2}(x_1^2-x_2^2).
$$

The physical gravitational-wave frequency is converted to dimensionless frequency using the
redshifted lens mass,

$$
w = 2\pi f\,\frac{4GM_{Lz}}{c^3}.
$$

The external convergence is restricted to \(\kappa\geq0\), and the shear is aligned with the
display axes. The app also reports \(\theta_E^{\rm MI}\): the angular radius of a source-plane
disk with the same area as the mapped negative-parity grid cells. Each cell is first mapped through
the lens equation and its four-corner polygon area is summed; equivalently this applies
\(dA_y=|\det A|dA_x\), rather than using the lens-plane area. If the negative-parity region reaches
the calculation boundary, the displayed radius is \(\infty\).

The BGO solver is the stationary-phase expansion documented in `bgo/REPORT.md` and follows
[Takahashi (2004), Appendix B](https://arxiv.org/abs/astro-ph/0402165). GLoW supplies the analytic
point-mass, semianalytic radial SIS, and resolution-checked contour calculations. BGO order zero
is the geometric-optics result, so it replaces a separate geometric-optics solver. Analytic and
series solvers are offered when the selected profile supports them.

The time panel displays GLoW's own internal \(I(t)\) response. The visualizer does not construct a
manual Fourier transform of \(F(w)\), preserving GLoW's specialised treatment of singular
responses.

## Interaction

- Choose any combination of available solvers. Line colour and dash pattern identify solvers.
- Move a slider and wait for the selected automatic delay, or press **Refresh**.
- Hover over control names or sliders for simple explanations.
- Hover over **Layers** in a sky panel to choose maps, curves, images, and mapped cells.
- Use the mouse wheel over a panel to zoom, drag to pan, and press the home button to reset only
  that panel.
- Drag the thin central grid edges to change the relative row or column sizes.

The page itself is fixed to the browser viewport and has no horizontal or vertical page scrolling.
