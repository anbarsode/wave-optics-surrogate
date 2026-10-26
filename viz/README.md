# Gravitational Lensing

`viz` contains a local interactive application for comparing geometric-optics and wave-optics
gravitational-lensing calculations. The Python backend uses the numerical work in `ab_initio` and
`bgo`; the browser frontend draws the returned arrays without requiring a JavaScript package install.

## Run

From the repository root:

```bash
/home/ankur/micromamba/envs/gw/bin/python -m pip install -r requirements.txt
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

The two source coordinates \((y_1,y_2)\), lens redshift, source redshift, redshifted mass,
external convergence, and external shear are independent controls. The only source/lens ordering
condition is \(z_s>z_l\). Mass and redshifts set the angular and physical-time conversion, while
the dimensionless source coordinates set the lens geometry.

External convergence is measured relative to a reference line of sight, so it may be negative for
an underdensity or positive for extra projected mass. The signed, axis-aligned shear chooses
whether the horizontal or vertical axis is stretched. The controls enforce
\(1-\kappa>|\gamma|\), so both far-field lens-equation eigenvalues remain positive. Any nonzero
\(\kappa\) or \(\gamma\) activates the external potential; there is no separate switch. The app
also reports \(\theta_E^{\rm MI}\): the angular radius of a source-plane
disk with the same area as the mapped negative-parity grid cells. Each cell is first mapped through
the lens equation and its four-corner polygon area is summed; equivalently this applies
\(dA_y=|\det A|dA_x\), rather than using the lens-plane area. If the negative-parity region reaches
the calculation boundary, the displayed radius is \(\infty\). A central point-mass singularity
alone does not trigger this flag: the reported value is the finite source-plane area sampled by
the displayed grid cells.

The BGO solver is the stationary-phase expansion documented in `bgo/REPORT.md` and follows
[Takahashi (2004), Appendix B](https://arxiv.org/abs/astro-ph/0402165). GLoW supplies the analytic
point-mass, semianalytic radial SIS, and resolution-checked contour calculations. BGO order zero
is the geometric-optics result, so it replaces a separate geometric-optics solver. Analytic and
series solvers are offered when the selected profile supports them. In particular, `ab_initio`
supplies the PM positive-parity external-field source-derivative series, the SIS radial analytic
integral with numerical angular quadrature, and Uddeepta's raw/repaired PM contour calculations.

The time panel displays the direct non-negative delay density \(F(t)\): GLoW's contour response
divided by \(2\pi\), or Uddeepta's raw Ulmer--Goodman-style contour histogram. The latter contains the expected
steps, logarithmic divergences, and smooth pieces between them. These are the quantities used
internally to obtain \(F(w)\); the visualizer does not construct an inverse Fourier transform of
\(F(w)\). The time-panel menu can use a linear or logarithmic time axis.

## Interaction

- Choose any combination of available solvers. Line colour and dash pattern identify solvers.
- Move a slider and wait for the selected automatic delay, or press **Refresh**.
- Hover over control names or sliders for simple explanations.
- Hover over **Layers** in a sky panel to choose maps, curves, images, and mapped cells. Hover
  over **Panels** in frequency, time, and complex panels to choose their displayed sub-panels.
- Use the mouse wheel over a panel to zoom, drag to pan, and press the home button to reset only
  that panel.
- Drag the thin central grid edges to change the relative row or column sizes.

The page itself is fixed to the browser viewport and has no horizontal or vertical page scrolling.
