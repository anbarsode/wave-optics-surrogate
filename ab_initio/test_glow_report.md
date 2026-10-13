# Amplification methods across lens models and image regimes

## Model and comparison

The point mass (PM) and singular isothermal sphere (SIS) with a general quadratic external
potential have lens potentials

$$\psi_{\rm PM}=\ln r+\frac{\kappa r^2}{2}+\frac{\gamma(x_1^2-x_2^2)}{2},\qquad
\psi_{\rm SIS}=r+\frac{\kappa r^2}{2}+\frac{\gamma(x_1^2-x_2^2)}{2}.$$

The source has two independent coordinates $(y_1,y_2)$ in the shear principal axes.
The external cases use a positive-parity quadratic field, $1-\kappa>\gamma$.
The NFW cases are spherical. Ordinary image count defines each row.
The amplification factor is
$F(w)=\frac{w}{2\pi i}\int d^2x\,\exp[iw(\phi(\mathbf x,\mathbf y)-\phi_{\min})]$,
with the first-arriving image setting the phase origin.

The methods are the isolated PM hypergeometric formula of
[Takahashi and Nakamura (2003)](https://arxiv.org/abs/astro-ph/0305055), a source-derivative
series for PM plus field from [Venumadhav (Appendix C)](relative_binning_chang_refsdal_v5.pdf),
the SIS expression with an analytic radial integral and 512-point numerical angular integral,
the local contour calculations in [helper_functions.py](helper_functions.py), and the dedicated
and FFT methods in [GLoW](https://arxiv.org/abs/2409.04606).
The first method column denotes the PM closed formula or the SIS radial formula with angular
quadrature. The latter is **semi-analytic**: no closed-form full $F(w)$ is used for SIS plus
a general quadratic field. An analytic series exists for the
[isolated SIS](https://arxiv.org/abs/astro-ph/0601701), but it is not the method used here.
“GLoW 5k” and “GLoW 20k” use 5,000 and 20,000 time-response points, respectively.
“Helper repaired” reconstructs the saddle peak before FFT; “Helper raw” does not.
For the PM field series, convergence is handled by an exact coordinate rescaling and
shear by the source operator
$\exp[\frac{i\gamma}{2w}(\partial_{y_1}^2-\partial_{y_2}^2)]$
acting on the isolated PM amplification before restoring the geometric source phase.
For the SIS, the radial Fresnel integral is evaluated analytically at each angle;
512-point Gauss–Legendre quadrature gives the remaining angular integral. The GLoW and
helper methods first build a time-delay response and then use an FFT to obtain $F(w)$.

## Implementation availability

A tick means that the method implements the lens model and image regime. It does not assert
accuracy or numerical success for every parameter choice. A cross means the case is not
implemented in that method.

| Lens and image regime | PM exact / SIS semi-analytic | Series | Helper raw | Helper repaired | GLoW dedicated | GLoW 5k | GLoW 20k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PM, 2 images | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PM + field, 2 images | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| PM + field, 4 images | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| SIS, 1 image | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| SIS, 2 images | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| SIS + field, 1 image | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| SIS + field, 2 images | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| SIS + field, 4 images | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| NFW, 1 image | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| NFW, 3 images | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |


## Computational cost

Each cell is the arithmetic mean for 10 generic parameter choices in that row. All methods
evaluate the same 1,024 geometrically spaced frequencies, $0.1\leq w\leq20$, with one thread.
The first table includes preparation and one grid evaluation. The second table times another
grid evaluation with the prepared object already available. Both helper variants pay the full
contour preparation cost when used alone. For a method without reusable preparation, the two
times are the same operation. These are wall times on x86_64, Python
3.10.20, GLoW 0.1; numerical settings and hardware affect them.

### First use: preparation plus 1,024 frequencies

| Lens and image regime | PM exact / SIS semi-analytic | Series | Helper raw | Helper repaired | GLoW dedicated | GLoW 5k | GLoW 20k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PM, 2 images | 332 ms | — | 2.3 s | 2.3 s | 0.465 ms | 17.7 ms | 61.2 ms |
| PM + field, 2 images | — | 26.6 s | 2.53 s | 2.53 s | — | 520 ms | 2.04 s |
| PM + field, 4 images | — | 23.3 s | 2.6 s | 2.6 s | — | 649 ms | 2.59 s |
| SIS, 1 image | 160 ms | — | — | — | 27 ms | 9.82 ms | 28.7 ms |
| SIS, 2 images | 153 ms | — | — | — | 4.27 ms | 10.1 ms | 29.5 ms |
| SIS + field, 1 image | 163 ms | — | — | — | — | 311 ms | 1.23 s |
| SIS + field, 2 images | 153 ms | — | — | — | — | 346 ms | 1.37 s |
| SIS + field, 4 images | 147 ms | — | — | — | — | 445 ms | 1.82 s |
| NFW, 1 image | — | — | — | — | — | 20 ms | 67.9 ms |
| NFW, 3 images | — | — | — | — | — | 21 ms | 69.2 ms |


### Repeated use: another 1,024 frequencies after preparation

| Lens and image regime | PM exact / SIS semi-analytic | Series | Helper raw | Helper repaired | GLoW dedicated | GLoW 5k | GLoW 20k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PM, 2 images | 332 ms | — | 0.0703 ms | 0.0379 ms | 0.253 ms | 1.2 ms | 1.14 ms |
| PM + field, 2 images | — | 24 s | 0.089 ms | 0.0667 ms | — | 1.03 ms | 1.06 ms |
| PM + field, 4 images | — | 20.7 s | 0.0751 ms | 0.065 ms | — | 1.26 ms | 1.31 ms |
| SIS, 1 image | 119 ms | — | — | — | 26.8 ms | 0.864 ms | 0.818 ms |
| SIS, 2 images | 114 ms | — | — | — | 3.97 ms | 1.17 ms | 1.25 ms |
| SIS + field, 1 image | 123 ms | — | — | — | — | 0.863 ms | 0.864 ms |
| SIS + field, 2 images | 114 ms | — | — | — | — | 1.28 ms | 1.32 ms |
| SIS + field, 4 images | 108 ms | — | — | — | — | 1.32 ms | 1.33 ms |
| NFW, 1 image | — | — | — | — | — | 0.88 ms | 0.895 ms |
| NFW, 3 images | — | — | — | — | — | 1.11 ms | 1.1 ms |


## Accuracy against the selected reference

For each parameter choice, $F_{\rm best}$ is the first successful applicable method in this
fixed order: independent formula, repaired helper, GLoW 20k, raw helper, GLoW dedicated,
GLoW 5k, series. “Repaired helper” is the `helper_corrected` calculation. The ordering is a
comparison convention, not a proof that the chosen method is physically most accurate.
In particular, the PM series or a GLoW result may be more accurate than the repaired helper
even when the latter is selected as $F_{\rm best}$. The error shown is the mean over 10 choices of the discrete sum
$\sum_{i=1}^{1024}|F(w_i)-F_{\rm best}(w_i)|$; it is not divided by 1,024.
A zero identifies the selected reference, not an independent error estimate. The NFW row has
no independent analytical or series reference, so its accuracy table measures agreement
between two GLoW resolutions.

| Lens and image regime | PM exact / SIS semi-analytic | Series | Helper raw | Helper repaired | GLoW dedicated | GLoW 5k | GLoW 20k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PM, 2 images | 0 | — | 10.3 | 6.68 | 0.000133 | 0.0126 | 0.0127 |
| PM + field, 2 images | — | 5.47 | 12 | 0 | — | 5.57 | 5.55 |
| PM + field, 4 images | — | 16.2 | 16.5 | 0 | — | 57.1 | 57.1 |
| SIS, 1 image | 0 | — | — | — | 6.89e-07 | 0.0135 | 0.0123 |
| SIS, 2 images | 0 | — | — | — | 1.14e-06 | 0.0156 | 0.0157 |
| SIS + field, 1 image | 0 | — | — | — | — | 0.56 | 0.696 |
| SIS + field, 2 images | 0 | — | — | — | — | 0.606 | 0.828 |
| SIS + field, 4 images | 0 | — | — | — | — | 51.6 | 52.1 |
| NFW, 1 image | — | — | — | — | — | 0.00325 | 0 |
| NFW, 3 images | — | — | — | — | — | 0.00604 | 0 |


## Amplification curves and residuals

Each figure shows one representative parameter choice from its row on the common frequency
grid. The tables above average all 10 choices. The three panels show amplitude, unwrapped
phase, and absolute complex residual against the selected $F_{\rm best}$. The reference
curve is omitted from the logarithmic residual panel because its residual is exactly zero.

### PM, 2 images

![Amplification and residuals for PM, 2 images](report_figures/case_01.jpg)

### PM + field, 2 images

![Amplification and residuals for PM + field, 2 images](report_figures/case_02.jpg)

### PM + field, 4 images

![Amplification and residuals for PM + field, 4 images](report_figures/case_03.jpg)

### SIS, 1 image

![Amplification and residuals for SIS, 1 image](report_figures/case_04.jpg)

### SIS, 2 images

![Amplification and residuals for SIS, 2 images](report_figures/case_05.jpg)

### SIS + field, 1 image

![Amplification and residuals for SIS + field, 1 image](report_figures/case_06.jpg)

### SIS + field, 2 images

![Amplification and residuals for SIS + field, 2 images](report_figures/case_07.jpg)

### SIS + field, 4 images

![Amplification and residuals for SIS + field, 4 images](report_figures/case_08.jpg)

### NFW, 1 image

![Amplification and residuals for NFW, 1 image](report_figures/case_09.jpg)

### NFW, 3 images

![Amplification and residuals for NFW, 3 images](report_figures/case_10.jpg)


## Reproducibility and limits

The 10 choices per row are deterministic small perturbations of an interior configuration,
checked to retain the stated image count; sampling uses NumPy seeds 20261013 and, if a
numerical evaluation fails, 20261014 for a replacement. No choice sits
at an image transition. [report_benchmark.py](report_benchmark.py) performs the calculations
using the formulas in [test_glow.ipynb](test_glow.ipynb), and
[report_benchmark_results.json](report_benchmark_results.json) contains the measured
configurations and per-choice results. The GLoW FFT uses $t_{\max}=1000$; the helper contour
uses $t_{\max}=100$, spatial step 0.06, and 32,768 interpolated time samples. A failed run
is reported as a failure rather than silently omitted from a 10-choice average. Four-image
FFT calculations can have substantial complex error despite a larger time-response grid.
