# Canonical image-curve surrogates for isolated point-mass and SIS lenses

## Model and domain

For an isolated lens with ordinary images indexed by $i$, the wave-optics amplification is represented as

$$F(w,y)=C(w,y)+\sum_i p_i(w,y)G_i(w,y),\qquad
p_i=\exp\!\left[iw\Delta T_i-i\pi n_i/2\right].$$

$\Delta T_i$ and $n_i$ are the image delay and Morse index. The point mass (PM) and the singular isothermal sphere (SIS) have a minimum and a saddle over the source ranges used here. The SIS also has a separate conical-center contribution $C$ at post-geometric order. The exact reference calculations follow [Takahashi and Nakamura (2003)](https://arxiv.org/abs/astro-ph/0305055) for PM and the radial SIS integral in [GLoW](https://arxiv.org/abs/2409.04606).

The surrogate covers $0.01\leq w\leq20$, with $0.01\leq y\leq10$ for PM and $0.03\leq y\leq0.9$ for SIS. These ranges stay within the two-image regime and avoid the SIS image-creation boundary. Values outside the stated ranges are extrapolations and are not assessed here.

## Canonical choice of image curves

Knowing $F$ alone does not uniquely determine its image factors: at each $w$, one complex equation constrains all $G_i$. The representative used here is the smoothest correction to geometric optics within a fixed error limit. Write

$$G_i(w,y)=a_i(y)+W(w,y)B_i^{(2)}(w,y)+u_i(w,y),\qquad
u_i(w,y)=\frac{S_i(\log_{10}w,y)}{1+w},\qquad a_i=\sqrt{|\mu_i|}.$$

$S_i$ is a complex cubic spline. All images use the **same 14 frequency knots at every source position**. In the geometric baseline, $W B_i^{(2)}=0$ and $C=0$. In the BGO version, $B_i^{(2)}$ contains the first two stationary-phase corrections for image $i$ from the [BGO derivation](../bgo/REPORT.md), following [Takahashi (2004), Appendix B](https://arxiv.org/abs/astro-ph/0402165). The SIS center term is added to $F$ separately; assigning it to either ordinary image would make that image's factor less smooth.

With $x=\log_{10}w$, the canonical convention minimizes

$$\mathcal R[u]=\sum_i\int_{x_{\min}}^{x_{\max}}
\left(0.02|u_i|^2+|\partial_xu_i|^2+0.02|\partial_x^2u_i|^2\right)\,dx$$

subject to weighted RMS complex amplification error at most $0.005$. The positive $|u_i|^2$ term makes the finite spline problem strictly convex and chooses one member of the image-mixing freedom. In spline coefficients $c$, each candidate is a Hermitian linear solve

$$c(\lambda)=(A^\dagger W A+\lambda K)^{-1}A^\dagger W r.$$

Bisection chooses the largest $\lambda$ meeting the error limit at each training position. Fitting these positions independently can cause an abrupt change in the image factors when the geometric or BGO baseline first meets the error limit and the optimal correction becomes zero. The fit therefore adds a source-position smoothness cost where the baseline alone meets the error limit. If $c_j$ contains all spline coefficients at $y_j$ and $K$ is the matrix of $\mathcal R$, the complete cost is

$$\mathcal J=\sum_j c_j^\dagger Kc_j+\gamma\sum_{j=2}^{m}q_j\frac{(c_j-c_{j-1})^\dagger K(c_j-c_{j-1})}{\log y_j-\log y_{j-1}},\qquad \gamma=1.$$

Here $q_j$ rises smoothly from 0 to 1 as the smaller baseline RMS error at the two ends falls from $1.2$ times the fitting limit to the limit itself. The same $0.005$ RMS limit applies separately at every $y_j$. Forward and backward constrained coordinate solves minimize this positive quadratic cost without an initial-guess choice. The additional cost prevents the fitted curves from jumping where a correction ceases to be needed for amplitude accuracy. A shared frequency spline basis and interpolation of its coefficients in $\log y$ give the full surrogate. The implementation accepts any number of images; the PM and SIS examples each have two.

## BGO transition

The raw stationary-phase terms grow as $w\rightarrow0$. At each trained $y$, compare the **complete** second-order BGO amplification, including the SIS center, with the exact wave result over $0.01\leq w\leq100$. The first onset condition is the earliest sampled frequency after which every sampled relative complex error stays below $1\%$. If none passes, use 100. Because a faint image can carry a large incorrect BGO correction while the full $F$ still agrees, impose a second condition on every ordinary image: the sum of the magnitudes of its first two corrections must be at most $25\%$ of its geometric magnitude. Define $w_*(y)$ as the larger onset from these two conditions. Smoothly switch the BGO terms from zero at $w_*$ to full strength at $4w_*$ using a cubic smoothstep in $\log w$. The onset values are interpolated in $\log y$ for new source positions. The spline $u_i$ retains its $1/(1+w)$ decay in both versions.

The BGO series is an asymptotic approximation, so the transition is a convention informed by the known training waveforms and the relative sizes of the image terms. The validity check extends well beyond the surrogate's upper frequency so that a temporary agreement near $w=20$ does not trigger the series.

## Validation

The PM uses 24 logarithmically spaced source positions. The SIS uses those 24 positions plus 12 additional positions around $0.09\leq y\leq0.16$, to resolve a narrow interpolation feature. The geometric midpoints between final training positions and an offset frequency grid are held out (23 PM positions and 35 SIS positions). The errors below use those withheld source positions. Complex and amplitude errors are relative; phase errors use the wrapped difference in radians. The complex-error denominator has a floor of $0.1$, which is inactive for these sampled waveforms. The training limit is an RMS limit, so it does not bound the maximum error at one frequency.

| Lens | Representation | Median complex RMS | Worst complex RMS | Worst pointwise complex error | Worst amplitude error | Worst phase error |
|:--|:--|--:|--:|--:|--:|--:|
| PM | geometric baseline | 0.50% | 0.54% | 2.29% | 2.29% | 0.014 rad |
| PM | two BGO orders | 0.50% | 0.58% | 2.30% | 2.29% | 0.015 rad |
| SIS | geometric baseline | 0.52% | 0.57% | 2.59% | 1.74% | 0.021 rad |
| SIS | two BGO orders | 0.52% | 0.57% | 2.59% | 1.74% | 0.021 rad |

The BGO representation is not uniformly better. At $y=0.5$, the PM versions have nearly the same complex RMS error, about $0.55\%$. At $y=5$, BGO reduces the PM complex RMS error over $5\leq w\leq20$ from about $0.27\%$ to $0.26\%$ with the conservative per-image onset. Across all withheld PM positions, the two methods have nearly equal median error. For SIS, the BGO onset is often at or beyond the surrogate's upper frequency; the two versions consequently have almost identical validation errors. The densified SIS source grid has a worst held-out pointwise complex error of about $2.6\%$. A pointwise $1\%$ guarantee would require a different fit constraint and likely more source samples.

The result is a stable convention for image curves, not a uniquely measurable physical split of $F$. Close to a caustic, merging image delays and divergent geometric magnifications call for a uniform approximation rather than separate ordinary-image factors.

## Point-mass lens

### Image factors over source position

Lines show source positions spanning the full range, including the upper training endpoint, colored by $y$. Factors are divided by their geometric magnitudes so the shape of the corrections remains visible across the full range.

![Point-mass lens image-factor curves](surrogate_results/pm_curves.jpg)

### Faint saddle at large source position

At $y=10$, the saddle geometric amplitude is only $\sqrt{|\mu_{\rm saddle}|}\simeq0.0098$. A test based only on the complete $F$ can therefore accept a large *relative* error in this faint image. The total-wave criterion alone would switch on BGO at $w\simeq0.21$, while the additional image-size criterion moves the onset to $w\simeq1.40$; the transition is complete at about $w=5.60$. Coupling the spline coefficients to the nearest trained source position prevents the correction from vanishing abruptly when the baseline alone meets the RMS error limit. The upper endpoint $y=10$ is included in the image-factor plot above. At the withheld position $y\simeq8.61$, both representations have complex RMS error about $0.43\%$ over the full frequency band.

### Error measures versus source position

The last two panels isolate $5\leq w\leq20$, where BGO has its clearest opportunity to help.

![Point-mass lens surrogate errors](surrogate_results/pm_errors.jpg)

### Amplification and image factors at $y=0.5$

The amplification, complex residual, amplitude, phase, and their errors are evaluated at a source position omitted from training.

![Point-mass lens amplification and residual](surrogate_results/pm_example.jpg)

![Point-mass lens image factors in the complex plane](surrogate_results/pm_components.jpg)

## Singular isothermal sphere

### Image factors over source position

Lines show source positions spanning the full range, including the upper training endpoint, colored by $y$. Factors are divided by their geometric magnitudes so the shape of the corrections remains visible across the full range.

![Singular isothermal sphere image-factor curves](surrogate_results/sis_curves.jpg)

### Error measures versus source position

The last two panels isolate $5\leq w\leq20$, where BGO has its clearest opportunity to help.

![Singular isothermal sphere surrogate errors](surrogate_results/sis_errors.jpg)

### Amplification and image factors at $y=0.5$

The amplification, complex residual, amplitude, phase, and their errors are evaluated at a source position omitted from training.

![Singular isothermal sphere amplification and residual](surrogate_results/sis_example.jpg)

![Singular isothermal sphere image factors in the complex plane](surrogate_results/sis_components.jpg)

## Using a saved surrogate

The four compressed coefficient files are `pm_geo.npz`, `pm_bgo2.npz`, `sis_geo.npz`, and `sis_bgo2.npz` in `surrogate_results`. Evaluation uses analytic image delays and magnifications, the saved BGO transition, and interpolation of the fitted spline values. It does not call the exact wave-optics integral.
