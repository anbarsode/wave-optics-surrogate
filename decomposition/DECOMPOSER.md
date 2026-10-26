# Adaptive decomposition of a sampled wave-optics amplification

## Inputs and convention

The function [`decompose`](decomposer.py) accepts increasing positive frequencies $w_j$, the sampled complex amplification $F_j$, and one entry per geometric image for its delay $t_i$, signed magnification $\mu_i$, and Morse **phase shift in radians** $\phi_i$. Thus a saddle has $\phi_i=-\pi/2$. The delays and the supplied $F$ must use the same arrival-time reference. The geometric amplification is

$$F_{\rm geo}(w)=\sum_i e^{i(wt_i+\phi_i)}\sqrt{|\mu_i|}.$$

The optional BGO array has shape `(number of images, number of orders)`. Its column $k-1$ holds $B_{ik}$ in

$$G_i(w)=\sqrt{|\mu_i|}+W(w)\sqrt{|\mu_i|}\sum_{k=1}^{K}B_{ik}\left(\frac{i}{w}\right)^k+\frac{S_i(\log_{10}w)}{1+w},$$

$$F_{\rm fit}(w)=C(w)+\sum_i e^{i(wt_i+\phi_i)}G_i(w).$$

$S_i$ is a complex cubic spline. Its $1/(1+w)$ factor and the inverse powers in BGO give $G_i\to\sqrt{|\mu_i|}$ as $w\to\infty$. The image BGO convention follows the stationary-phase derivation in [the BGO report](../bgo/REPORT.md) and [Takahashi (2004), Appendix B](https://arxiv.org/abs/astro-ph/0402165). The optional `known_term` supplies sampled values of $C(w)$, for example the separate SIS center contribution. It is interpolated in $\log w$ when evaluating between samples, so that interpolation should be checked if $C$ oscillates rapidly. If no known term is supplied, $C=0$ and the image splines absorb any center contribution over the fitted range.

### Data-selected ghost terms

Set `ghost_count=k` to search for up to $k$ additional coherent phases that are not represented by the ordinary images. A ghost has

$$F_g(w)=e^{i(wt_g+\phi_g)}\left[\frac{\sqrt{\mu_g}+S_g(\log_{10}w)}{1+w}\right].$$

The extra factor $1/(1+w)$ makes every ghost disappear in the geometric limit. The code first makes a cheap no-ghost fit with relative $L^2$ limit `ghost_prefit_l2`, which defaults to one percent. It interpolates onto a uniform $w$ grid and Fourier transforms $(1+w)(F-F_{\rm prefit})$ after removing a straight-line trend and applying a Hann window. The strongest separated peaks that are farther than one Fourier-resolution bin from every ordinary image delay initialize the ghosts. For fixed delays, all complex coefficients in $c_g e^{iwt_g}/(1+w)$ are solved together by linear least squares, with $\phi_g=\arg c_g$ and $\mu_g=|c_g|^2$.

As a diagnostic, the code also transforms $(1+w)[G_i-G_i^{\rm base}]e^{i(wt_i+\phi_i)}$ for every physical image. A peak shared by all these spectra is strong confirmation of an omitted coherent term. Because the decomposition is non-unique, a smooth fit may place most of that term in only a subset of the factors, so the residual spectrum selects the delay and the shared spectra are reported separately. A candidate is accepted when either its normalized residual strength reaches `ghost_min_residual_strength` or its weakest-image support reaches `ghost_min_shared_support`; both default to 0.1. Consequently the returned ghost count can be smaller than the requested maximum. `ghost_delay_bounds` restricts the delay search. Known initial delays can instead be supplied through `ghost_delay_guesses`; explicit guesses are retained without this threshold.

The returned object exposes `ghost_delays`, `ghost_phases`, `ghost_magnifications`, the delay resolution, the residual Fourier strengths, and the individual image spectra. `components(w)` returns the physical image factors first and the ghost factors last. `physical_image_count` records where this split occurs.

The BGO window $W$ rises from zero at $w_*$ to one at $4w_*$ with a cubic smoothstep in $\log w$. Unless `bgo_start` is supplied, $w_*$ is the first sampled frequency after which the complete supplied-image BGO model stays within `bgo_tolerance` of the ground truth **and** the sum of the magnitudes of the BGO terms is at most `bgo_image_fraction` of the geometric magnitude for every image. The second check matters when one image is faint enough that an erroneous correction is hidden in the total $F$. If no frequency passes, BGO remains off within the fitted range.

## Error and smoothness

The requested `l2_limit` bounds the dimensionless discrete norm

$$E=\frac{\left[\sum_j|F_{\rm fit}(w_j)-F_j|^2\right]^{1/2}}{\left[\sum_j|F_j|^2\right]^{1/2}}.$$

All images enter the same smoothness cost. Let $x=\log_{10}w$ and $u_i=S_i(x)/(1+w)$. For each knot set, the fit minimizes

$$\begin{aligned}
J={}&\sum_i\left\langle\sqrt{\epsilon^2+|\partial_xG_i|^2}
+\eta_0|u_i|^2+\eta_2|\partial_x^2u_i|^2\right\rangle_x\\
&+\alpha(E/E_{\max})^2,
\end{aligned}$$

subject to $E\leq E_{\max}$. The defaults are $\epsilon=10^{-5}$, $\eta_0=\eta_2=0.02$, and $\alpha=0.01$. The first term measures the length of **every** image curve in the complex plane as frequency changes. The positive $|u_i|^2$ term removes flat directions in the image split and gives a unique minimum for a fixed knot set. The curvature cost discourages artificial wiggles. An optional `reference` decomposition adds a small cost for changing the image curves when fitting a nearby lens setting; it is a declared convention, rather than a property inferred from $F$ alone.

The brackets $\langle\cdot\rangle_x$ are a trapezoidal average over the fitted $\log_{10}w$ interval. Thus the numerical length is normalized by the interval width, but it is **not** divided by $|G_i|$, magnification, or the number of images. A factor that becomes $10^{11}$ times its geometric value receives an enormous absolute length and amplitude cost.

When the chosen knot space cannot meet the error limit, the code uses its smallest-$L^2$ residual only to decide where a new knot is needed. Once the requested error becomes feasible, a singular-value ridge path supplies a stable point strictly inside the allowed error ball. SLSQP then minimizes the stated smoothness objective and records its lowest-cost feasible iterate. Thus an exploratory step outside the constraint cannot replace a valid curve. Optimization uses analytic derivatives and a deterministic initial curve obtained by refitting the preceding spline at the new knots.

The hard error constraint still takes precedence over smoothness. If the requested error is below the numerical uncertainty of the supplied $F$, every feasible decomposition may require large cancellations. The correct response is to use an error limit consistent with the reference accuracy or to improve the reference curve; a length cost cannot make inconsistent data smooth while also satisfying a tighter hard constraint.

## Adaptive knots

The number of strict local maxima in $|F_j|$ and $|F_{{\rm geo},j}|$ is counted with neighboring-array comparisons. Write these counts as $P_F$ and $P_{\rm geo}$. The initial number of common knots is

$$K_0=\operatorname{clip}\!\left(P_F+2|P_F-P_{\rm geo}|,K_{\min},K_{\rm initial,max}\right).$$

The default limits are 3 and 20. Every candidate has $K$ knots uniformly spaced in $\log_{10}w$. If the initial fit misses the limit, the code increases $K$ by one until the limit is reached. If the initial fit succeeds, it decreases $K$ by one until the next count fails. There is no optimization over individual knot locations.

When $K$ changes, the preceding cubic splines are evaluated directly at the new uniform knot positions. These values provide the analytically transferred initial curve for the new constrained fit. The default absolute maximum is 80 knots or the number of supplied samples, whichever is smaller; `max_knots` can override it. If the limit remains unattained, the function raises an error that includes the best attained value.

Set `factor_scale_limit` to reject cancelling decompositions whose largest $|G_i|$ exceeds a chosen multiple of the largest physical geometric amplitude. The knot count then increases until both the L2 limit and this conditioning limit hold. Set `optimizer_maxiter=0` to use the singular-value ridge solution directly. In that mode every knot-count refit is analytic and no iterative numerical optimization is called.

Because there is exactly one uniform knot grid for each integer $K$, the accepted count is reproducible and cannot develop local knot clusters. The knot count can still change discretely with lens parameters even though each fixed-count fit has a unique smoothness minimum.

## Example

```python
import numpy as np
from decomposition.decomposer import decompose

# w and F_wave are sampled at the same frequencies. Here the images are a
# minimum and a saddle; the signed magnifications are supplied directly.
fit = decompose(
    w=w,
    wave=F_wave,
    delays=np.array([0.0, saddle_delay]),
    magnifications=np.array([mu_minimum, mu_saddle]),
    morse_shifts=np.array([0.0, -np.pi/2]),
    l2_limit=0.005,
)

image_factors = fit.components(w)
recomposed = fit.amplification(w)
print(len(fit.knots), fit.relative_l2)
```

For one unknown coherent contribution, add `ghost_count=1`. Optional physical delay bounds are useful:

```python
fit = decompose(
    w=w,
    wave=F_wave,
    delays=image_delays,
    magnifications=image_magnifications,
    morse_shifts=image_morse_phases,
    l2_limit=0.005,
    ghost_count=1,
    ghost_delay_bounds=(-1.0, 1.0),
)
print(fit.ghost_delays, fit.ghost_phases, fit.ghost_magnifications)
```

The exact PM reference used in this project follows [Takahashi and Nakamura (2003)](https://arxiv.org/abs/astro-ph/0305055). The SIS radial reference follows [GLoW](https://arxiv.org/abs/2409.04606).

## Numerical check

Using 160 logarithmically spaced frequencies over $0.01\leq w\leq20$, $y=0.5$, and a relative $L^2$ limit of $0.005$, the adaptive fit selected 3 knots for PM and 7 for SIS. Both recomposed amplifications met the limit. A synthetic three-image example required additional knots; a ten-peak signal that was already geometric was reduced from 10 knots to the minimum of 3. The checks verify an automatically inferred residual-Fourier delay to the available Fourier resolution and verify two user-supplied ghost delay guesses exactly. They are implemented in [`test_decomposer.py`](test_decomposer.py).

The external-potential report computes every GLoW curve at two consecutive contour resolutions. The relative $L^2$ difference is used directly as the decomposition limit, so fit errors are compared on the scale actually resolved by the reference. In the tested regimes the measured GLoW uncertainties are below $5\times10^{-4}$, while each higher-resolution calculation of the complete 121-frequency array remains below one second. SIS fits search for a ghost to represent diffraction from the conical centre. The four-image PM and SIS tests search for one additional ghost, subject to the Fourier support threshold. A factor-scale limit of 10 rejects ill-conditioned multi-image solutions before accepting a knot count.
