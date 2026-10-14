# Smooth image contributions for isolated PM and SIS lenses

For a source at $y=0.5$, both an isolated point mass (PM) and an isolated singular isothermal sphere (SIS) produce a minimum and a saddle image. Subtracting the minimum arrival time gives

$$F(w)=G_{\rm min}(w)-iG_{\rm sad}(w)e^{iw\Delta T},\qquad
G_i(w)\longrightarrow\sqrt{|\mu_i|}\quad (w\longrightarrow\infty).$$

The factor $-i$ is the saddle's Morse phase. Equivalently, it can be absorbed into the corresponding complex $G_i$ in $F=\sum_i G_i e^{iw\Delta T_i}$. Writing it separately makes both limits positive. This decomposition has a freedom: one may move a frequency-dependent term from one image contribution to the other while preserving $F$. Smoothness and the geometric limits select a useful representative; they do not prove a unique physical division of the diffraction field.

## Reference amplification and image geometry

The PM reference uses the confluent hypergeometric expression of [Takahashi and Nakamura (2003)](https://arxiv.org/abs/astro-ph/0305055). For the isolated SIS, the lens potential is $\psi(r)=r$ and the radial part of the diffraction integral is evaluated analytically, leaving a numerical angular integral as in [GLoW, Eq. 65](https://arxiv.org/abs/2409.04606). Both results use the phase convention $F(w)\to 1$ as $w\to 0$. The geometric result is $F_{\rm geo}=\sqrt{\mu_+}-i\sqrt{|\mu_-|}e^{iw\Delta T}$.

For the SIS at $0<y<1$, $x_+=1+y$, $x_-=y-1$, $\mu_+=1+1/y$, $\mu_-=1-1/y$, and $\Delta T=2y$. The PM image positions and magnifications follow the point-lens equation $y=x-1/x$.

    Install LaTeX, dvipng, and txfonts separately for LaTeX plot labels.


## Joint smooth decomposition

Set $x=\log_{10}w$ and represent each contribution by a complex cubic spline $S_i(x)$:

$$G_i(w)=\sqrt{|\mu_i|}+\frac{S_i(\log_{10}w)}{1+w}.$$

The factor $1/(1+w)$ enforces the geometric limit, including beyond the last fit frequency. For a fixed number $k$ of equally spaced knots in $x$, the fitted values minimize

$$\begin{aligned}
\mathcal L_k={}&\left\langle\left|F-\sum_i G_i e^{iw\Delta T_i-i\pi n_i/2}\right|^2\right\rangle_x\\
&+\lambda_L\sum_i\frac{1}{x_{\max}-x_{\min}}\int_{x_{\min}}^{x_{\max}}
\sqrt{\epsilon^2+\left|\frac{dG_i}{dx}\right|^2}\,dx .
\end{aligned}$$

Here $n_{\rm min}=0$, $n_{\rm sad}=1$, $\lambda_L=10^{-4}$, and $\epsilon=10^{-5}$. The length term acts on **both** $G_i$. The small $\epsilon$ smooths the absolute value near zero slope. The optimizer varies all real and imaginary spline values simultaneously. It fits each integer knot count $k=4,\ldots,12$ and chooses the smallest $\mathcal L_k+2k\lambda_K$ with $\lambda_K=3\times10^{-6}$. Thus the number of knots varies by an explicit discrete search. The weights set a tradeoff between reconstruction error, curve length, and complexity; they are settings rather than physical constants.

    PM: delay=1.010321, sqrt(|mu|)=[1.26151112 0.76903206], least knots at 1% error=5
      knots   RMS |delta F|   max relative error   curve length   acceleration
          4       0.0050683              2.14%        0.28194        0.42766
          5         0.00251              0.99%        0.26947        0.44223
          6      0.00084128              0.38%        0.24049        0.41274
          7      0.00067727              0.46%        0.23961        0.37451
          8      0.00033735              0.24%         0.2386        0.36523
          9      0.00021723              0.14%        0.23827        0.30773
         10      0.00023422              0.14%        0.23821        0.33211
         11      0.00022622              0.17%        0.23795        0.33313
         12      0.00021706              0.18%         0.2378        0.32717
    SIS: delay=1.000000, sqrt(|mu|)=[1.73205081 1.        ], least knots at 1% error=9
      knots   RMS |delta F|   max relative error   curve length   acceleration
          4        0.025275             14.49%         1.5462         3.3792
          5        0.013509              6.76%        0.61459         1.5133
          6       0.0091452              4.24%        0.37469        0.81794
          7       0.0061383              2.76%        0.34879         0.8615
          8       0.0042312              1.59%        0.36427        0.78262
          9       0.0028246              0.85%        0.36523        0.80646
         10       0.0020266              0.60%        0.35661         0.7069
         11       0.0014694              0.49%        0.35018        0.64707
         12        0.001067              0.40%        0.34557         0.5714


    PM: 5 knots per component, RMS |delta F|=0.00251, max relative |delta F|=0.99%
    SIS: 9 knots per component, RMS |delta F|=0.002825, max relative |delta F|=0.85%


The knot count below is the smallest count meeting a **1% maximum relative complex-error limit** on 321 comparison frequencies. Those frequencies were not used in the fit. This selection rule is shared by all four methods below; the original score remains available for other uses.

## Plots

The first plot for each lens compares the wave result, geometric result, and fitted reconstruction in the complex plane. The second shows the two smooth image contributions and their geometric limits. The third shows their real and imaginary parts against frequency, with faint vertical lines marking the knot positions. Color along each complex curve indicates increasing $w$.

### Point mass

![PM amplification in the complex plane](figures/pm_amplification.jpg)

![PM smooth contributions in the complex plane](figures/pm_components.jpg)

![PM real and imaginary contributions](figures/pm_parts.jpg)

### Singular isothermal sphere

![SIS amplification in the complex plane](figures/sis_amplification.jpg)

![SIS smooth contributions in the complex plane](figures/sis_components.jpg)

![SIS real and imaginary contributions](figures/sis_parts.jpg)

## Scope of the geometric-only decomposition

The length penalty discourages transferring a rapidly winding term from one image contribution to the other. The cubic spline factor $1/(1+w)$ guarantees geometric limits, but the fitted finite-frequency curve is intended for $0.01\le w\le20$. The SIS also has diffraction from its singular center, which the two smooth image components absorb over this band.

## Penalizing acceleration

The original smoothness cost contains a normalized complex-curve length, $\sum_i\langle\sqrt{\epsilon^2+|G'_i(x)|^2}\rangle_x$, with $x=\log_{10}w$. The new variant adds

$$\lambda_A\sum_i\left\langle\sqrt{\epsilon^2+|G''_i(x)|^2}\right\rangle_x,$$

where $\lambda_A=10^{-6}$ and $\epsilon=10^{-5}$. Thus both image contributions are penalized for speed **and** acceleration in log frequency. The derivatives include the frequency dependence of the $1/(1+w)$ factor. Knot counts are again searched from 4 to 12.

## Constraints from one and two BGO orders

The high-frequency expression in [the BGO derivation](../bgo/REPORT.md) and [implementation](../bgo/expansion.py), based on [Takahashi (2004), Appendix B](https://arxiv.org/abs/astro-ph/0402165), is

$$F_{\rm BGO}^{(K)}(w)=\sum_{j=1}^{2}\sqrt{|\mu_j|}\,e^{iw\Delta T_j-i\pi n_j/2}
\sum_{k=0}^{K}B_{jk}(i/w)^k+F_{\rm center}^{(K)}(w).$$

$F_{\rm center}$ is present for the SIS and absent for PM. Here $K=1$ and $K=2$ are tested separately. These asymptotic terms diverge at low frequency. To choose their onset from the known wave-optics result, evaluate each BGO order on $0.01\le w\le100$ and find the **earliest** $w_{\rm full}$ for which $|F_{\rm BGO}-F_{\rm wave}|/|F_{\rm wave}|\le1\%$ at that and every larger sampled frequency. The BGO constraint is zero below $w_{\rm full}/2$ and rises by cubic smoothstep in $\log w$ to full weight at $w_{\rm full}$.

For the fitted band $0.01\le w\le20$, add $\lambda_B\langle W(w)|F_{\rm fit}-F_{\rm BGO}^{(K)}|^2\rangle$ with $\lambda_B=0.01$. The spline formula remains $G_i=\sqrt{|\mu_i|}+S_i(\log_{10}w)/(1+w)$: **the splines still describe deviations from geometric optics**. BGO enters only as a soft high-frequency constraint. The length penalty and the 1% comparison error limit remain the same.

    PM: BGO order 1 zero below w=2.58, full at 5.15; order 2 zero below w=2.65, full at 5.30
      method               knots   RMS |delta F|   max relative error
      Geometric spline       5         0.00251              0.99%
      Acceleration           5       0.0025237              0.99%
      BGO order 1            5       0.0025049              0.97%
      BGO order 2            5       0.0025165              0.96%
    SIS: BGO order 1 zero below w=18.25, full at 36.51; order 2 zero below w=12.10, full at 24.19
      method               knots   RMS |delta F|   max relative error
      Geometric spline       9       0.0028246              0.85%
      Acceleration           9       0.0028566              0.87%
      BGO order 1            9       0.0028213              0.85%
      BGO order 2            9       0.0028415              0.97%


At the same 1% maximum relative error limit, no tested penalty reduced the minimum knot count. The first BGO order gave small improvements for both lenses. The second order improved PM maximum error, while worsening the SIS result within this band.

| Lens | Method | Knots per $G_i$ | RMS $|\Delta F|$ | Maximum relative error |
|:--|:--|--:|--:|--:|
| PM | Geometric spline | 5 | 0.002510 | 0.986% |
| PM | Acceleration | 5 | 0.002524 | 0.993% |
| PM | BGO order 1 | 5 | 0.002505 | 0.969% |
| PM | BGO order 2 | 5 | 0.002517 | 0.964% |
| SIS | Geometric spline | 9 | 0.002825 | 0.854% |
| SIS | Acceleration | 9 | 0.002857 | 0.872% |
| SIS | BGO order 1 | 9 | 0.002821 | 0.845% |
| SIS | BGO order 2 | 9 | 0.002842 | 0.971% |

The BGO transition is selected from the ground truth separately for each lens and each order. The 1% criterion is checked out to $w=100$, even though the fitted band ends at $w=20$. Consequently, the reliable first-order SIS region begins beyond the fitted band; its window has only begun to turn on by $w=20$. This limits how strongly the first BGO order can constrain the present SIS fit. A smaller BGO coefficient is not automatically better after it is combined with finite-knot splines and a length penalty.

The lower panels show errors of the fitted methods; geometric optics is omitted there so subpercent differences remain visible.

### Point mass: residuals, amplitude, phase, and BGO validity

![PM residual in the complex plane](figures/pm_residuals.jpg)

![PM amplification, phase, and errors](figures/pm_diagnostics.jpg)

![PM BGO validity and transition](figures/pm_bgo_validity.jpg)

### Singular isothermal sphere: residuals, amplitude, phase, and BGO validity

![SIS residual in the complex plane](figures/sis_residuals.jpg)

![SIS amplification, phase, and errors](figures/sis_diagnostics.jpg)

![SIS BGO validity and transition](figures/sis_bgo_validity.jpg)

## Interpretation

The acceleration penalty controls curvature of both image contributions, but does not by itself reduce the number of knots needed for the stated 1% error limit. The BGO constraints improve some finite-band errors relative to the geometric-only spline fit, with different effects for PM and SIS. The SIS center term and the late onset of reliable BGO behavior restrict how much high-frequency information is available within $w\le20$. The decomposition remains a smooth two-component representation over the stated band, not a unique physical separation of diffraction by image.
