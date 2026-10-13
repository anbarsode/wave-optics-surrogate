# Wave-optics amplification through sixth order beyond geometric optics

## Conventions and range of the expansion

The thin-lens amplification factor is

\[
 F(w,\boldsymbol y)=\frac{w}{2\pi i}\int_{\mathbb R^2}d^2x\,
    \exp\{iw[\tau(\boldsymbol x,\boldsymbol y)-\tau_0]\},
 \qquad w=2\pi f t_0,
\]

where \(\tau_0\) is the smallest ordinary-image Fermat potential. If
\(\theta_0\) is the angular scale used for \(\boldsymbol x=\boldsymbol\theta/\theta_0\)
and \(\boldsymbol y=\boldsymbol\beta/\theta_0\), then

\[
 t_0=\frac{(1+z_L)D_LD_S}{cD_{LS}}\theta_0^2,
 \quad a=\left(\frac{\theta_{E,\rm PM}}{\theta_0}\right)^2,
 \quad b=\frac{\theta_{E,\rm SIS}}{\theta_0},
\]

with \(\theta_{E,\rm PM}^2=4GM D_{LS}/(c^2D_LD_S)\) and
\(\theta_{E,\rm SIS}=4\pi(\sigma_v/c)^2D_{LS}/D_S\).
For a PM with \(\theta_0=\theta_{E,\rm PM}\), this gives
\(t_0=4GM_{Lz}/c^3\). Here \(D_L,D_S,D_{LS}\) are angular-diameter
distances, \(z_L\) is lens redshift, \(M_{Lz}=(1+z_L)M\), and \(\sigma_v\)
is the SIS velocity dispersion. All plots use \(w\), so no physical mass or
distance is assumed. The potential and normalization follow [Takahashi and Nakamura
(2003)](https://arxiv.org/abs/astro-ph/0305055) and [Takahashi
(2004)](https://arxiv.org/abs/astro-ph/0402165).

Let \(a>0\) be the point-mass strength, \(b>0\) the SIS deflection strength,
and

\[
 \Gamma=\begin{pmatrix}\gamma_1&\gamma_2\\\gamma_2&-\gamma_1\end{pmatrix},
 \qquad A=I-\Gamma,\qquad \gamma=(\gamma_1^2+\gamma_2^2)^{1/2}.
\]

The full parameter set is \((w,\boldsymbol y,a,\gamma_1,\gamma_2)\) for a
point mass and \((w,\boldsymbol y,b,\gamma_1,\gamma_2)\) for an SIS, with
\(\gamma_1=\gamma_2=0\) for isolated lenses. Rotation of axes removes one
orientation parameter, but the formulae retain both shear components. The
standard bounded external-field setting is \(0\leq\gamma<1\). The image
formulae below also give local solutions for other shears whenever their
Hessians are nonsingular, although the diffraction integral at infinity then
needs a separate normalization prescription.

These are **high-frequency, fixed-parameter asymptotic formulae**. They apply
to every ordinary-image region where all relevant images are isolated and have
nonzero Hessian determinant. At a caustic, that determinant is zero, so an
ordinary saddle expansion diverges. The SIS has an additional center
transition at \(|\boldsymbol y|=b\), where its central term diverges. A finite
second-order ordinary-saddle formula cannot hold *at* these physically allowed
parameter values. Airy or Pearcey uniform approximations, together with a
uniform treatment of the SIS center transition, are separate calculations.
This distinction is essential when interpreting the four-image comparisons.

## Fermat potentials and geometric optics

Writing \(r=|\boldsymbol x|\), the four potentials are

| Lens | Fermat potential \(\tau(\boldsymbol x,\boldsymbol y)\) |
|---|---|
| PM | \(\tfrac12|\boldsymbol x-\boldsymbol y|^2-a\log r\) |
| SIS | \(\tfrac12|\boldsymbol x-\boldsymbol y|^2-br\) |
| PM + external shear | \(\tfrac12|\boldsymbol x-\boldsymbol y|^2-a\log r-\tfrac12\boldsymbol x^T\Gamma\boldsymbol x\) |
| SIS + external shear | \(\tfrac12|\boldsymbol x-\boldsymbol y|^2-br-\tfrac12\boldsymbol x^T\Gamma\boldsymbol x\) |

An image \(\boldsymbol x_j\ne0\) obeys

\[
 A\boldsymbol x_j-\frac{a\boldsymbol x_j}{r_j^2}=\boldsymbol y
 \quad\text{(PM)},\qquad
 A\boldsymbol x_j-\frac{b\boldsymbol x_j}{r_j}=\boldsymbol y
 \quad\text{(SIS)}.
\]

The Hessian, signed magnification, and Morse count are

\[
 H_j^{\rm PM}=A-\frac{a}{r_j^2}I+\frac{2a}{r_j^2}\boldsymbol e_j\boldsymbol e_j^T,
 \qquad
 H_j^{\rm SIS}=A-\frac b{r_j}(I-\boldsymbol e_j\boldsymbol e_j^T),
 \qquad \mu_j=\frac1{\det H_j},
\]

where \(\boldsymbol e_j=\boldsymbol x_j/r_j\), and \(m_j\in\{0,1,2\}\) counts
the negative eigenvalues of \(H_j\). Thus every model has the same geometric
formula,

\[
 F_{\rm geo}=\sum_j\frac{\exp[iw(\tau_j-\tau_0)-i\pi m_j/2]}
                             {\sqrt{|\det H_j|}}.
\]

For an isolated PM, orient the first axis along the source and set
\(y=|\boldsymbol y|>0\). Its two images lie at
\(x_\pm=(y\pm\sqrt{y^2+4a})/2\), have
\(\mu_\pm=[1-a^2/x_\pm^4]^{-1}\), and have delay difference
\[
 \Delta\tau=\frac{y\sqrt{y^2+4a}}2+
 a\log\frac{\sqrt{y^2+4a}+y}{\sqrt{y^2+4a}-y}.
\]
For an isolated SIS, \(x_+=y+b\) always exists. If \(0<y<b\), a saddle
\(x_-=y-b\) also exists; \(\mu_+=1+b/y\), \(\mu_-=1-b/y\), and
\(\Delta\tau=2by\). For \(y>b\), only \(x_+\) is an ordinary image.

For either sheared lens, rotate into the eigenbasis of \(A\), with
\(\lambda_1=1-\gamma\), \(\lambda_2=1+\gamma\), and source components
\((y_1,y_2)\). Write \(t=a/r^2\) for PM or \(t=b/r\) for SIS. Away from a
zero denominator, each image is \(x_i=y_i/(\lambda_i-t)\), where the positive
real values of \(t\) solve the respective quartic

\[
 \begin{aligned}
 \text{PM: }&t\{y_1^2(\lambda_2-t)^2+y_2^2(\lambda_1-t)^2\}
     -a(\lambda_1-t)^2(\lambda_2-t)^2=0,\\
 \text{SIS: }&t^2\{y_1^2(\lambda_2-t)^2+y_2^2(\lambda_1-t)^2\}
     -b^2(\lambda_1-t)^2(\lambda_2-t)^2=0.
 \end{aligned}
\]

On a symmetry axis, \(y_i=0\) can allow \(t=\lambda_i\); those images are
obtained directly from the lens equation and are included by the code. The
quartics therefore specify the complete ordinary-image sum in the one, two,
and four-image regions without assuming that the source lies on a shear axis.

## First- and second-order image terms

At each image, choose local radial and transverse displacements \((u,v)\).
The Fermat potential has the expansion
\(\tau=\tau_j+\tfrac12(u,v)H_j^{\rm local}(u,v)^T+
P_3+P_4+P_5+P_6+\cdots\), where \(P_n\) is homogeneous of degree \(n\).
External shear is quadratic, so its effect on orders three through six enters
only through the image position and the Hessian. SageMath's polynomial
expansion in [derive.sage](derive.sage) gives

| Model | \(P_3\) | \(P_4\) |
|---|---|---|
| PM | \(a(-u^3/3+uv^2)/r^3\) | \(a(u^4/4-3u^2v^2/2+v^4/4)/r^4\) |
| SIS | \(buv^2/(2r^2)\) | \(b(-u^2v^2/2+v^4/8)/r^3\) |

| Model | \(P_5\) | \(P_6\) |
|---|---|---|
| PM | \(a(-u^5/5+2u^3v^2-uv^4)/r^5\) | \(a(u^6/6-5u^4v^2/2+5u^2v^4/2-v^6/6)/r^6\) |
| SIS | \(b(u^3v^2/2-3uv^4/8)/r^4\) | \(b(-u^4v^2/2+3u^2v^4/4-v^6/16)/r^5\) |

Here \(r=r_j\). These polynomials already contain their Taylor factorials.
Define a formal Gaussian average \(\langle\cdot\rangle_j\) by covariance
\(G_j=(H_j^{\rm local})^{-1}\). It is valid algebraically for minima,
saddles, and maxima, including indefinite \(G_j\). An explicit finite formula
for every required monomial is

\[
 \langle u^m v^n\rangle_j=m!n!
 \sum_{\substack{2k+\ell=m\\\ell+2h=n}}
 \frac{(G_{11}/2)^kG_{12}^{\ell}(G_{22}/2)^h}{k!\ell!h!}.
\]

Consequently the coefficients are analytical functions of the complete lens
parameters through the quartic image positions:

\[
 \boxed{\begin{aligned}
 c_{1j}&=-\langle P_4\rangle_j+\tfrac12\langle P_3^2\rangle_j,\\
 c_{2j}&=\langle P_6\rangle_j-\langle P_3P_5\rangle_j
 -\tfrac12\langle P_4^2\rangle_j
 +\tfrac12\langle P_3^2P_4\rangle_j
 -\tfrac1{24}\langle P_3^4\rangle_j.
 \end{aligned}}\tag{1}
\]

This compact finite-polynomial expression avoids a many-line rational
polynomial in \(a,b,r,G_{11},G_{12},G_{22}\). No integrals or derivatives remain
in Eq. (1). The ordinary-image contribution through second order is

\[
 \boxed{F_{\rm images}^{(2)}=
 \sum_j\frac{e^{iw(\tau_j-\tau_0)-i\pi m_j/2}}{\sqrt{|\det H_j|}}
 \left(1+\frac{i c_{1j}}w+\frac{c_{2j}}{w^2}\right).}\tag{2}
\]

For PM, Eq. (1) reproduces the published first-order coefficient
\((4x_j^2-1)/[3(x_j^2+1)^3(x_j^2-1)]\) when \(a=1\) and the shear vanishes
([Takahashi 2004, Eq. 27](https://arxiv.org/html/astro-ph/0402165)).
For the isolated lenses, SageMath also reduces Eq. (1) to short rational
formulae. With \(\rho=r_j/\sqrt a\) for PM,

\[
 c_{1j}^{\rm PM}=\frac1a\frac{4\rho^2-1}
 {3(\rho^2+1)^3(\rho^2-1)},\qquad
 c_{2j}^{\rm PM}=-\frac1{a^2}
 \frac{108\rho^8-144\rho^6+124\rho^4-8\rho^2+1}
 {18(\rho^2+1)^6(\rho^2-1)^2}.
\]

With \(\rho=r_j/b\) for SIS,

\[
 c_{1j}^{\rm SIS}=\frac1{b^2}\frac1{8\rho^2(\rho-1)},\qquad
 c_{2j}^{\rm SIS}=-\frac1{b^4}
 \frac{3(8\rho-5)}{128\rho^4(\rho-1)^2}.
\]

The SIS expressions are the *ordinary-image* coefficients and must be used
together with Eq. (3). The sheared coefficients remain exactly specified by
the finite monomial formula in Eq. (1), with no assumption about orientation.

## The SIS center contribution

The cusp at \(\boldsymbol x=0\) is not an ordinary image. It contributes at
order \(w^{-1}\), so omitting it would make both SIS formulae wrong at their
stated order. Put
\(z=b^2-|\boldsymbol y|^2-i0\) and
\(\tau_c=|\boldsymbol y|^2/2\). For \(|\boldsymbol y|\ne b\), the result through
second order is

\[
 \boxed{F_{\rm center}^{(2)}=
 e^{iw(\tau_c-\tau_0)}\left[
 \frac{ib}{wz^{3/2}}+
 \frac b{2w^2}\left\{
 \frac{3\operatorname{tr}A}{z^{5/2}}+
 \frac{15\boldsymbol y^TA\boldsymbol y}{z^{7/2}}
 \right\}\right].}\tag{3}
\]

The \(-i0\) gives the outgoing-wave branch: the first coefficient is
\(ib/(b^2-y^2)^{3/2}\) for \(y<b\), and
\(b/(y^2-b^2)^{3/2}\) for \(y>b\), agreeing with
[Takahashi's leading center result](https://arxiv.org/html/astro-ph/0402165).
To derive the second coefficient, scale \(\boldsymbol x=\boldsymbol u/w\)
at the center and expand
\(\exp[i\boldsymbol u^TA\boldsymbol u/(2w)]\). Multiplication by
\(u_i u_j\) is \(-\partial_{y_i}\partial_{y_j}\) on the leading conical
integral, yielding Eq. (3). The SageMath derivative identity is checked in
`derive.sage`. Thus, for either SIS model,
\(F^{(2)}=F_{\rm images}^{(2)}+F_{\rm center}^{(2)}+O(w^{-3})\).
For either PM model, \(F^{(2)}=F_{\rm images}^{(2)}+O(w^{-3})\).

## Third through sixth order

The local Taylor polynomials through degree $14$ give six orders beyond geometric optics. Write $h=n-2j$ and let $[u^h v^{2j}]P_n$ denote the coefficient of that monomial. For every $n\ge3$, the two lens families have explicit coefficients

\[
 [u^h v^{2j}]P_n^{\rm PM}=\frac{a}{r^n}
 \begin{cases}(-1)^n/n,&j=0,\\
 (-1)^{j+h}\binom{n-1}{h}/(2j),&j\ge1,
 \end{cases}
\qquad
 [u^h v^{2j}]P_n^{\rm SIS}=
 \begin{cases}0,&j=0,\\
 -\dfrac{b}{r^{n-1}}\binom{1/2}{j}(-1)^h
 \binom{2j+h-2}{h},&j\ge1.
 \end{cases}
\tag{4}
\]

All odd powers of $v$ vanish. These formulas follow from the Taylor series of
$-a\log\sqrt{(r+u)^2+v^2}$ and $-b\sqrt{(r+u)^2+v^2}$. [SageMath's direct polynomial expansion](derive.sage) verifies Eq. (4) for every $P_3,\ldots,P_{14}$, with exact rational coefficients. External shear has no cubic or higher derivative, so Eq. (4) applies unchanged to both sheared models at each of their image positions.

Define $B_{0j}=1$. For $1\le k\le6$, an explicit finite analytical expression for all higher-order image coefficients is

\[
 \boxed{B_{kj}=\sum_{\substack{m_3,\ldots,m_{2k+2}\ge0\\
                  \sum_{n=3}^{2k+2}(n-2)m_n=2k}}
 \frac{(-1)^{\sum_n m_n}}{\prod_n m_n!}
 \left\langle\prod_{n=3}^{2k+2}P_n^{m_n}\right\rangle_j.}
 \tag{5}
\]

The average is the explicit Wick sum preceding Eq. (1); thus Eq. (5) requires no integration or numerical differentiation. It specifies the third, fourth, fifth, and sixth terms for arbitrary $a$ or $b$, source coordinates, and shear components. The first two terms satisfy $B_{1j}=c_{1j}$ and $B_{2j}=-c_{2j}$. The complete ordinary-image result is

\[
 \boxed{F_{\rm images}^{(K)}(w)=
 \sum_j\frac{e^{iw(\tau_j-\tau_0)-i\pi m_j/2}}{\sqrt{|\det H_j|}}
 \sum_{k=0}^K\frac{i^k B_{kj}}{w^k},\qquad 0\le K\le6.}
 \tag{6}
\]

For an SIS, the conical center must be expanded to the same order. Let
$D_A=A_{11}\partial_{y_1}^2+2A_{12}\partial_{y_1}\partial_{y_2}+A_{22}\partial_{y_2}^2$ and $C_0=i b(b^2-|\boldsymbol y|^2-i0)^{-3/2}$. Scaling $\boldsymbol x=\boldsymbol u/w$ in the diffraction integral and expanding its quadratic factor gives

\[
 \boxed{F_{\rm center}^{(K)}(w)=
 e^{iw(\tau_c-\tau_0)}
 \sum_{m=0}^{K-1}\frac{(-i/2)^m}{m!\,w^{m+1}}D_A^m C_0,
 \qquad \tau_c=|\boldsymbol y|^2/2.}
 \tag{7}
\]

Equation (7) includes ten source derivatives at sixth order. The code evaluates these derivatives analytically by a sparse polynomial recurrence. The final result is Eq. (6) for PM and Eq. (6) plus Eq. (7) for SIS. The coefficients depend on every physical parameter through the quartic roots, Hessians, and source derivatives.

### Mathematical range

These are asymptotic series as $w\to\infty$ at *fixed, nondegenerate* lens parameters. At an ordinary-image caustic, $\det H=0$, and at the isolated PM alignment a whole Einstein ring is stationary. The SIS center expansion also becomes singular at $|\boldsymbol y|=b$. Thus no finite-order version of Eqs. (5)–(7) is valid at every physically possible parameter value. Uniform catastrophe or center-transition expansions are needed on those boundaries. An asymptotic series also need not converge: at fixed finite $w$, truncating after the smallest term can be more accurate than including all six terms.

## Numerical comparisons with GLoW

Each plot shows the full GLoW wave-optics result, cumulative orders zero through six, and the normalized complex error. The error denominator is $\max(|F_{\rm GLoW}|,0.2)$ to avoid artificial spikes near interference zeros. The [numerical record](comparison_metrics.json) includes all 121 frequencies per case, the complete order-dependent errors, selected complex $F$ values, and timings. All lenses use unit strength, $a=b=1$; frequencies are dimensionless $w=2\pi f t_0$.

For isolated PM, GLoW uses its exact analytic point-lens function. For isolated SIS, it uses its semi-analytic frequency solver. For sheared lenses, it uses the multi-contour time response followed by GLoW’s **regularized FFT**, with its singular contribution evaluated exactly. The plotted sheared reference has $N_t=20{,}000$, $t_{\max}=10^4$, and a frequency range extending 20% above the plot. The gray curve is its difference from $N_t=10{,}000$; an $N_t=5{,}000$ run is also timed. GLoW issued contour-bracketing and GSL warnings for some sheared calculations. Resolution spread measures one numerical sensitivity, but it is not a rigorous bound on systematic error.

The eight comparison plots cover the one-, two-, and four-image regions:

| Model | Isolated or shear | Plot |
|---|---|---|
| PM | isolated, two images | ![PM isolated two](pm_isolated_two_orders_0_to_6.jpg) |
| PM | shear, two images | ![PM shear two](pm_shear_two_orders_0_to_6.jpg) |
| PM | shear, four images | ![PM shear four](pm_shear_four_orders_0_to_6.jpg) |
| SIS | isolated, two images | ![SIS isolated two](sis_isolated_two_orders_0_to_6.jpg) |
| SIS | isolated, one image | ![SIS isolated one](sis_isolated_one_orders_0_to_6.jpg) |
| SIS | shear, two images | ![SIS shear two](sis_shear_two_orders_0_to_6.jpg) |
| SIS | shear, four images | ![SIS shear four](sis_shear_four_orders_0_to_6.jpg) |
| SIS | shear, one image | ![SIS shear one](sis_shear_one_orders_0_to_6.jpg) |

The table below gives $F_{\rm GLoW}$ near $w=20$, the normalized complex error of each cumulative order in percent, and the numerical resolution spread in percent. Zero spread means a dedicated isolated-lens solver was used.

| Lens and region | $w$ | $F_{\rm GLoW}$ | Errors, orders 0–6 (%) | GLoW spread (%) |
|---|---:|---:|---|---:|
| PM / isolated, two images | 19.84 | 0.826-0.858i | 4, 0.15, 0.071, 0.0041, 0.0059, 0.00039, 0.0012 | 0 |
| PM / shear, two images | 20.05 | 1.087-1.304i | 2.3, 1, 0.49, 0.2, 0.31, 0.48, 0.42 | 0.0024 |
| PM / shear, four images | 20.05 | 5.988+3.024i | 14, 6.5, 50, 45, 1e+03, 3.2e+03, 7.3e+04 | 0.25 |
| SIS / isolated, two images | 19.84 | 1.130-1.171i | 2, 0.63, 0.15, 0.14, 0.09, 0.092, 0.1 | 0 |
| SIS / isolated, one image | 19.84 | 1.303-0.028i | 2.4, 0.89, 0.61, 0.63, 0.87, 1.5, 3.2 | 0 |
| SIS / shear, two images | 20.05 | 2.106-1.740i | 1.5, 0.69, 0.2, 0.18, 0.062, 0.13, 0.22 | 0.015 |
| SIS / shear, four images | 20.05 | 8.447+4.094i | 15, 4.9, 50, 51, 1.1e+03, 3.5e+03, 8.1e+04 | 1.4 |
| SIS / shear, one image | 20.05 | 1.223+0.007i | 2.5, 0.81, 0.73, 0.62, 0.58, 1.2, 2.6 | 0.1 |

For the isolated examples, a conservative 1% onset is the lowest sampled frequency after which *every* higher plotted frequency has error below 1%. This avoids choosing a favorable interference fringe. The onsets for cumulative orders 0–6 are:

| Isolated example | 1% onset $w$ for orders 0–6 |
|---|---|
| PM / isolated, two images | 126.9, 8.7, 9.0, 8.1, 8.1, 7.8, 5.4 |
| SIS / isolated, two images | 160.0, 19.8, 10.9, 10.2, 10.9, 11.3, 12.5 |
| SIS / isolated, one image | 53.6, 18.6, 16.3, 18.0, 19.8, 21.9, 24.2 |

The isolated PM gains a lower 1% onset overall, reaching about $w=5.4$ at order six, although intermediate onsets are not monotone. The isolated SIS two-image series is most useful at moderate order near $w=10$; order six needs about $w=12.5$ for the same 1% bound. The isolated SIS one-image series reaches 1% near $w=16.3$ at order two, while order six needs about $w=24.2$. These are actual counterexamples to a claim that each added asymptotic term must extend the agreement to lower frequency. At higher $w$, the sixth-order error is far smaller than lower-order errors, which checks the coefficients independently against the GLoW isolated-lens solvers.

For the weak-shear four-image examples, order-two and higher corrections become very large around $w=20$. The largest PM image has $|B_6|^{1/6}\simeq66$ and the largest SIS image has $|B_6|^{1/6}\simeq67$. At $w=20$ their sixth-order factors $|B_6|/w^6$ exceed $10^3$: the expansion is being used far below its local high-frequency scale. An optimal truncation there can favor geometric or first order. A uniform caustic approximation is needed to extend these examples toward the caustic. In the PM shear two-image case, the regularized GLoW result agrees closely with the high-frequency expansion. At $w\simeq80$ the change from fifth to sixth order is only about $10^{-6}$ in complex $F$, below the difference between the $N_t=10{,}000$ and $20{,}000$ GLoW results. The visually merged high-order curves therefore reflect the expected powers of $1/w$ and finite reference accuracy. The SIS sheared one- and two-image cases show the same numerical floor at high order.

### Check of GLoW transform methods

A direct transform and a regularized FFT were applied to *the same* PM plus shear time response ($y=0.3$, $\gamma_1=0.05$, $\gamma_2=0$, $N_t=10{,}000$, $t_{\max}=10^4$). Their complex differences from the sixth-order result are:

| $w$ | Direct transform difference | Regularized FFT difference |
|---:|---:|---:|
| 19.95 | 0.3165 | 0.007163 |
| 39.81 | 0.3133 | 0.0006737 |
| 79.43 | 0.3135 | 0.0001647 |

![GLoW transform comparison](reference_diagnostic.jpg)

The direct transform differs by about 0.31 throughout this high-frequency sample, while the regularized FFT difference falls as the asymptotic calculation predicts. The direct transform of a sampled time response is therefore unsuitable as the high-frequency reference for these settings. The [diagnostic script](reference_diagnostic.py) and [values](reference_diagnostic.json) reproduce this check. The regularized FFT includes analytical image singularities, so close agreement with geometric optics alone is partly built into this method; agreement of the higher-order residuals is the more relevant comparison.

## Computational cost and rough break-even order

At cumulative order $K$, the ordinary-image calculation needs Taylor terms through $P_{2K+2}$. The number of product types in Eq. (5) is the integer-partition count of $2K$; orders three through six require 11, 22, 42, and 77 types per image. Their largest Gaussian degrees are 18, 24, 30, and 36, respectively. A memoized Wick recurrence avoids explicit enumeration of $35!!$ pairings at sixth order. The SIS center needs $2(K-1)$ source derivatives at order $K$.

The plot compares the measured wall time for each *cumulative* formula, orders zero through six, with the corresponding GLoW result in all eight examples. Each analytic point is the median of three single-thread evaluations of 121 frequencies and includes image finding and coefficient construction. The GLoW line includes solver construction and evaluation of the same frequencies; sheared cases use the $N_t=20{,}000$ regularized FFT. The dashed curves fit

$$
T_K=A K^p,\qquad K=3,4,5,6,
$$

by least squares in log time. Diamonds mark the fitted crossings. The gray strips show how the sheared crossing shifts when orders 2–6 are fitted instead. The horizontal axis displays $K+1$ on a log scale so order zero can appear.

![Measured and extrapolated computational costs](computational_costs.jpg)

The isolated PM special-function solver is already faster than the generic geometric calculation. The isolated SIS two-image example crosses between measured orders 2 and 3; its fit predicts $K\simeq2.7$. The isolated SIS one-image example crosses between measured orders 5 and 6; its fit predicts $K\simeq5.9$.

For sheared lenses, the order-3-to-6 power laws predict rough crossings at $K\simeq39$ (PM, two images), $36$ (PM, four), $42$ (SIS, two), $25$ (SIS, four), and $53$ (SIS, one). Changing the fit window to orders 2–6 moves these estimates to about 73, 62, 58, 35, and 59. The [fit parameters and measured times](cost_fit_metrics.json) are saved separately, and [plot_costs.py](plot_costs.py) reproduces the figure.

These sheared crossings are extrapolations far beyond the six implemented orders. The number of partition terms grows faster than any fixed power law at large $K$, so a power-law fit to four measured points may place the true crossing too late. Caching coefficients across repeated frequency calls, changing the number of frequencies, or optimizing the contractions would also change it. The plotted estimates describe this implementation and benchmark, not an intrinsic lens-model limit.

## Reproduction and checks

Run in the `gw` micromamba environment:

```bash
DOT_SAGE=/tmp/wave_optics_sage /home/ankur/micromamba/envs/gw/bin/sage bgo/derive.sage
/home/ankur/micromamba/envs/gw/bin/python bgo/validate.py
/home/ankur/micromamba/envs/gw/bin/python bgo/compare.py
/home/ankur/micromamba/envs/gw/bin/python bgo/reference_diagnostic.py
/home/ankur/micromamba/envs/gw/bin/python bgo/plot_costs.py
```

`derive.sage` verifies the exact Taylor coefficients $P_3$ through $P_{14}$, the lower-order isolated rational expressions, exact symbolic Wick contractions through sixth order, and floating-point coefficients against 120-bit arithmetic in a generic indefinite Hessian. `validate.py` checks published PM first order, all represented image counts, rotation covariance, isolated-lens errors through order six, and high-frequency agreement with GLoW’s regularized FFT in sheared two-image cases. The plotting code uses one CPU thread and saves 200-dpi JPEGs. If `dvipng` is unavailable, it prints an installation instruction and uses Matplotlib's built-in mathematics renderer.
