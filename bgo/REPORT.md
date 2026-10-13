# PM and SIS wave optics with convergence and shear: ten orders beyond geometric optics

## Scope and conventions

A constant external convergence and arbitrary two-component shear have potential
\[
 \psi_{\rm ext}(\boldsymbol x)
 =\frac{\kappa}{2}|\boldsymbol x|^2+
   \frac{\gamma_1}{2}(x_1^2-x_2^2)+\gamma_2x_1x_2,\qquad
 A=(1-\kappa)I-
 \begin{pmatrix}\gamma_1&\gamma_2\\\gamma_2&-\gamma_1\end{pmatrix}.
\]
Thus the analysis covers the full quadratic external tidal potential, including all orientations and convergence. It does not assume that the source lies on a shear axis. A spatially varying external potential would need its own third through twenty-second derivatives.

Write \(a>0\) for PM strength, \(b>0\) for SIS strength, and \(w=2\pi f t_0>0\). In the angular normalization \(\boldsymbol x=\boldsymbol\theta/\theta_0\), \(\boldsymbol y=\boldsymbol\beta/\theta_0\),
\[
 t_0=\frac{(1+z_L)D_LD_S}{cD_{LS}}\theta_0^2,\quad
 a=(\theta_{E,\rm PM}/\theta_0)^2,\quad b=\theta_{E,\rm SIS}/\theta_0.
\]
The normalization used by the code and plots is
\[
 F(w)=\frac{w}{2\pi i}\int_{\mathbb R^2}
       e^{iw[\tau(\boldsymbol x,\boldsymbol y)-\tau_{\min}]}d^2x,
\]
where \(\tau_{\min}\) is the smallest ordinary-image Fermat value. The local calculations below are valid for any \((\kappa,\gamma_1,\gamma_2,\boldsymbol y,a/b)\) with isolated nonsingular ordinary images and, for the SIS center series, \(|\boldsymbol y|\ne b\). In particular, the formulae do not impose \(\gamma_2=0\), \(\kappa=0\), or unit strength. A globally conventional oscillatory diffraction integral is simplest when \(A\) is positive definite, \(\kappa+\sqrt{\gamma_1^2+\gamma_2^2}<1\); outside that sector its normalization at infinity needs care. Caustics, the aligned PM Einstein ring, and the SIS center transition are physical parameter values requiring uniform approximations. An ordinary saddle expansion of any finite order diverges there.

## Four lens models: Fermat potential and geometric optics

The four named families are recovered from two general formulae:
\[
 \begin{aligned}
 \tau_{\rm PM}&=\tfrac12|\boldsymbol x-\boldsymbol y|^2-a\log r
             -\tfrac{\kappa}{2}r^2-\tfrac12\boldsymbol x^T\Gamma\boldsymbol x,\\
 \tau_{\rm SIS}&=\tfrac12|\boldsymbol x-\boldsymbol y|^2-br
             -\tfrac{\kappa}{2}r^2-\tfrac12\boldsymbol x^T\Gamma\boldsymbol x,
 \quad r=|\boldsymbol x|.
 \end{aligned}
\]
Set \((\kappa,\Gamma)=(0,0)\) for isolated PM/SIS; set \(\kappa=0\) for their shear-only versions; retain both for the general external-potential versions. Images solve
\[
 A\boldsymbol x-\frac{a}{r^2}\boldsymbol x=\boldsymbol y\quad(\rm PM),\qquad
 A\boldsymbol x-\frac b r\boldsymbol x=\boldsymbol y\quad(\rm SIS).
\]
Their Hessians and geometric contribution are
\[
 H_{\rm PM}=A-\frac a{r^2}I+\frac{2a}{r^2}\boldsymbol e_r\boldsymbol e_r^T,\qquad
 H_{\rm SIS}=A-\frac b r(I-\boldsymbol e_r\boldsymbol e_r^T),
\]
\[
 F_{\rm geo}=\sum_j\frac{
 e^{iw(\tau_j-\tau_{\min})-i\pi m_j/2}}{\sqrt{|\det H_j|}},
 \qquad m_j=\#\{\text{negative Hessian eigenvalues}\}.
\]
These equations give the image magnification \(\mu_j=1/\det H_j\) and phase in every one-, two-, or four-image region.

For an analytic specification of all ordinary image locations, rotate into the eigenbasis of \(A\). Put \(\gamma=(\gamma_1^2+\gamma_2^2)^{1/2}\), \(\lambda_{1,2}=1-\kappa\mp\gamma\), and \(t=a/r^2\) (PM) or \(t=b/r\) (SIS). Unless a denominator vanishes, \(x_i=y_i/(\lambda_i-t)\), where the positive real roots satisfy
\[
 \begin{array}{ll}
 {\rm PM}:&
 t[y_1^2(\lambda_2-t)^2+y_2^2(\lambda_1-t)^2]
 -a(\lambda_1-t)^2(\lambda_2-t)^2=0,\\
 {\rm SIS}:&
 t^2[y_1^2(\lambda_2-t)^2+y_2^2(\lambda_1-t)^2]
 -b^2(\lambda_1-t)^2(\lambda_2-t)^2=0.
 \end{array}
\]
The code also solves symmetry-axis roots with \(t=\lambda_i\) directly. It checks every candidate against the original vector lens equation. For zero external field, PM has \(x_\pm=(y\pm\sqrt{y^2+4a})/2\). SIS has \(x_+=y+b\), and \(x_-=y-b\) when \(0<y<b\).

## Explicit analytical terms through order ten

At each ordinary image use local radial/transverse coordinates \((u,v)\), let \(r=r_j\), and write
\(\tau=\tau_j+\frac12(u,v)H_j^{\rm local}(u,v)^T+\sum_{n=3}^{22}P_n(u,v)+\cdots\).
The external potential is quadratic, so every \(P_n\), \(n\ge3\), comes from the PM or SIS itself. The following finite formula gives every coefficient required for \(K\le10\). Put \(h=n-2q\):
\[
 [u^h v^{2q}]P_n^{\rm PM}=\frac a{r^n}
 \begin{cases}(-1)^n/n,&q=0,\\
 (-1)^{q+h}\binom{n-1}{h}/(2q),&q\ge1,\end{cases}
\]
\[
 [u^h v^{2q}]P_n^{\rm SIS}=
 \begin{cases}0,&q=0,\\
 -\dfrac b{r^{n-1}}\binom{1/2}{q}(-1)^h
 \binom{2q+h-2}{h},&q\ge1.\end{cases}
\]
All coefficients with odd transverse degree vanish. For clarity, the first local terms are
\[
 \begin{array}{c|cc}
 &P_3&P_4\\ \hline
 {\rm PM}&a(-u^3/3+uv^2)/r^3&
 a(u^4/4-3u^2v^2/2+v^4/4)/r^4\\
 {\rm SIS}&buv^2/(2r^2)&b(-u^2v^2/2+v^4/8)/r^3 .
 \end{array}
\]
[SageMath verification](derive.sage) expands the potentials directly with exact rational arithmetic through \(P_{22}\).

Let \(G_j=(H_j^{\rm local})^{-1}\). The Wick average below is an explicit polynomial, valid also for indefinite saddle Hessians:
\[
 \langle u^p v^q\rangle_j=p!q!
 \sum_{\substack{2l+m=p\\m+2n=q}}
 \frac{(G_{11}/2)^lG_{12}^{m}(G_{22}/2)^n}{l!m!n!}.
\]
Put \(B_{0j}=1\). For every integer \(1\le k\le10\), define the fully analytical finite sum
\[
 \boxed{B_{kj}=
 \sum_{\substack{m_3,\ldots,m_{2k+2}\ge0\\
                  \sum_{n=3}^{2k+2}(n-2)m_n=2k}}
 \frac{(-1)^{\sum m_n}}{\prod m_n!}
 \left\langle\prod_{n=3}^{2k+2}P_n^{m_n}\right\rangle_j.}
 \tag{1}
\]
This specifies each of the ten requested coefficients in terms of
\((a\ {\rm or}\ b,\boldsymbol y,\kappa,\gamma_1,\gamma_2)\) through algebraic image roots. There are no unevaluated integrals or derivatives in Eq. (1). The first two are
\[
 B_{1j}=-\langle P_4\rangle_j+\tfrac12\langle P_3^2\rangle_j,
\]
\[
 B_{2j}=\langle P_6\rangle_j-\langle P_3P_5\rangle_j
 -\tfrac12\langle P_4^2\rangle_j+
 \tfrac12\langle P_3^2P_4\rangle_j-\tfrac1{24}\langle P_3^4\rangle_j.
\]
For isolated unit-strength PM, \(B_1=(4r^2-1)/[3(r^2+1)^3(r^2-1)]\), agreeing with [Takahashi (2004)](https://arxiv.org/abs/astro-ph/0402165). For isolated unit-strength SIS, \(B_1=1/[8r^2(r-1)]\). Higher terms are best kept in Eq. (1): already at \(k=10\), the exact generic PM coefficient has 2,475 distinct monomials in the three entries of \(G\), and the SIS one has 1,005.

The ordinary-image amplification at cumulative order \(K\) is
\[
 \boxed{F_{\rm images}^{(K)}=
 \sum_j\frac{e^{iw(\tau_j-\tau_{\min})-i\pi m_j/2}}{\sqrt{|\det H_j|}}
 \sum_{k=0}^{K}\frac{i^k B_{kj}}{w^k},\qquad 0\le K\le10.}
 \tag{2}
\]
Thus Eq. (2) directly covers PM, PM with shear, and PM with convergence plus shear.

### SIS conical center

The SIS has a contribution from its nondifferentiable center, even if there is no central ordinary image. It starts at order \(w^{-1}\); omitting it would invalidate the requested accuracy. Define
\[
 z=b^2-|\boldsymbol y|^2-i0,\quad
 C_0=ibz^{-3/2},\quad
 D_A=A_{11}\partial_{y_1}^2+2A_{12}\partial_{y_1}\partial_{y_2}
       +A_{22}\partial_{y_2}^2,\quad
 \tau_c=|\boldsymbol y|^2/2.
\]
Then the SIS center series through the same cumulative order is
\[
 \boxed{F_{\rm center}^{(K)}=
 e^{iw(\tau_c-\tau_{\min})}
 \sum_{m=0}^{K-1}\frac{(-i/2)^m}{m!\,w^{m+1}}D_A^m C_0.}
 \tag{3}
\]
The \(m=9\) term contains eighteen source derivatives. The code evaluates them by a sparse derivative recurrence, so Eq. (3) is ready for arbitrary inputs. For example its first two terms are
\[
 e^{iw(\tau_c-\tau_{\min})}
 \left[\frac{ib}{wz^{3/2}}+
 \frac b{2w^2}\left(\frac{3\,{\rm tr}A}{z^{5/2}}+
       \frac{15\boldsymbol y^TA\boldsymbol y}{z^{7/2}}\right)\right].
\]
The \(-i0\) fixes the outgoing-wave branch. The complete SIS result is Eqs. (2) plus (3).

### Exact mass-sheet check

For \(s=1-\kappa>0\), a change of integration coordinates proves the exact identity
\[
 F_{\rm PM}(w,\boldsymbol y,a,\kappa,\Gamma)
 =s^{-1}F_{\rm PM}(w,\boldsymbol y/\sqrt s,a,0,\Gamma/s),
\]
\[
 F_{\rm SIS}(w,\boldsymbol y,b,\kappa,\Gamma)
 =s^{-1}F_{\rm SIS}(w,\boldsymbol y/\sqrt s,b/\sqrt s,0,\Gamma/s).
 \tag{4}
\]
The constant phase induced by this change cancels under the \(\tau_{\min}\) normalization. The code checks Eq. (4) separately at orders 0, 1, 6, 8, and 10 for off-axis sources, nonzero shear,
and \(s=0.6,0.82,1.2\). At order ten the relative discrepancy in that test is about \(10^{-12}\). This also checks all ten SIS center terms.

## Comparison with full GLoW wave optics

The [comparison script](compare.py) and [machine-readable measurements](comparison_metrics.json) use \(y=0.5\) in all thirteen examples. PM always has \(a=1\). The two- and four-image SIS examples have \(b=1\); the one-image SIS examples use \(b=0.4\), because a unit SIS at \(y=0.5\) necessarily has two images. Weak-shear two-image cases use \((\gamma_1,\gamma_2)=(0.05,0.03)\), while four-image cases use \((0.40,0.02)\). The latter is strong enough to place the same \(y=0.5\) source inside the caustic while retaining a positive-definite external quadratic form for both \(\kappa=0\) and \(0.2\).

Comparing a common numerical \(w\) can still compare different numbers of fringes. For each case define
\[
 \Delta\tau=\max_j\tau_j-\min_j\tau_j,
 \qquad N_\Delta=\frac{w\Delta\tau}{2\pi}.
\]
For SIS, the center phase \(\tau_c=y^2/2\) is included among the \(\tau_j\); this is essential in a one-image case. Every new plot spans the same \(N_\Delta=0.5\) to \(20\), samples \(N_\Delta=10\) exactly, and states the corresponding \(w\) range. Thus the comparison now controls both source position and the number of phase oscillations. Each plot contains selected cumulative orders and an error heat map for every order 0 through 10. The plotted error is \(|F_K-F_{\rm GLoW}|/\max(|F_{\rm GLoW}|,0.2)\).

The isolated and convergence-only references use GLoW's analytic PM or semi-analytic SIS solver with Eq. (4). Generic shear cases use the multi-contour response and regularized FFT at \(N_t=20{,}000\), with the gray curve showing its difference from \(N_t=16{,}000\). GLoW's contour initialization fails for the matched-y four-image PM example, so that one reference uses GLoW's robust area integral at \(n_\rho=80{,}000\), \(n_\theta=4{,}000\), followed by the same regularized FFT; its gray curve compares \(n_\rho=40{,}000\). Resolution spread is a diagnostic rather than a rigorous error bound. GLoW emitted contour-bracketing or saddle-integration warnings in some shear cases.

The following heat map compares every case at exactly ten phase cycles. It makes the effect of matched oscillation range directly visible.

![Matched-y errors at ten phase cycles](matched_y_error_at_ten_cycles.jpg)

### PM: isolated, convergence, shear, and convergence plus shear

![PM isolated, two images](pm_isolated_two_orders_0_to_10.jpg)

![PM convergence, two images](pm_convergence_two_orders_0_to_10.jpg)

![PM shear, two images](pm_shear_two_orders_0_to_10.jpg)

![PM convergence and shear, two images](pm_convergence_shear_two_orders_0_to_10.jpg)

![PM convergence and shear, four images](pm_convergence_shear_four_orders_0_to_10.jpg)

### SIS: one, two, and four-image regimes

![SIS isolated, two images](sis_isolated_two_orders_0_to_10.jpg)

![SIS isolated, one image](sis_isolated_one_orders_0_to_10.jpg)

![SIS convergence, two images](sis_convergence_two_orders_0_to_10.jpg)

![SIS convergence, one image](sis_convergence_one_orders_0_to_10.jpg)

![SIS shear, four images](sis_shear_four_orders_0_to_10.jpg)

![SIS convergence and shear, two images](sis_convergence_shear_two_orders_0_to_10.jpg)

![SIS convergence and shear, four images](sis_convergence_shear_four_orders_0_to_10.jpg)

![SIS convergence and shear, one image](sis_convergence_shear_one_orders_0_to_10.jpg)

Matching \(y\) and \(N_\Delta\) removes a substantial confounding effect. At ten cycles, isolated PM and PM plus convergence have geometric errors \(3.36\times10^{-3}\) and \(3.21\times10^{-3}\), respectively, and both reach roughly \(10^{-9}\) by order six. The isolated and convergence-only two-image SIS cases have identical order-dependent errors, as required by the exact mass-sheet scaling: their geometric error is \(3.28\times10^{-2}\), and order six gives \(7.58\times10^{-5}\).

It does not remove the qualitative difference between regular and singular parameter regions. The convergence-plus-shear four-image PM example is best at order three at ten cycles; order ten has already diverged. The corresponding SIS four-image example is also best at order three. The shear-only four-image SIS is best at order one. In the one-image SIS examples, the center expansion is best at order one and later center derivatives grow rapidly. These failures therefore survive the matched-y, matched-fringe comparison.

The reason is that \(N_\Delta\) measures a global interference scale, whereas stationary-phase accuracy is controlled locally by the inverse Hessian and higher derivatives at each image, and for SIS by distance from the center transition \(|y|=b\). Near a caustic, a small Hessian eigenvalue makes the coefficients large even if the signal has completed many fringes. An asymptotic series improves only up to its frequency-dependent optimal truncation order; adding every available term is not expected to improve it monotonically. Where the gray GLoW spread is comparable to the plotted error, no order-by-order numerical claim is made.

## Cumulative computational cost

The [cost plot](computational_costs_orders_0_to_10.jpg) measures complete one-thread evaluation of 121 frequencies, including image solving and construction of all coefficients through \(K\). The horizontal line is the complete GLoW reference computation used for that example. The [fit data](cost_fit_metrics.json) fit \(T_K=A K^p\) over \(K=7,\ldots,10\), and record sensitivity to fitting \(K=6,\ldots,10\). These are empirical local fits, not complexity laws: the partition count and high-precision cancellations make distant extrapolations unreliable. For the isolated lenses, GLoW's dedicated analytic or semi-analytic solver is already faster than high-order symbolic evaluation. For generic shear, measured order ten is generally below the cost of constructing the GLoW time response; the fitted shear-case crossing is roughly order 11 to 15 in these matched-y configurations and remains only a rough estimate.

![Cumulative computational cost and power-law fits](computational_costs_orders_0_to_10.jpg)

## Reproduction and limits

Use the gw environment: sage bgo/derive.sage, python bgo/validate.py, python bgo/compare.py, and python bgo/plot_costs.py. Call the amplification function from [expansion.py](expansion.py); model is PM or SIS, source is a two-vector, and \(0\le K\le10\). Orders seven through ten use 80-digit contractions because double precision suffers severe cancellation. In the GLoW plots, \(w\) is dimensionless; a physical \(f\) requires the chosen mass, distances, and \(\theta_0\). The method is a high-frequency fixed-parameter expansion and does not replace uniform caustic or SIS center-transition wave optics.
