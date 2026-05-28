# Mathematical notes

Derivations behind the code, with the implementing module noted for each.

---

## 1. Continuous-time Ornstein–Uhlenbeck MLE
*Module: `stage1/ou_mle.py`*

The spread is modelled as

$$ dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t, \qquad \kappa > 0. $$

### Exact transition density

Unlike the Euler scheme (which regresses $\Delta X$ on $X$ and carries an
$O(\Delta t)$ bias in $\kappa$), the OU process has a **closed-form Gaussian
transition**:

$$ X_{t+\Delta t}\mid X_t \sim \mathcal{N}\!\left(\mu + (X_t-\mu)e^{-\kappa\Delta t},\;\; \frac{\sigma^2}{2\kappa}\bigl(1-e^{-2\kappa\Delta t}\bigr)\right). $$

This is structurally an AR(1), $X_{i+1} = a + b X_i + \varepsilon_i$, with

$$ b = e^{-\kappa\Delta t}, \qquad a = \mu(1-b), \qquad \mathrm{Var}(\varepsilon) = \frac{\sigma^2}{2\kappa}(1-b^2). $$

Because the likelihood factorises into these Gaussian transitions, **OLS on
the AR(1) is the exact MLE** for any $\Delta t$. The point estimates back out as

$$ \hat\kappa = -\frac{\ln \hat b}{\Delta t}, \qquad \hat\mu = \frac{\hat a}{1-\hat b}, \qquad \hat\sigma^2 = \hat v\,\frac{2\hat\kappa}{1-\hat b^2}, $$

where $\hat v = \tfrac1n\sum \hat\varepsilon_i^2$ is the MLE residual variance.

### Stationarity and half-life

The fit is stationary iff $\kappa>0$ **and** the lower confidence bound on
$\kappa$ exceeds 0 (so mean reversion is statistically significant — the
roadmap's "stationarity condition" gate). The half-life follows directly:

$$ t_{1/2} = \frac{\ln 2}{\kappa}, $$

which sets the natural holding period with no hand-tuning.

### Confidence intervals (Fisher information + delta method)

For the Gaussian AR(1) with parameters $(a, b, v)$, the information matrix is
block-diagonal in $v$. The OLS covariance of $(a,b)$ is $v\,(X'X)^{-1}$ with
$X=[\mathbf 1, x_{prev}]$, and $\mathrm{Var}(\hat v)=2v^2/n$. We propagate this
to $(\kappa,\mu,\sigma)$ with the Jacobian of the transform above (the *delta
method*), giving 95% intervals reported on every fit. (A subtle bug — dividing
by $v$ once instead of $v^2$ — inflates the $\kappa$ interval by a factor of
$1/v$; the test suite's CI-coverage test guards against its return.)

Note $\kappa$ has a known **finite-sample upward bias** (Tang & Chen 2009);
the coverage test relaxes nominal coverage for $\kappa$ accordingly.

---

## 2. Cointegration screening
*Module: `stage1/cointegration.py`*

* **Engle–Granger (1987).** OLS $Y_t = \alpha + \beta X_t + u_t$, then test
  $\hat u_t$ for a unit root. The residual is the *tradable spread*. We obtain
  the p-value from `statsmodels.tsa.stattools.coint`, which applies MacKinnon
  critical values appropriate for a **pre-estimated** cointegrating vector —
  plain `adfuller` on the residual over-rejects because it ignores that
  $\beta$ was fitted from the same data.
* **Johansen (1988, 1991).** Trace test on a VECM; symmetric, allows multiple
  cointegrating vectors, and lays the groundwork for Stage 6's multivariate
  VECM work. Rank is read off the 5% trace critical values.

---

## 3. Optimal double-stopping (Leung & Li, 2015)
*Module: `stage2/optimal_stopping.py`*

Hold the OU spread; pay $c_s$ to exit, $c_b$ to enter; discount at rate $r>0$.

### Fundamental solutions

The ODE $\mathcal{L}V - rV = 0$ (with $\mathcal{L}$ the OU generator) has two
positive, linearly independent solutions:

$$ F(x) = \int_0^\infty u^{\frac{r}{\kappa}-1}\exp\!\Bigl(\tfrac{\sqrt{2\kappa}}{\sigma}(x-\mu)u - \tfrac{u^2}{2}\Bigr)du \;\; (\uparrow), $$
$$ G(x) = \int_0^\infty u^{\frac{r}{\kappa}-1}\exp\!\Bigl(-\tfrac{\sqrt{2\kappa}}{\sigma}(x-\mu)u - \tfrac{u^2}{2}\Bigr)du \;\; (\downarrow). $$

We evaluate these (and their $x$-derivatives, which only differ by a factor
$u$ in the integrand) by adaptive quadrature, with the upper limit placed
several standard deviations beyond the integrand's peak.

### Liquidation (sell) boundary

$$ V(x) = \sup_\tau \mathbb{E}_x\!\left[e^{-r\tau}(X_\tau - c_s)\right]. $$

Smooth-pasting gives $V(x) = (b^\*-c_s)\,F(x)/F(b^\*)$ for $x \le b^\*$, with the
optimal take-profit $b^\*$ solving

$$ F(b^\*) = (b^\* - c_s)\,F'(b^\*). $$

### Entry boundary

$$ J(x) = \sup_\tau \mathbb{E}_x\!\left[e^{-r\tau}\bigl(V(X_\tau) - X_\tau - c_b\bigr)\right], $$

with optimal entry $d^\*$ solving

$$ G(d^\*)\bigl(V'(d^\*)-1\bigr) = G'(d^\*)\bigl(V(d^\*) - d^\* - c_b\bigr). $$

Both conditions are solved by bracketed root-finding (`scipy.optimize.brentq`).

### Short side and feasibility

OU is symmetric about **its own mean** $\mu$; with symmetric costs the short
position is the long position reflected about $\mu$: $x \mapsto 2\mu - x$.
Hence `short_entry = 2μ − long_entry`, `short_exit = 2μ − long_exit` — no
second solve. Because OU is unbounded, a feasible boundary *always* exists;
high transaction cost does not make trading "infeasible", it pushes the entry
many standard deviations from the mean (so the strategy only acts on rare,
large dislocations).

---

## 4. Sharpe ratio inference (Lo, 2002)
*Module: `validation/sharpe.py`*

For iid returns, $\mathrm{SE}(\widehat{SR}) \approx \sqrt{(1+\tfrac12\widehat{SR}^2)/T}$.
Autocorrelation inflates this by $\sqrt{\eta(q)}$ with

$$ \eta(q) = 1 + 2\sum_{k=1}^{q}\Bigl(1-\tfrac{k}{q}\Bigr)\rho_k, $$

$\rho_k$ the lag-$k$ autocorrelation and $q$ a Newey–West truncation lag
($q=\lfloor 4(T/100)^{2/9}\rfloor$, Andrews 1991). We report a **confidence
interval**, never a bare point estimate.

---

## 5. Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
*Module: `validation/deflated_sharpe.py`*

Trying $N$ strategies inflates the expected maximum Sharpe under the null:

$$ \mathbb{E}[\max_i \widehat{SR}_i] = \sqrt{V[\widehat{SR}]}\Bigl((1-\gamma_E)\Phi^{-1}\!\bigl(1-\tfrac1N\bigr) + \gamma_E\Phi^{-1}\!\bigl(1-\tfrac{1}{Ne}\bigr)\Bigr), $$

with $\gamma_E$ the Euler–Mascheroni constant and $V[\widehat{SR}]$ the
cross-sectional variance of the trial Sharpes. The DSR is the probability the
*true* skill exceeds this null benchmark:

$$ \mathrm{DSR} = \Phi\!\left(\frac{(\widehat{SR}-SR_0)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\tfrac{\gamma_4-1}{4}\widehat{SR}^2}}\right), $$

incorporating the return skew $\gamma_3$ and kurtosis $\gamma_4$. Reporting the
winning Sharpe without this deflation is the single most common stat-arb error.

---

## 6. Probability of Backtest Overfitting (Bailey et al., 2017)
*Module: `validation/pbo.py`*

Combinatorially-symmetric cross-validation (CSCV): partition the returns into
$S$ blocks, and for every split into in-sample / out-of-sample halves, pick the
best in-sample strategy and record its OOS rank percentile $\omega$ as
$\lambda=\log\frac{\omega}{1-\omega}$. PBO is the fraction of splits with
$\lambda\le 0$ — i.e. how often the in-sample winner underperforms the OOS
median. For pure-noise strategies PBO $\approx 0.5$ (tested explicitly).

---

## 7. Purged k-fold with embargo (López de Prado, 2018)
*Module: `validation/purged_cv.py`*

Naive k-fold leaks when observations are serially correlated. **Purging**
removes training points whose label window overlaps the test fold;
**embargo** drops a further band immediately after the test fold (its features
may touch test data). Both corrections are essential for honest time-series CV
and are reused throughout the validation spine.

---

### References
Engle & Granger (1987); Johansen (1988, 1991); Phillips (1972); Tang & Chen
(2009); Lo (2002); Bailey & López de Prado (2014); Bailey, Borwein, López de
Prado & Zhu (2017); López de Prado, *Advances in Financial Machine Learning*
(2018); Leung & Li, *Optimal Mean Reversion Trading* (2015); Avellaneda & Lee
(2010).
