Defínela como **mixtura** en \([0,\infty)\) de una normal truncada en 0 y una ley de cola (Weibull o Pareto):

\[
X\sim w\,\text{TN}_0(\mu,\sigma^2)+(1-w)\,G,\qquad w\in(0,1),\;G\in\{\text{Weibull}(k{=}10,\lambda),\ \text{Pareto}(\alpha,x_m)\}.
\]

---

# 1) Truncación y componentes

## Normal truncada en 0
Sea \(Y\sim\mathcal N(\mu,\sigma^2)\) y \(X=Y\,|\,Y\ge 0\).  
PDF:
\[
f_{\mathrm{TN}}(x)=\frac{\phi\!\left(\frac{x-\mu}{\sigma}\right)}{\sigma\,\bigl[1-\Phi\!\left(\frac{0-\mu}{\sigma}\right)\bigr]}\,\mathbf 1_{x\ge 0},
\]
CDF:
\[
F_{\mathrm{TN}}(x)=\frac{\Phi\!\left(\frac{x-\mu}{\sigma}\right)-\Phi\!\left(\frac{0-\mu}{\sigma}\right)}{1-\Phi\!\left(\frac{0-\mu}{\sigma}\right)}\,\mathbf 1_{x\ge 0}.
\]
Momentos:
\[
\lambda_0:=\frac{\phi(a)}{1-\Phi(a)},\quad a=\frac{0-\mu}{\sigma}.
\]
\[
\mathbb E[X]=\mu+\sigma\,\lambda_0,\qquad 
\mathrm{Var}[X]=\sigma^2\!\left(1-a\lambda_0-\lambda_0^2\right).
\]

## Weibull \(k=10\) (escala \(\lambda>0\))
\[
f_W(x)=\frac{10}{\lambda}\Big(\frac{x}{\lambda}\Big)^{9}e^{-(x/\lambda)^{10}}\mathbf 1_{x\ge 0},\quad 
F_W(x)=1-e^{-(x/\lambda)^{10}}.
\]
Modo \(x^\star=\lambda\,(9/10)^{1/10}\). Colas superligeras (\(e^{-x^{10}}\)).  
Momentos: \(\mathbb E[X^r]=\lambda^r\,\Gamma(1+r/10)\).

## Pareto \((\alpha,x_m)\) con \(x_m>0\)
\[
f_P(x)=\frac{\alpha x_m^\alpha}{x^{\alpha+1}}\mathbf 1_{x\ge x_m},\quad
F_P(x)=\bigl(1-(x_m/x)^\alpha\bigr)\mathbf 1_{x\ge x_m}.
\]
Cola pesada \(x^{-(\alpha+1)}\).  
Existencia de momentos: \(\mathbb E[X]<\infty\iff \alpha>1\), \(\mathrm{Var}[X]<\infty\iff \alpha>2\).

---

# 2) Mixtura: PDF, CDF, cola, momentos

PDF y CDF:
\[
f(x)=w\,f_{\mathrm{TN}}(x)+(1-w)\,g(x),\qquad 
F(x)=w\,F_{\mathrm{TN}}(x)+(1-w)\,G(x),
\]
donde \(g,G\) son de Weibull o Pareto.

Supervivencia y hazard:
\[
\bar F(x)=w\,\bar F_{\mathrm{TN}}(x)+(1-w)\,\bar G(x),\qquad 
h(x)=\frac{f(x)}{\bar F(x)}.
\]

Momentos (si existen):
\[
\mathbb E[X]=w\,\mathbb E_{\mathrm{TN}}[X]+(1-w)\,\mathbb E_G[X],\quad
\mathrm{Var}[X]=w\,\mathrm{Var}_{\mathrm{TN}}+ (1-w)\,\mathrm{Var}_{G}+w(1-w)\big(\mathbb E_{\mathrm{TN}}-\mathbb E_G\big)^2.
\]

Comportamiento en cola:
- Con **Weibull k=10**: cola dominada por Weibull, ultraligera.  
- Con **Pareto**: cola pesada; si \(\alpha\) es pequeña, \(F\) hereda la pesadez.

Modalidad: puede ser unimodal o bimodal según la separación entre el modo TN (≈\(\mu+\sigma\lambda_0\) cercano a 18) y el modo de \(G\), y según \(w\).

Soporte: \([0,\infty)\). Con Pareto habrá densidad nula en \([0,x_m)\) por su componente.

---

# 3) Generación de muestras

1) Muestrea \(Z\sim\mathrm{Bernoulli}(w)\).  
2) Si \(Z=1\): muestrea \(Y\sim\mathcal N(\mu,\sigma^2)\) y acepta \(Y\) si \(Y\ge 0\) (o usa CDF truncada por inversión).  
3) Si \(Z=0\):  
   - Weibull: \(X=\lambda\,(-\ln U)^{1/10}\).  
   - Pareto: \(X=x_m\,U^{-1/\alpha}\).  
Aquí \(U\sim\mathrm{Unif}(0,1)\).

---

# 4) Estimación de parámetros (MLE por EM)

Datos \(x_{1:n}\). Latente \(Z_i\in\{0,1\}\) indica componente TN.

Log-verosimilitud completa:
\[
\ell=\sum_{i=1}^n \big[Z_i\big(\log w+\log f_{\mathrm{TN}}(x_i|\mu,\sigma)\big)+(1-Z_i)\big(\log(1-w)+\log g(x_i|\theta_G)\big)\big].
\]

**E-step** (responsabilidades):
\[
\gamma_i:=\mathbb P(Z_i=1\,|\,x_i)=\frac{w\,f_{\mathrm{TN}}(x_i)}{w\,f_{\mathrm{TN}}(x_i)+(1-w)\,g(x_i)}.
\]

**M-step**:
\[
w^{\text{new}}=\frac{1}{n}\sum_i \gamma_i.
\]

- **TN(\(\mu,\sigma\)) truncada**: maximiza
\[
Q_{\mathrm{TN}}=\sum_i \gamma_i\Big[-\log\sigma-\tfrac12\big(\tfrac{x_i-\mu}{\sigma}\big)^2-\log\big(1-\Phi(a)\big)\Big],\quad a=\tfrac{0-\mu}{\sigma}.
\]
No hay fórmula cerrada por la dependencia de \(\mu,\sigma\) en \(a\) dentro de \(\log(1-\Phi(a))\). Usar Newton o quasi-Newton sobre \((\mu,\log\sigma)\).

- **Weibull \(k=10\) con escala \(\lambda\)**: si fijas \(k=10\), el M-step para \(\lambda\) con pesos \((1-\gamma_i)\) resuelve
\[
\lambda^{10}=\frac{\sum_i (1-\gamma_i)\,x_i^{10}}{\sum_i (1-\gamma_i)}\;\Rightarrow\;\lambda=\left(\frac{\sum_i (1-\gamma_i)\,x_i^{10}}{\sum_i (1-\gamma_i)}\right)^{1/10}.
\]

- **Pareto \((\alpha,x_m)\)**: si \(x_m\) se fija, el MLE ponderado es
\[
\alpha=\frac{\sum_i (1-\gamma_i)}{\sum_{i:\,x_i\ge x_m}(1-\gamma_i)\,\log(x_i/x_m)}.
\]
Si también se estima \(x_m\), el MLE clásico toma \(x_m=\min\{x_i: \gamma_i\neq 1\}\); en mixturas conviene fijarlo o perfilarlo con restricción.

Criterio de parada: incremento de log-verosimilitud marginal \(\sum_i \log\big(w f_{\mathrm{TN}}(x_i)+(1-w)g(x_i)\big)\) por debajo de tolerancia.

Regularización útil: penalizar \(\alpha\) pequeñas o imponer priors suaves para estabilidad.

---

# 5) Identificabilidad y diagnóstico

- Reescalado y solapamiento pueden inducir **no identificabilidad local**. Inicia EM desde varios puntos.  
- Pareto fuerza colas polinomiales; verifica linealidad de \(\log\bar F\) vs \(\log x\) en cauda.  
- Con Weibull \(k=10\) la cola es tan corta que la mezcla puede “parecer” una sola campana si \(w\) alto.

---

# 6) Ajuste de \(w\) por media objetivo (opcional)

Si fijas una media deseada \(m\), resuelve
\[
m=w\,(\mu+\sigma\lambda_0)+(1-w)\,m_G \;\Rightarrow\; 
w=\frac{m-m_G}{\mu+\sigma\lambda_0-m_G},
\]
donde \(m_G=\lambda\,\Gamma(1+1/10)\) para Weibull o \(m_G=\frac{\alpha x_m}{\alpha-1}\) si \(\alpha>1\) para Pareto.

---

# 7) Implementación práctica

- PDF/CDF de mezcla: combinaciones convexas.  
- Derivadas para optimización: usa \(\partial_\mu \log(1-\Phi(a))=\tfrac{\phi(a)}{\sigma\,[1-\Phi(a)]}\), \(\partial_\sigma \log(1-\Phi(a))=\tfrac{a\,\phi(a)}{1-\Phi(a)}\,\frac{1}{\sigma}\).  
- Evita underflow en colas: trabaja en log-densidades.  
- En Pareto, valida \(x_i\ge x_m\) en la contribución de esa componente; de lo contrario su término es 0 (o \(-\infty\) en log).

---

# 8) Resumen operativo

1) Especifica \(f=f_{\mathrm{TN}}\) y \(g\in\{f_W,f_P\}\).  
2) \(f_{\text{mix}}=w f+(1-w)g\), \(F_{\text{mix}}=w F+(1-w)G\).  
3) Muestras: mezcla de componentes.  
4) Estima con EM: responsabilidades, \(w\) cerrado, \(G\) casi cerrado, TN por optimización numérica.  
5) Elige \(G\) según cola deseada: Weibull para colas cortas, Pareto para colas pesadas.