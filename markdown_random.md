
# From neuronal least action to prospective dynamics with memory

## 1. Start from what NLA is actually trying to solve

The neuronal least-action principle begins with a somato-dendritic mismatch energy. In the simplest scalar case, write the mismatch as

\[
r(t)=s(t)-f(t),
\]

where \(s(t)\) is the neural state and \(f(t)\) is the instantaneous feedforward target.

The corresponding mismatch functional is

\[
\boxed{
A_{\rm NLA}
=
\frac12\int dt\,|s-f|^2 .
}
\]

Biophysically, NLA interprets this quadratic mismatch as proportional to a somato-dendritic “virtual power”; its action is the time integral of this mismatch quantity. The Appendix-6 circuit interpretation relates it to an RC-like soma–dendrite reduction.

The distinctive NLA step is that this action is **not varied with respect to the instantaneous voltage \(s\)**.

Instead NLA introduces a prospective coordinate \(\tilde s\), defined as an exponentially discounted future voltage, satisfying

\[
\boxed{
s=(1-\tau D)\tilde s,
\qquad D=\frac{d}{dt}.
}
\]

This is the core NLA prospective transformation. The NLA paper explicitly defines the future-discounted voltage as its canonical variational coordinate.

Therefore

\[
\delta s=(1-\tau D)\delta\tilde s.
\]

Varying the mismatch action gives

\[
\delta A_{\rm NLA}
=
\int dt\,r(1-\tau D)\delta\tilde s.
\]

After integration by parts,

\[
\delta A_{\rm NLA}
=
\int dt\,
\left[(1+\tau D)r\right]
\delta\tilde s.
\]

Hence stationarity requires

\[
\boxed{
(1+\tau D)(s-f)=0.
}
\]

That is,

\[
\boxed{
\tau\dot s+s=f+\tau\dot f.
}
\]

This is the essential first-order prospective equation.

The right-hand side

\[
\boxed{
y_{\rm pros}
\equiv f+\tau\dot f
}
\]

is the NLA prospective target.

So we can rewrite NLA as

\[
\boxed{
\tau\dot s+s=y_{\rm pros}.
}
\]

---

## 2. NLA can therefore be viewed as a first-order gradient flow

Define the time-dependent prospective mismatch potential

\[
\boxed{
F_t(s)
=
\frac12
\left|s-y_{\rm pros}(t)\right|^2.
}
\]

Then

\[
\nabla_sF_t=s-y_{\rm pros},
\]

and the NLA equation becomes

\[
\boxed{
\tau\dot s
=
-\nabla_sF_t(s).
}
\]

This is an important reinterpretation.

NLA gives us two things:

\[
\boxed{
\text{the objective: }
F_t(s)=\frac12|s-y_{\rm pros}|^2
}
\]

and

\[
\boxed{
\text{the first-order dynamics: }
\tau\dot s=-\nabla F_t.
}
\]

The biological prospective mechanism determines **where the minimum should be**:

\[
y_{\rm pros}=f+\tau\dot f.
\]

At this point there is still only one dynamical neural degree of freedom.

---

## 3. Now the connection to WWJ and Nesterov becomes natural

Wibisono, Wilson and Jordan show that first-order gradient-flow-like optimization can be embedded in a larger variational family containing inertial, accelerated trajectories. Their Bregman Lagrangian generates second-order optimization dynamics, and suitable members/discretizations connect to Nesterov-type accelerated methods.

Importantly, WWJ explicitly discuss a “massless” limit in which the second-order Bregman dynamics reduces to natural gradient flow.

So the natural question is:

\[
\boxed{
\text{What happens if NLA is the massless limit of a WWJ dynamics?}
}
\]

We should not change the NLA prospective objective.

We keep

\[
F_t(s)
=
\frac12|s-y_{\rm pros}|^2,
\qquad
y_{\rm pros}=f+\tau\dot f,
\]

and simply give the neural trajectory a finite inertial degree of freedom.

Introduce

\[
M>0.
\]

Then consider the Euclidean WWJ-type action

\[
\boxed{
A_{\rm mem}[s]
=
\int dt\,
e^{\tau t/M}
\left[
\frac{M}{2}\dot s^2
-
\frac12
\left(
s-f-\tau\dot f
\right)^2
\right].
}
\]

This is precisely the Euclidean exponential WWJ form, up to an irrelevant overall constant, with mass parameter

\[
m=\frac{M}{\tau}.
\]

We have therefore made only one conceptual modification to NLA:

\[
\boxed{
\text{NLA first-order prospective flow}
\quad\longrightarrow\quad
\text{finite-mass WWJ prospective flow}.
}
\]

Crucially, the prospective target itself remains exactly the NLA one.

---

## 4. Euler–Lagrange now gives the generalized prospective dynamics

For

\[
L=
e^{\tau t/M}
\left[
\frac M2\dot s^2
-\frac12(s-y_{\rm pros})^2
\right],
\]

we have

\[
\frac{\partial L}{\partial\dot s}
=
e^{\tau t/M}M\dot s.
\]

Therefore

\[
\frac{d}{dt}
\frac{\partial L}{\partial\dot s}
=
e^{\tau t/M}
\left(
M\ddot s+\tau\dot s
\right).
\]

And

\[
\frac{\partial L}{\partial s}
=
-e^{\tau t/M}(s-y_{\rm pros}).
\]

Euler–Lagrange gives

\[
\boxed{
M\ddot s+\tau\dot s+s
=
y_{\rm pros}.
}
\]

Using the NLA target,

\[
\boxed{
M\ddot s+\tau\dot s+s
=
f+\tau\dot f.
}
\]

This is the generalized prospective equation.

Nothing has been added to the target beyond what NLA already required.

The new term is entirely on the neural side:

\[
\boxed{M\ddot s.}
\]

That is the new memory degree of freedom.

---

## 5. Ordinary NLA is recovered as the massless limit

Take

\[
M\rightarrow0.
\]

Then

\[
M\ddot s+\tau\dot s+s
=
f+\tau\dot f
\]

becomes

\[
\boxed{
\tau\dot s+s=f+\tau\dot f,
}
\]

which is exactly the first-order NLA prospective equation.

Therefore

\[
\boxed{
\text{NLA}
=
\text{massless limit of the generalized prospective action}.
}
\]

Schematically,

\[
\boxed{
\begin{array}{ccc}
M=0
&&
M>0
\\
\text{NLA}
&&
\text{WWJ-NLA}
\\
\text{one neural state}
&&
\text{state + momentum}
\\
\text{prospective, no memory mode}
&&
\text{prospective + dynamical memory}.
\end{array}}
\]

This mirrors precisely the WWJ observation that a suitable massless limit of its Bregman mechanics reduces to first-order gradient flow.

---

## 6. The transfer function shows what the extension actually accomplished

For zero initial conditions,

\[
M\ddot s+\tau\dot s+s=f+\tau\dot f
\]

gives

\[
\boxed{
H(p)
=
\frac{S(p)}{F(p)}
=
\frac{1+\tau p}
{1+\tau p+Mp^2}.
}
\]

This is the central result.

Ordinary passive second-order filtering would give

\[
H_{\rm passive}
=
\frac{1}
{1+\tau p+Mp^2}.
\]

NLA supplies the prospective zero

\[
1+\tau p.
\]

But WWJ supplies the additional poles through

\[
Mp^2.
\]

Therefore the prospective mechanism removes the ordinary first-order membrane lag **without removing the new inertial memory**.

Indeed,

\[
H(p)
=
1-Mp^2+O(p^3).
\]

There is no term linear in \(p\).

Thus the low-frequency first-order delay vanishes:

\[
\boxed{
\tau_g(0)=0.
}
\]

But

\[
H(p)\neq1.
\]

The second-order poles remain.

Hence

\[
\boxed{
\text{prospective coding}
+
\text{transfer-function memory}.
}
\]

This is fundamentally different from simply making the response instantaneous.

---

## 7. Where is the memory physically stored?

The generalized dynamics requires two initial conditions:

\[
s(0),\qquad \dot s(0).
\]

Equivalently, the WWJ action has canonical momentum

\[
\boxed{
P
=
\frac{\partial L}{\partial\dot s}
=
e^{\tau t/M}M\dot s.
}
\]

The neural state is therefore no longer characterized by \(s\) alone.

It is characterized by

\[
\boxed{(s,P).}
\]

Two neurons can have exactly the same instantaneous voltage \(s\) but different momentum \(P\), and therefore different subsequent trajectories.

That extra variable is genuine dynamical memory.

The Hamiltonian corresponding to the action is

\[
\boxed{
H(s,P,t)
=
e^{-\tau t/M}
\frac{P^2}{2M}
+
e^{\tau t/M}
F_t(s).
}
\]

So the NLA prospective objective and the WWJ momentum become the two complementary pieces of one variational system.

---

## 8. What does the residual remember?

Let

\[
r=s-f.
\]

The generalized equation gives

\[
M(\ddot r+\ddot f)
+\tau(\dot r+\dot f)
+r+f
=
f+\tau\dot f.
\]

Therefore

\[
\boxed{
M\ddot r+\tau\dot r+r
=
-M\ddot f.
}
\]

This equation gives the memory a particularly clear interpretation.

A static input has

\[
\dot f=\ddot f=0.
\]

A constant-velocity trajectory has

\[
\ddot f=0.
\]

For both, after transients,

\[
r=0.
\]

So the system prospectively compensates constant position and constant velocity.

But when

\[
\ddot f\neq0,
\]

the inertial state is excited.

The additional dynamical mode therefore carries information specifically about **changes in the trajectory**, rather than simply imposing an ordinary delay on everything.

---

## 9. Now return to the NLA biological circuit

The same structure emerges if we stop eliminating the dendritic dynamics instantaneously.

Take a linear soma–dendrite system

\[
C_s\dot s=-as+bv+\eta_s f,
\]

\[
C_d\dot v=ks-dv+\eta_d f.
\]

NLA's usual fast-dendrite reduction removes \(v\) as an independent dynamical state and gives an effectively first-order RC equation. The biological interpretation of this reduction and its relation to mismatch power is discussed in NLA Appendix 6.

If instead \(C_d\) is retained, exact elimination of \(v\) gives

\[
\boxed{
M\ddot s+\Gamma\dot s+s
=
f+T_z\dot f,
}
\]

after normalization, where \(M,\Gamma,T_z\) are determined by the circuit's capacitances, conductances and input routing. The retained-compartment derivation is exactly the origin of the extra second-order state discussed previously.

The transfer function is

\[
\boxed{
H_{\rm bio}(p)
=
\frac{1+T_zp}
{1+\Gamma p+Mp^2}.
}
\]

Thus the circuit naturally has precisely the structural form predicted by the WWJ extension:

\[
\boxed{
\text{first-order prospective numerator}
\over
\text{second-order memory denominator}.
}
\]

If the biological prospective pathway is matched so that

\[
T_z=\Gamma,
\]

then the biocircuit realizes

\[
\boxed{
H(p)
=
\frac{1+\Gamma p}
{1+\Gamma p+Mp^2},
}
\]

which is exactly the memory-preserving generalized prospective dynamics.

So the same model is reached independently from both directions:

\[
\boxed{
\text{NLA}
\rightarrow
\text{WWJ finite inertia}
}
\]

and

\[
\boxed{
\text{NLA biocircuit}
\rightarrow
\text{retain dendritic dynamics}.
}
\]

That agreement is the important bridge.

---

## 10. Where Nesterov enters—and where we should be precise

WWJ's general result is broader than the constant-coefficient equation above. Their Bregman Lagrangian

\[
\mathcal L(X,V,t)
=
e^{\alpha_t+\gamma_t}
\left[
D_h(X+e^{-\alpha_t}V,X)
-
e^{\beta_t}F