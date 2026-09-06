# Parametric Curve Parameter Estimation

## Project Overview

This project estimates the unknown parameters of a given parametric curve from a set of observed \((x,y)\) data points.

The unknown parameters are:

- \(\theta\) — rotation angle
- \(M\) — exponential growth/decay parameter
- \(X\) — horizontal translation parameter

The estimated parameters are then used to reconstruct the original parametric curve and evaluate the quality of the reconstruction using the L1 distance.

---

## Problem Statement

The given parametric curve is defined as

\[
x(t)=t\cos(\theta)-e^{M|t|}\sin(0.3t)\sin(\theta)+X
\]

\[
y(t)=42+t\sin(\theta)+e^{M|t|}\sin(0.3t)\cos(\theta)
\]

subject to the constraints

\[
0^\circ<\theta<50^\circ
\]

\[
-0.05<M<0.05
\]

\[
0<X<100
\]

and

\[
6<t<60.
\]

The provided `xy_data.csv` file contains observed points generated from this curve, but the corresponding \(t\)-values are not explicitly provided.

---

## Objective

The main objectives of this project are:

1. Estimate the unknown parameters \(\theta\), \(M\), and \(X\).
2. Recover the latent \(t\)-values corresponding to the observed points.
3. Reconstruct the parametric curve using the estimated parameters.
4. Compare the reconstructed curve with the observed data.
5. Evaluate the reconstruction using the L1 distance.
6. Verify that the estimated parameters satisfy all given constraints.

---

## Dataset

The dataset is provided as `xy_data.csv`.

It contains two columns:

- `x`
- `y`

There are 1500 observed data points.

The dataset does not contain the corresponding \(t\)-values, so the latent parameter values must be recovered during the estimation process.

---

## Mathematical Reformulation

Since the given range satisfies \(t>0\), we have

\[
|t|=t.
\]

Define

\[
A(t)=e^{Mt}\sin(0.3t).
\]

The original equations can then be written as

\[
x-X=t\cos(\theta)-A(t)\sin(\theta)
\]

\[
y-42=t\sin(\theta)+A(t)\cos(\theta).
\]

This represents a rotation of the vector

\[
\begin{bmatrix}
t\\
A(t)
\end{bmatrix}.
\]

Applying the inverse rotation gives

\[
t=(x-X)\cos(\theta)+(y-42)\sin(\theta)
\]

and

\[
A=-(x-X)\sin(\theta)+(y-42)\cos(\theta).
\]

The model therefore requires

\[
A=e^{Mt}\sin(0.3t).
\]

This transformation allows the unknown \(t\)-values to be recovered from the observed \((x,y)\) coordinates for any candidate set of parameters.

---

## Parameter Estimation

For every candidate parameter vector

\[
(\theta,M,X),
\]

the transformed values of \(t\) and \(A\) are calculated.

The residual is defined as

\[
r_i=A_i-e^{Mt_i}\sin(0.3t_i).
\]

The mean squared fitting error is

\[
E(\theta,M,X)
=
\frac{1}{N}
\sum_{i=1}^{N}r_i^2.
\]

A penalty is also applied when recovered \(t\)-values fall outside the required range \(6<t<60\).

The optimization is performed using the **Differential Evolution** algorithm from SciPy.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Jupyter Notebook / Kaggle Notebook

---

## Optimization Results

The optimization consistently converged to parameter values very close to:

\[
\boxed{\theta=30^\circ}
\]

\[
\boxed{M=0.03}
\]

\[
\boxed{X=55}
\]

The estimated numerical values from the optimization were approximately:

| Parameter | Estimated Value |
|---|---:|
| \(\theta\) | 29.999973° |
| \(M\) | 0.030000 |
| \(X\) | 54.999998 |

All estimated parameters satisfy the specified constraints.

---

## Reconstructed Parametric Curve

Using the estimated parameters, the final reconstructed curve is

\[
x(t)
=
t\cos(30^\circ)
-
e^{0.03|t|}
\sin(0.3t)
\sin(30^\circ)
+
55
\]

\[
y(t)
=
42
+
t\sin(30^\circ)
+
e^{0.03|t|}
\sin(0.3t)
\cos(30^\circ)
\]

with

\[
6<t<60.
\]

---

## L1 Distance Evaluation

The reconstructed curve is evaluated against the observed data using the pointwise L1 distance

\[
L_{1,i}
=
|x_{\text{expected},i}-x_{\text{predicted},i}|
+
|y_{\text{expected},i}-y_{\text{predicted},i}|.
\]

Using 10,000 uniformly sampled evaluation points over the range supported by the observed data, the obtained results were approximately:

| Metric | Value |
|---|---:|
| Number of samples | 10,000 |
| Total L1 distance | 1.715145 |
| Mean L1 distance | \(1.715145\times10^{-4}\) |
| Maximum pointwise L1 | \(9.083963\times10^{-3}\) |

The small mean L1 distance indicates that the reconstructed curve closely matches the observed data.

---

## Robustness

The optimization was also tested using multiple random seeds.

The optimizer converged to essentially the same parameter values for the different seeds, indicating that the obtained solution is stable and reproducible.

---

## Conclusion

The unknown parameters of the given parametric curve were successfully estimated from the observed \((x,y)\) data.

The final parameter estimates are

\[
\boxed{\theta=30^\circ,\quad M=0.03,\quad X=55}.
\]

The estimated parameters satisfy all the specified constraints, and the recovered \(t\)-values lie within the required range.

The reconstructed curve closely follows the observed data, and the small mean L1 distance confirms a high level of agreement between the observed and reconstructed curves.

---

## Files in This Repository

- `parametric-curve-parameter-estimation.ipynb` — Complete analysis, implementation, visualizations, optimization, and evaluation.

## How to Run

The notebook can be opened and executed using:

- Kaggle Notebooks
- Jupyter Notebook
- JupyterLab

The notebook contains the complete workflow from dataset loading through parameter estimation, curve reconstruction, and L1 evaluation.
