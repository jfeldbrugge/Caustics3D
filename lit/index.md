---
title: Caustic skeleton of 3D gradient deformation fields
author: Feldbrugge & Hidding
---

Lagrangian map:

$$\vec{x}(t) = \vec{q} + D \vec{v}(D).$$

Hessian:

$$H_{ij} = \frac{\partial v_i}{\partial x_j}.$$

We have eigenvalues $\alpha \le \beta \le \gamma$ and corresponding eigenvectors $v_{\alpha}$, $v_{\beta}$, $v_{\gamma}$. Alternatively we may write $\lambda_1 \dots \lambda_3$ and $v_1 \dots v_3$.

# Singularities

$A_2$ or fold singularities can be found where $D = 1/\lambda_i$.

$A_3$ or cusp singularities can be found where $v_i \cdot \grad lambda_i = 0.$

We know how to find these in 2D from Hidding 2014, which can be extended to 3D.

# Numeric methods
We assume the values of the Hessian are known at grid points. By means of interpolation we can compute continuous functional representations of the eigenvalue and eigenvector fields.

Each cube on the grid is divided into six tetrahedra. To compute $A_2$ singularities we want to extract the level set for $\lambda_i = 1/D$. We use marching tetrahedra algorithm on the interpolated eigenvalue function. This gives us a guaranteed manifold with correct structure (i.e. no boundary).

## Marching tetrahedra
Given a tetrahedron we may find the species of resulting triangle by comparing vertex values to the iso contour level, where earch vertex is assigned a bit (e.g. 0 for smaller than x, 1 for larger or equal to x), resulting in one of 16 possible geometries, ranging from empty, to triangle or quadrilangle. Due to symmetry only 8 unique cases arise.

First $A_2$ is computed for each tetrahedron separately, then the intersection of $A_3$ surfaces with $A_2$ can be computed for that specific time $D$.

In a similar way we can compute the entire $D_3$ surface (the bigcaustic), and recursively $A_4$ lines are found on this surface, and $A_5$ on the $A_4$ lines. For each step in this cascade we have the following constraints:

| caustic | constraint |
|---|---|
| $A_3$ | $v_i \cdot \grad \lambda_i = 0$ |
| $A_4$ | $v_i \cdot \grad (v_i \cdot \grad \lambda_i) = 0$ |
| $A_5$ | $v_i \cdot \grad (v_i \cdot \grad (v_i \cdot \lambda_i)) = 0$ |

## $D_4$
This is the hard one. Feldbrugge (eq 4.15-17) derived a set of conditions which we can use to detect those places where
$$\alpha = \beta\quad {\rm or}\quad \beta = \gamma.$$
To remind ourselves, why is this so hard? We have methods of computing the eigenvalues such that $\alpha \ge \beta \ge \gamma$. We used Cardano's formula on the characteristic equation of the Hessian to find these. It is not possible however, to find a good continuous representation where these fields are disambiguated such that we can compute the level set of say ($\alpha - \beta$). To understand this we should learn about the geometry of the eigenvalue fields: near points where we expect $\alpha = \beta$, on a 2D domain these fields assume the shape of a cone. From generic fields with transverse crossings we would expect these to intersect in a line, however, we find only points. In 3D the equation $\alpha = \beta$ yields curves, again one dimension less than we would expect from a naive level set. We need to expand the condition to a set of at least two conditions, to arrive at curves.

$$\begin{align}
C_1: &H_{11} H_{13} H_{23} + H_{12} H_{23}^2 = H_{22} H_{13} H_{23} + H_{12} H_{13}^2\\
C_2: &H_{22} H_{12} H_{13} + H_{23} H_{13}^2 = H_{33} H_{12} H_{13} + H_{23} H_{12}^2\\
C_3: &H_{33} H_{23} H_{12} + H_{13} H_{12}^2 = H_{11} H_{23} H_{12} + H_{13} H_{23}^2
\end{align}$$

Two of these conditions suffice to compute the $D_4$ singularities, but where they differ is in their numerical stability. If say $H_{12}$ is large compared to $H_{13}$ and $H_{23}$, it makes more sense to use $C_2$ and $C_3$ over $C_1$. Note that the relations are completely cyclic.
