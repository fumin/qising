package math

import (
	"math"
	"math/cmplx"
)

// SVD128 performs singular value decomposition.
// https://github.com/ktye/svd.
func SVD128(A [][]complex128) ([][]complex128, [][]complex128, [][]complex128) {
	transposed := false
	if len(A) < len(A[0]) {
		transposed = true
		A = Transpose128(A)
	}

	// The original code is written in single precision with
	//
	//	eta = 1.1920929e-07
	//	tol = 1.5e-31
	//
	// While the original paper uses
	//
	//	eta = 1.5E-8
	//	tol = 1.E-31
	const eta = 2.8e-16  // Relative machine precision. In C this is DBL_EPSILON. // 2.2204460492503131E-16
	const tol = 4.0e-293 // The smallest normalized positive number, divided by eta.
	// with the smallest normalized positive number: 2.225073858507201e-308 this would be 1.0020841800044862e-292

	const zero complex128 = complex(0, 0)
	const one complex128 = complex(1, 0)

	norm := func(z complex128) float64 {
		return real(z)*real(z) + imag(z)*imag(z)
	}

	var b, c, t []float64
	var sn, w, x, y, z, cs, eps, f, g, h float64
	var i, j, k, k1, L, L1 int
	var q complex128
	var U, V [][]complex128
	var S []float64

	m := len(A)
	if m < 1 {
		panic("svd: matrix a has no rows")
	}
	n := len(A[0])
	if n < 1 {
		panic("svd: input has no columns")
	}
	for _, v := range A {
		if len(v) != n {
			panic("svd: input is not a uniform matrix")
		}
	}
	if m < n {
		panic("svd: input matrix has less rows than cols")
	}

	// Allocate temporary and result storage.
	b = make([]float64, n)
	c = make([]float64, n)
	t = make([]float64, n)

	U = make([][]complex128, m)
	for i = range U {
		U[i] = make([]complex128, n)
	}

	S = make([]float64, n)

	V = make([][]complex128, n)
	for i = range V {
		V[i] = make([]complex128, n)
	}

	// Householder Reduction.
	for {
		k1 = k + 1

		// Elimination of A[i][k], i = k, ..., m-1
		z = 0.0
		for i = k; i < m; i++ {
			z += norm(A[i][k])
		}
		b[k] = 0.0
		if z > tol {
			z = math.Sqrt(z)
			b[k] = z
			w = cmplx.Abs(A[k][k])
			q = one
			if w != 0.0 {
				q = A[k][k] / complex(w, 0)
			}
			A[k][k] = q * complex(z+w, 0)
			if k != n-1 {
				for j = k1; j < n; j++ {
					q = zero
					for i = k; i < m; i++ {
						q += cmplx.Conj(A[i][k]) * A[i][j]
					}
					q /= complex(z*(z+w), 0)
					for i = k; i < m; i++ {
						A[i][j] -= q * A[i][k]
					}
				}
			}

			// Phase Transformation.
			q = -cmplx.Conj(A[k][k]) / complex(cmplx.Abs(A[k][k]), 0)
			for j = k1; j < n; j++ {
				A[k][j] *= q
			}
		}

		// Elimination of A[k][j], j=k+2, ..., n-1
		if k == n-1 {
			break
		}
		z = 0.0
		for j = k1; j < n; j++ {
			z += norm(A[k][j])
		}
		c[k1] = 0.0
		if z > tol {
			z = math.Sqrt(z)
			c[k1] = z
			w = cmplx.Abs(A[k][k1])
			q = one
			if w != 0.0 {
				q = A[k][k1] / complex(w, 0)
			}
			A[k][k1] = q * complex(z+w, 0)
			for i = k1; i < m; i++ {
				q = zero
				for j = k1; j < n; j++ {
					q += cmplx.Conj(A[k][j]) * A[i][j]
				}
				q /= complex(z*(z+w), 0)
				for j = k1; j < n; j++ {
					A[i][j] -= q * A[k][j]
				}
			}

			// Phase Transformation.
			q = -cmplx.Conj(A[k][k1]) / complex(cmplx.Abs(A[k][k1]), 0)
			for i = k1; i < m; i++ {
				A[i][k1] *= q
			}
		}
		k = k1
	}

	// Tolerance for negligible elements.
	eps = 0.0
	for k = 0; k < n; k++ {
		S[k] = b[k]
		t[k] = c[k]
		if S[k]+t[k] > eps {
			eps = S[k] + t[k]
		}
	}
	eps *= eta

	// Initialization of U and V.
	for j = 0; j < n; j++ {
		U[j][j] = one
		V[j][j] = one
	}

	// QR Diagonalization.
	for k = n - 1; k >= 0; k-- {

		// Test for split.
		for {
			for L = k; L >= 0; L-- {
				if math.Abs(t[L]) <= eps {
					goto Test
				}
				if math.Abs(S[L-1]) <= eps {
					break
				}
			}

			// Cancellation of E(L)
			cs = 0.0
			sn = 1.0
			L1 = L - 1
			for i = L; i <= k; i++ {
				f = sn * t[i]
				t[i] *= cs
				if math.Abs(f) <= eps {
					goto Test
				}
				h = S[i]
				w = math.Sqrt(f*f + h*h)
				S[i] = w
				cs = h / w
				sn = -f / w
				for j = 0; j < n; j++ {
					x = real(U[j][L1])
					y = real(U[j][i])
					U[j][L1] = complex(x*cs+y*sn, 0)
					U[j][i] = complex(y*cs-x*sn, 0)
				}
			}

			// Test for convergence.
		Test:
			w = S[k]
			if L == k {
				break
			}

			// Origin shift.
			x = S[L]
			y = S[k-1]
			g = t[k-1]
			h = t[k]
			f = ((y-w)*(y+w) + (g-h)*(g+h)) / (2.0 * h * y)
			g = math.Sqrt(f*f + 1.0)
			if f < 0.0 {
				g = -g
			}
			f = ((x-w)*(x+w) + (y/(f+g)-h)*h) / x

			// QR Step.
			cs = 1.0
			sn = 1.0
			L1 = L + 1
			for i = L1; i <= k; i++ {
				g = t[i]
				y = S[i]
				h = sn * g
				g = cs * g
				w = math.Sqrt(h*h + f*f)
				t[i-1] = w
				cs = f / w
				sn = h / w
				f = x*cs + g*sn
				g = g*cs - x*sn
				h = y * sn
				y = y * cs
				for j = 0; j < n; j++ {
					x = real(V[j][i-1])
					w = real(V[j][i])
					V[j][i-1] = complex(x*cs+w*sn, 0)
					V[j][i] = complex(w*cs-x*sn, 0)
				}
				w = math.Sqrt(h*h + f*f)
				S[i-1] = w
				cs = f / w
				sn = h / w
				f = cs*g + sn*y
				x = cs*y - sn*g
				for j = 0; j < n; j++ {
					y = real(U[j][i-1])
					w = real(U[j][i])
					U[j][i-1] = complex(y*cs+w*sn, 0)
					U[j][i] = complex(w*cs-y*sn, 0)
				}
			}
			t[L] = 0.0
			t[k] = f
			S[k] = x
		}

		// Convergence
		if w >= 0.0 {
			continue
		}
		S[k] = -w
		for j = 0; j < n; j++ {
			V[j][k] = -V[j][k]
		}
	}

	// Sort singular values.
	for k = 0; k < n; k++ {
		g = -1.0
		j = k
		for i = k; i < n; i++ {
			if S[i] <= g {
				continue
			}
			g = S[i]
			j = i
		}
		if j == k {
			continue
		}
		S[j] = S[k]
		S[k] = g
		for i = 0; i < n; i++ {
			q = V[i][j]
			V[i][j] = V[i][k]
			V[i][k] = q
		}
		for i = 0; i < n; i++ {
			q = U[i][j]
			U[i][j] = U[i][k]
			U[i][k] = q
		}
	}

	// Back transformation.
	for k = n - 1; k >= 0; k-- {
		if b[k] == 0.0 {
			continue
		}
		q = -A[k][k] / complex(cmplx.Abs(A[k][k]), 0)
		for j = 0; j < n; j++ {
			U[k][j] *= q
		}
		for j = 0; j < n; j++ {
			q = zero
			for i = k; i < m; i++ {
				q += cmplx.Conj(A[i][k]) * U[i][j]
			}
			q /= complex(cmplx.Abs(A[k][k])*b[k], 0)
			for i = k; i < m; i++ {
				U[i][j] -= q * A[i][k]
			}
		}
	}

	if n > 1 {
		for k = n - 2; k >= 0; k-- {
			k1 = k + 1
			if c[k1] == 0.0 {
				continue
			}
			q = -cmplx.Conj(A[k][k1]) / complex(cmplx.Abs(A[k][k1]), 0)
			for j = 0; j < n; j++ {
				V[k1][j] *= q
			}
			for j = 0; j < n; j++ {
				q = zero
				for i = k1; i < n; i++ {
					q += A[k][i] * V[i][j]
				}
				q /= complex(cmplx.Abs(A[k][k1])*c[k1], 0)
				for i = k1; i < n; i++ {
					V[i][j] -= q * cmplx.Conj(A[k][i])
				}
			}
		}
	}

	// Compute full basis for U and S.
	diag := func(v []float64) [][]complex128 {
		A := make([][]complex128, len(v))
		for i := range A {
			A[i] = make([]complex128, len(v))
			A[i][i] = complex(v[i], 0)
		}
		return A
	}
	Sigma := diag(S)
	if m > n {
		for i := range len(U) {
			U[i] = append(U[i], make([]complex128, m-n)...)
		}
		for j := n; j < m; j++ {
			U[j][j] = 1
		}
		// Gram-Schmidt.
		for j := n; j < m; j++ {
			for i := 0; i < m; i++ {
				v := U[i][j]
				for k := 0; k < j; k++ {
					v -= cmplx.Conj(U[j][k]) * U[i][k]
				}
				U[i][j] = v
			}

			var ujnormf float64
			for i := 0; i < m; i++ {
				ujnormf += norm(U[i][j])
			}
			ujnorm := complex(math.Sqrt(ujnormf), 0)
			for i := 0; i < m; i++ {
				U[i][j] /= ujnorm
			}

			// Check orthogonal.
			for k := 0; k < j; k++ {
				var dot complex128
				for i := 0; i < m; i++ {
					dot += U[i][k] * cmplx.Conj(U[i][j])
				}
				if cmplx.Abs(dot) > 1e-12 {
					panic("not orthogonal")
				}
			}
		}

		for _ = range m - n {
			Sigma = append(Sigma, make([]complex128, n))
		}
	}

	if transposed {
		U, Sigma, V = Conjugate128(V), Transpose128(Sigma), Conjugate128(U)
	}

	return U, Sigma, V
}
