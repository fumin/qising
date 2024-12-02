package tensor

import (
	"cmp"
	"fmt"
	"log"
	"math/cmplx"
	"slices"
)

type Eig struct {
	qr         *QR
	hessenberg *Hessenberg

	eigvals *Dense
	eigvecs *Dense

	a           *Dense
	q           *Dense
	permutation *Dense
	buf         *Dense
	valPos      []valuePosition
}

func NewEig() *Eig {
	solver := &Eig{
		qr:          NewQR(),
		hessenberg:  NewHessenberg(),
		eigvals:     Zeros(1),
		eigvecs:     Zeros(1),
		a:           Zeros(1),
		q:           Zeros(1),
		permutation: Zeros(1),
		buf:         Zeros(1),
	}
	return solver
}

func (solver *Eig) Solve(a *Dense) (*Dense, *Dense) {
	//hess, q := solver.hessenberg.Solve(a)
	//log.Printf("hess %#v %#v", hess.ToSlice2(), q.ToSlice2())
	//for i := range 9999 {
	//
	//}

	// a0 = q0 @ q1 @ q2 @... a3 ...@ q2.H @ q1.H @ q0.H
	// solver.q = q0 @ q1 @ q2 @...
	m := a.shape[0]
	solver.a.Copy(a)
	solver.q.Eye(m, 0)
	for i := range 20 {
		log.Printf("a %d %#v", i, solver.a.ToSlice2())
		if lowerTriangle(solver.a) < 1e-3 {
			log.Printf("done")
			break
		}
		q, r := solver.qr.Solve(solver.a)
		MatMul(solver.a, r, q)
		solver.q.Copy(MatMul(solver.buf, solver.q, q))
	}

	mya := Zeros(1)
	MatMul(mya, MatMul(solver.buf, solver.q, solver.a), solver.q.H())
	log.Printf("qqqq %#v", solver.q.ToSlice2())
	log.Printf("mymya aaa %#v", mya.ToSlice2())
	log.Printf("tiaggi %#v", solver.a.ToSlice2())

	// Now solver.a is triangle, get its eigenvectors.
	solver.eigvals = Zeros(m)
	solver.eigvecs = Zeros(m, m)
	for i := range m {
		lambda := solver.a.data[solver.a.at([]int{i, i})]
		solver.eigvals.data[solver.eigvals.at([]int{i})] = lambda

		aMinusLambda := solver.buf.Copy(solver.a)
		for j := range m {
			aMinusLambda.data[aMinusLambda.at([]int{j, j})] -= lambda
		}
		vec := solver.eigvecs.Slice([][2]int{{0, m}, {i, i + 1}})
		log.Printf("eigenvalue %v", lambda)
		backSubstitution(vec, aMinusLambda, i)
	}

	log.Printf("tridiago %#v", solver.a.ToSlice2())
	log.Printf("tri vals %#v", solver.eigvals.ToSlice1())
	log.Printf("tri eigvecs %#v", solver.eigvecs.ToSlice2())

	solver.eigvecs.Copy(MatMul(solver.buf, solver.q, solver.eigvecs))

	log.Printf("unsorted vecs %#v", solver.eigvecs.ToSlice2())

	solver.sortByEigenvalue()

	return solver.eigvals, solver.eigvecs
}

func backSubstitution(x *Dense, l *Dense, zeroIndex int) {
	log.Printf("LLLLLLLL %#v", l.ToSlice2())
	m := x.shape[0]
	for i := m - 1; i >= 0; i-- {
		var v complex64
		for j := m - 1; j > i; j-- {
			lv := l.data[l.at([]int{i, j})]
			xv := x.data[x.at([]int{j, 0})]
			v -= lv * xv
		}
		lii := l.data[l.at([]int{i, i})]
		if abs(lii) < 1e-6 {
			// Only set to 1 if specified to achive independent vectors in the null space.
			if i == zeroIndex {
				v = 1
			} else {
				v = 0
			}
		} else {
			v /= lii
		}
		log.Printf("%d %v vvvv %v", i, l.data[l.at([]int{i, i})], v)
		x.data[x.at([]int{i, 0})] = v
	}
}

type valuePosition struct {
	value    complex64
	position int
}

func (solver *Eig) sortByEigenvalue() {
	m := solver.eigvals.shape[0]
	solver.valPos = solver.valPos[:0]
	for i := range m {
		ev := solver.eigvals.At(i, 0)
		vp := valuePosition{value: ev, position: i}
		solver.valPos = append(solver.valPos, vp)
	}
	slices.SortFunc(solver.valPos, func(a, b valuePosition) int {
		return cmp.Compare(real(a.value), real(b.value))
	})
	solver.permutation.Zeros(len(solver.valPos), len(solver.valPos))
	for i, vp := range solver.valPos {
		ptr := solver.permutation.at([]int{i, vp.position})
		solver.permutation.data[ptr] = 1
	}

	log.Printf("permutation %#v", solver.permutation.ToSlice2())

	MatMul(solver.eigvals, solver.permutation, MatMul(solver.buf, solver.eigvals.ToDiag(), solver.permutation.H()))
	solver.eigvecs.Copy(MatMul(solver.buf, solver.eigvecs, solver.permutation.H()))
}

type Hessenberg struct {
	hess           *Dense
	q              *Dense
	identity       *Dense
	householderSub *Dense
	householder    *Dense
	u              *Dense
	v              *Dense
	buf            *Dense
}

func NewHessenberg() *Hessenberg {
	solver := &Hessenberg{
		hess:           Zeros(1),
		q:              Zeros(1),
		identity:       Zeros(1),
		householderSub: Zeros(1),
		householder:    Zeros(1),
		u:              Zeros(1),
		v:              Zeros(1),
		buf:            Zeros(1),
	}
	return solver
}

func (solver *Hessenberg) Solve(a *Dense) (*Dense, *Dense) {
	n := a.shape[0]
	solver.hess.Copy(a)
	solver.q.Eye(n, 0)
	solver.identity.Eye(n, 0)

	for i := 1; i <= n-2; i++ {
		// Note that we take [i:,i-1], whereas QR takes [i:,i].
		x := solver.hess.Slice([][2]int{{i, n}, {i - 1, i}})
		e := solver.identity.Slice([][2]int{{i, n}, {i, i + 1}})

		// Make v.
		xk := x.At(0, 0)
		eixk := complex64(cmplx.Rect(1, cmplx.Phase(complex128(xk))))
		alpha := eixk * complex(x.FrobeniusNorm(), 0)
		Add(solver.u, x, Mul(solver.buf, -alpha, e))
		Mul(solver.v, complex(1/solver.u.FrobeniusNorm(), 0), solver.u)

		// Make householder matrix from v.
		MatMul(solver.householderSub, solver.v, solver.v.H())
		Mul(solver.householderSub, -2, solver.householderSub)
		hm := solver.householderSub.Shape()[0]
		identity := solver.identity.Slice([][2]int{{0, hm}, {0, hm}})
		Add(solver.householderSub, identity, solver.householderSub)

		// Multiply householder.
		solver.householder.Eye(n, 0)
		solver.householder.Set([]int{i, i}, solver.householderSub)
		// hess = householder @ hess @ householder.H.
		MatMul(solver.buf, solver.hess, solver.householder.H())
		MatMul(solver.hess, solver.householder, solver.buf)
		// q = q @ householder.H.
		solver.q.Copy(MatMul(solver.buf, solver.q, solver.householder.H()))
	}

	return solver.hess, solver.q
}

type QR struct {
	q *Dense
	r *Dense

	identity *Dense
	u        *Dense
	v        *Dense
	hh       *Dense
	fullHh   *Dense
	buf      *Dense
}

func NewQR() *QR {
	solver := &QR{
		q: Zeros(1),
		r: Zeros(1),

		identity: Zeros(1),
		u:        Zeros(1),
		v:        Zeros(1),
		hh:       Zeros(1),
		fullHh:   Zeros(1),
		buf:      Zeros(1),
	}
	return solver
}

func (solver *QR) Solve(a *Dense) (*Dense, *Dense) {
	m, n := a.shape[0], a.shape[1]
	solver.q.Eye(m, 0)
	solver.r.Copy(a)
	solver.identity.Eye(m, 0)

	last := n
	if m == n {
		last--
	}
	for i := range last {
		sliceAx := [][2]int{{i, m}, {i, i + 1}}
		x := solver.r.Slice(sliceAx)
		e := solver.identity.Slice(sliceAx)

		solver.householder(i, solver.hh, x, e)
		solver.fullHh.Eye(m, 0)
		solver.fullHh.Set([]int{i, i}, solver.hh)

		solver.q.Copy(MatMul(solver.buf, solver.q, solver.fullHh.H()))
		solver.r.Copy(MatMul(solver.buf, solver.fullHh, solver.r))
	}

	// Make all diagonals of R positive.
	for i := range solver.r.shape[1] {
		rv := solver.r.At(i, i)
		phase := complex64(cmplx.Rect(1, -cmplx.Phase(complex128(rv))))

		ptr := solver.identity.at([]int{i, i})
		solver.identity.data[ptr] = phase
	}
	solver.r.Copy(MatMul(solver.buf, solver.identity, solver.r))
	solver.q.Copy(MatMul(solver.buf, solver.q, solver.identity.H()))

	return solver.q, solver.r
}

func (solver *QR) householder(iteration int, hh, x, e *Dense) {
	xk := x.At(0, 0)
	eixk := complex64(cmplx.Rect(1, cmplx.Phase(complex128(xk))))
	alpha := eixk * complex(x.FrobeniusNorm(), 0)

	Add(solver.u, x, Mul(solver.buf, -alpha, e))
	Mul(solver.v, complex(1/solver.u.FrobeniusNorm(), 0), solver.u)

	MatMul(hh, solver.v, solver.v.H())
	Mul(hh, -2, hh)
	hm := hh.Shape()[0]
	identity := solver.identity.Slice([][2]int{{0, hm}, {0, hm}})
	Add(hh, identity, hh)
}

type SVD struct {
	u *Dense
	s *Dense
	v *Dense

	A [][]complex64
	U [][]complex64
	S []float32
	V [][]complex64

	b []float32
	c []float32
	t []float32

	Sigma [][]complex64
}

func NewSVD() *SVD {
	solver := &SVD{u: Zeros(1), s: Zeros(1), v: Zeros(1)}
	return solver
}

func (solver *SVD) Solve(aT *Dense) (*Dense, *Dense, *Dense) {
	transposed := false
	if aT.Shape()[0] < aT.Shape()[1] {
		transposed = true
		aT = aT.Transpose(1, 0)
	}

	// Prepare the solver.A matrix.
	alen := len(solver.A)
	for i := 0; i < aT.Shape()[0]-alen; i++ {
		solver.A = append(solver.A, make([]complex64, aT.Shape()[1]))
	}
	solver.A = solver.A[:aT.Shape()[0]]
	for i := range solver.A {
		solver.A[i] = solver.A[i][:0]
		solver.A[i] = append(solver.A[i], make([]complex64, aT.Shape()[1])...)
	}
	for i := 0; i < len(solver.A); i++ {
		for j := 0; j < len(solver.A[i]); j++ {
			solver.A[i][j] = aT.At(i, j)
		}
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
	// const eta = 2.8e-16  // Relative machine precision. In C this is DBL_EPSILON. // 2.2204460492503131E-16
	// const tol = 4.0e-293 // The smallest normalized positive number, divided by eta.
	// with the smallest normalized positive number: 2.225073858507201e-308 this would be 1.0020841800044862e-292
	const eta = 1.5e-8 // Relative machine precision. In C this is DBL_EPSILON. // 2.2204460492503131E-16
	const tol = 1.e-31 // The smallest normalized positive number, divided by eta.

	const zero complex64 = complex(0, 0)
	const one complex64 = complex(1, 0)

	norm := func(z complex64) float32 {
		return real(z)*real(z) + imag(z)*imag(z)
	}

	var sn, w, x, y, z, cs, eps, f, g, h float32
	var i, j, k, k1, L, L1 int
	var q complex64

	m := len(solver.A)
	if m < 1 {
		panic("svd: matrix a has no rows")
	}
	n := len(solver.A[0])
	if n < 1 {
		panic("svd: input has no columns")
	}
	for _, v := range solver.A {
		if len(v) != n {
			panic("svd: input is not a uniform matrix")
		}
	}
	if m < n {
		panic("svd: input matrix has less rows than cols")
	}

	// Allocate temporary and result storage.
	solver.b = solver.b[:0]
	solver.b = append(solver.b, make([]float32, n)...)
	solver.c = solver.c[:0]
	solver.c = append(solver.c, make([]float32, n)...)
	solver.t = solver.t[:0]
	solver.t = append(solver.t, make([]float32, n)...)

	ulen := len(solver.U)
	for i := 0; i < m-ulen; i++ {
		solver.U = append(solver.U, make([]complex64, n))
	}
	solver.U = solver.U[:m]
	for i := range solver.U {
		solver.U[i] = solver.U[i][:0]
		solver.U[i] = append(solver.U[i], make([]complex64, n)...)
	}

	solver.S = solver.S[:0]
	solver.S = append(solver.S, make([]float32, n)...)

	vlen := len(solver.V)
	for i := 0; i < n-vlen; i++ {
		solver.V = append(solver.V, make([]complex64, n))
	}
	solver.V = solver.V[:n]
	for i := range solver.V {
		solver.V[i] = solver.V[i][:0]
		solver.V[i] = append(solver.V[i], make([]complex64, n)...)
	}

	// Householder Reduction.
	for {
		k1 = k + 1

		// Elimination of A[i][k], i = k, ..., m-1
		z = 0.0
		for i = k; i < m; i++ {
			z += norm(solver.A[i][k])
		}
		solver.b[k] = 0.0
		if z > tol {
			z = sqrt(z)
			solver.b[k] = z
			w = abs(solver.A[k][k])
			q = one
			if w != 0.0 {
				q = solver.A[k][k] / complex(w, 0)
			}
			solver.A[k][k] = q * complex(z+w, 0)
			if k != n-1 {
				for j = k1; j < n; j++ {
					q = zero
					for i = k; i < m; i++ {
						q += conj(solver.A[i][k]) * solver.A[i][j]
					}
					q /= complex(z*(z+w), 0)
					for i = k; i < m; i++ {
						solver.A[i][j] -= q * solver.A[i][k]
					}
				}
			}

			// Phase Transformation.
			q = -conj(solver.A[k][k]) / complex(abs(solver.A[k][k]), 0)
			for j = k1; j < n; j++ {
				solver.A[k][j] *= q
			}
		}

		// Elimination of A[k][j], j=k+2, ..., n-1
		if k == n-1 {
			break
		}
		z = 0.0
		for j = k1; j < n; j++ {
			z += norm(solver.A[k][j])
		}
		solver.c[k1] = 0.0
		if z > tol {
			z = sqrt(z)
			solver.c[k1] = z
			w = abs(solver.A[k][k1])
			q = one
			if w != 0.0 {
				q = solver.A[k][k1] / complex(w, 0)
			}
			solver.A[k][k1] = q * complex(z+w, 0)
			for i = k1; i < m; i++ {
				q = zero
				for j = k1; j < n; j++ {
					q += conj(solver.A[k][j]) * solver.A[i][j]
				}
				q /= complex(z*(z+w), 0)
				for j = k1; j < n; j++ {
					solver.A[i][j] -= q * solver.A[k][j]
				}
			}

			// Phase Transformation.
			q = -conj(solver.A[k][k1]) / complex(abs(solver.A[k][k1]), 0)
			for i = k1; i < m; i++ {
				solver.A[i][k1] *= q
			}
		}
		k = k1
	}

	// Tolerance for negligible elements.
	eps = 0.0
	for k = 0; k < n; k++ {
		solver.S[k] = solver.b[k]
		solver.t[k] = solver.c[k]
		if solver.S[k]+solver.t[k] > eps {
			eps = solver.S[k] + solver.t[k]
		}
	}
	eps *= eta

	// Initialization of U and V.
	for j = 0; j < n; j++ {
		solver.U[j][j] = one
		solver.V[j][j] = one
	}

	// QR Diagonalization.
	for k = n - 1; k >= 0; k-- {

		// Test for split.
		for {
			for L = k; L >= 0; L-- {
				if absf(solver.t[L]) <= eps {
					goto Test
				}
				if absf(solver.S[L-1]) <= eps {
					break
				}
			}

			// Cancellation of E(L)
			cs = 0.0
			sn = 1.0
			L1 = L - 1
			for i = L; i <= k; i++ {
				f = sn * solver.t[i]
				solver.t[i] *= cs
				if absf(f) <= eps {
					goto Test
				}
				h = solver.S[i]
				w = sqrt(f*f + h*h)
				solver.S[i] = w
				cs = h / w
				sn = -f / w
				for j = 0; j < n; j++ {
					x = real(solver.U[j][L1])
					y = real(solver.U[j][i])
					solver.U[j][L1] = complex(x*cs+y*sn, 0)
					solver.U[j][i] = complex(y*cs-x*sn, 0)
				}
			}

			// Test for convergence.
		Test:
			w = solver.S[k]
			if L == k {
				break
			}

			// Origin shift.
			x = solver.S[L]
			y = solver.S[k-1]
			g = solver.t[k-1]
			h = solver.t[k]
			f = ((y-w)*(y+w) + (g-h)*(g+h)) / (2.0 * h * y)
			g = sqrt(f*f + 1.0)
			if f < 0.0 {
				g = -g
			}
			f = ((x-w)*(x+w) + (y/(f+g)-h)*h) / x

			// QR Step.
			cs = 1.0
			sn = 1.0
			L1 = L + 1
			for i = L1; i <= k; i++ {
				g = solver.t[i]
				y = solver.S[i]
				h = sn * g
				g = cs * g
				w = sqrt(h*h + f*f)
				solver.t[i-1] = w
				cs = f / w
				sn = h / w
				f = x*cs + g*sn
				g = g*cs - x*sn
				h = y * sn
				y = y * cs
				for j = 0; j < n; j++ {
					x = real(solver.V[j][i-1])
					w = real(solver.V[j][i])
					solver.V[j][i-1] = complex(x*cs+w*sn, 0)
					solver.V[j][i] = complex(w*cs-x*sn, 0)
				}
				w = sqrt(h*h + f*f)
				solver.S[i-1] = w
				cs = f / w
				sn = h / w
				f = cs*g + sn*y
				x = cs*y - sn*g
				for j = 0; j < n; j++ {
					y = real(solver.U[j][i-1])
					w = real(solver.U[j][i])
					solver.U[j][i-1] = complex(y*cs+w*sn, 0)
					solver.U[j][i] = complex(w*cs-y*sn, 0)
				}
			}
			solver.t[L] = 0.0
			solver.t[k] = f
			solver.S[k] = x
		}

		// Convergence
		if w >= 0.0 {
			continue
		}
		solver.S[k] = -w
		for j = 0; j < n; j++ {
			solver.V[j][k] = -solver.V[j][k]
		}
	}

	// Sort singular values.
	for k = 0; k < n; k++ {
		g = -1.0
		j = k
		for i = k; i < n; i++ {
			if solver.S[i] <= g {
				continue
			}
			g = solver.S[i]
			j = i
		}
		if j == k {
			continue
		}
		solver.S[j] = solver.S[k]
		solver.S[k] = g
		for i = 0; i < n; i++ {
			q = solver.V[i][j]
			solver.V[i][j] = solver.V[i][k]
			solver.V[i][k] = q
		}
		for i = 0; i < n; i++ {
			q = solver.U[i][j]
			solver.U[i][j] = solver.U[i][k]
			solver.U[i][k] = q
		}
	}

	// Back transformation.
	for k = n - 1; k >= 0; k-- {
		if solver.b[k] == 0.0 {
			continue
		}
		q = -solver.A[k][k] / complex(abs(solver.A[k][k]), 0)
		for j = 0; j < n; j++ {
			solver.U[k][j] *= q
		}
		for j = 0; j < n; j++ {
			q = zero
			for i = k; i < m; i++ {
				q += conj(solver.A[i][k]) * solver.U[i][j]
			}
			q /= complex(abs(solver.A[k][k])*solver.b[k], 0)
			for i = k; i < m; i++ {
				solver.U[i][j] -= q * solver.A[i][k]
			}
		}
	}

	if n > 1 {
		for k = n - 2; k >= 0; k-- {
			k1 = k + 1
			if solver.c[k1] == 0.0 {
				continue
			}
			q = -conj(solver.A[k][k1]) / complex(abs(solver.A[k][k1]), 0)
			for j = 0; j < n; j++ {
				solver.V[k1][j] *= q
			}
			for j = 0; j < n; j++ {
				q = zero
				for i = k1; i < n; i++ {
					q += solver.A[k][i] * solver.V[i][j]
				}
				q /= complex(abs(solver.A[k][k1])*solver.c[k1], 0)
				for i = k1; i < n; i++ {
					solver.V[i][j] -= q * conj(solver.A[k][i])
				}
			}
		}
	}

	// Compute full basis for U and S.
	diag := func(m [][]complex64, v []float32) [][]complex64 {
		alen := len(m)
		for i := 0; i < len(v)-alen; i++ {
			m = append(m, make([]complex64, len(v)))
		}
		m = m[:len(v)]
		for i := range m {
			m[i] = m[i][:0]
			m[i] = append(m[i], make([]complex64, len(v))...)
		}

		for i := range m {
			m[i][i] = complex(v[i], 0)
		}
		return m
	}
	solver.Sigma = diag(solver.Sigma, solver.S)
	if m > n {
		for i := range len(solver.U) {
			solver.U[i] = append(solver.U[i], make([]complex64, m-n)...)
		}
		for j := n; j < m; j++ {
			solver.U[j][j] = 1
		}
		// Gram-Schmidt.
		for j := n; j < m; j++ {
			for i := 0; i < m; i++ {
				v := solver.U[i][j]
				for k := 0; k < j; k++ {
					v -= conj(solver.U[j][k]) * solver.U[i][k]
				}
				solver.U[i][j] = v
			}

			var ujnormf float32
			for i := 0; i < m; i++ {
				ujnormf += norm(solver.U[i][j])
			}
			ujnorm := complex(sqrt(ujnormf), 0)
			for i := 0; i < m; i++ {
				solver.U[i][j] /= ujnorm
			}

			// Check orthogonal.
			for k := 0; k < j; k++ {
				var dot complex64
				for i := 0; i < m; i++ {
					dot += solver.U[i][k] * conj(solver.U[i][j])
				}
				if abs(dot) > 1e-6 {
					panic(fmt.Sprintf("not orthogonal %f", abs(dot)))
				}
			}
		}

		for _ = range m - n {
			solver.Sigma = append(solver.Sigma, make([]complex64, n))
		}
	}

	solver.u.T2(solver.U)
	solver.s.T2(solver.Sigma)
	solver.v.T2(solver.V)
	if transposed {
		solver.u, solver.v = solver.v.Conj(), solver.u.Conj()

		// Transpose solver.s.
		solver.s.axis[0], solver.s.axis[1] = solver.s.axis[1], solver.s.axis[0]
		solver.s.updateShape()
		for i := range solver.s.shape[0] {
			for j := range solver.s.shape[1] {
				solver.s.data[i*solver.s.shape[1]+j] = solver.Sigma[j][i]
			}
		}
	}

	return solver.u, solver.s, solver.v
}
