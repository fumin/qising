package tensor

import (
	"cmp"
	"fmt"
	"math/cmplx"
	"math/rand"
	"slices"

	"github.com/pkg/errors"
)

const (
	epsilon = 1e-6
)

type Arnoldi struct {
	eig *Eig
	qr  *QR

	eigvals *Dense
	eigvecs *Dense
	q       *Dense
	h       *Dense
	r       *Dense
	buf0    *Dense
}

func NewArnoldi() *Arnoldi {
	solver := &Arnoldi{
		eig:     NewEig(),
		qr:      NewQR(),
		eigvals: Zeros(1),
		eigvecs: Zeros(1),
		q:       Zeros(1),
		h:       Zeros(1),
		r:       Zeros(1),
		buf0:    Zeros(1),
	}
	return solver
}

type ArnoldiOptions struct {
	K              int
	KrylovSpaceDim int
}

func NewArnoldiOptions(k int) ArnoldiOptions {
	opt := ArnoldiOptions{K: k}
	opt.KrylovSpaceDim = max(2*k+1, 20)
	return opt
}

func (solver *Arnoldi) Solve(a *Dense, opt ArnoldiOptions) (*Dense, *Dense, error) {
	m := a.Shape()[0]

	opt.KrylovSpaceDim = min(m, opt.KrylovSpaceDim)
	solver.h.Zeros(opt.KrylovSpaceDim, opt.KrylovSpaceDim)
	solver.q.Zeros(m, opt.KrylovSpaceDim)
	solver.q.Set([]int{0, 0}, solver.initVec(m))
	start := 1

	converged := false
	for _ = range 128 {
		q, h, r := solver.iterate(a, opt.KrylovSpaceDim, start)
		if h.Shape()[0] < opt.K {
			solver.q.Set([]int{0, 0}, solver.initVec(m))
			start = 1
			continue
		}
		eigvals, hvecs, err := solver.eig.Solve(h)
		if err != nil {
			return nil, nil, errors.Wrap(err, "")
		}
		hvecs = hvecs.Slice([][2]int{{0, hvecs.Shape()[0]}, {0, opt.K}})
		if arnoldiConverged(h, r, hvecs) {
			solver.eigvals.Zeros(opt.K)
			solver.eigvals.Set([]int{0}, eigvals.Slice([][2]int{{0, opt.K}}))
			MatMul(solver.eigvecs, q, hvecs)
			converged = true
			break
		}

		unwanted := eigvals.Slice([][2]int{{opt.K, hvecs.Shape()[0]}})
		solver.implicitQR(unwanted, q, h)
		start = opt.K
	}
	if !converged {
		return nil, nil, errors.Errorf("not converged")
	}

	return solver.eigvals, solver.eigvecs, nil
}

func (solver *Arnoldi) iterate(a *Dense, n, start int) (*Dense, *Dense, *Dense) {
	m := a.Shape()[0]
	k := start
	for ; k <= n; k++ {
		// Gram Schimdt.
		MatMul(solver.r, a, solver.q.Slice([][2]int{{0, m}, {k - 1, k}}))
		for j := range k {
			qj := solver.q.Slice([][2]int{{0, m}, {j, j + 1}})
			hjk1 := MatMul(solver.buf0, qj.H(), solver.r).At(0, 0)
			solver.h.SetAt([]int{j, k - 1}, hjk1)
			solver.buf0.Zeros(qj.Shape()...)
			Add(solver.r, solver.r, Mul(solver.buf0, -hjk1, qj))
		}
		hkk1 := solver.r.FrobeniusNorm()

		if k == n {
			break
		}
		solver.h.SetAt([]int{k, k - 1}, complex(hkk1, 0))
		if hkk1 < epsilon {
			break
		}
		solver.q.Set([]int{0, k}, Mul(solver.r, complex(1/hkk1, 0), solver.r))
	}
	q := solver.q.Slice([][2]int{{0, m}, {0, k}})
	h := solver.h.Slice([][2]int{{0, k}, {0, k}})
	return q, h, solver.r
}

func (solver *Arnoldi) implicitQR(shifts, v, h *Dense) {
	eye := solver.eigvals.Eye(h.Shape()[0], 0)
	buf0 := solver.buf0.Zeros(eye.Shape()...)
	buf1 := solver.eigvecs

	for i := range shifts.Shape()[0] {
		shift := shifts.At(i)
		Add(h, h, Mul(buf0, -shift, eye))
		q, r := solver.qr.Solve(h)

		Add(h, MatMul(buf1, r, q), Mul(buf0, shift, eye))
		v.Set([]int{0, 0}, MatMul(buf1, v, q))
	}
}

func (solver *Arnoldi) initVec(m int) *Dense {
	b0 := solver.buf0.Zeros(m, 1)
	for i := range m {
		b0.SetAt([]int{i, 0}, complex(rand.Float32(), 0))
	}
	Mul(b0, complex(1/b0.FrobeniusNorm(), 0), b0)
	return b0
}

func arnoldiConverged(h, r, vecs *Dense) bool {
	hNorm := h.FrobeniusNorm()
	rNorm := r.FrobeniusNorm()
	for i := range vecs.Shape()[1] {
		diff := rNorm * abs(vecs.At(vecs.Shape()[0]-1, i))
		if diff > 1*epsilon*hNorm {
			return false
		}
	}
	return true
}

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

func (solver *Eig) Solve(a *Dense) (*Dense, *Dense, error) {
	if err := solver.solve(a); err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	solver.sortByEigenvalue(func(a, b complex64) int {
		return cmp.Compare(real(a), real(b))
	})

	return solver.eigvals, solver.eigvecs, nil
}

func (solver *Eig) solve(a *Dense) error {
	hess, hessq := solver.hessenberg.Solve(a)

	// QR algorithm.
	// a0 = q0 @ q1 @ q2 @... a3 ...@ q2.H @ q1.H @ q0.H
	// solver.q = q0 @ q1 @ q2 @...
	m := a.shape[0]
	solver.a.Zeros(m, m)
	solver.a.Set([]int{0, 0}, hess)
	solver.q.Eye(m, 0)

	for mrow := m - 1; mrow >= 1; mrow-- {
		// Use deflation and get the active portion.
		ax := [][2]int{{0, mrow + 1}, {0, mrow + 1}}
		aActive := solver.a.Slice(ax)
		var converged bool
		for _ = range 16 {
			if abs(aActive.At(mrow, mrow-1)) < epsilon {
				converged = true
				break
			}
			// Calculate Wilkinson's shift.
			lambda0, lambda1 := eig22(aActive.Slice([][2]int{{mrow - 1, mrow + 1}, {mrow - 1, mrow + 1}}))
			amm := aActive.At(mrow, mrow)
			shift := lambda0
			if abs(lambda0-amm) > abs(lambda1-amm) {
				shift = lambda1
			}

			// Q, R = A.
			identity := solver.permutation.Eye(mrow+1, 0)
			Add(aActive, aActive, Mul(identity, -shift, identity))
			q, r := solver.qr.Solve(aActive)

			// A = RQ.
			identity = solver.permutation.Eye(mrow+1, 0)
			Add(aActive, MatMul(solver.buf, r, q), Mul(identity, shift, identity))
			if mrow+1 < m {
				topRight := solver.a.Slice([][2]int{{0, mrow + 1}, {mrow + 1, m}})
				topRight.Set([]int{0, 0}, MatMul(solver.buf, q.H(), topRight))
			}
			fullQ := solver.permutation.Eye(m, 0).Set([]int{0, 0}, q)
			solver.q.Set([]int{0, 0}, MatMul(solver.buf, solver.q, fullQ))
		}
		if !converged {
			return errors.Errorf("not converged %d %v", mrow, aActive.At(mrow, mrow-1))
		}
	}

	// Now solver.a is triangle, get its eigenvectors.
	zeros := solver.permutation.Zeros(m)
	solver.eigvals.Zeros(m)
	solver.eigvecs.Zeros(m, m)
	for i := range m {
		lambda := solver.a.At(i, i)
		solver.eigvals.SetAt([]int{i}, lambda)

		aMinusLambda := solver.buf.Zeros(solver.a.Shape()...).Set([]int{0, 0}, solver.a)
		for j := range m {
			aMinusLambda.data[aMinusLambda.at([]int{j, j})] -= lambda
		}
		vec := solver.eigvecs.Slice([][2]int{{0, m}, {i, i + 1}})
		if err := backSubstitution(vec, aMinusLambda, zeros, i); err != nil {
			return errors.Wrap(err, "")
		}
	}

	// Transform eigenvectors to original space.
	solver.eigvecs.Set([]int{0, 0}, MatMul(solver.buf, solver.q, solver.eigvecs))
	solver.eigvecs.Set([]int{0, 0}, MatMul(solver.buf, hessq, solver.eigvecs))

	// Normalize eigenvectors
	for j := range solver.eigvecs.Shape()[1] {
		vec := solver.eigvecs.Slice([][2]int{{0, solver.eigvecs.Shape()[0]}, {j, j + 1}})
		Mul(vec, complex(1/vec.FrobeniusNorm(), 0), vec)
	}

	return nil
}

type valuePosition struct {
	value    complex64
	position int
}

func (solver *Eig) sortByEigenvalue(fn func(a, b complex64) int) {
	// Sort eigenvalues.
	m := solver.eigvals.shape[0]
	solver.valPos = solver.valPos[:0]
	for i := range m {
		ev := solver.eigvals.At(i, 0)
		vp := valuePosition{value: ev, position: i}
		solver.valPos = append(solver.valPos, vp)
	}
	slices.SortFunc(solver.valPos, func(a, b valuePosition) int {
		return fn(a.value, b.value)
	})

	// Use a permutation matrix to apply the sorting.
	solver.permutation.Zeros(len(solver.valPos), len(solver.valPos))
	for i, vp := range solver.valPos {
		ptr := solver.permutation.at([]int{i, vp.position})
		solver.permutation.data[ptr] = 1
	}
	// Permute eigenvalues.
	MatMul(solver.eigvals, solver.permutation, MatMul(solver.buf, solver.eigvals.ToDiag(), solver.permutation.H()))
	solver.buf.Zeros(m)
	for i := range m {
		solver.buf.SetAt([]int{i}, solver.eigvals.At(i, i))
	}
	solver.eigvals.Zeros(m)
	solver.eigvals.Set([]int{0}, solver.buf)
	// Permute eigenvectors.
	solver.eigvecs.Set([]int{0, 0}, MatMul(solver.buf, solver.eigvecs, solver.permutation.H()))
}

type Hessenberg struct {
	hess *Dense
	q    *Dense

	householder *Dense
	buf0        *Dense
	buf1        *Dense
}

func NewHessenberg() *Hessenberg {
	solver := &Hessenberg{
		hess:        Zeros(1),
		q:           Zeros(1),
		householder: Zeros(1),
		buf0:        Zeros(1),
		buf1:        Zeros(1),
	}
	return solver
}

func (solver *Hessenberg) Solve(a *Dense) (*Dense, *Dense) {
	n := a.shape[0]
	solver.hess.Zeros(n, n)
	solver.hess.Set([]int{0, 0}, a)
	solver.q.Eye(n, 0)

	for i := 1; i <= n-2; i++ {
		// Note that we take [i:,i-1], whereas QR takes [i:,i].
		x := solver.hess.Slice([][2]int{{i, n}, {i - 1, i}})
		if x.FrobeniusNorm() < epsilon {
			continue
		}

		solver.householder.Eye(n, 0)
		hh := solver.householder.Slice([][2]int{{i, n}, {i, n}})
		householder(hh, x, solver.buf0, solver.buf1)

		// hess = householder @ hess @ householder.H.
		MatMul(solver.buf0, solver.hess, solver.householder.H())
		MatMul(solver.hess, solver.householder, solver.buf0)
		// q = q @ householder.H.
		solver.q.Set([]int{0, 0}, MatMul(solver.buf0, solver.q, solver.householder.H()))
	}

	return solver.hess, solver.q
}

type LinearEquations struct {
	qr        *QR
	x         *Dense
	qHb       *Dense
	zeroDiags []int
}

func NewLinearEquations(qr *QR) *LinearEquations {
	solver := &LinearEquations{qr: qr}
	solver.x = Zeros(1)
	solver.qHb = Zeros(1)
	return solver
}

func (solver *LinearEquations) Solve(a, b *Dense) (*Dense, error) {
	if a.Shape()[0] < a.Shape()[1] {
		return nil, errors.Errorf("under determined")
	}
	q, r := solver.qr.Solve(a)
	MatMul(solver.qHb, q.H(), b)

	// Since the bottom of r is zero, so should qHb.
	for i := r.Shape()[1]; i < solver.qHb.Shape()[0]; i++ {
		if abs(solver.qHb.At(i)) > float32(b.Shape()[0])*epsilon {
			return nil, errors.Errorf("no solution")
		}
	}

	// Solve the triangular matrix by back substitution.
	triangular := r.Slice([][2]int{{0, r.Shape()[1]}, {0, r.Shape()[1]}})
	solver.zeroDiags = solver.zeroDiags[:0]
	solver.zeroDiags = append(solver.zeroDiags, -1)
	for i := range r.Shape()[1] {
		if abs(r.At(i, i)) < epsilon {
			solver.zeroDiags = append(solver.zeroDiags, i)
		}
	}
	solver.x.Zeros(r.Shape()[1], len(solver.zeroDiags))
	for i, zeroIdx := range solver.zeroDiags {
		xi := solver.x.Slice([][2]int{{0, solver.x.Shape()[0]}, {i, i + 1}})
		if err := backSubstitution(xi, triangular, solver.qHb, zeroIdx); err != nil {
			return nil, errors.Wrap(err, "")
		}
	}
	return solver.x, nil
}

func backSubstitution(x, l, b *Dense, zeroIndex int) error {
	m := x.Shape()[0]
	for i := m - 1; i >= 0; i-- {
		var v complex64 = b.At(i)
		for j := m - 1; j > i; j-- {
			v -= l.At(i, j) * x.At(j, 0)
		}
		if abs(l.At(i, i)) < epsilon {
			if abs(v) > float32(m-i)*epsilon {
				return errors.Errorf("%d %v", i, v)
			}
			// Only set to 1 if specified to achive independent vectors in the null space.
			if i == zeroIndex {
				v = 1
			} else {
				v = 0
			}
		} else {
			v /= l.At(i, i)
		}
		x.SetAt([]int{i, 0}, v)
	}
	return nil
}

type QR struct {
	q *Dense
	r *Dense

	householder *Dense
	buf0        *Dense
	buf1        *Dense
}

func NewQR() *QR {
	solver := &QR{
		q: Zeros(1),
		r: Zeros(1),

		householder: Zeros(1),
		buf0:        Zeros(1),
		buf1:        Zeros(1),
	}
	return solver
}

func (solver *QR) Solve(a *Dense) (*Dense, *Dense) {
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		panic(fmt.Sprintf("%d %d", m, n))
	}
	solver.q.Eye(m, 0)
	solver.r.Zeros(m, n)
	solver.r.Set([]int{0, 0}, a)

	last := n
	if m == n {
		last--
	}
	for i := range last {
		// Skip column if it is already {alpha, 0, 0, 0...}.
		x := solver.r.Slice([][2]int{{i, m}, {i, i + 1}})
		if x.Slice([][2]int{{1, m - i}, {0, 1}}).FrobeniusNorm() < epsilon {
			continue
		}

		// Perform Householder reflection.
		solver.householder.Eye(m, 0)
		hh := solver.householder.Slice([][2]int{{i, m}, {i, m}})
		householder(hh, x, solver.buf0, solver.buf1)

		// Update Q and R.
		solver.q.Set([]int{0, 0}, MatMul(solver.buf0, solver.q, solver.householder.H()))
		solver.r.Set([]int{0, 0}, MatMul(solver.buf0, solver.householder, solver.r))
	}

	// Make all diagonals of R positive.
	identity := solver.buf0.Eye(m, 0)
	for i := range solver.r.shape[1] {
		rv := solver.r.At(i, i)
		phase := complex64(cmplx.Rect(1, -cmplx.Phase(complex128(rv))))

		identity.SetAt([]int{i, i}, phase)
	}
	solver.r.Set([]int{0, 0}, MatMul(solver.buf1, identity, solver.r))
	solver.q.Set([]int{0, 0}, MatMul(solver.buf1, solver.q, identity.H()))

	return solver.q, solver.r
}

func (solver *QR) AreIndependent(x *Dense) bool {
	m, n := x.Shape()[0], x.Shape()[1]
	if m < n {
		return false
	}
	_, r := solver.Solve(x)

	for j := range n {
		rjj := r.At(j, j)
		if abs(rjj) < epsilon {
			return false
		}
	}

	return true
}

func householder(hh, x, buf0, buf1 *Dense) {
	xk := x.At(0, 0)
	eixk := complex64(cmplx.Rect(1, cmplx.Phase(complex128(xk))))
	alpha := -eixk * complex(x.FrobeniusNorm(), 0)

	u := buf0.Zeros(x.Shape()...)
	u.SetAt([]int{0, 0}, -alpha)
	Add(u, x, u)
	Mul(u, complex(1/u.FrobeniusNorm(), 0), u)

	buf1.Zeros(1)
	Mul(buf1, -2, MatMul(buf1, u, u.H()))
	Add(hh, hh, buf1)
}

type SVD struct {
	u          *Dense
	s          *Dense
	v          *Dense
	golubKahan *GolubKahan
	eig        *Eig
	buf0       *Dense
}

func NewSVD() *SVD {
	solver := &SVD{}
	solver.u = Zeros(1)
	solver.s = Zeros(1)
	solver.v = Zeros(1)
	solver.golubKahan = NewGolubKahan()
	solver.eig = NewEig()
	solver.buf0 = Zeros(1)
	return solver
}

func (solver *SVD) Solve(a *Dense) (*Dense, *Dense, *Dense, error) {
	hermitiated := false
	if a.Shape()[0] < a.Shape()[1] {
		hermitiated = true
		a = a.H()
	}

	if err := solver.solve(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "")
	}

	if hermitiated {
		// s = s.H()
		solver.buf0.Zeros(solver.s.Shape()...)
		solver.buf0.Set([]int{0, 0}, solver.s)
		solver.s.Zeros(solver.buf0.H().Shape()...)
		solver.s.Set([]int{0, 0}, solver.buf0.H())
		// u, v = v, u
		solver.buf0.Zeros(solver.u.Shape()...)
		solver.buf0.Set([]int{0, 0}, solver.u)
		solver.u.Zeros(solver.v.Shape()...)
		solver.u.Set([]int{0, 0}, solver.v)
		solver.v.Zeros(solver.buf0.Shape()...)
		solver.v.Set([]int{0, 0}, solver.buf0)
	}

	return solver.u, solver.s, solver.v, nil
}

func (solver *SVD) solve(a *Dense) error {
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		return errors.Errorf("%#v", a.Shape())
	}
	// Transform to bidiagonal.
	bidiagU, bidiag, bidiagV := solver.golubKahan.Solve(a)

	// Solve the Jordan-Wielandt matrix.
	bd := bidiag.Slice([][2]int{{0, n}, {0, n}})
	solver.buf0.Zeros(2*n, 2*n)
	solver.buf0.Set([]int{0, n}, bd)
	solver.buf0.Set([]int{n, 0}, bd.H())
	if err := solver.eig.solve(solver.buf0); err != nil {
		return errors.Wrap(err, "")
	}
	solver.eig.sortByEigenvalue(func(a, b complex64) int {
		return -cmp.Compare(real(a), real(b))
	})

	// Extract from the Jordan-Wielandt matrix.
	solver.s.Zeros(m, n)
	for i := range n {
		solver.s.SetAt([]int{i, i}, solver.eig.eigvals.At(i))
	}
	solver.buf0.Zeros(n, n)
	solver.u.Eye(m, 0)
	solver.u.Set([]int{0, 0}, Mul(solver.buf0, complex(sqrt(2), 0), solver.eig.eigvecs.Slice([][2]int{{0, n}, {0, n}})))
	solver.v.Eye(n, 0)
	solver.v.Set([]int{0, 0}, Mul(solver.buf0, complex(sqrt(2), 0), solver.eig.eigvecs.Slice([][2]int{{n, 2 * n}, {0, n}})))

	// Apply bidiagonal transforms.
	solver.u.Set([]int{0, 0}, MatMul(solver.buf0, bidiagU, solver.u))
	solver.v.Set([]int{0, 0}, MatMul(solver.buf0, bidiagV, solver.v))

	return nil
}

type GolubKahan struct {
	u *Dense
	b *Dense
	v *Dense

	householder *Dense
	buf0        *Dense
	buf1        *Dense
}

func NewGolubKahan() *GolubKahan {
	solver := &GolubKahan{}
	solver.u = Zeros(1)
	solver.b = Zeros(1)
	solver.v = Zeros(1)
	solver.householder = Zeros(1)
	solver.buf0 = Zeros(1)
	solver.buf1 = Zeros(1)
	return solver
}

func (solver *GolubKahan) Solve(a *Dense) (*Dense, *Dense, *Dense) {
	m, n := a.Shape()[0], a.Shape()[1]
	solver.u.Eye(m, 0)
	solver.b.Zeros(m, n).Set([]int{0, 0}, a)
	solver.v.Eye(n, 0)

	for i := range max(m, n) - 1 {
		// Column.
		if m-i >= 2 && i < n {
			x := solver.b.Slice([][2]int{{i, m}, {i, i + 1}})
			if x.Slice([][2]int{{1, m - i}, {0, 1}}).FrobeniusNorm() < epsilon {
				continue
			}
			// Perform Householder reflection.
			solver.householder.Eye(m, 0)
			hh := solver.householder.Slice([][2]int{{i, m}, {i, m}})
			householder(hh, x, solver.buf0, solver.buf1)
			// Update Q and R.
			solver.u.Set([]int{0, 0}, MatMul(solver.buf0, solver.u, solver.householder.H()))
			solver.b.Set([]int{0, 0}, MatMul(solver.buf0, solver.householder, solver.b))
		}

		// Row.
		if i < m && n-(i+1) >= 2 {
			x := solver.b.Slice([][2]int{{i, i + 1}, {i + 1, n}})
			if x.Slice([][2]int{{0, 1}, {1, n - i - 1}}).FrobeniusNorm() < epsilon {
				continue
			}
			// Perform Householder reflection.
			solver.householder.Eye(n, 0)
			hh := solver.householder.Slice([][2]int{{i + 1, n}, {i + 1, n}})
			householder(hh, x.H(), solver.buf0, solver.buf1)
			// Update Q and R.
			solver.v.Set([]int{0, 0}, MatMul(solver.buf0, solver.v, solver.householder.H()))
			solver.b.Set([]int{0, 0}, MatMul(solver.buf0, solver.b, solver.householder.H()))
		}
	}

	return solver.u, solver.b, solver.v
}
