package tensor

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"slices"

	"github.com/pkg/errors"
)

const (
	// Machine precision.
	epsilon = 0x1p-23
	// Safe minimum such that 1/safmin does not overflow.
	safmin = 0x1p-126
)

type Arnoldi struct {
	eig *Eig

	eigvals *Dense
	eigvecs *Dense
	q       *Dense
	h       *Dense
	r       *Dense
	buf0    *Dense

	debug bool
	a     *Dense
	initQ *Dense
}

func NewArnoldi() *Arnoldi {
	solver := &Arnoldi{
		eig:     NewEig(),
		eigvals: Zeros(1),
		eigvecs: Zeros(1),
		q:       Zeros(1),
		h:       Zeros(1),
		r:       Zeros(1),
		buf0:    Zeros(1),
		initQ:   Zeros(1),
	}
	return solver
}

type ArnoldiOptions struct {
	K              int
	KrylovSpaceDim int
	MaxIterations  int
}

func NewArnoldiOptions(k int) ArnoldiOptions {
	opt := ArnoldiOptions{K: k}
	opt.KrylovSpaceDim = max(2*k+1, 20)
	opt.MaxIterations = 32
	return opt
}

func (solver *Arnoldi) Solve(a *Dense, opt ArnoldiOptions) (*Dense, *Dense, error) {
	m := a.Shape()[0]

	solver.a = a
	opt.KrylovSpaceDim = min(m, opt.KrylovSpaceDim)
	solver.h.Zeros(opt.KrylovSpaceDim, opt.KrylovSpaceDim)
	solver.q.Zeros(m, opt.KrylovSpaceDim)
	solver.q.Set([]int{0, 0}, randVec(solver.eigvecs.Zeros(m, 1)))
	MatMul(solver.r, a, solver.q)
	start := 1

	solver.initQ.Zeros(m, 1)
	solver.initQ.Set([]int{0, 0}, solver.q.Slice([][2]int{{0, m}, {0, 1}}))

	converged := false
	for _ = range opt.MaxIterations {
		q, h, r := solver.iterate(a, opt.K, start)
		eigvals, hvecs, err := solver.eig.Solve(h, EigOption{Vectors: true})
		if err != nil {
			return nil, nil, errors.Wrap(err, "")
		}
		hvecs = hvecs.Slice([][2]int{{0, hvecs.Shape()[0]}, {0, opt.K}})
		numConverged := arnoldiConverged(r, hvecs, eigvals)
		if numConverged == opt.K {
			solver.eigvals.Zeros(opt.K)
			solver.eigvals.Set([]int{0}, eigvals.Slice([][2]int{{0, opt.K}}))
			MatMul(solver.eigvecs, q, hvecs)
			converged = true
			break
		}

		// Prevent stagnation by increasing the wanted set.
		// For more details, see Section 5.1.2 XYaup2, ARPACK Users' Guide, Lehoucq et al.
		start = opt.K + min(numConverged, (opt.KrylovSpaceDim-opt.K)/2)

		unwanted := eigvals.Slice([][2]int{{start, hvecs.Shape()[0]}})
		solver.implicitlyRestart(unwanted, q, h, r)
	}
	if !converged {
		return nil, nil, errors.Errorf("not converged")
	}

	solver.checkEigenvectors()
	return solver.eigvals, solver.eigvecs, nil
}

func (solver *Arnoldi) iterate(a *Dense, wants, start int) (*Dense, *Dense, *Dense) {
	bufs := []*Dense{solver.eigvals, solver.eigvecs}
	f := solver.buf0
	m := a.Shape()[0]

	k := start
	for ; k <= solver.q.Shape()[1]; k++ {
		qk := solver.q.Slice([][2]int{{0, m}, {0, k}})
		hk := solver.h.Slice([][2]int{{0, k}, {k - 1, k}})

		// Modified Gram Schimdt with re-orthogonalization.
		MatMul(f, a, solver.q.Slice([][2]int{{0, m}, {k - 1, k}}))
		fNorm, ok := gramSchimdt(f, hk, qk, bufs)
		if !ok {
			panic(fmt.Sprintf("cannot orthognalize %d", k))
			// If a @ q[:, k-1] collapses, simply use a random vector.
			// Section 5.1.3 XYaitr, ARPACK Users' Guide, Lehoucq et al.
			f.Set([]int{0, 0}, randVec(solver.eigvecs.Zeros(m, 1)))
			fNorm, ok = gramSchimdt(f, hk, qk, bufs)
			if !ok {
				panic(fmt.Sprintf("cannot orthogonalize %d", k))
			}
		}

		if solver.debug {
			solver.r.Zeros(solver.r.Shape()...)
			solver.r.Set([]int{0, k - 1}, f)
			solver.checkAQQH("iterating", k)
		}

		if k == solver.q.Shape()[1] {
			break
		}
		solver.h.SetAt([]int{k, k - 1}, complex(fNorm, 0))
		solver.q.Set([]int{0, k}, Mul(f, complex(1/fNorm, 0), f))
	}

	q := solver.q.Slice([][2]int{{0, m}, {0, k}})
	h := solver.h.Slice([][2]int{{0, k}, {0, k}})
	r := solver.r.Zeros(solver.r.Shape()...).Slice([][2]int{{0, m}, {0, k}})
	r.Set([]int{0, k - 1}, f)
	solver.checkAQQH("iterate end", k)

	return q, h, r
}

// gramSchimdt orthogonalizes vector f against vectors in q.
// Coefficients are stored in h as in:
// f_{out} = f_{in} - q @ h.
// Re-orthogonalization used here is explained in
// Remark 11.1, Chapter 11, Lecture notes of Numerical Methods for Solving Large Scale Eigenvalue Problems, Peter Arbenz.
func gramSchimdt(f, h, q *Dense, bufs []*Dense) (float32, bool) {
	// Angle of sin(pi/4) is explained in Section 5.1.3 XYaitr, ARPACK Users' Guide, Lehoucq et al.
	var sinPi4 = 1 * float32(math.Sin(math.Pi/4))
	for i := range h.Shape()[0] {
		h.SetAt([]int{i, 0}, 0)
	}

	for _ = range 3 {
		f0 := f.FrobeniusNorm()

		c := MatMul(bufs[0], q.H(), f)
		Add(f, f, Mul(bufs[1], -1, MatMul(bufs[1], q, c)))
		Add(h, h, c)

		fn := f.FrobeniusNorm()
		if fn > sinPi4*f0 {
			return fn, true
		}
		// Perhaps we can handle this case using clascl as in cnaitr ARPACK routine.
		if fn < epsilon {
			return -1, false
		}
	}

	return -1, false
}

// implicitlyRestart purges the subspace of the unwanted shifts.
// For a graphical explanation, conule Figure 4.5, ARPACK Users' Guide, Lehoucq et al.
// Also, see Section 5.1.8 XYapps, ARPACK Users' Guide.
func (solver *Arnoldi) implicitlyRestart(shifts, v, h, f *Dense) {
	eye := solver.eigvals.Eye(h.Shape()[0], 0)
	buf0 := solver.buf0
	buf1 := solver.eigvecs

	n := h.Shape()[0]

	for i := range shifts.Shape()[0] {
		shift := shifts.At(i)
		Add(h, h, Mul(buf1.Zeros(eye.Shape()...), -shift, eye))

		q, r := buf0, buf1
		chaseBulgeHessenberg(h, q, r)
		Add(h, h, Mul(buf1.Zeros(eye.Shape()...), shift, eye))
		h.Triu(-1)

		v.Set([]int{0, 0}, MatMul(buf1, v, q))
		f.Set([]int{0, 0}, MatMul(buf1, f, q))

		solver.checkAQQH(fmt.Sprintf("implicit %d", i), n)
	}
}

// arnoldiConverged checks the convergence of an Arnoldi iteration.
// For more details, consult Section 4.6 Stopping Criterion, ARPACK Users' Guide, Lehoucq et al.
// Also, see Section 5.1.7 YConv, ARPACK Users' Guide.
func arnoldiConverged(r, vecs, vals *Dense) int {
	const tol = epsilon
	rNorm := r.FrobeniusNorm()
	m := vecs.Shape()[0]
	numVecs := vecs.Shape()[1]

	var numConverged int
	for i := range numVecs {
		lambda := vals.At(i)
		diff := rNorm * abs(vecs.At(m-1, i))

		// log.Printf("|r|*v[-1] %d %v diff %f %f %f", i, lambda, diff, rNorm, abs(vecs.At(vecs.Shape()[0]-1, i)))
		if diff < tol*max(1, abs(lambda)) {
			numConverged++
		}
	}

	return numConverged
}

// checkAQQH checks for the Arnoldi relation a@q = q@h + r.
func (solver *Arnoldi) checkAQQH(prefix string, n int) {
	if !solver.debug {
		return
	}

	a := solver.a
	q := solver.q.Slice([][2]int{{0, solver.q.Shape()[0]}, {0, n}})
	h := solver.h.Slice([][2]int{{0, n}, {0, n}})
	r := solver.r.Slice([][2]int{{0, solver.r.Shape()[0]}, {0, n}})

	aq := Zeros(1)
	MatMul(aq, a, q)
	qh := Zeros(1)
	MatMul(qh, q, h)
	diff, buf1 := Zeros(aq.Shape()...), Zeros(aq.Shape()...)
	Add(diff, aq, Mul(buf1, -1, qh))
	Add(diff, diff, Mul(buf1, -1, r))

	// log.Printf("Arnoldi relation %s %d %f", prefix, n, diff.FrobeniusNorm())
	// Empirically, diff is around 1e-2 using float32.
	if diff.FrobeniusNorm() > 2e-1 {
		panic(fmt.Sprintf("Arnoldi relation violated %s %d %f", prefix, n, diff.FrobeniusNorm()))
	}
}

// checkEigenvectors checks for the eigenvector relation a @ v = lambda * v.
// If the Arnoldi relation holds, then a@v - lambda*v = r@s, where r is the residue in the Arnoldi relation, and s is the eigenvector in Krylov space.
func (solver *Arnoldi) checkEigenvectors() {
	if !solver.debug {
		return
	}

	for i := range solver.eigvals.Shape()[0] {
		lambda := solver.eigvals.At(i)
		vec := solver.eigvecs.Slice([][2]int{{0, solver.eigvecs.Shape()[0]}, {i, i + 1}})

		av := Zeros(1)
		MatMul(av, solver.a, vec)
		lambdaVec := Zeros(vec.Shape()...)
		Mul(lambdaVec, -lambda, vec)

		diff := Zeros(vec.Shape()...)
		Add(diff, av, lambdaVec)

		if diff.FrobeniusNorm() > 1000*epsilon*abs(lambda) {
			panic(fmt.Sprintf("%v %f", lambda, diff.FrobeniusNorm()))
		}
	}
}

type EigOption struct {
	Vectors bool
}

type Eig struct {
	eigvals *Dense
	eigvecs *Dense

	a      *Dense
	q      *Dense
	buf0   *Dense
	buf1   *Dense
	valPos []valuePosition
}

func NewEig() *Eig {
	solver := &Eig{
		eigvals: Zeros(1),
		eigvecs: Zeros(1),
		a:       Zeros(1),
		q:       Zeros(1),
		buf0:    Zeros(1),
		buf1:    Zeros(1),
	}
	return solver
}

func (solver *Eig) Solve(a *Dense, opt EigOption) (*Dense, *Dense, error) {
	if err := solver.solve(a, opt); err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	solver.valPos = sortByEigenvalue(solver.eigvals, solver.eigvecs, func(a, b complex64) int {
		return cmp.Compare(real(a), real(b))
	}, solver.valPos, solver.buf0)

	return solver.eigvals, solver.eigvecs, nil
}

func (solver *Eig) solve(a *Dense, opt EigOption) error {
	m := a.shape[0]
	solver.a.Zeros(a.Shape()...)
	solver.a.Set([]int{0, 0}, a)
	solver.q.Eye(m, 0)
	balance(solver.a, solver.q)
	Hessenberg(solver.a, solver.q, []*Dense{solver.buf0, solver.buf1})

	for mrow := m - 1; mrow >= 1; mrow-- {
		// Use deflation and get the active portion.
		ax := [][2]int{{0, mrow + 1}, {0, mrow + 1}}
		aActive := solver.a.Slice(ax)
		var converged bool
		for _ = range 32 {
			// Calculate Wilkinson's shift.
			lambda0, lambda1 := eig22(aActive.Slice([][2]int{{mrow - 1, mrow + 1}, {mrow - 1, mrow + 1}}))
			amm := aActive.At(mrow, mrow)
			shift := lambda0
			if abs(lambda0-amm) > abs(lambda1-amm) {
				shift = lambda1
			}

			// Apply implicit QR via bulge chasing.
			eye := solver.buf1.Eye(aActive.Shape()[0], 0)
			solver.buf0.Zeros(eye.Shape()...)
			Add(aActive, aActive, Mul(solver.buf0, -shift, eye))

			q, r := solver.eigvecs, solver.eigvals
			chaseBulgeHessenberg(aActive, q, r)
			Add(aActive, aActive, Mul(solver.buf0, shift, eye))
			aActive.Triu(-1)

			// Update solver.a.
			if mrow+1 < m {
				topRight := solver.a.Slice([][2]int{{0, mrow + 1}, {mrow + 1, m}})

				topRight.Set([]int{0, 0}, MatMul(solver.buf0, q.H(), topRight))
			}
			// Update solver.q.
			if opt.Vectors {
				qActive := solver.q.Slice([][2]int{{0, m}, {0, mrow + 1}})
				qActive.Set([]int{0, 0}, MatMul(solver.buf0, qActive, q))
			}

			// Criterion for when a subdiagonal entry is considered small:
			// Section 7.5.1 Deflation, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
			// Section 5.1.8 XYapps, ARPACK Users' Guide, Lehoucq et al.
			sd := abs(aActive.At(mrow, mrow-1))
			d := abs(aActive.At(mrow, mrow)) + abs(aActive.At(mrow-1, mrow-1))
			if sd < epsilon*d {
				converged = true
				break
			}
		}
		if !converged {
			return errors.Errorf("not converged %d %v", mrow, aActive.At(mrow, mrow-1))
		}
	}

	// Collect eigenvalues.
	solver.eigvals.Zeros(m)
	solver.eigvecs.Zeros(m, m)
	for i := range m {
		solver.eigvals.SetAt([]int{i}, solver.a.At(i, i))
	}
	if !opt.Vectors {
		return nil
	}

	// Now solver.a is triangle, get its eigenvectors.
	aMinusLambda := solver.buf0.Zeros(solver.a.Shape()...).Set([]int{0, 0}, solver.a)
	zeros := solver.buf1.Zeros(m, 1)
	for i := range m {
		for j := range m {
			aMinusLambda.SetAt([]int{j, j}, solver.a.At(j, j)-solver.a.At(i, i))
		}
		vec := solver.eigvecs.Slice([][2]int{{0, m}, {i, i + 1}})
		backSubstitution(vec, aMinusLambda, zeros, i)
	}

	// Transform eigenvectors to original space.
	solver.eigvecs.Set([]int{0, 0}, MatMul(solver.buf0, solver.q, solver.eigvecs))

	// Normalize eigenvectors
	for j := range solver.eigvecs.Shape()[1] {
		vec := solver.eigvecs.Slice([][2]int{{0, solver.eigvecs.Shape()[0]}, {j, j + 1}})
		Mul(vec, complex(1/vec.FrobeniusNorm(), 0), vec)
	}

	return nil
}

// InverseIteration computes the eigenvector whose eigenvalue is closest to mu.
// See Section 7.6.1 Selected Eigenvectors via Inverse Iteration, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
func (solver *Eig) InverseIteration(a *Dense, mu complex64) (complex64, *Dense, error) {
	m := a.Shape()[0]
	aNorm := a.InfNorm()
	solver.a.Zeros(a.Shape()...).Set([]int{0, 0}, a)
	for i := range m {
		solver.a.SetAt([]int{i, i}, solver.a.At(i, i)-mu)
	}
	QR(solver.a, solver.q, []*Dense{solver.buf0, solver.buf1})
	zeroIndex := -1
	for i := range m {
		if abs(solver.a.At(i, i)) < epsilon {
			zeroIndex = i
			break
		}
	}

	// Prepare the initial vector.
	q := solver.eigvecs.Zeros(m, 1)
	for i := range m {
		q.SetAt([]int{i, 0}, 1)
	}
	Mul(q, complex(1/q.FrobeniusNorm(), 0), q)

	var lambda complex64
	// r is the residue, (a - mu)q.
	var r *Dense
	var converged bool
	for _ = range 16 {
		// Solve (a - mu)z = q
		MatMul(solver.buf0, solver.q.H(), q)
		backSubstitution(q, solver.a, solver.buf0, zeroIndex)

		// Normalize q and compute residue.
		Mul(q, complex(1/q.FrobeniusNorm(), 0), q)
		lambda = MatMul(solver.buf1, q.H(), MatMul(solver.buf0, a, q)).At(0, 0)
		r = MatMul(solver.buf0, a, q)
		for i := range m {
			r.SetAt([]int{i, 0}, r.At(i, 0)-mu*q.At(i, 0))
		}

		// Check convergence.
		if r.InfNorm() < epsilon*aNorm {
			converged = true
			break
		}
	}
	if !converged {
		return complex64(cmplx.NaN()), nil, errors.Errorf("not converged %f %f", r.InfNorm(), epsilon*aNorm)
	}

	return lambda, q, nil
}

type valuePosition struct {
	value    complex64
	position int
}

func sortByEigenvalue(eigvals, eigvecs *Dense, fn func(a, b complex64) int, valPos []valuePosition, buf0 *Dense) []valuePosition {
	// Sort eigenvalues.
	m := eigvals.shape[0]
	valPos = valPos[:0]
	for i := range m {
		ev := eigvals.At(i)
		vp := valuePosition{value: ev, position: i}
		valPos = append(valPos, vp)
	}
	slices.SortFunc(valPos, func(a, b valuePosition) int {
		return fn(a.value, b.value)
	})

	// Apply sorted permutation.
	buf0.Zeros(m, 1)
	for i := range m {
		otherIdx := valPos[i].position
		for otherIdx < i {
			otherIdx = valPos[otherIdx].position
		}

		// Swap eigenvalues.
		tmp := eigvals.At(i)
		eigvals.SetAt([]int{i}, eigvals.At(otherIdx))
		eigvals.SetAt([]int{otherIdx}, tmp)

		// Swap eigenvectors.
		axi := [][2]int{{0, m}, {i, i + 1}}
		axOther := [][2]int{{0, m}, {otherIdx, otherIdx + 1}}
		buf0.Set([]int{0, 0}, eigvecs.Slice(axi))
		eigvecs.Slice(axi).Set([]int{0, 0}, eigvecs.Slice(axOther))
		eigvecs.Slice(axOther).Set([]int{0, 0}, buf0)
	}

	return valPos
}

// balance reduces the norm of a matrix.
// Fore more details, see Algorithm 3, On Matrix Balancing and Eigenvector Computation, R. James, J. Langou, B. R. Lowery.
// Section 7.5.7 Balancing, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
func balance(a, d *Dense) {
	const b float32 = 2 // float32 radix base.
	m := a.Shape()[0]
	d.Eye(m, 0)

	var converged bool
	for !converged {
		converged = true
		for i := range m {
			iCol := a.Slice([][2]int{{0, m}, {i, i + 1}})
			iRow := a.Slice([][2]int{{i, i + 1}, {0, m}})
			c := iCol.FrobeniusNorm()
			r := iRow.FrobeniusNorm()
			s := c + r

			var f float32 = 1
			for c < r/b && (max(absf(c), absf(f)) < 1/b/epsilon && absf(r) > b*epsilon) {
				c *= b
				r /= b
				f *= b
			}
			for c >= r*b && (absf(r) < 1/b/epsilon && max(absf(c), absf(f)) > b*epsilon) {
				c /= b
				r *= b
				f /= b
			}

			cf := complex(f, 0)
			if c+r < 0.95*s && (abs(d.At(i, i))*f > epsilon && abs(d.At(i, i)) < 1/f/epsilon) {
				converged = false
				Mul(iCol, cf, iCol)
				Mul(iRow, 1/cf, iRow)
				d.SetAt([]int{i, i}, d.At(i, i)*cf)
			}
		}
	}
}

func Hessenberg(a, q *Dense, bufs []*Dense) {
	m := a.shape[0]
	vs := bufs[0].Zeros(a.Shape()...)
	for i := 1; i <= m-2; i++ {
		// Note that we take [i:,i-1], whereas QR takes [i:,i].
		ax := [][2]int{{0, m}, {i - 1, i}}
		x := a.Slice(ax)
		v := vs.Slice(ax)
		h := newHouseholder(v, x, i)

		h.applyLeft(a, bufs[1])
		h.applyRight(a, bufs[1])
		h.applyRight(q, bufs[1])
	}
}

func backSubstitution(x, l, b *Dense, zeroIndex int) {
	m := x.Shape()[0]
	for i := m - 1; i >= 0; i-- {
		var v complex64 = b.At(i, 0)
		for j := m - 1; j > i; j-- {
			v -= l.At(i, j) * x.At(j, 0)
		}
		if abs(l.At(i, i)) < epsilon {
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
}

func QR(a, q *Dense, bufs []*Dense) {
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		panic(fmt.Sprintf("%d %d", m, n))
	}
	q.Eye(m, 0)
	bufs[0].Zeros(a.Shape()...)

	last := n
	if m == n {
		last--
	}
	for i := range last {
		ax := [][2]int{{i, m}, {i, i + 1}}
		x := a.Slice(ax)
		v := bufs[0].Slice(ax)
		h := newHouseholder(v, x, 0)

		h.applyLeft(a.Slice([][2]int{{i, m}, {i, n}}), bufs[1])
		h.applyRight(q.Slice([][2]int{{0, m}, {i, m}}), bufs[1])
	}

	// Make all diagonals of R positive.
	phase := bufs[0].Eye(m, 0)
	for i := range a.shape[1] {
		rv := a.At(i, i)
		phs := complex64(cmplx.Rect(1, -cmplx.Phase(complex128(rv))))

		phase.SetAt([]int{i, i}, phs)
	}
	a.Set([]int{0, 0}, MatMul(bufs[1], phase, a))
	q.Set([]int{0, 0}, MatMul(bufs[1], q, phase.H()))
}

type SVD struct {
	u       *Dense
	s       *Dense
	v       *Dense
	bidiagU *Dense
	bidiagV *Dense
	eig     *Eig
	buf0    *Dense
	buf1    *Dense
}

func NewSVD() *SVD {
	solver := &SVD{}
	solver.u = Zeros(1)
	solver.s = Zeros(1)
	solver.v = Zeros(1)
	solver.bidiagU = Zeros(1)
	solver.bidiagV = Zeros(1)
	solver.eig = NewEig()
	solver.buf0 = Zeros(1)
	solver.buf1 = Zeros(1)
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
	bidiagU, bidiag, bidiagV := solver.bidiagU, a, solver.bidiagV
	GolubKahan(a, bidiagU, bidiagV, []*Dense{solver.buf0, solver.buf1})

	// Solve the Jordan-Wielandt matrix.
	bd := bidiag.Slice([][2]int{{0, n}, {0, n}})
	solver.buf0.Zeros(2*n, 2*n)
	solver.buf0.Set([]int{0, n}, bd)
	solver.buf0.Set([]int{n, 0}, bd.H())
	if err := solver.eig.solve(solver.buf0, EigOption{Vectors: true}); err != nil {
		return errors.Wrap(err, "")
	}
	solver.eig.valPos = sortByEigenvalue(solver.eig.eigvals, solver.eig.eigvecs, func(a, b complex64) int {
		return -cmp.Compare(real(a), real(b))
	}, solver.eig.valPos, solver.eig.buf0)

	// Extract from the Jordan-Wielandt matrix.
	solver.s.Zeros(m, n)
	for i := range n {
		solver.s.SetAt([]int{i, i}, solver.eig.eigvals.At(i))
	}
	solver.buf0.Zeros(n, n)
	solver.u.Eye(m, 0)
	solver.u.Set([]int{0, 0}, Mul(solver.buf0, complex(sqrtf(2), 0), solver.eig.eigvecs.Slice([][2]int{{0, n}, {0, n}})))
	solver.v.Eye(n, 0)
	solver.v.Set([]int{0, 0}, Mul(solver.buf0, complex(sqrtf(2), 0), solver.eig.eigvecs.Slice([][2]int{{n, 2 * n}, {0, n}})))

	// Apply bidiagonal transforms.
	solver.u.Set([]int{0, 0}, MatMul(solver.buf0, bidiagU, solver.u))
	solver.v.Set([]int{0, 0}, MatMul(solver.buf0, bidiagV, solver.v))

	return nil
}

func GolubKahan(a, u, v *Dense, bufs []*Dense) {
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		panic(fmt.Sprintf("%d %d", m, n))
	}
	u.Eye(m, 0)
	v.Eye(n, 0)
	bufs[0].Zeros(a.Shape()...)

	for j := range n {
		ax := [][2]int{{j, m}, {j, j + 1}}
		x := a.Slice(ax)
		hv := bufs[0].Slice(ax)
		h := newHouseholder(hv, x, 0)
		h.applyLeft(a.Slice([][2]int{{j, m}, {j, n}}), bufs[1])
		h.applyRight(u.Slice([][2]int{{0, m}, {j, m}}), bufs[1])
		if j+1 < n {
			ax := [][2]int{{j, j + 1}, {j + 1, n}}
			x := a.Slice(ax).H()
			hv := bufs[0].Slice(ax).H()
			h := newHouseholder(hv, x, 0)
			h.applyRight(a.Slice([][2]int{{j, m}, {j + 1, n}}), bufs[1])
			h.applyRight(v.Slice([][2]int{{0, n}, {j + 1, n}}), bufs[1])
		}
	}
}

func randVec(vec *Dense) *Dense {
	m := vec.Shape()[0]
	for i := range m {
		vec.SetAt([]int{i, 0}, complex(rand.Float32(), rand.Float32()))
	}
	Mul(vec, complex(1/vec.FrobeniusNorm(), 0), vec)
	return vec
}
