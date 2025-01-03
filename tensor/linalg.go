package tensor

import (
	"cmp"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"

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
	krylovSpaceDim int
	maxIterations  int
}

func NewArnoldiOptions() ArnoldiOptions {
	opt := ArnoldiOptions{}
	opt.krylovSpaceDim = -1
	opt.maxIterations = 64
	return opt
}

func (opt ArnoldiOptions) KrylovSpaceDim(v int) ArnoldiOptions {
	opt.krylovSpaceDim = v
	return opt
}

func (opt ArnoldiOptions) MaxIterations(v int) ArnoldiOptions {
	opt.maxIterations = v
	return opt
}

func (solver *Arnoldi) Solve(a *Dense, k int, options ...ArnoldiOptions) (*Dense, *Dense, error) {
	opt := NewArnoldiOptions()
	if len(options) > 0 {
		opt = options[0]
	}
	if opt.krylovSpaceDim < 0 {
		opt.krylovSpaceDim = max(2*k+1, 20)
	}

	m := a.Shape()[0]
	solver.a = a
	opt.krylovSpaceDim = min(m, opt.krylovSpaceDim)
	solver.h.Zeros(opt.krylovSpaceDim+1, opt.krylovSpaceDim)
	solver.q.Zeros(m, opt.krylovSpaceDim+1)
	solver.q.Set([]int{0, 0}, randVec(solver.eigvecs.Zeros(m, 1)))
	MatMul(solver.r, a, solver.q)
	start := 1

	solver.initQ.Zeros(m, 1)
	solver.initQ.Set([]int{0, 0}, solver.q.Slice([][2]int{{0, m}, {0, 1}}))

	var cvg arnoldiConvergence
	for _ = range opt.maxIterations {
		q, h, r := solver.iterate(a, k, start)
		eigvals, hvecs, err := solver.eig.Solve(h)
		if err != nil {
			return nil, nil, errors.Wrap(err, "")
		}
		hvecs = hvecs.Slice([][2]int{{0, hvecs.Shape()[0]}, {0, k}})
		cvg = arnoldiConverged(r, hvecs, eigvals)
		if cvg.converged {
			solver.eigvals.Zeros(k)
			solver.eigvals.Set([]int{0}, eigvals.Slice([][2]int{{0, k}}))
			MatMul(solver.eigvecs, q, hvecs)
			break
		}

		// Prevent stagnation by increasing the wanted set.
		// For more details, see Section 5.1.2 XYaup2, ARPACK Users' Guide, Lehoucq et al.
		start = k + cvg.numConverged
		start = min(start, k+(opt.krylovSpaceDim-k)/2)

		unwanted := eigvals.Slice([][2]int{{start, eigvals.Shape()[0]}})
		solver.implicitlyRestart(unwanted, q, h, r)
	}
	if !cvg.converged {
		return nil, nil, errors.Errorf("not converged %#v", cvg)
	}

	solver.checkEigenvectors()
	return solver.eigvals, solver.eigvecs, nil
}

func (solver *Arnoldi) iterate(a *Dense, wants, start int) (*Dense, *Dense, *Dense) {
	bufs := []*Dense{solver.eigvals, solver.eigvecs}
	f := solver.buf0
	m := a.Shape()[0]

	k := solver.h.Shape()[1]
	for i := start; i <= k; i++ {
		vi1 := solver.q.Slice([][2]int{{0, m}, {i - 1, i}})
		v := solver.q.Slice([][2]int{{0, m}, {0, i}})
		h := solver.h.Slice([][2]int{{0, i}, {i - 1, i}})

		// Modified Gram Schimdt with re-orthogonalization.
		MatMul(f, a, vi1)
		fNorm := gramSchimdt(f, h, v, bufs)

		if solver.debug {
			solver.r.Zeros(solver.r.Shape()...)
			solver.r.Set([]int{0, i - 1}, f)
			solver.checkAQQH("iterating", i)
		}

		solver.h.SetAt([]int{i, i - 1}, complex(fNorm, 0))
		vi := solver.q.Slice([][2]int{{0, m}, {i, i + 1}})
		if fNorm < epsilon {
			// If a @ q[:, i-1] collapses, simply use a random vector.
			// Section 5.1.3 XYaitr, ARPACK Users' Guide, Lehoucq et al.
			vi.Set([]int{0, 0}, randVec(solver.eigvecs.Zeros(m, 1)))
		} else {
			Mul(vi, complex(1/fNorm, 0), f)
		}
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
func gramSchimdt(f, h, q *Dense, bufs []*Dense) float32 {
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
			return fn
		}
	}

	return 0
}

// implicitlyRestart purges the subspace of the unwanted shifts.
// For a graphical explanation, conule Figure 4.5, ARPACK Users' Guide, Lehoucq et al.
// Also, see Section 5.1.8 XYapps, ARPACK Users' Guide.
func (solver *Arnoldi) implicitlyRestart(shifts, v, h, f *Dense) {
	buf0 := solver.buf0
	buf1 := solver.eigvecs

	for i := range shifts.Shape()[0] {
		shift := shifts.At(i)

		deflate(h)

		eye := solver.eigvals.Eye(h.Shape()[0], 0)
		Add(h, h, Mul(buf1.Zeros(eye.Shape()...), -shift, eye))
		q, r := buf0, buf1
		chaseBulgeHessenberg(h, q, r)
		Add(h, h, Mul(buf1.Zeros(eye.Shape()...), shift, eye))

		v.Set([]int{0, 0}, MatMul(buf1, v, q))
		f.Set([]int{0, 0}, MatMul(buf1, f, q))

		solver.checkAQQH(fmt.Sprintf("implicit %d", i), h.Shape()[0])
	}
}

type arnoldiConvergence struct {
	converged      bool
	numConverged   int
	largestDiffIdx int
	largestDiff    float32
}

// arnoldiConverged checks the convergence of an Arnoldi iteration.
// For more details, consult Section 4.6 Stopping Criterion, ARPACK Users' Guide, Lehoucq et al.
// Also, see Section 5.1.7 YConv, ARPACK Users' Guide.
func arnoldiConverged(r, vecs, vals *Dense) arnoldiConvergence {
	const tol = 2 * epsilon
	rNorm := r.FrobeniusNorm()
	m := vecs.Shape()[0]
	numVecs := vecs.Shape()[1]

	c := arnoldiConvergence{largestDiffIdx: -1}
	for i := range numVecs {
		lambda := vals.At(i)
		diff := rNorm * abs(vecs.At(m-1, i))

		// log.Printf("|r|*v[-1] %d %v diff %f %f %f", i, lambda, diff, rNorm, abs(vecs.At(vecs.Shape()[0]-1, i)))
		if diff < tol*max(1, abs(lambda)) {
			c.numConverged++
		} else {
			if diff > c.largestDiff {
				c.largestDiffIdx = i
				c.largestDiff = diff
			}
		}
	}

	if c.numConverged == numVecs {
		c.converged = true
	}

	return c
}

// checkAQQH checks for the Arnoldi relation a@q = q@h + r.
func (solver *Arnoldi) checkAQQH(prefix string, n int) *Dense {
	if !solver.debug {
		return nil
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
	if diff.FrobeniusNorm() > 20*epsilon*a.FrobeniusNorm() {
		panic(fmt.Sprintf("Arnoldi relation violated %s %d %f", prefix, n, diff.FrobeniusNorm()))
	}

	return diff
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

		if diff.FrobeniusNorm() > 100*epsilon*abs(lambda) {
			panic(fmt.Sprintf("%v %f", lambda, diff.FrobeniusNorm()))
		}
	}
}

type EigOptions struct {
	Vectors bool
}

type Eig struct {
	eigvals *Dense
	eigvecs *Dense

	a    *Dense
	q    *Dense
	buf0 *Dense
	buf1 *Dense
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

func (solver *Eig) Solve(a *Dense, options ...EigOptions) (*Dense, *Dense, error) {
	opt := EigOptions{Vectors: true}
	if len(options) > 0 {
		opt = options[0]
	}

	if err := solver.solve(a, opt); err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	sortEigen(solver.eigvals, solver.eigvecs, nil, func(a, b complex64) int { return cmp.Compare(real(a), real(b)) }, solver.buf0)

	return solver.eigvals, solver.eigvecs, nil
}

func (solver *Eig) solve(a *Dense, opt EigOptions) error {
	m := a.shape[0]
	solver.a.Zeros(a.Shape()...)
	solver.a.Set([]int{0, 0}, a)
	solver.q.Eye(m, 0)
	balance(solver.a, solver.q)
	Hessenberg(solver.a, solver.q, []*Dense{solver.buf0, solver.buf1})

	for {
		p, q := findUnreducedHessenberg(solver.a)
		if q == m {
			break
		}
		h22 := solver.a.Slice([][2]int{{p, m - q}, {p, m - q}})
		hm := h22.Shape()[0]

		var converged bool
		for _ = range 32 {
			shift := wilkinsonsShift(h22)

			// Apply implicit QR via bulge chasing.
			eye := solver.buf1.Eye(hm, 0)
			solver.buf0.Zeros(eye.Shape()...)
			Add(h22, h22, Mul(solver.buf0, -shift, eye))

			z, r := solver.eigvecs, solver.eigvals
			chaseBulgeHessenberg(h22, z, r)
			Add(h22, h22, Mul(solver.buf0, shift, eye))

			if opt.Vectors {
				// Update solver.a.
				if p > 0 {
					h12 := solver.a.Slice([][2]int{{0, p}, {p, m - q}})
					h12.Set([]int{0, 0}, MatMul(solver.buf0, h12, z))
				}
				if q > 0 {
					h23 := solver.a.Slice([][2]int{{p, m - q}, {m - q, m}})

					h23.Set([]int{0, 0}, MatMul(solver.buf0, z.H(), h23))
				}
				// Update solver.q.
				q2 := solver.q.Slice([][2]int{{0, m}, {p, m - q}})
				q2.Set([]int{0, 0}, MatMul(solver.buf0, q2, z))
			}

			p22, q22 := findUnreducedHessenberg(h22)
			if !(p22 == 0 && q22 == 0) {
				converged = true
				break
			}
		}
		if !converged {
			return errors.Errorf("not converged %d %d %v", p, q, h22.At(hm-1, hm-2))
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

	// Decompose (a-mu) = u @ t, where t is triangular.
	solver.a.Zeros(a.Shape()...).Set([]int{0, 0}, a)
	for i := range m {
		solver.a.SetAt([]int{i, i}, solver.a.At(i, i)-mu)
	}
	t, u := solver.a, solver.q
	QR(t, u, []*Dense{solver.buf0, solver.buf1})
	// Find the zero index so that back-substitution does not return a zero vector.
	zeroIndex := -1
	for i := range m {
		if abs(t.At(i, i)) < epsilon {
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

	// r is the residue, (a - mu)q.
	var r *Dense
	var converged bool
	for _ = range 16 {
		// Solve (a - mu)z = q
		MatMul(solver.buf0, u.H(), q)
		backSubstitution(q, t, solver.buf0, zeroIndex)

		// Normalize q.
		Mul(q, complex(1/q.FrobeniusNorm(), 0), q)
		// Compute residue r = (a-mu)q.
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

	// Compute a refined eigenvalue.
	lambda := MatMul(solver.buf1, q.H(), MatMul(solver.buf0, a, q)).At(0, 0)

	return lambda, q, nil
}

func wilkinsonsShift(a *Dense) complex64 {
	m := a.Shape()[0]
	lambda0, lambda1 := eig22(a.Slice([][2]int{{m - 2, m}, {m - 2, m}}))
	amm := a.At(m-1, m-1)

	shift := lambda0
	if abs(lambda0-amm) > abs(lambda1-amm) {
		shift = lambda1
	}
	return shift
}

// deflate sets to zero all subdiagonals that satisfy |a[i, i-1]| < tol*(|a[i, i]| + |a[i-1, i-1]|)
// For more details about this criterion, see Section 7.5.1 Deflation, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
// Section 5.1.8 XYapps, ARPACK Users' Guide, Lehoucq et al.
func deflate(a *Dense) {
	m := a.Shape()[0]
	for i := 1; i < m; i++ {
		sd := abs(a.At(i, i-1))
		d := abs(a.At(i, i)) + abs(a.At(i-1, i-1))
		if sd < epsilon*d {
			a.SetAt([]int{i, i - 1}, 0)
		}
	}
}

// findUnreducedHessenberg finds the largest submatrix that is unreduced Hessenberg.
// See Algorithm 7.5.2, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
func findUnreducedHessenberg(a *Dense) (int, int) {
	m := a.Shape()[0]

	// Deflate so that finding p and q below can compare against zero.
	deflate(a)

	var q int = m
	for i := m - 1; i >= 1; i-- {
		if a.At(i, i-1) != 0 {
			q = m - 1 - i
			break
		}
	}

	var p int
	for i := m - 1 - q - 1; i >= 1; i-- {
		if a.At(i, i-1) == 0 {
			p = i
			break
		}
	}

	return p, q
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
	for i := 1; i <= m-2; i++ {
		// Note that we take [i:,i-1], whereas QR takes [i:,i].
		x := a.Slice([][2]int{{i, m}, {i - 1, i}})
		v := bufs[0].Zeros(x.Shape()...)
		h := newHouseholder(v, x, 0)

		h.applyLeft(a.Slice([][2]int{{i, m}, {i - 1, m}}), bufs[1])
		a.SetAt([]int{i, i - 1}, h.beta)

		h.applyRight(a.Slice([][2]int{{0, m}, {i, m}}), bufs[1])
		h.applyRight(q.Slice([][2]int{{0, m}, {i, m}}), bufs[1])
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
			// See Section 7.6.4 Eigenvector Bases, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
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
		a.SetAt([]int{i, i}, h.beta)

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

func SVD(s, u, v *Dense, bufs []*Dense) error {
	m, n := s.Shape()[0], s.Shape()[1]
	if m >= n {
		return svd(s, u, v, bufs)
	}
	return svd(s.H(), v, u, bufs)
}

func svd(s, u, v *Dense, bufs []*Dense) error {
	tol := max(10, min(100, float32(math.Pow(epsilon, -1./8)))) * epsilon

	m, n := s.Shape()[0], s.Shape()[1]
	GolubKahan(s, u, v, bufs)
	b := s.Slice([][2]int{{0, n}, {0, n}})
	b.Triu(0).Tril(1)

	smin, _ := calcSMinMax(b)
	thresh := epsilon * smin / sqrtf(float32(n))

	for {
		p, q := findBidiagonal(b, tol, thresh)
		if q == n {
			break
		}
		b22 := b.Slice([][2]int{{p, n - q}, {p, n - q}})
		bm := b22.Shape()[0]

		// Special case for 2x2.
		if b22.Shape()[0] == 2 {
			bufs[0].Zeros(4, 2)
			u22 := bufs[0].Slice([][2]int{{0, 2}, {0, 2}})
			v22 := bufs[0].Slice([][2]int{{2, 4}, {0, 2}})
			svd22(b22, u22, v22)
			u2 := u.Slice([][2]int{{0, m}, {p, n - q}})
			u2.Set([]int{0, 0}, MatMul(bufs[1], u2, u22))
			v2 := v.Slice([][2]int{{0, n}, {p, n - q}})
			v2.Set([]int{0, 0}, MatMul(bufs[1], v2, v22))
			continue
		}

		smax, smin := calcSMinMax(b22)
		// t holds the bottom right corner of b22.H() @ b22.
		t := bufs[0].Zeros(2, 2)

		var converged bool
		for _ = range max(n-p-q, 32) {
			// Compute shift.
			t.SetAt([]int{0, 0}, MatMul(bufs[1], b22.H().Slice([][2]int{{bm - 2, bm - 1}, {0, bm}}), b22.Slice([][2]int{{0, bm}, {bm - 2, bm - 1}})).At(0, 0))
			t.SetAt([]int{0, 1}, MatMul(bufs[1], b22.H().Slice([][2]int{{bm - 2, bm - 1}, {0, bm}}), b22.Slice([][2]int{{0, bm}, {bm - 1, bm}})).At(0, 0))
			t.SetAt([]int{1, 0}, MatMul(bufs[1], b22.H().Slice([][2]int{{bm - 1, bm}, {0, bm}}), b22.Slice([][2]int{{0, bm}, {bm - 2, bm - 1}})).At(0, 0))
			t.SetAt([]int{1, 1}, MatMul(bufs[1], b22.H().Slice([][2]int{{bm - 1, bm}, {0, bm}}), b22.Slice([][2]int{{0, bm}, {bm - 1, bm}})).At(0, 0))
			shift := wilkinsonsShift(t)
			// Use a zero shift if shifting will ruin relative accuracy.
			if float32(n)*tol*(smin/smax) < max(epsilon, 0.01*tol) {
				shift = 0
			}

			y := MatMul(bufs[1], b22.H().Slice([][2]int{{0, 1}, {0, bm}}), b22.Slice([][2]int{{0, bm}, {0, 1}})).At(0, 0) - shift
			z := MatMul(bufs[1], b22.H().Slice([][2]int{{0, 1}, {0, bm}}), b22.Slice([][2]int{{0, bm}, {1, 2}})).At(0, 0)
			for k := range bm - 1 {
				// Remove top right bulge.
				g := newGivens(conj(y), conj(z), k, k+1)
				g.applyRight(b22.Slice([][2]int{{k, k + 2}, {0, bm}}))
				if k > 0 {
					b22.SetAt([]int{k - 1, k}, g.r)
					b22.SetAt([]int{k - 1, k + 1}, 0)
				}

				g.applyRight(v.Slice([][2]int{{0, n}, {p, n - q}}))

				// Remove bottom left bulge.
				y = b22.At(k, k)
				z = b22.At(k+1, k)
				g = newGivens(y, z, k, k+1)
				g.applyLeft(b22.Slice([][2]int{{0, bm}, {k + 1, min(k+3, bm)}}))
				b22.SetAt([]int{k, k}, g.r)
				b22.SetAt([]int{k + 1, k}, 0)

				g.applyRight(u.Slice([][2]int{{0, m}, {p, n - q}}))

				if k+2 < bm {
					y = b22.At(k, k+1)
					z = b22.At(k, k+2)
				}
			}

			p22, q22 := findBidiagonal(b22, tol, thresh)
			if !(p22 == 0 && q22 == 0) {
				converged = true
				break
			}
		}
		if !converged {
			f := abs(b22.At(bm-2, bm-1))
			d := abs(b22.At(bm-2, bm-2)) + abs(b22.At(bm-1, bm-1))
			return errors.Errorf("not converged %f %f %f", f/epsilon/d, f, d)
		}
	}

	// Make s non-negative.
	for i := range n {
		if sii := s.At(i, i); real(sii) < 0 {
			s.SetAt([]int{i, i}, -sii)
			ui := u.Slice([][2]int{{0, m}, {i, i + 1}})
			Mul(ui, -1, ui)
		}
	}

	// Sort s descending.
	sdiag := bufs[0].Zeros(s.Shape()[1])
	for i := range sdiag.Shape()[0] {
		sdiag.SetAt([]int{i}, s.At(i, i))
	}
	sortEigen(sdiag, u, v, func(a, b complex64) int { return -cmp.Compare(real(a), real(b)) }, bufs[1])
	for i := range sdiag.Shape()[0] {
		s.SetAt([]int{i, i}, sdiag.At(i))
	}

	return nil
}

func checkAllReal(prefix string, a *Dense) {
	n := a.Shape()[1]
	if imag(a.At(0, 0)) != 0 {
		panic(fmt.Sprintf("%s 0 0 %v", prefix, a.At(0, 0)))
	}
	for j := 1; j < n; j++ {
		if imag(a.At(j, j)) != 0 {
			panic(fmt.Sprintf("%s %d %d %v", prefix, j, j, a.At(j, j)))
		}
		if imag(a.At(j, j-1)) != 0 {
			panic(fmt.Sprintf("%s %d %d %v", prefix, j, j-1, a.At(j, j-1)))
		}
	}
	log.Printf("%s all real", prefix)
}

func calcSMinMax(a *Dense) (float32, float32) {
	n := a.Shape()[1]
	smax := abs(a.At(0, 0))
	for j := 1; j < n; j++ {
		smax = max(smax, abs(a.At(j, j)))
		smax = max(smax, abs(a.At(j, j-1)))
	}

	// Equation 2.4, Accurate Singular Values of Bidiagonal Matrices, James Demmel and W. Kahan.
	mu := abs(a.At(0, 0))
	smin := mu
	for j := 1; j < n; j++ {
		mu = abs(a.At(j, j)) * (mu / (mu + abs(a.At(j-1, j))))
		smin = min(smin, mu)
	}

	return smin, smax
}

func findBidiagonal(a *Dense, tol, thresh float32) (int, int) {
	m := a.Shape()[0]
	for i := range m - 1 {
		f := abs(a.At(i, i+1))
		d := abs(a.At(i, i)) + abs(a.At(i+1, i+1))
		if f < tol*d || f < thresh {
			a.SetAt([]int{i, i + 1}, 0)
		}
	}

	var q int = m
	for i := m - 2; i >= 0; i-- {
		if a.At(i, i+1) != 0 {
			q = m - 2 - i
			break
		}
	}

	var p int
	for i := m - 2 - q - 1; i >= 0; i-- {
		if a.At(i, i+1) == 0 {
			p = i + 1
			break
		}
	}

	return p, q
}

type valRightLeft struct {
	val   *Dense
	right *Dense
	left  *Dense
	fn    func(complex64, complex64) int
	buf   *Dense
}

func (vrl valRightLeft) Len() int { return vrl.val.Shape()[0] }
func (vrl valRightLeft) Swap(i, j int) {
	tmp := vrl.val.At(i)
	vrl.val.SetAt([]int{i}, vrl.val.At(j))
	vrl.val.SetAt([]int{j}, tmp)

	for _, vec := range []*Dense{vrl.right, vrl.left} {
		if vec == nil {
			continue
		}
		m := vec.Shape()[0]
		vrl.buf.Zeros(m, 1).Set([]int{0, 0}, vec.Slice([][2]int{{0, m}, {i, i + 1}}))
		vec.Set([]int{0, i}, vec.Slice([][2]int{{0, m}, {j, j + 1}}))
		vec.Set([]int{0, j}, vrl.buf)
	}
}
func (vrl valRightLeft) Less(i, j int) bool {
	return vrl.fn(vrl.val.At(i), vrl.val.At(j)) < 0
}

func sortEigen(val, right, left *Dense, fn func(complex64, complex64) int, buf *Dense) {
	vrl := valRightLeft{val: val, right: right, left: left, fn: fn, buf: buf}
	sort.Sort(vrl)
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
		a.SetAt([]int{j, j}, h.beta)

		h.applyRight(u.Slice([][2]int{{0, m}, {j, m}}), bufs[1])

		if j+1 < n {
			ax := [][2]int{{j, j + 1}, {j + 1, n}}
			x := a.Slice(ax).H()
			hv := bufs[0].Slice(ax).H()
			h := newHouseholder(hv, x, 0)

			h.applyRight(a.Slice([][2]int{{j, m}, {j + 1, n}}), bufs[1])
			a.SetAt([]int{j, j + 1}, h.beta)

			h.applyRight(v.Slice([][2]int{{0, n}, {j + 1, n}}), bufs[1])
		}
	}
}

func randVec(vec *Dense) *Dense {
	m := vec.Shape()[0]
	for i := range m {
		vec.SetAt([]int{i, 0}, complex(rand.Float32()*2-1, rand.Float32()*2-1))
	}
	Mul(vec, complex(1/vec.FrobeniusNorm(), 0), vec)
	return vec
}
