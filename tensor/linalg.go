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
	// Radix of float32.
	radix = 2
	// Machine precision.
	epsilon = 0x1p-23
	// Safe minimum such that 1/safmin does not overflow.
	safmin = 0x1p-126
)

type ArnoldiOptions struct {
	krylovSpaceDim int
	maxIterations  int

	debug bool
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

func Arnoldi(eigvals, eigvecs, a *Dense, k int, bufs []*Dense, options ...ArnoldiOptions) error {
	opt := NewArnoldiOptions()
	if len(options) > 0 {
		opt = options[0]
	}
	if opt.krylovSpaceDim < 0 {
		opt.krylovSpaceDim = max(2*k+1, 20)
	}
	m := a.Shape()[0]
	opt.krylovSpaceDim = min(m, opt.krylovSpaceDim)

	// Prepare buffers for Q, H, R in the Arnoldi relation A@Q = Q@H + R.
	bQ := bufs[0]
	bH := bufs[1]
	bR := bufs[2]
	bufs = bufs[3:]

	bH.Reset(opt.krylovSpaceDim+1, opt.krylovSpaceDim)

	// Start with a random vector for the first Arnoldi iteration.
	bQ.Reset(m, opt.krylovSpaceDim)
	bQ.Set([]int{0, 0}, randVec(bufs[0].Reset(m, 1)))
	start := 1

	// hvecs are eigenvectors in the Krylov space.
	var hvecs *Dense
	var cvg arnoldiConvergence
	for _ = range opt.maxIterations {
		q, h, r, err := arnoldiIterate(a, bQ, bH, bR, start, bufs, opt.debug)
		if err != nil {
			return errors.Wrap(err, "")
		}
		hcopy := bufs[0].Reset(h.Shape()...).Set([]int{0, 0}, h)
		if err := Eig(eigvals, eigvecs, hcopy, bufs[1:]); err != nil {
			return errors.Wrap(err, "")
		}
		hvecs = eigvecs.Slice([][2]int{{0, eigvecs.Shape()[0]}, {0, k}})
		cvg = arnoldiConverged(r, hvecs, eigvals)
		if cvg.converged {
			break
		}

		// Prevent stagnation by increasing the wanted set.
		// For more details, see Section 5.1.2 XYaup2, ARPACK Users' Guide, Lehoucq et al.
		start = k + cvg.numConverged
		start = min(start, k+(opt.krylovSpaceDim-k)/2)

		unwanted := eigvals.Slice([][2]int{{start, eigvals.Shape()[0]}})
		implicitlyRestart(unwanted, a, q, h, r, bufs, opt.debug)
	}
	if !cvg.converged {
		return errors.Errorf("not converged %#v", cvg)
	}

	// Set eigvals.
	bufs[0].Reset(k).Set([]int{0}, eigvals.Slice([][2]int{{0, k}}))
	eigvals.Reset(k)
	eigvals.Set([]int{0}, bufs[0])
	// Set eigvecs.
	bufs[0].Reset(hvecs.Shape()...).Set([]int{0, 0}, hvecs)
	q := bQ.Slice([][2]int{{0, m}, {0, opt.krylovSpaceDim}})
	MatMul(eigvecs, q, bufs[0])

	checkEigenvectors(eigvals, eigvecs, a, opt.debug)
	return nil
}

func arnoldiIterate(a, q, h, r *Dense, start int, bufs []*Dense, debug bool) (*Dense, *Dense, *Dense, error) {
	m := a.Shape()[0]
	k := h.Shape()[1]

	f := bufs[0]
	bufs = bufs[1:]

	for i := start; i <= k; i++ {
		vi1 := q.Slice([][2]int{{0, m}, {i - 1, i}})
		v := q.Slice([][2]int{{0, m}, {0, i}})
		hi := h.Slice([][2]int{{0, i}, {i - 1, i}})

		// Modified Gram Schimdt with re-orthogonalization.
		MatMul(f, a, vi1)
		fNorm, err := gramSchimdt(f, hi, v, bufs)
		if err != nil {
			fNorm = 0
		}

		if debug {
			r.Reset(m, i).Set([]int{0, i - 1}, f)
			checkAQQH("iterating", a, q, h, r, i)
		}

		h.SetAt([]int{i, i - 1}, complex(fNorm, 0))
		if i == k {
			break
		}
		vi := q.Slice([][2]int{{0, m}, {i, i + 1}})
		if fNorm < epsilon {
			// If a @ q[:, i-1] collapses, simply use a random vector.
			// Section 5.1.3 XYaitr, ARPACK Users' Guide, Lehoucq et al.
			if err := randOrthogonal(vi, v, bufs); err != nil {
				return nil, nil, nil, errors.Wrap(err, "")
			}
		} else {
			vi.Set([]int{0, 0}, f).Mul(complex(1/fNorm, 0))
		}
	}

	outQ := q.Slice([][2]int{{0, m}, {0, k}})
	outH := h.Slice([][2]int{{0, k}, {0, k}})
	outR := r.Reset(m, k).Set([]int{0, k - 1}, f)
	if debug {
		checkAQQH("iterate end", a, q, h, r, k)
	}

	return outQ, outH, outR, nil
}

// gramSchimdt orthogonalizes vector f against vectors in q.
// Coefficients are stored in h as in:
// f_{out} = f_{in} - q @ h.
// Re-orthogonalization used here is explained in
// Remark 11.1, Chapter 11, Lecture notes of Numerical Methods for Solving Large Scale Eigenvalue Problems, Peter Arbenz.
func gramSchimdt(f, h, q *Dense, bufs []*Dense) (float32, error) {
	// Angle of sin(pi/4) is explained in Section 5.1.3 XYaitr, ARPACK Users' Guide, Lehoucq et al.
	var sinPi4 = 1 * float32(math.Sin(math.Pi/4))
	if h != nil {
		h.Set([]int{0, 0}, bufs[0].Reset(h.Shape()...))
	}

	for _ = range 3 {
		f0 := f.FrobeniusNorm()

		c := MatMul(bufs[0], q.H(), f)
		f.Add(-1, MatMul(bufs[1], q, c))
		if h != nil {
			h.Add(1, c)
		}

		fn := f.FrobeniusNorm()
		if fn > sinPi4*f0 {
			return fn, nil
		}
	}

	return -1, errors.Errorf("Gram-Schimdt fail")
}

func randOrthogonal(qi, q *Dense, bufs []*Dense) error {
	for _ = range 3 {
		randVec(qi)

		qNorm, err := gramSchimdt(qi, nil, q, bufs)
		if err == nil {
			qi.Mul(complex(1/qNorm, 0))
			return nil
		}
	}
	return errors.Errorf("fail to orthogonalize")
}

// implicitlyRestart purges the subspace of the unwanted shifts.
// For a graphical explanation, conule Figure 4.5, ARPACK Users' Guide, Lehoucq et al.
// Also, see Section 5.1.8 XYapps, ARPACK Users' Guide.
func implicitlyRestart(shifts, a, qM, h, r *Dense, bufs []*Dense, debug bool) {
	m := h.Shape()[0]
	for i := range shifts.Shape()[0] {
		shift := shifts.At(i)

		p, q := m, -1
		for {
			remainder := h.Slice([][2]int{{0, p}, {0, p}})
			skipped := m - p
			p, q = findUnreducedHessenberg(remainder)
			q += skipped
			if q == m {
				break
			}
			h22 := h.Slice([][2]int{{p, m - q}, {p, m - q}})

			eye := bufs[0].Eye(h22.Shape()[0], 0)
			h22.Add(-shift, eye)
			z, unused := bufs[1], bufs[2]
			chaseBulgeHessenberg(h22, z, unused)
			h22.Add(shift, eye)

			if p > 0 {
				h12 := h.Slice([][2]int{{0, p}, {p, m - q}})
				h12.Set([]int{0, 0}, MatMul(bufs[0], h12, z))
			}
			if q > 0 {
				h23 := h.Slice([][2]int{{p, m - q}, {m - q, m}})
				h23.Set([]int{0, 0}, MatMul(bufs[0], z.H(), h23))
			}
			q2 := qM.Slice([][2]int{{0, qM.Shape()[0]}, {p, m - q}})
			q2.Set([]int{0, 0}, MatMul(bufs[0], q2, z))

			if debug {
				r2 := r.Slice([][2]int{{0, r.Shape()[0]}, {p, m - q}})
				r2.Set([]int{0, 0}, MatMul(bufs[0], r2, z))
			}
		}

		if debug {
			checkAQQH(fmt.Sprintf("implicit %d", i), a, qM, h, r, h.Shape()[0])
		}
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
func checkAQQH(prefix string, a, q, h, r *Dense, n int) *Dense {
	q = q.Slice([][2]int{{0, q.Shape()[0]}, {0, n}})
	h = h.Slice([][2]int{{0, n}, {0, n}})
	r = r.Slice([][2]int{{0, r.Shape()[0]}, {0, n}})

	// Check q orthogonality.
	qq := MatMul(Zeros(1), q.H(), q)
	eye := Zeros(1).Eye(qq.Shape()[0], 0)
	diff := Zeros(qq.Shape()...)
	diff.Add(1, qq).Add(-1, eye)
	if dn := diff.FrobeniusNorm(); dn > 100*epsilon {
		log.Printf("q %#v", q.ToSlice2())
		log.Printf("diff %#v", diff.ToSlice2())
		panic(fmt.Sprintf("not orthogonal %s %d %f", prefix, n, dn))
	}

	aq := MatMul(Zeros(1), a, q)
	qh := MatMul(Zeros(1), q, h)

	diff.Reset(aq.Shape()...)
	diff.Add(1, aq)
	diff.Add(-1, qh)
	diff.Add(-1, r)

	// log.Printf("Arnoldi relation %s %d %f", prefix, n, diff.FrobeniusNorm())
	if dn := diff.FrobeniusNorm(); dn > 20*epsilon*a.FrobeniusNorm() {
		log.Printf("q %#v", q.ToSlice2())
		log.Printf("h %#v", h.ToSlice2())
		log.Printf("r %#v", r.ToSlice2())
		panic(fmt.Sprintf("Arnoldi relation violated %s %d %f", prefix, n, dn))
	}

	return diff
}

// checkEigenvectors checks for the eigenvector relation a @ v = lambda * v.
// If the Arnoldi relation holds, then a@v - lambda*v = r@s, where r is the residue in the Arnoldi relation, and s is the eigenvector in Krylov space.
func checkEigenvectors(eigvals, eigvecs, a *Dense, debug bool) {
	if !debug {
		return
	}

	m := eigvecs.Shape()[0]
	for i := range eigvals.Shape()[0] {
		lambda := eigvals.At(i)
		vec := eigvecs.Slice([][2]int{{0, m}, {i, i + 1}})

		av := Zeros(1)
		MatMul(av, a, vec)
		lambdaVec := Zeros(vec.Shape()...).Set([]int{0, 0}, vec)
		lambdaVec.Mul(lambda)

		diff := Zeros(vec.Shape()...)
		diff.Add(1, av)
		diff.Add(-1, lambdaVec)

		if diff.FrobeniusNorm() > 100*epsilon*max(abs(lambda), 1) {
			panic(fmt.Sprintf("%v %f", lambda, diff.FrobeniusNorm()))
		}
	}
}

func Eig(eigvals, eigvecs, a *Dense, bufs []*Dense) error {
	if err := eig(eigvals, eigvecs, a, bufs); err != nil {
		return errors.Wrap(err, "")
	}
	sortEigen(eigvals, eigvecs, nil, func(a, b complex64) int { return cmp.Compare(real(a), real(b)) }, bufs[0])
	return nil
}

func eig(eigvals, eigvecs, a *Dense, bufs []*Dense) error {
	m := a.Shape()[0]

	// d is the balancing matrix.
	// It is held separately from the subsequent QR iteration, since it may contain very large or small values which are not suitable for frequent calculations.
	d := bufs[0].Eye(m, 0)
	balance(a, d)

	hq := bufs[1]
	hbufs := []*Dense{eigvals, bufs[2]}
	hessenberg(a, hq, hbufs)

	for {
		p, q := findUnreducedHessenberg(a)
		if q == m {
			break
		}
		h22 := a.Slice([][2]int{{p, m - q}, {p, m - q}})
		hm := h22.Shape()[0]

		var converged bool
		for _ = range 32 {
			shift := wilkinsonsShift(h22)

			// Apply implicit QR via bulge chasing.
			eye := bufs[2].Eye(hm, 0)
			h22.Add(-shift, eye)
			z, r := eigvals, bufs[2]
			chaseBulgeHessenberg(h22, z, r)
			eye = bufs[2].Eye(hm, 0)
			h22.Add(shift, eye)

			if eigvecs != nil {
				// Update h.
				if p > 0 {
					h12 := a.Slice([][2]int{{0, p}, {p, m - q}})
					h12.Set([]int{0, 0}, MatMul(bufs[2], h12, z))
				}
				if q > 0 {
					h23 := a.Slice([][2]int{{p, m - q}, {m - q, m}})

					h23.Set([]int{0, 0}, MatMul(bufs[2], z.H(), h23))
				}
				// Update hq.
				q2 := hq.Slice([][2]int{{0, m}, {p, m - q}})
				q2.Set([]int{0, 0}, MatMul(bufs[2], q2, z))
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
	eigvals.Reset(m)
	for i := range m {
		eigvals.SetAt([]int{i}, a.At(i, i))
	}
	if eigvecs == nil {
		return nil
	}

	// Now h is triangle, get its eigenvectors.
	eigvecs.Reset(m, m)
	zeros := bufs[2].Reset(m, 1)
	for i := range m {
		for j := range m {
			a.SetAt([]int{j, j}, eigvals.At(j, j)-eigvals.At(i, i))
		}
		vec := eigvecs.Slice([][2]int{{0, m}, {i, i + 1}})
		backSubstitution(vec, a, zeros, i)
	}

	// Transform eigenvectors to original space.
	eigvecs.Set([]int{0, 0}, MatMul(bufs[2], hq, eigvecs))
	eigvecs.Set([]int{0, 0}, MatMul(bufs[2], d, eigvecs))

	// Normalize eigenvectors
	for j := range eigvecs.Shape()[1] {
		vec := eigvecs.Slice([][2]int{{0, eigvecs.Shape()[0]}, {j, j + 1}})
		vec.Mul(complex(1/vec.FrobeniusNorm(), 0))
	}

	return nil
}

// InverseIteration computes the eigenvector whose eigenvalue is closest to mu.
// See Section 7.6.1 Selected Eigenvectors via Inverse Iteration, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
func InverseIteration(q, a *Dense, mu complex64, bufs []*Dense) (complex64, error) {
	m := a.Shape()[0]
	aNorm := a.InfNorm()

	// Decompose (a-mu) = u @ t, where t is triangular.
	t := bufs[0].Reset(a.Shape()...).Set([]int{0, 0}, a)
	for i := range m {
		t.SetAt([]int{i, i}, t.At(i, i)-mu)
	}
	u := bufs[1]
	qrBufs := []*Dense{bufs[2], bufs[3]}
	t = QR(u, t, qrBufs)
	// Find the zero index so that back-substitution does not return a zero vector.
	zeroIndex := -1
	for i := range m {
		if abs(t.At(i, i)) < epsilon {
			zeroIndex = i
			break
		}
	}

	// Prepare the initial vector.
	q.Reset(m, 1)
	for i := range m {
		q.SetAt([]int{i, 0}, 1)
	}
	q.Mul(complex(1/q.FrobeniusNorm(), 0))

	// r is the residue, (a - mu)q.
	var r *Dense
	var converged bool
	for _ = range 16 {
		// Solve (a - mu)z = q, where (a-mu) = t @ u
		uHq := MatMul(bufs[2], u.H(), q)
		backSubstitution(q, t, uHq, zeroIndex)

		// Normalize q.
		q.Mul(complex(1/q.FrobeniusNorm(), 0))
		// Compute residue r = (a-mu)q.
		r = MatMul(bufs[2], a, q)
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
		return complex64(cmplx.NaN()), errors.Errorf("not converged %f %f", r.InfNorm(), epsilon*aNorm)
	}

	// Compute a refined eigenvalue.
	lambda := MatMul(bufs[1], q.H(), MatMul(bufs[0], a, q)).At(0, 0)

	return lambda, nil
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

	const ulp = radix * epsilon
	smlnum := safmin * (float32(m) / ulp)

	for i := 1; i < m; i++ {
		sd := abs(a.At(i, i-1))
		// See LAPACK clahqr routine for this criterion.
		if sd < smlnum {
			a.SetAt([]int{i, i - 1}, 0)
		}

		d := abs(a.At(i, i)) + abs(a.At(i-1, i-1))
		if sd < ulp*d {
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
	const b float32 = radix
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
				iCol.Mul(cf)
				iRow.Mul(1 / cf)
				d.SetAt([]int{i, i}, d.At(i, i)*cf)
			}
		}
	}
}

func hessenberg(a, q *Dense, bufs []*Dense) {
	m := a.shape[0]
	bufs[0].Reset(a.Shape()...)

	if m-2 < 1 {
		q.Eye(a.Shape()[0], 0)
		return
	}
	hhs := make([]householder, 0, m-2)

	for i := 1; i <= m-2; i++ {
		// Note that we take [i:,i-1], whereas QR takes [i:,i].
		ax := [][2]int{{i, m}, {i - 1, i}}
		x := a.Slice(ax)
		v := bufs[0].Slice(ax)
		h := newHouseholder(v, x, 0)

		h.applyLeft(a.Slice([][2]int{{i, m}, {i, m}}), bufs[1])
		a.SetAt([]int{i, i - 1}, h.beta)
		for j := i + 1; j < m; j++ {
			a.SetAt([]int{j, i - 1}, 0)
		}

		h.applyRight(a.Slice([][2]int{{0, m}, {i, m}}), bufs[1])

		hhs = append(hhs, h)
	}

	// Compute q.
	q.Eye(m, 0)
	for i := m - 2; i >= 1; i-- {
		h := hhs[i-1]

		h.tau = conj(h.tau)
		h.applyLeft(q.Slice([][2]int{{i, m}, {i, m}}), bufs[1])
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

type QROptions struct {
	Full bool
}

func QR(q, a *Dense, bufs []*Dense, options ...QROptions) *Dense {
	m, n := a.Shape()[0], a.Shape()[1]
	if m >= n {
		return qrTall(q, a, bufs, options...)
	}
	return qrShort(q, a, bufs)
}

func qrShort(q, a *Dense, bufs []*Dense) *Dense {
	// Split a into {aLeft, aRight}, where aLeft is square.
	m, n := a.Shape()[0], a.Shape()[1]
	aLeft := a.Slice([][2]int{{0, m}, {0, m}})
	qrTall(q, aLeft, bufs)

	// Apply q.H to aRight.
	aRight := a.Slice([][2]int{{0, m}, {m, n}})
	aRight.Set([]int{0, 0}, MatMul(bufs[0], q.H(), aRight))

	return a
}

func qrTall(q, a *Dense, bufs []*Dense, options ...QROptions) *Dense {
	opt := QROptions{}
	if len(options) > 0 {
		opt = options[0]
	}

	m, n := a.Shape()[0], a.Shape()[1]
	var k int
	switch {
	case opt.Full:
		k = m
	default:
		k = n
	}

	// Compute the triangular matrix R.
	bufs[0].Reset(a.Shape()...)
	last := n
	if m == n {
		last--
	}
	hhs := make([]householder, 0, last)
	for i := range last {
		ax := [][2]int{{i, m}, {i, i + 1}}
		x := a.Slice(ax)
		v := bufs[0].Slice(ax)
		h := newHouseholder(v, x, 0)

		if i+1 < n {
			h.applyLeft(a.Slice([][2]int{{i, m}, {i + 1, n}}), bufs[1])
		}
		a.SetAt([]int{i, i}, h.beta)
		for j := i + 1; j < m; j++ {
			a.SetAt([]int{j, i}, 0)
		}

		hhs = append(hhs, h)
	}
	if k != m {
		a = a.Slice([][2]int{{0, k}, {0, min(k, n)}})
	}

	// Compute Q by backward accumulation.
	// See Section 5.1.6 The Factored-Form Representation, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
	q.Reset(m, k)
	for i := range k {
		q.SetAt([]int{i, i}, 1)
	}
	for i := last - 1; i >= 0; i-- {
		h := hhs[i]

		// Conjugate tau since what we want is h @ q, but applyLeft does h.H @ q.
		h.tau = conj(h.tau)
		h.applyLeft(q.Slice([][2]int{{i, m}, {i, k}}), bufs[1])
	}

	// Make all diagonals of R positive.
	phase := bufs[0].Eye(k, 0)
	for i := range a.shape[1] {
		rv := a.At(i, i)
		phs := complex64(cmplx.Rect(1, -cmplx.Phase(complex128(rv))))

		phase.SetAt([]int{i, i}, phs)
	}
	a.Set([]int{0, 0}, MatMul(bufs[1], phase, a))
	q.Set([]int{0, 0}, MatMul(bufs[1], q, phase.H()))

	return a
}

type SVDOptions struct {
	Full bool
}

func SVD(u, v, s *Dense, bufs []*Dense, options ...SVDOptions) (*Dense, error) {
	opt := SVDOptions{}
	if len(options) > 0 {
		opt = options[0]
	}

	m, n := s.Shape()[0], s.Shape()[1]
	var err error
	if m >= n {
		err = rsvd(u, v, s, bufs, opt)
	} else {
		err = rsvd(v, u, s.H(), bufs, opt)
	}
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	if !opt.Full {
		minD := min(m, n)
		s = s.Slice([][2]int{{0, minD}, {0, minD}})
	}

	return s, nil
}

// rsvd performs SVD with R-bidiagonalization.
// See Figure 8.6.1, Matrix Computations 4th Ed., G. H. Golub, C. F. Van Loan.
func rsvd(u, v, s *Dense, bufs []*Dense, opt SVDOptions) error {
	m, n := s.Shape()[0], s.Shape()[1]
	if m < 3*n/2 {
		return svd(u, v, s, bufs)
	}

	r := QR(u, s, bufs, QROptions{Full: opt.Full})
	r = r.Slice([][2]int{{0, n}, {0, n}})

	ur := bufs[0]
	bufs = bufs[1:]
	if err := svd(ur, v, r, bufs); err != nil {
		return errors.Wrap(err, "")
	}

	un := u.Slice([][2]int{{0, m}, {0, n}})
	un.Set([]int{0, 0}, MatMul(bufs[0], un, ur))

	return nil
}

func svd(u, v, s *Dense, bufs []*Dense) error {
	tol := max(10, min(100, float32(math.Pow(epsilon, -1./8)))) * epsilon

	m, n := s.Shape()[0], s.Shape()[1]
	bidiagonalize(u, v, s, bufs)
	b := s.Slice([][2]int{{0, n}, {0, n}})

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
			bufs[0].Reset(4, 2)
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
		t := bufs[0].Reset(2, 2)

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
			ui.Mul(-1)
		}
	}

	// Sort s descending.
	sdiag := bufs[0].Reset(s.Shape()[1])
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
		vrl.buf.Reset(m, 1).Set([]int{0, 0}, vec.Slice([][2]int{{0, m}, {i, i + 1}}))
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

func bidiagonalize(u, v, a *Dense, bufs []*Dense) {
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		panic(fmt.Sprintf("%d %d", m, n))
	}
	bufs[0].Reset(a.Shape()...)

	uhs := make([]householder, 0, n-1)
	vhs := make([]householder, 0, n-2)
	for j := range n {
		ax := [][2]int{{j, m}, {j, j + 1}}
		x := a.Slice(ax)
		hv := bufs[0].Slice(ax)
		h := newHouseholder(hv, x, 0)
		uhs = append(uhs, h)

		if j+1 < n {
			h.applyLeft(a.Slice([][2]int{{j, m}, {j + 1, n}}), bufs[1])
		}
		a.SetAt([]int{j, j}, h.beta)
		for i := j + 1; i < m; i++ {
			a.SetAt([]int{i, j}, 0)
		}

		if j+1 < n {
			ax := [][2]int{{j, j + 1}, {j + 1, n}}
			x := a.Slice(ax).H()
			hv := bufs[0].Slice(ax).H()
			h := newHouseholder(hv, x, 0)
			vhs = append(vhs, h)

			h.applyRight(a.Slice([][2]int{{j + 1, m}, {j + 1, n}}), bufs[1])
			a.SetAt([]int{j, j + 1}, h.beta)
			for k := j + 2; k < n; k++ {
				a.SetAt([]int{j, k}, 0)
			}
		}
	}

	// Compute u.
	u.Eye(m, 0)
	for j := n - 1; j >= 0; j-- {
		h := uhs[j]

		h.tau = conj(h.tau)
		h.applyLeft(u.Slice([][2]int{{j, m}, {j, m}}), bufs[1])
	}

	// Compute v.
	v.Eye(n, 0)
	for j := n - 2; j >= 0; j-- {
		h := vhs[j]

		h.tau = conj(h.tau)
		h.applyLeft(v.Slice([][2]int{{j + 1, n}, {j + 1, n}}), bufs[1])
	}
}

func randVec(vec *Dense) *Dense {
	m := vec.Shape()[0]
	for i := range m {
		vec.SetAt([]int{i, 0}, complex(rand.Float32()*2-1, rand.Float32()*2-1))
	}

	// If we unfortunately got a zero vector, simply make the first element unity.
	norm := vec.FrobeniusNorm()
	if norm < epsilon {
		v := complex64(cmplx.Rect(1, rand.Float64()*2*math.Pi))
		vec.SetAt([]int{0, 0}, v)
		norm = vec.FrobeniusNorm()
	}

	vec.Mul(complex(1/norm, 0))
	return vec
}
