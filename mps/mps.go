// Why we need to normalize: Eq. 211, page 66
package mps

import (
	"fmt"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"

	"qising/tensor"

	"github.com/pkg/errors"
)

const (
	// mpsLeftAxis is the axis of a_{l-1} in Figure 6.
	mpsLeftAxis  = 0
	mpsUpAxis    = 1
	mpsRightAxis = 2
	// mpoLeftAxis is the axis of b_{l-1} in Figure 35.
	mpoLeftAxis  = 0
	mpoRightAxis = 1
	mpoUpAxis    = 2
	mpoDownAxis  = 3

	// Machine precision.
	epsilon = 0x1p-23
)

// NewMPS create a matrix product representation of a general state.
func NewMPS(state *tensor.Dense, bufs []*tensor.Dense) []*tensor.Dense {
	shape := state.Shape()

	sites := make([]*tensor.Dense, 0, len(shape))
	var leftD int = 1
	for _, physD := range shape[:len(shape)-1] {
		q := tensor.Zeros(1)
		r := tensor.QR(q, state.Reshape(leftD*physD, -1), bufs)

		leftD = r.Shape()[0]
		state = r

		sites = append(sites, q.Reshape(-1, physD, leftD))
	}

	state = state.Reshape(leftD, shape[len(shape)-1], 1)
	sites = append(sites, resetCopy(tensor.Zeros(1), state))

	return sites
}

// RandMPS creates a random matrix product state.
// d is D below equation 71 in section 4.1.4.
// Consult Figure 37 for details.
func RandMPS(mpo []*tensor.Dense, maxD int) []*tensor.Dense {
	sites := make([]*tensor.Dense, 0, len(mpo))

	// First site.
	physD := mpo[0].Shape()[mpoDownAxis]
	leftD := physD
	sites = append(sites, randTensor(1, physD, min(physD, maxD)))

	for i := 1; i <= len(mpo)-2; i++ {
		physD := mpo[i].Shape()[mpoDownAxis]
		var rightD int
		switch {
		case i < len(mpo)/2:
			rightD = leftD * physD
		case i > len(mpo)/2:
			rightD = leftD / physD
		case len(mpo)%2 == 0:
			rightD = leftD / physD
		default:
			rightD = leftD
		}
		leftD = rightD

		si1 := sites[i-1].Shape()
		sites = append(sites, randTensor(si1[mpsRightAxis], physD, min(rightD, maxD)))
	}

	// Last site.
	physD = mpo[len(mpo)-1].Shape()[mpoDownAxis]
	si1 := sites[len(mpo)-2].Shape()
	sites = append(sites, randTensor(si1[mpsRightAxis], physD, 1))

	return sites
}

func InnerProduct(x, y []*tensor.Dense, bufs []*tensor.Dense) complex64 {
	if len(x) != len(y) {
		panic(fmt.Sprintf("%d %d", len(x), len(y)))
	}

	f := ones(bufs[0], 1, 1)
	const fTopAxis, fBottomAxis = 0, 1
	bufs = bufs[1:]
	for i, xi := range x {
		yi := y[i]

		fyi := tensor.Contract(bufs[0], f, yi, [][2]int{{fBottomAxis, mpsLeftAxis}})
		tensor.Contract(f, xi.Conj(), fyi, [][2]int{{mpsLeftAxis, fTopAxis}, {mpsUpAxis, mpsUpAxis}})
	}

	if !slices.Equal(f.Shape(), []int{1, 1}) {
		panic(fmt.Sprintf("%#v", f.Shape()))
	}
	return f.At(0, 0)
}

func LExpressions(fs, ws, ms, bufs []*tensor.Dense) complex64 {
	if len(fs) != len(ws) {
		panic(fmt.Sprintf("%d %d", len(fs), len(ws)))
	}
	if len(ws) != len(ms) {
		panic(fmt.Sprintf("%d %d", len(ws), len(ms)))
	}

	fi1 := ones(fs[0], 1, 1, 1)
	for i, w := range ws {
		m := ms[i]
		fi1 = lExpression(fs[i], fi1, w, m, bufs)
	}

	if !slices.Equal(fi1.Shape(), []int{1, 1, 1}) {
		panic(fmt.Sprintf("%#v", fi1.Shape()))
	}
	return fi1.At(0, 0, 0)
}

func lExpression(fi, fi1, w, m *tensor.Dense, bufs []*tensor.Dense) *tensor.Dense {
	// fi1 is of shape {fTop, fMid, fBot}.
	// fm is of shape {fTop, fMid, mpsTop, mpsRight}.
	fm := tensor.Contract(bufs[0], fi1, m, [][2]int{{2, mpsLeftAxis}})

	// wfm is of shape {mpoRight, mpoUp, fTop, mpsRight}.
	wfm := tensor.Contract(bufs[1], w, fm, [][2]int{{mpoDownAxis, 2}, {mpoLeftAxis, 1}})

	// fi is of shape {mpsRight.conj, mpoRight, mpsRight}.
	tensor.Contract(fi, m.Conj(), wfm, [][2]int{{mpsLeftAxis, 2}, {mpsUpAxis, 1}})

	return fi
}

func RExpressions(fs, ws, ms, bufs []*tensor.Dense) complex64 {
	if len(fs) != len(ws) {
		panic(fmt.Sprintf("%d %d", len(fs), len(ws)))
	}
	if len(ws) != len(ms) {
		panic(fmt.Sprintf("%d %d", len(ws), len(ms)))
	}

	fi1 := ones(fs[len(fs)-1], 1, 1, 1)
	for i := len(fs) - 1; i >= 0; i-- {
		w, m := ws[i], ms[i]
		fi1 = rExpression(fs[i], fi1, w, m, bufs)
	}

	if !slices.Equal(fi1.Shape(), []int{1, 1, 1}) {
		panic(fmt.Sprintf("%#v", fi1.Shape()))
	}
	return fi1.At(0, 0, 0)
}

func rExpression(fi, fi1, w, m *tensor.Dense, bufs []*tensor.Dense) *tensor.Dense {
	// fi1 is of shape {fTop, fMid, fBot}.
	// fm is of shape {fTop, fMid, mpsLeft, mpsTop}.
	fm := tensor.Contract(bufs[0], fi1, m, [][2]int{{2, mpsRightAxis}})

	// wfm is of shape {mpoLeft, mpoUp, fTop, mpsLeft}.
	wfm := tensor.Contract(bufs[1], w, fm, [][2]int{{mpoDownAxis, 3}, {mpoRightAxis, 1}})

	// fi is of shape {mpsLeft.conj, mpoLeft, mpsLeft}.
	tensor.Contract(fi, m.Conj(), wfm, [][2]int{{mpsRightAxis, 2}, {mpsUpAxis, 1}})

	return fi
}

// page 74.
func H2(ws, ms, bufs []*tensor.Dense) complex64 {
	if len(ws) != len(ms) {
		panic(fmt.Sprintf("%d %d", len(ws), len(ms)))
	}

	// fi1 is the F expression at site i-1, and is of shape {fTop, fMid2, fMid, fBot}.
	fi1 := ones(bufs[0], 1, 1, 1, 1)
	for i, w := range ws {
		m := ms[i]

		// fm is of shape {fTop, fMid2, fMid, mpsTop, mpsRight}.
		fm := tensor.Contract(bufs[1], fi1, m, [][2]int{{3, mpsLeftAxis}})

		// wfm is of shape {mpoRight, mpoUp, fTop, fMid2, mpsRight}.
		wfm := tensor.Contract(bufs[0], w, fm, [][2]int{{mpoDownAxis, 3}, {mpoLeftAxis, 2}})

		// wwfm is of shape {mpoRight2, mpoUp2, mpoRight, fTop, mpsRight}.
		wwfm := tensor.Contract(bufs[1], w, wfm, [][2]int{{mpoDownAxis, 1}, {mpoLeftAxis, 3}})

		// fi1 is of shape {mpsRight.conj, mpoRight2, mpoRight, mpsRight}.
		fi1 = tensor.Contract(bufs[0], m.Conj(), wwfm, [][2]int{{mpsLeftAxis, 3}, {mpsUpAxis, 1}})
	}

	if !slices.Equal(fi1.Shape(), []int{1, 1, 1, 1}) {
		panic(fmt.Sprintf("%#v", fi1.Shape()))
	}
	return fi1.At(0, 0, 0, 0)
}

type SearchGroundStateOptions struct {
	maxIterations int
	tol           float32
}

func NewSearchGroundStateOptions() SearchGroundStateOptions {
	opt := SearchGroundStateOptions{}
	opt.maxIterations = 32
	opt.tol = 1e-6
	return opt
}

func (opt SearchGroundStateOptions) MaxIterations(i int) SearchGroundStateOptions {
	opt.maxIterations = i
	return opt
}

func (opt SearchGroundStateOptions) Tol(tol float32) SearchGroundStateOptions {
	opt.tol = tol
	return opt
}

func SearchGroundState(fs, ws, ms, bufs []*tensor.Dense, options ...SearchGroundStateOptions) error {
	opt := NewSearchGroundStateOptions()
	if len(options) > 0 {
		opt = options[0]
	}

	rightNormalizeAll(ms, bufs)
	RExpressions(fs, ws, ms, bufs)
	convergence := struct {
		ok bool
		h2 complex64
	}{}
	for i := range opt.maxIterations {
		if err := rightSweep(fs, ws, ms, bufs); err != nil {
			return errors.Wrap(err, fmt.Sprintf("%d", i))
		}
		if err := leftSweep(fs, ws, ms, bufs); err != nil {
			return errors.Wrap(err, fmt.Sprintf("%d", i))
		}

		// Test for convergence.
		psiIP := InnerProduct(ms, ms, bufs)
		if abs(psiIP) < epsilon {
			return errors.Errorf("%f", psiIP)
		}
		// Since leftSweep built R expression to fs[1], we need only further build fs[0].
		rExpression(fs[0], fs[1], ws[0], ms[0], bufs)
		h := fs[0].At(0, 0, 0) / psiIP
		// Compute h2 and use the criterion h2 - h*h.
		h2 := H2(ws, ms, bufs) / psiIP
		convergence.h2 = h2 - h*h
		if abs(convergence.h2) < opt.tol*max(abs(h2), 1) {
			convergence.ok = true
			break
		}
	}
	if !convergence.ok {
		return errors.Errorf("%#v", convergence)
	}
	return nil
}

func leftSweep(fs, ws, ms, bufs []*tensor.Dense) error {
	for l := len(ms) - 1; l >= 1; l-- {
		fRight := ones(fs[l], 1, 1, 1)
		if l+1 <= len(ms)-1 {
			fRight = fs[l+1]
		}
		h := getH(bufs[0], fs[l-1], fRight, ws[l], l, bufs[1:])

		eigvals, eigvecs := bufs[1], bufs[2]
		if err := tensor.Arnoldi(eigvals, eigvecs, h, 1, bufs[3:]); err != nil {
			return errors.Wrap(err, "")
		}
		resetCopy(ms[l], eigvecs.Reshape(ms[l].Shape()...))

		// Right normalize ms[l], and multiply into ms[l-1].
		// Since ms[l-1] is modified, reset fs[l-1].
		rightNormalize(ms, l, bufs)
		fs[l-1].Reset(1)

		rExpression(fs[l], fRight, ws[l], ms[l], bufs)
	}
	return nil
}

func rightSweep(fs, ws, ms, bufs []*tensor.Dense) error {
	for l := range len(ms) - 1 {
		fLeft := ones(fs[l], 1, 1, 1)
		if l-1 >= 0 {
			fLeft = fs[l-1]
		}
		h := getH(bufs[0], fLeft, fs[l+1], ws[l], l, bufs[1:])

		eigvals, eigvecs := bufs[1], bufs[2]
		if err := tensor.Arnoldi(eigvals, eigvecs, h, 1, bufs[3:]); err != nil {
			return errors.Wrap(err, "")
		}
		resetCopy(ms[l], eigvecs.Reshape(ms[l].Shape()...))

		// Left normalize ms[l], and multiply into ms[l+1].
		// Since ms[l+1] is modified, reset fs[l+1].
		leftNormalize(ms, l, bufs)
		fs[l+1].Reset(1)

		lExpression(fs[l], fLeft, ws[l], ms[l], bufs)
	}
	return nil
}

func getH(h, left, right, w *tensor.Dense, l int, bufs []*tensor.Dense) *tensor.Dense {
	// right is of shape {rightTop, rightMid, rightBot}.
	// wRight is of shape {mpoLeft, mpoUp, mpoDown, rightTop, rightBot}.
	wRight := tensor.Contract(bufs[0], w, right, [][2]int{{mpoRightAxis, 1}})

	// left is of shape {leftTop, leftMid, leftBot}.
	// lwr is of shape {leftTop, leftBot, mpoUp, mpoDown, rightTop, rightBot}.
	lwr := tensor.Contract(bufs[1], left, wRight, [][2]int{{1, 0}})

	// h is of shape {leftTop, mpoUp, rightTop, leftBot, mpoDown, rightBot}.
	resetCopy(h, lwr.Transpose(0, 2, 4, 1, 3, 5))

	// Reshape h to square matrix.
	ls, ws, rs := left.Shape(), w.Shape(), right.Shape()
	if ls[0] != ls[2] || ws[mpoUpAxis] != ws[mpoDownAxis] || rs[0] != rs[2] {
		panic(fmt.Sprintf("%#v %#v %#v", ls, ws, rs))
	}
	return h.Reshape(ls[0]*ws[mpoUpAxis]*rs[0], ls[2]*ws[mpoDownAxis]*rs[2])
}

func rightNormalizeAll(ms []*tensor.Dense, bufs []*tensor.Dense) {
	for i := len(ms) - 1; i >= 1; i-- {
		rightNormalize(ms, i, bufs)
	}
}

// rightNormalize normalizes a MPS site from the right.
// Consult Section 4.4.2 for details.
func rightNormalize(ms []*tensor.Dense, i int, bufs []*tensor.Dense) {
	s := ms[i].Shape()
	dUp, dRight := s[mpsUpAxis], s[mpsRightAxis]

	// Decompose ms[i] = l @ q.H.
	mi := ms[i].Reshape(s[mpsLeftAxis], dUp*dRight)
	q := bufs[0]
	l := lq(q, mi, bufs[1:])

	// ms[i-1] = ms[i-1] @ l.
	axes := [][2]int{{mpsRightAxis, 0}}
	resetCopy(ms[i-1], tensor.Contract(bufs[1], ms[i-1], l, axes))

	// ms[i] = q.H.
	ms[i] = resetCopy(ms[i], q.H()).Reshape(-1, dUp, dRight)
}

func leftNormalizeAll(ms []*tensor.Dense, bufs []*tensor.Dense) {
	for i := range len(ms) - 1 {
		leftNormalize(ms, i, bufs)
	}
}

func leftNormalize(ms []*tensor.Dense, i int, bufs []*tensor.Dense) {
	s := ms[i].Shape()
	dLeft, dUp := s[mpsLeftAxis], s[mpsUpAxis]

	// Decompose ms[i] = q @ r.
	mi := ms[i].Reshape(dLeft*dUp, s[mpsRightAxis])
	q := bufs[0]
	r := tensor.QR(q, mi, bufs[1:])

	// ms[i+1] = r @ ms[i+1].
	axes := [][2]int{{1, mpsLeftAxis}}
	resetCopy(ms[i+1], tensor.Contract(bufs[1], r, ms[i+1], axes))

	// ms[i] = q.
	ms[i] = resetCopy(ms[i], q).Reshape(dLeft, dUp, -1)
}

func lq(q, a *tensor.Dense, bufs []*tensor.Dense) *tensor.Dense {
	r := tensor.QR(q, a.H(), bufs)
	return r.H()
}

func product(p *tensor.Dense, ms []*tensor.Dense, buf *tensor.Dense) *tensor.Dense {
	// mmi is the product of m0 @ m1 @ ... mi.
	var mmi *tensor.Dense

	// Do mmi = mmi @ mi.
	mmiPrev := buf
	resetCopy(mmiPrev, ms[0])
	for _, mi := range ms[1:] {
		if mmiPrev == buf {
			mmi = p
		} else {
			mmi = buf
		}
		axes := [][2]int{{len(mmiPrev.Shape()) - 1, 0}}
		tensor.Contract(mmi, mmiPrev, mi, axes)

		mmiPrev = mmi
	}

	if mmi == buf {
		resetCopy(p, mmi)
	}
	return p
}

func format(a *tensor.Dense) string {
	shapeStrs := make([]string, 0, len(a.Shape()))
	for _, d := range a.Shape() {
		shapeStrs = append(shapeStrs, strconv.Itoa(d))
	}
	shapeS := strings.Join(shapeStrs, ",")

	ss := make([]string, 0)
	for _, v := range a.All() {
		s := fmt.Sprintf("%v", v)
		s = strings.ReplaceAll(s, "i", "j")
		ss = append(ss, s)
	}
	s := strings.Join(ss, ",")

	return fmt.Sprintf("[%s][%s]", shapeS, s)
}

func resetCopy(dst, src *tensor.Dense) *tensor.Dense {
	shape := src.Shape()
	zeroDigit := make([]int, len(shape))
	dst.Reset(shape...).Set(zeroDigit, src)
	return dst
}

func ones(t *tensor.Dense, shape ...int) *tensor.Dense {
	t.Reset(shape...)
	for ijk := range t.All() {
		t.SetAt(ijk, 1)
	}
	return t
}

func abs(x complex64) float32 {
	return float32(cmplx.Abs(complex128(x)))
}

func randTensor(shape ...int) *tensor.Dense {
	t := tensor.Zeros(shape...)
	for ijk := range t.All() {
		v := complex(rand.Float32()*2-1, rand.Float32()*2-1)
		t.SetAt(ijk, v)
	}
	return t
}
