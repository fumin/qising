package mps

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/fumin/tensor"
)

func TestNewMPS(t *testing.T) {
	t.Parallel()
	type testcase struct {
		state  *tensor.Dense
		shapes [][]int
		tol    float32
	}
	tests := []testcase{}

	var tc testcase
	tc.state = tensor.Zeros(2, 2, 2)
	tc.state.SetAt([]int{0, 0, 0}, 1)
	tc.shapes = [][]int{{1, 2, 2}, {2, 2, 2}, {2, 2, 1}}
	tc.tol = epsilon
	tests = append(tests, tc)

	tc = testcase{}
	tc.state = randTensor(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
	tc.shapes = [][]int{{1, 2, 2}, {2, 2, 4}, {4, 2, 8}, {8, 2, 16}, {16, 2, 32}, {32, 2, 64}, {64, 2, 32}, {32, 2, 16}, {16, 2, 8}, {8, 2, 4}, {4, 2, 2}, {2, 2, 1}}
	tc.tol = 5e-6
	tests = append(tests, tc)

	tc = testcase{}
	tc.state = randTensor(3, 3, 3, 3, 3, 3, 3)
	tc.shapes = [][]int{{1, 3, 3}, {3, 3, 9}, {9, 3, 27}, {27, 3, 27}, {27, 3, 9}, {9, 3, 3}, {3, 3, 1}}
	tc.tol = 5e-6
	tests = append(tests, tc)

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			var bufs [2]*tensor.Dense
			for i := range len(bufs) {
				bufs[i] = tensor.Zeros(1)
			}
			state := resetCopy(tensor.Zeros(1), test.state)

			mps := NewMPS(test.state, bufs)

			// Check shapes.
			if len(mps) != len(test.shapes) {
				t.Fatalf("%d %d", len(mps), len(test.shapes))
			}
			for j, m := range mps {
				if !slices.Equal(m.Shape(), test.shapes[j]) {
					t.Fatalf("%d %#v %#v", j, m.Shape(), test.shapes[j])
				}
			}

			// Check the product of mps is equal to state.
			mpsState := product(tensor.Zeros(1), mps, bufs[0])
			s := mpsState.Shape()
			mpsState = mpsState.Reshape(s[1 : len(s)-1]...)
			if err := mpsState.Equal(state, test.tol); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check sites are unitary.
			axes := [][2]int{
				{mpsLeftAxis, mpsLeftAxis},
				{mpsUpAxis, mpsUpAxis},
			}
			for _, m := range mps[:len(mps)-1] {
				mm := tensor.Product(bufs[0], m.Conj(), m, axes)
				eye := bufs[1].Eye(mm.Shape()[0], 0)
				if err := mm.Equal(eye, 10*epsilon); err != nil {
					t.Fatalf("%+v", err)
				}
			}
		})
	}
}

func TestRandMPS(t *testing.T) {
	t.Parallel()
	type testcase struct {
		mpo     []*tensor.Dense
		bondDim int
		shapes  [][]int
	}
	tests := []testcase{
		{
			mpo:     Ising([2]int{12, 1}, 1),
			bondDim: 999,
			shapes:  [][]int{{1, 2, 2}, {2, 2, 4}, {4, 2, 8}, {8, 2, 16}, {16, 2, 32}, {32, 2, 64}, {64, 2, 32}, {32, 2, 16}, {16, 2, 8}, {8, 2, 4}, {4, 2, 2}, {2, 2, 1}},
		},
		{
			mpo:     Ising([2]int{7, 1}, 1),
			bondDim: 999,
			shapes:  [][]int{{1, 2, 2}, {2, 2, 4}, {4, 2, 8}, {8, 2, 8}, {8, 2, 4}, {4, 2, 2}, {2, 2, 1}},
		},
		{
			mpo:     Ising([2]int{12, 1}, 1),
			bondDim: 7,
			shapes:  [][]int{{1, 2, 2}, {2, 2, 4}, {4, 2, 7}, {7, 2, 7}, {7, 2, 7}, {7, 2, 7}, {7, 2, 7}, {7, 2, 7}, {7, 2, 7}, {7, 2, 4}, {4, 2, 2}, {2, 2, 1}},
		},
		{
			mpo:     Ising([2]int{7, 1}, 1),
			bondDim: 5,
			shapes:  [][]int{{1, 2, 2}, {2, 2, 4}, {4, 2, 5}, {5, 2, 5}, {5, 2, 4}, {4, 2, 2}, {2, 2, 1}},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			mps := RandMPS(test.mpo, test.bondDim)

			if len(mps) != len(test.shapes) {
				t.Fatalf("%d %d", len(mps), len(test.shapes))
			}
			for i, shape := range test.shapes {
				if !slices.Equal(mps[i].Shape(), shape) {
					t.Fatalf("%d %#v %#v", i, mps[i].Shape(), shape)
				}
			}
		})
	}
}

func TestExpectation(t *testing.T) {
	t.Parallel()
	type testcase struct {
		state       *tensor.Dense
		normSquare  float32
		op          []*tensor.Dense
		expectation complex64
		h2          complex64
		tol         float32
	}
	tests := []testcase{}

	// qubits: up, up, up.
	var tc testcase
	tc.state = tensor.Zeros(2, 2, 2)
	tc.state.SetAt([]int{0, 0, 0}, 1)
	tc.normSquare = 1
	tc.op = MagnetizationZ([2]int{3, 1})
	tc.expectation = 3
	tc.h2 = 9
	tc.tol = epsilon
	tests = append(tests, tc)

	// qubits: up, down, down.
	tc = testcase{}
	tc.state = tensor.Zeros(2, 2, 2)
	tc.state.SetAt([]int{0, 1, 1}, 1)
	tc.normSquare = 1
	tc.op = MagnetizationZ([2]int{3, 1})
	tc.expectation = -1
	tc.h2 = 1
	tc.tol = epsilon
	tests = append(tests, tc)

	tc = testcase{}
	tc.state = tensor.Zeros(2, 2, 2)
	for k := range 2 {
		tc.state.SetAt([]int{k, 0, 0}, 4i/5)
		tc.state.SetAt([]int{k, 0, 1}, -3i/5)
		tc.state.SetAt([]int{k, 1, 0}, (3+4i)/5)
		tc.state.SetAt([]int{k, 1, 1}, 0)
	}
	tc.normSquare = 4
	tc.op = MagnetizationZ([2]int{3, 1})
	tc.expectation = (2 * (7./25 + 1))
	tc.h2 = 9.12
	tc.tol = 5 * epsilon
	tests = append(tests, tc)

	for _ = range 128 {
		tc = testcase{}
		tc.state = randTensor(2, 2, 2, 2, 2, 2, 2)
		psi := tc.state.Reshape(-1, 1)
		tc.normSquare = real(tensor.MatMul(tensor.Zeros(1), psi.H(), psi).At(0, 0))

		tc.op = MagnetizationZ([2]int{len(tc.state.Shape()), 1})
		// Compute expectation explicitly.
		i := -1
		for basis := range tc.state.All() {
			i++
			var m int
			for _, spin := range basis {
				if spin == 0 {
					m++
				} else {
					m--
				}
			}
			mf := float32(m)

			c := psi.At(i, 0)
			tc.expectation += complex(abs(c)*abs(c)*mf, 0)
			tc.h2 += complex(abs(c)*abs(c)*mf*mf, 0)
		}

		tc.tol = 1e-3
		tests = append(tests, tc)
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			var bufs [2]*tensor.Dense
			for i := range len(bufs) {
				bufs[i] = tensor.Zeros(1)
			}
			mps := NewMPS(test.state, bufs)

			// Check norm square.
			ns := real(InnerProduct(mps, mps, bufs))
			if diff := absf(ns - test.normSquare); diff > test.tol {
				t.Fatalf("%f %f %f", diff, ns, test.normSquare)
			}

			fs := make([]*tensor.Dense, 0, len(mps))
			for _ = range len(mps) {
				fs = append(fs, tensor.Zeros(1))
			}
			expectation := LExpressions(fs, test.op, mps, bufs)

			// Check L-expression.
			if diff := abs(expectation - test.expectation); diff > test.tol {
				t.Fatalf("%f %f %f", diff, expectation, test.expectation)
			}
			last := fs[len(fs)-1].At(0, 0, 0)
			if diff := abs(last - test.expectation); diff > test.tol {
				t.Fatalf("%f %f %f", diff, last, test.expectation)
			}

			// Check R-expression.
			expectation = RExpressions(fs, test.op, mps, bufs)
			if diff := abs(expectation - test.expectation); diff > test.tol {
				t.Fatalf("%f %f %f", diff, expectation, test.expectation)
			}
			first := fs[0].At(0, 0, 0)
			if diff := abs(first - test.expectation); diff > test.tol {
				t.Fatalf("%f %f %f", diff, first, test.expectation)
			}

			// Check expectation value of H @ H.
			h2 := H2(test.op, mps, bufs)
			if diff := abs(h2 - test.h2); diff > test.tol {
				t.Fatalf("%f %f %f", diff, h2, test.h2)
			}
		})
	}
}

func TestSearchGroundState(t *testing.T) {
	t.Parallel()
	type testcase struct {
		h   []*tensor.Dense
		e0  complex64
		mz  []*tensor.Dense
		m   complex64
		tol float32
	}
	tests := []testcase{
		{
			h:   Ising([2]int{4, 1}, 0.031623),
			e0:  -3.001501,
			mz:  MagnetizationZ([2]int{4, 1}),
			m:   0.999765,
			tol: 2e-6,
		},
		{
			h:   Ising([2]int{16, 1}, 0.031623),
			e0:  -15.004505,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.999839,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 0.1),
			e0:  -15.045036,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.998380,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 0.316228),
			e0:  -15.453211,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.982974,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 0.501187),
			e0:  -16.151592,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.952991,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 0.630957),
			e0:  -16.847988,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.916530,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 0.794328),
			e0:  -18.003237,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.825456,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 0.891251),
			e0:  -18.869982,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.723979,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 1.122018),
			e0:  -21.471106,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.502189,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 1.258925),
			e0:  -23.240532,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.439212,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 1.584893),
			e0:  -27.780334,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.369304,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 1.995262),
			e0:  -33.830681,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.332405,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 3.162278),
			e0:  -51.788876,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.294568,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 10),
			e0:  -160.375198,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.262319,
			tol: 2e-4,
		},
		{
			h:   Ising([2]int{16, 1}, 100),
			e0:  -1600.037598,
			mz:  MagnetizationZ([2]int{16, 1}),
			m:   0.251177,
			tol: 2e-4,
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			fs := make([]*tensor.Dense, 0, len(test.h))
			for _ = range test.h {
				fs = append(fs, tensor.Zeros(1))
			}
			var bufs [10]*tensor.Dense
			for i := range len(bufs) {
				bufs[i] = tensor.Zeros(1)
			}

			const bondDim = 8
			mps := RandMPS(test.h, bondDim)
			if err := SearchGroundState(fs, test.h, mps, bufs); err != nil {
				t.Fatalf("%+v", err)
			}
			bufs2 := [2]*tensor.Dense(bufs[:2])
			psiIP := InnerProduct(mps, mps, bufs2)

			e0 := LExpressions(fs, test.h, mps, bufs2) / psiIP
			if diff := abs(e0 - test.e0); diff > test.tol*max(abs(test.e0), 1) {
				t.Fatalf("%f %f %f", diff, e0, test.e0)
			}

			m2 := H2(test.mz, mps, bufs2) / psiIP
			m := sqrt(m2) / complex(float32(len(mps)), 0) // per spin
			if diff := abs(m - test.m); diff > test.tol*max(abs(test.m), 1) {
				t.Fatalf("%f %f %f", diff, m, test.m)
			}
		})
	}
}

func TestNormlize(t *testing.T) {
	t.Parallel()
	type testcase struct {
		mps    []*tensor.Dense
		isLeft bool
		tol    float32
	}
	tests := []testcase{
		{
			mps: []*tensor.Dense{
				tensor.T3([][][]complex64{{{1, 2}, {3, 4}}}),
				tensor.T3([][][]complex64{{{5}, {6}}, {{7}, {8}}}),
			},
			tol: epsilon,
		},
	}

	var tc testcase
	const bondDim = 3
	tc.mps = RandMPS(Ising([2]int{4, 1}, -100), bondDim)
	tc.tol = 1e-5
	tests = append(tests, tc)

	// Add tests for left normalization.
	testsLen := len(tests)
	for i := range testsLen {
		var tc testcase

		tc.mps = make([]*tensor.Dense, 0, len(tests[i].mps))
		for _, m := range tests[i].mps {
			tc.mps = append(tc.mps, resetCopy(tensor.Zeros(1), m))
		}

		tc.tol = tests[i].tol
		tc.isLeft = true
		tests = append(tests, tc)
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			bufs := make([]*tensor.Dense, 0)
			for _ = range 3 {
				bufs = append(bufs, tensor.Zeros(1))
			}
			state := product(tensor.Zeros(1), test.mps, bufs[0])

			if test.isLeft {
				leftNormalizeAll(test.mps, bufs)
			} else {
				rightNormalizeAll(test.mps, bufs)
			}

			// Check that a normalized state represents the same state.
			normed := product(tensor.Zeros(1), test.mps, bufs[0])
			if err := normed.Equal(state, test.tol); err != nil {
				t.Fatalf("%d %+v", i, err)
			}

			// Check that sites are unitary.
			var axes [][2]int
			var sites []*tensor.Dense
			if test.isLeft {
				axes = [][2]int{
					{mpsLeftAxis, mpsLeftAxis},
					{mpsUpAxis, mpsUpAxis},
				}
				sites = test.mps[:len(test.mps)-1]
			} else {
				axes = [][2]int{
					{mpsRightAxis, mpsRightAxis},
					{mpsUpAxis, mpsUpAxis},
				}
				sites = test.mps[1:]
			}
			for i, b := range sites {
				bb := tensor.Product(tensor.Zeros(1), b.Conj(), b, axes)
				eye := tensor.Zeros(1).Eye(bb.Shape()[0], 0)
				if err := bb.Equal(eye, 10*epsilon); err != nil {
					t.Fatalf("%d %+v", i, err)
				}
			}
		})
	}
}

func TestLQ(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *tensor.Dense
		l *tensor.Dense
	}{
		{
			a: tensor.T2([][]complex64{{1, 3, 5, 1 - 3i}, {1 + 2i, 4, 6, 4 - 1i}}),
			l: tensor.T2([][]complex64{{6.7082, 0}, {7.45356 + 1.93793i, 3.83261}}),
		},
		{
			a: tensor.T2([][]complex64{{1 - 1i, -2 - 7i}, {5 - 3i, -4}, {-1, 2 - 1i}, {4 + 1i, 5}, {3 + 2i, -1 - 3i}}),
			l: tensor.T2([][]complex64{{7.41619849, 0}, {2.15743956 - 3.50583928i, 5.74930826}, {0.26967994 + 2.02259959i, -1.12898958 - 0.74949728i}, {-0.94387981 + 5.3935989i, 3.12132412 + 1.50848187i}, {3.23615934 + 0.53935989i, 1.37565957 + 3.21619719i}}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			a := resetCopy(tensor.Zeros(1), test.a)
			var bufs [2]*tensor.Dense
			for i := range len(bufs) {
				bufs[i] = tensor.Zeros(1)
			}
			q := tensor.Zeros(1)
			l := lq(q, test.a, bufs)

			// Check shape of l.
			m, n := a.Shape()[0], a.Shape()[1]
			lShape := []int{m, n}
			if m < n {
				lShape = []int{m, m}
			}
			if !slices.Equal(l.Shape(), lShape) {
				t.Fatalf("%#v %#v", l.Shape(), lShape)
			}

			// Check a = l @ q.H.
			lqh := tensor.MatMul(tensor.Zeros(1), l, q.H())
			if err := lqh.Equal(a, 2*epsilon*a.FrobeniusNorm()); err != nil {
				t.Fatalf("%+v", err)
			}

			// Check q.H @ q = I.
			qq := tensor.MatMul(tensor.Zeros(1), q.H(), q)
			if err := qq.Equal(tensor.Zeros(1).Eye(qq.Shape()[0], 0), 2*epsilon); err != nil {
				t.Fatalf("%+v", err)
			}

			if err := l.Equal(test.l, 5*epsilon*a.FrobeniusNorm()); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestProduct(t *testing.T) {
	t.Parallel()
	tests := []struct {
		ms []*tensor.Dense
		p  *tensor.Dense
	}{
		{
			ms: []*tensor.Dense{
				rangeT(1, 2, 2),
				rangeT(2, 2, 4).Add(-1i, rangeT(2, 2, 4)),
				rangeT(4, 2, 8).Add(3i, rangeT(4, 2, 8)),
			},
			p: tensor.T1([]complex64{2.529296875 + 1.2646484375i, 2.62109375 + 1.310546875i, 2.712890625 + 1.3564453125i, 2.8046875 + 1.40234375i, 2.896484375 + 1.4482421875i, 2.98828125 + 1.494140625i, 3.080078125 + 1.5400390625i, 3.171875 + 1.5859375i, 3.263671875 + 1.6318359375i, 3.35546875 + 1.677734375i, 3.447265625 + 1.7236328125i, 3.5390625 + 1.76953125i, 3.630859375 + 1.8154296875i, 3.72265625 + 1.861328125i, 3.814453125 + 1.9072265625i, 3.90625 + 1.953125i, 3.701171875 + 1.8505859375i, 3.83984375 + 1.919921875i, 3.978515625 + 1.9892578125i, 4.1171875 + 2.05859375i, 4.255859375 + 2.1279296875i, 4.39453125 + 2.197265625i, 4.533203125 + 2.2666015625i, 4.671875 + 2.3359375i, 4.810546875 + 2.4052734375i, 4.94921875 + 2.474609375i, 5.087890625 + 2.5439453125i, 5.2265625 + 2.61328125i, 5.365234375 + 2.6826171875i, 5.50390625 + 2.751953125i, 5.642578125 + 2.8212890625i, 5.78125 + 2.890625i, 5.380859375 + 2.6904296875i, 5.57421875 + 2.787109375i, 5.767578125 + 2.8837890625i, 5.9609375 + 2.98046875i, 6.154296875 + 3.0771484375i, 6.34765625 + 3.173828125i, 6.541015625 + 3.2705078125i, 6.734375 + 3.3671875i, 6.927734375 + 3.4638671875i, 7.12109375 + 3.560546875i, 7.314453125 + 3.6572265625i, 7.5078125 + 3.75390625i, 7.701171875 + 3.8505859375i, 7.89453125 + 3.947265625i, 8.087890625 + 4.0439453125i, 8.28125 + 4.140625i, 8.115234375 + 4.0576171875i, 8.41796875 + 4.208984375i, 8.720703125 + 4.3603515625i, 9.0234375 + 4.51171875i, 9.326171875 + 4.6630859375i, 9.62890625 + 4.814453125i, 9.931640625 + 4.9658203125i, 10.234375 + 5.1171875i, 10.537109375 + 5.2685546875i, 10.83984375 + 5.419921875i, 11.142578125 + 5.5712890625i, 11.4453125 + 5.72265625i, 11.748046875 + 5.8740234375i, 12.05078125 + 6.025390625i, 12.353515625 + 6.1767578125i, 12.65625 + 6.328125i}).Reshape(1, 2, 2, 2, 8),
		},
		{
			ms: []*tensor.Dense{
				rangeT(1, 2, 2),
				rangeT(2, 2, 4).Add(-1i, rangeT(2, 2, 4)),
				rangeT(4, 2, 8).Add(3i, rangeT(4, 2, 8)),
				rangeT(8, 2, 4).Mul(-1),
				rangeT(4, 2, 2).Mul(5),
				rangeT(2, 2, 1).Mul(-1i),
			},
			p: tensor.T1([]complex64{-56.089067459106445 + 112.17813491821289i, -83.24689865112305 + 166.4937973022461i, -70.27630805969238 + 140.55261611938477i, -104.52775955200195 + 209.0555191040039i, -62.99283027648926 + 125.98566055297852i, -93.49119186401367 + 186.98238372802734i, -78.9616870880127 + 157.9233741760254i, -117.44447708129883 + 234.88895416259766i, -69.93748664855957 + 139.87497329711914i, -103.8007926940918 + 207.6015853881836i, -87.62448310852051 + 175.24896621704102i, -130.3312873840332 + 260.6625747680664i, -78.61981391906738 + 157.23962783813477i, -116.68424606323242 + 233.36849212646484i, -98.54741096496582 + 197.09482192993164i, -146.57564163208008 + 293.15128326416016i, -82.47342109680176 + 164.94684219360352i, -122.40629196166992 + 244.81258392333984i, -103.3348560333252 + 206.6697120666504i, -153.69844436645508 + 307.39688873291016i, -92.61265754699707 + 185.22531509399414i, -137.45161056518555 + 274.9032211303711i, -116.09066963195801 + 232.18133926391602i, -172.66862869262695 + 345.3372573852539i, -103.39337348937988 + 206.78674697875977i, -153.45579147338867 + 306.91158294677734i, -129.54167366027832 + 259.08334732055664i, -192.67824172973633 + 385.35648345947266i, -116.2193775177002 + 232.4387550354004i, -172.4879264831543 + 344.9758529663086i, -145.67761421203613 + 291.35522842407227i, -216.6752815246582 + 433.3505630493164i, -119.14811134338379 + 238.29622268676758i, -176.8385887145996 + 353.6771774291992i, -149.28536415100098 + 298.57072830200195i, -222.0444679260254 + 444.0889358520508i, -133.81890296936035 + 267.6378059387207i, -198.60815048217773 + 397.21630096435547i, -167.74216651916504 + 335.4843330383301i, -249.49304580688477 + 498.98609161376953i, -148.3181858062744 + 296.6363716125488i, -220.13296127319336 + 440.2659225463867i, -185.8272647857666 + 371.6545295715332i, -276.39657974243164 + 552.7931594848633i, -166.73531532287598 + 333.47063064575195i, -247.46160507202148 + 494.92321014404297i, -208.99720191955566 + 417.9944038391113i, -310.854434967041 + 621.708869934082i, -180.71160316467285 + 361.4232063293457i, -268.210506439209 + 536.421012878418i, -226.42197608947754 + 452.8439521789551i, -336.776065826416 + 673.552131652832i, -202.9318332672119 + 405.8636665344238i, -301.1824607849121 + 602.3649215698242i, -254.3764591217041 + 508.7529182434082i, -378.3493995666504 + 756.6987991333008i, -226.38192176818848 + 452.76384353637695i, -335.99462509155273 + 671.9892501831055i, -283.63404273986816 + 567.2680854797363i, -421.87280654907227 + 843.7456130981445i, -254.46763038635254 + 508.9352607727051i, -377.67019271850586 + 755.3403854370117i, -318.9676761627197 + 637.9353523254395i, -474.42026138305664 + 948.8405227661133i}).Reshape(1, 2, 2, 2, 2, 2, 2, 1),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			buf := tensor.Zeros(1)
			p := product(tensor.Zeros(1), test.ms, buf)

			if err := p.Equal(test.p, epsilon*test.p.FrobeniusNorm()); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func rangeT(shape ...int) *tensor.Dense {
	var volume float32 = 1
	for _, d := range shape {
		volume *= float32(d)
	}

	t := tensor.Zeros(shape...)
	var p int
	for ijk := range t.All() {
		p++
		t.SetAt(ijk, complex(float32(p)/volume, 0))
	}
	return t
}

func absf(x float32) float32 {
	return float32(math.Abs(float64(x)))
}

func sqrt(x complex64) complex64 {
	return complex64(cmplx.Sqrt(complex128(x)))
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}
