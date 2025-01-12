package exactdiag

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"testing"

	"github.com/fumin/qising/exactdiag/mat"
)

func TestTransverseFieldIsing(t *testing.T) {
	t.Parallel()
	type matrixSlice struct {
		y [2]int
		x [2]int
		s *mat.COO
	}
	tests := []struct {
		n                [2]int
		h                complex64
		hamiltonianShape [2]int
		hamiltonian      []matrixSlice
	}{
		{
			n:                [2]int{4, 1},
			h:                1,
			hamiltonianShape: [2]int{16, 16},
			hamiltonian: []matrixSlice{
				{
					y: [2]int{0, 16},
					x: [2]int{0, 16},
					s: mat.M([][]complex64{
						{-3, -1, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0},
						{-1, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0},
						{-1, 0, 1, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0},
						{0, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0},
						{-1, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0},
						{0, -1, 0, 0, -1, 3, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0},
						{0, 0, -1, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, -1, 0},
						{0, 0, 0, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1},
						{-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0},
						{0, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, -1, 0, -1, 0, 0},
						{0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 3, -1, 0, 0, -1, 0},
						{0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, -1},
						{0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0},
						{0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 1, 0, -1},
						{0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, -1, -1},
						{0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, -1, -3},
					}),
				},
			},
		},
		{
			n:                [2]int{8, 1},
			h:                1,
			hamiltonianShape: [2]int{256, 256},
			hamiltonian: []matrixSlice{
				{
					y: [2]int{0, 10},
					x: [2]int{0, 9},
					s: mat.M([][]complex64{
						{-7, -1, -1, 0, -1, 0, 0, 0, -1},
						{-1, -5, 0, -1, 0, -1, 0, 0, 0},
						{-1, 0, -3, -1, 0, 0, -1, 0, 0},
						{0, -1, -1, -5, 0, 0, 0, -1, 0},
						{-1, 0, 0, 0, -3, -1, -1, 0, 0},
						{0, -1, 0, 0, -1, -1, 0, -1, 0},
						{0, 0, -1, 0, -1, 0, -3, -1, 0},
						{0, 0, 0, -1, 0, -1, -1, -5, 0},
						{-1, 0, 0, 0, 0, 0, 0, 0, -3},
						{0, -1, 0, 0, 0, 0, 0, 0, -1},
					}),
				},
				{
					y: [2]int{0, 10},
					x: [2]int{-9, 256},
					s: mat.COOZeros(10, 9),
				},
				{
					y: [2]int{-10, 256},
					x: [2]int{0, 9},
					s: mat.COOZeros(10, 9),
				},
				{
					y: [2]int{-9, 256},
					x: [2]int{-9, 256},
					s: mat.M([][]complex64{
						{-3, 0, 0, 0, 0, 0, 0, 0, -1},
						{0, -5, -1, -1, 0, -1, 0, 0, 0},
						{0, -1, -3, 0, -1, 0, -1, 0, 0},
						{0, -1, 0, -1, -1, 0, 0, -1, 0},
						{0, 0, -1, -1, -3, 0, 0, 0, -1},
						{0, -1, 0, 0, 0, -5, -1, -1, 0},
						{0, 0, -1, 0, 0, -1, -3, 0, -1},
						{0, 0, 0, -1, 0, -1, 0, -5, -1},
						{-1, 0, 0, 0, -1, 0, -1, -1, -7},
					}),
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v %#v", test.n, test.h), func(t *testing.T) {
			t.Parallel()
			hamiltonian := mat.M([][]complex64{{0}})
			buf := mat.M([][]complex64{{0}})
			TransverseFieldIsing(hamiltonian, buf, test.n, test.h)
			if !(hamiltonian.Rows() == test.hamiltonianShape[0] && hamiltonian.Cols() == test.hamiltonianShape[1]) {
				t.Fatalf("%d %d, expected %v", hamiltonian.Rows(), hamiltonian.Cols(), test.hamiltonianShape)
			}
			for _, th := range test.hamiltonian {
				s := hamiltonian.COO().Slice(th.y, th.x)
				if !s.Equal(th.s) {
					t.Fatalf("%s, expected %s", s, th.s)
				}
			}
		})
	}
}

func TestTransverseFieldIsingExplicit(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n [2]int
	}{
		{
			n: [2]int{8, 1},
		},
		{
			n: [2]int{2, 2},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test.n), func(t *testing.T) {
			t.Parallel()
			dir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("%+v", err)
			}
			defer os.RemoveAll(dir)

			m := mat.M([][]complex64{{0}})
			buf := mat.M([][]complex64{{0}})
			TransverseFieldIsing(m, buf, test.n, 1)

			TransverseFieldIsingExplicit(dir, test.n, 1)
			mExplicit, err := mat.ReadCOO(dir)
			if err != nil {
				t.Fatalf("%+v", err)
			}

			if !mExplicit.Equal(m) {
				t.Fatalf("\n%s, expected \n\n%s", mExplicit, m)
			}
		})
	}
}

func TestEigen(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", t.Name())
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	h, buf := mat.M([][]complex64{{0}}), mat.M([][]complex64{{0}})
	TransverseFieldIsing(h, buf, [2]int{8, 1}, 1)
	vvs := h.COO().Eigen()

	// Check eigenvalues.
	// Values are from https://juliaphysics.github.io/PhysicsTutorials.jl/tutorials/general/quantum_ising/quantum_ising.html
	vals := []float64{-9.837951447459426, -9.46887800960621, -8.7432994871710, -8.374226049317867, -8.054998024353266, -7.685924586500063, -7.427412901942416, -7.058339464089192, -6.960346064064927, -6.881915778576785}
	for i, v := range vvs[0:10] {
		if math.Abs(real(v.Val)-vals[i]) > 1e-6 {
			t.Fatalf("%d %v %f", i, v.Val, vals[i])
		}
	}
	vals = []float64{6.960346064064934, 7.0583394640891886, 7.427412901942393, 7.685924586500062, 8.054998024353269, 8.374226049317883, 8.74329948717109, 9.468878009606211, 9.83795144745942}
	for i, v := range vvs[len(vvs)-9:] {
		if math.Abs(real(v.Val)-vals[i]) > 1e-6 {
			t.Fatalf("%d %v %f", i, v.Val, vals[i])
		}
	}

	// Check eigenvectors.
	var probSum float64
	for _, v := range vvs[0].Vec {
		probSum += real(v)*real(v) + imag(v)*imag(v)
	}
	if math.Abs(probSum-1) > 1e-6 {
		t.Fatalf("%f", probSum)
	}
	vec := []float64{0.11623105759942885, 0.030073150814502212, 0.0119388989548912, 0.01836268922781065, 0.010306563749646199, 0.0036432311839576883, 0.005695810419718821, 0.014593393364127294, 0.009913022568277332, 0.002835013679521494}
	for i, v := range vvs[0].Vec[:10] {
		prob := real(v)*real(v) + imag(v)*imag(v)
		if math.Abs(prob-vec[i]) > 1e-6 {
			t.Fatalf("%d %v %f %f", i, v, prob, vec[i])
		}
	}
	vec = []float64{0.009913022568277134, 0.014593393364126966, 0.005695810419718817, 0.003643231183957665, 0.010306563749646001, 0.018362689227810196, 0.01193889895489093, 0.030073150814501577, 0.11623105759942208}
	for i, v := range vvs[0].Vec[len(vvs[0].Vec)-9:] {
		prob := real(v)*real(v) + imag(v)*imag(v)
		if math.Abs(prob-vec[i]) > 1e-6 {
			t.Fatalf("%d %v %f %f", i, v, prob, vec[i])
		}
	}
}

func TestEigs(t *testing.T) {
	t.Parallel()
	type vectorSlice struct {
		from int
		to   int
		vec  []complex128
	}
	type basis struct {
		id  int
		val complex128
		vec []vectorSlice
	}
	type testcase struct {
		n [2]int
		h complex64

		basis []basis
	}
	tests := []testcase{
		{
			n: [2]int{20, 1},
			h: 1,
			basis: []basis{
				{
					id:  0,
					val: complex(-25.1078, 0),
					vec: []vectorSlice{
						{
							from: 0,
							to:   10,
							vec: []complex128{
								complex(-0.1491614167055709, 0),
								complex(-0.07480019098307723, 0),
								complex(-0.046806700861319696, 0),
								complex(-0.056487650378552155, 0),
								complex(-0.043310375894986604, 0),
								complex(-0.02501935464440957, 0),
								complex(-0.03071772369181276, 0),
								complex(-0.047588699647192347, 0),
								complex(-0.04222143781233176, 0),
								complex(-0.02203858865080983, 0),
							},
						},
						{
							from: 1048576 - 9,
							to:   1048576,
							vec: []complex128{
								complex(-0.04222144411793357, 0),
								complex(-0.047588693656718646, 0),
								complex(-0.030717723337439166, 0),
								complex(-0.025019355620736743, 0),
								complex(-0.04331038177954525, 0),
								complex(-0.05648765162278255, 0),
								complex(-0.0468067067983888, 0),
								complex(-0.0748002016414607, 0),
								complex(-0.14916144668864517, 0),
							},
						},
					},
				},
			},
		},
	}

	n := [2]int{8, 1}
	var h complex64 = 1
	h8, buf := mat.M([][]complex64{{0}}), mat.M([][]complex64{{0}})
	TransverseFieldIsing(h8, buf, n, h)
	vvs := h8.COO().Eigen()
	tc := testcase{n: n, h: h}
	for i := range 3 {
		vv := vvs[i]
		vslc := vectorSlice{from: 0, to: len(vv.Vec), vec: vv.Vec}
		tc.basis = append(tc.basis, basis{id: i, val: vv.Val, vec: []vectorSlice{vslc}})
	}
	tests = append(tests, tc)

	for _, test := range tests {
		t.Run(fmt.Sprintf("%v %v", test.n, test.h), func(t *testing.T) {
			t.Parallel()
			dir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("%+v", err)
			}
			defer os.RemoveAll(dir)

			TransverseFieldIsingExplicit(dir, test.n, test.h)

			vvs := mat.EigsDir(dir)

			for i, basis := range test.basis {
				vv := vvs[basis.id]

				if cmplx.Abs(vv.Val-basis.val)/cmplx.Abs(basis.val) > 1e-3 {
					t.Fatalf("%f %f", vv.Val, basis.val)
				}
				for j, vslc := range basis.vec {
					for k, v := range vv.Vec[vslc.from:vslc.to] {
						vProb := math.Pow(cmplx.Abs(v), 2)
						tProb := math.Pow(cmplx.Abs(vslc.vec[k]), 2)
						if math.Abs(vProb-tProb) > 1e-2 {
							t.Fatalf("%d %d %d %f %f", i, j, k, vProb, tProb)
						}
					}
				}
			}
		})
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}
