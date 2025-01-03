package tensor

import (
	"flag"
	"fmt"
	"log"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/pkg/errors"
)

func TestTn(t *testing.T) {
	t.Parallel()
	tests := []struct {
		slice2 [][]complex64
		slice3 [][][]complex64
		slice4 [][][][]complex64
	}{
		{
			slice4: [][][][]complex64{
				{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}},
				{{{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}, {{21, 22}, {23, 24}}},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			var ts *Dense
			switch {
			case test.slice2 != nil:
				ts = T2(test.slice2)
				if err := equal2(ts, test.slice2, 0); err != nil {
					t.Fatalf("%+v", err)
				}
			case test.slice3 != nil:
				ts = T3(test.slice3)
				if err := equal3(ts, test.slice3, 0); err != nil {
					t.Fatalf("%+v", err)
				}
			default:
				ts = T4(test.slice4)
				if err := equal4(ts, test.slice4, 0); err != nil {
					t.Fatalf("%+v", err)
				}
			}
		})
	}
}

func TestSetAt(t *testing.T) {
	t.Parallel()
	type testcase struct {
		aFull      *Dense
		a          *Dense
		digits     []int
		ca         complex64
		aAfter     *Dense
		aFullAfter *Dense
	}
	tests := []testcase{}

	var tc testcase
	tc.aFull = T2([][]complex64{{99, 99, 99, 99}, {99, 1i, 2i, 99}, {99, 3i, 4i, 99}, {99, 99, 99, 99}})
	tc.a = tc.aFull.Conj().Transpose(1, 0).Slice([][2]int{{1, 3}, {1, 3}})
	tc.digits = []int{0, 1}
	tc.ca = 5 + 5i
	tc.aAfter = T2([][]complex64{{-1i, 5 + 5i}, {-2i, -4i}})
	tc.aFullAfter = T2([][]complex64{{99, 99, 99, 99}, {99, -1i, -2i, 99}, {99, 5 + 5i, -4i, 99}, {99, 99, 99, 99}})

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			test.a.SetAt(test.digits, test.ca)
			if err := equal2(test.a, test.aAfter.ToSlice2(), 1e-6); err != nil {
				t.Fatalf("%+v", err)
			}
			if err := equal2(test.aFull, test.aFullAfter.ToSlice2(), 1e-6); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestSet(t *testing.T) {
	t.Parallel()
	tests := []struct {
		b     *Dense
		at    []int
		a     *Dense
		bSetA *Dense
	}{
		{
			b: T2([][]complex64{
				{0, 0, 0, 0, 0, 0},
				{0, 1i, 2, 3, 4, 0},
				{0, 5i, 6, 7, 8, 0},
				{0, 9, 10, 11, 12, 0},
				{0, 13, 14, 15, 16, 0},
				{0, 0, 0, 0, 0, 0},
			}).Conj().Slice([][2]int{{1, 5}, {1, 5}}).Transpose(1, 0),
			at: []int{1, 1},
			a: T2([][]complex64{
				{100, 200, 300, 400},
				{500, 600, 700i, 800},
				{900, 1000, 1100, 1200},
				{1300, 1400, 1500, 1600},
			}).Conj().Transpose(1, 0).Slice([][2]int{{1, 3}, {1, 3}}),
			bSetA: T2([][]complex64{
				{-1i, -5i, 9, 13},
				{2, 600, 1000, 14},
				{3, -700i, 1100, 15},
				{4, 8, 12, 16},
			}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			test.b.Set(test.at, test.a)
			if err := test.b.Equal(test.bSetA, 0); err != nil {
				t.Fatalf("%#v %#v", test.b.ToSlice2(), test.bSetA.ToSlice2())
			}
		})
	}
}

func TestAll(t *testing.T) {
	t.Parallel()
	type testcase struct {
		a *Dense
		b []complex64
	}
	tests := []testcase{}

	var tc testcase
	tc.a = Zeros(5, 5, 5, 5)
	for i := range tc.a.Shape()[0] {
		for j := range tc.a.Shape()[1] {
			for k := range tc.a.Shape()[2] {
				for l := range tc.a.Shape()[3] {
					re := float32(i + 2*j + 3*k + 4*l)
					im := float32(i - j - k + l)
					tc.a.SetAt([]int{i, j, k, l}, complex(re, im))
				}
			}
		}
	}
	tc.a = tc.a.Transpose(2, 1, 3, 0)
	tc.a = tc.a.Conj()
	tc.a = tc.a.Slice([][2]int{{1, 4}, {1, 3}, {2, 4}, {3, 5}})
	tc.b = []complex64{16 - 3i, 17 - 4i, 20 - 4i, 21 - 5i, 18 - 2i, 19 - 3i, 22 - 3i, 23 - 4i, 19 - 2i, 20 - 3i, 23 - 3i, 24 - 4i, 21 - 1i, 22 - 2i, 25 - 2i, 26 - 3i, 22 - 1i, 23 - 2i, 26 - 2i, 27 - 3i, 24, 25 - 1i, 28 - 1i, 29 - 2i}
	tests = append(tests, tc)

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			var j int = -1
			for _, v := range test.a.All() {
				j++
				if v != test.b[j] {
					t.Fatalf("%d %v %v", j, v, test.b[j])
				}
			}
		})
	}
}

func TestSlice(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a          *Dense
		transpose0 []int
		slice0     [][2]int
		transpose1 []int
		slice1     [][2]int
		b          *Dense
	}{
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				{{26, 27, 28, 29, 30}, {31, 32, 33, 34, 35}, {36, 37, 38, 39, 40}, {41, 42, 43, 44, 45}, {46, 47, 48, 49, 50}},
				{{51, 52, 53, 54, 55}, {56, 57, 58, 59, 60}, {61, 62, 63, 64, 65}, {66, 67, 68, 69, 70}, {71, 72, 73, 74, 75}},
				{{76, 77, 78, 79, 80}, {81, 82, 83, 84, 85}, {86, 87, 88, 89, 90}, {91, 92, 93, 94, 95}, {96, 97, 98, 99, 100}},
				{{101, 102, 103, 104, 105}, {106, 107, 108, 109, 110}, {111, 112, 113, 114, 115}, {116, 117, 118, 119, 120}, {121, 122, 123, 124, 125}},
			}),
			transpose0: []int{0, 1, 2},
			slice0:     [][2]int{{1, 4}, {1, 4}, {1, 4}},
			transpose1: []int{0, 1, 2},
			slice1:     [][2]int{{0, 3}, {0, 3}, {0, 3}},
			b: T3([][][]complex64{
				{{32, 33, 34}, {37, 38, 39}, {42, 43, 44}},
				{{57, 58, 59}, {62, 63, 64}, {67, 68, 69}},
				{{82, 83, 84}, {87, 88, 89}, {92, 93, 94}},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				{{26, 27, 28, 29, 30}, {31, 32, 33, 34, 35}, {36, 37, 38, 39, 40}, {41, 42, 43, 44, 45}, {46, 47, 48, 49, 50}},
				{{51, 52, 53, 54, 55}, {56, 57, 58, 59, 60}, {61, 62, 63, 64, 65}, {66, 67, 68, 69, 70}, {71, 72, 73, 74, 75}},
				{{76, 77, 78, 79, 80}, {81, 82, 83, 84, 85}, {86, 87, 88, 89, 90}, {91, 92, 93, 94, 95}, {96, 97, 98, 99, 100}},
				{{101, 102, 103, 104, 105}, {106, 107, 108, 109, 110}, {111, 112, 113, 114, 115}, {116, 117, 118, 119, 120}, {121, 122, 123, 124, 125}},
			}),
			transpose0: []int{0, 1, 2},
			slice0:     [][2]int{{1, 4}, {1, 4}, {1, 4}},
			transpose1: []int{2, 0, 1},
			slice1:     [][2]int{{0, 3}, {0, 3}, {0, 3}},
			b: T3([][][]complex64{
				{{32, 37, 42}, {57, 62, 67}, {82, 87, 92}},
				{{33, 38, 43}, {58, 63, 68}, {83, 88, 93}},
				{{34, 39, 44}, {59, 64, 69}, {84, 89, 94}},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				{{26, 27, 28, 29, 30}, {31, 32, 33, 34, 35}, {36, 37, 38, 39, 40}, {41, 42, 43, 44, 45}, {46, 47, 48, 49, 50}},
				{{51, 52, 53, 54, 55}, {56, 57, 58, 59, 60}, {61, 62, 63, 64, 65}, {66, 67, 68, 69, 70}, {71, 72, 73, 74, 75}},
				{{76, 77, 78, 79, 80}, {81, 82, 83, 84, 85}, {86, 87, 88, 89, 90}, {91, 92, 93, 94, 95}, {96, 97, 98, 99, 100}},
				{{101, 102, 103, 104, 105}, {106, 107, 108, 109, 110}, {111, 112, 113, 114, 115}, {116, 117, 118, 119, 120}, {121, 122, 123, 124, 125}},
			}),
			transpose0: []int{2, 0, 1},
			slice0:     [][2]int{{1, 4}, {1, 4}, {1, 4}},
			transpose1: []int{0, 1, 2},
			slice1:     [][2]int{{0, 3}, {0, 3}, {0, 3}},
			b: T3([][][]complex64{
				{{32, 37, 42}, {57, 62, 67}, {82, 87, 92}},
				{{33, 38, 43}, {58, 63, 68}, {83, 88, 93}},
				{{34, 39, 44}, {59, 64, 69}, {84, 89, 94}},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				{{26, 27, 28, 29, 30}, {31, 32, 33, 34, 35}, {36, 37, 38, 39, 40}, {41, 42, 43, 44, 45}, {46, 47, 48, 49, 50}},
				{{51, 52, 53, 54, 55}, {56, 57, 58, 59, 60}, {61, 62, 63, 64, 65}, {66, 67, 68, 69, 70}, {71, 72, 73, 74, 75}},
				{{76, 77, 78, 79, 80}, {81, 82, 83, 84, 85}, {86, 87, 88, 89, 90}, {91, 92, 93, 94, 95}, {96, 97, 98, 99, 100}},
				{{101, 102, 103, 104, 105}, {106, 107, 108, 109, 110}, {111, 112, 113, 114, 115}, {116, 117, 118, 119, 120}, {121, 122, 123, 124, 125}},
			}),
			transpose0: []int{2, 0, 1},
			slice0:     [][2]int{{1, 4}, {1, 4}, {1, 4}},
			transpose1: []int{1, 2, 0},
			slice1:     [][2]int{{0, 3}, {0, 3}, {0, 3}},
			b: T3([][][]complex64{
				{{32, 33, 34}, {37, 38, 39}, {42, 43, 44}},
				{{57, 58, 59}, {62, 63, 64}, {67, 68, 69}},
				{{82, 83, 84}, {87, 88, 89}, {92, 93, 94}},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}},
				{{26, 27, 28, 29, 30}, {31, 32, 33, 34, 35}, {36, 37, 38, 39, 40}, {41, 42, 43, 44, 45}, {46, 47, 48, 49, 50}},
				{{51, 52, 53, 54, 55}, {56, 57, 58, 59, 60}, {61, 62, 63, 64, 65}, {66, 67, 68, 69, 70}, {71, 72, 73, 74, 75}},
				{{76, 77, 78, 79, 80}, {81, 82, 83, 84, 85}, {86, 87, 88, 89, 90}, {91, 92, 93, 94, 95}, {96, 97, 98, 99, 100}},
				{{101, 102, 103, 104, 105}, {106, 107, 108, 109, 110}, {111, 112, 113, 114, 115}, {116, 117, 118, 119, 120}, {121, 122, 123, 124, 125}},
			}),
			transpose0: []int{2, 0, 1},
			slice0:     [][2]int{{1, 4}, {1, 4}, {1, 4}},
			transpose1: []int{1, 0, 2},
			slice1:     [][2]int{{1, 3}, {0, 2}, {0, 3}},
			b: T3([][][]complex64{
				{{57, 62, 67}, {58, 63, 68}},
				{{82, 87, 92}, {83, 88, 93}},
			}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()

			b := test.a
			b = b.Transpose(test.transpose0...)
			b = b.Slice(test.slice0)
			b = b.Transpose(test.transpose1...)
			b = b.Slice(test.slice1)

			if err := b.Equal(test.b, 0); err != nil {
				t.Fatalf("%#v", err)
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	t.Parallel()
	type digitsV struct {
		digits []int
		v      complex64
	}
	type testcase struct {
		a               *Dense
		axes            []int
		transposedShape []int
		dvs             []digitsV
	}
	tests := []testcase{
		{
			a:               trange(0, 1680, 1).Reshape(5, 6, 7, 8),
			axes:            []int{2, 0, 3, 1},
			transposedShape: []int{7, 5, 8, 6},
			dvs: []digitsV{
				{digits: []int{6, 4, 7, 5}, v: 1679},
				{digits: []int{6, 4, 6, 5}, v: 1678},
				{digits: []int{0, 0, 1, 0}, v: 1},
				{digits: []int{1, 0, 0, 0}, v: 8},
				{digits: []int{4, 2, 3, 1}, v: 763},
			},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			aT := test.a.Transpose(test.axes...)
			if !slices.Equal(aT.Shape(), test.transposedShape) {
				t.Fatalf("%#v %#v", aT.Shape(), test.transposedShape)
			}
			for _, dv := range test.dvs {
				if v := aT.At(dv.digits...); v != dv.v {
					t.Fatalf("%#v %v %v", dv.digits, v, dv.v)
				}
			}
		})
	}
}

func reshapeWithError(a *Dense, newShape []int) (b *Dense, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.Errorf("%#v", r)
		}
	}()

	b = a.Reshape(newShape...)
	return
}

func TestReshape(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a         *Dense
		shape     []int
		b         *Dense
		shouldErr bool
	}{
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			shape: []int{4, 3, 2, 2},
			b: T4([][][][]complex64{
				{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}},
				{{{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}, {{21, 22}, {23, 24}}},
				{{{25, 26}, {27, 28}}, {{29, 30}, {31, 32}}, {{33, 34}, {35, 36}}},
				{{{37, 38}, {39, 40}}, {{41, 42}, {43, 44}}, {{45, 46}, {47, 48}}},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			shape: []int{4, -1, 2, 2},
			b: T4([][][][]complex64{
				{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}},
				{{{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}, {{21, 22}, {23, 24}}},
				{{{25, 26}, {27, 28}}, {{29, 30}, {31, 32}}, {{33, 34}, {35, 36}}},
				{{{37, 38}, {39, 40}}, {{41, 42}, {43, 44}}, {{45, 46}, {47, 48}}},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			shape: []int{4, 12},
			b: T2([][]complex64{
				{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
				{25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
				{37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
			}),
		},
		{
			a:     T2([][]complex64{{-2, -1}, {1, 2}, {3, 4}, {5, 6}}).Transpose(1, 0).Slice([][2]int{{0, 2}, {1, 3}}).Transpose(1, 0),
			shape: []int{4},
			b:     T1([]complex64{1, 2, 3, 4}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			// shape should be 4, 3, 2, 2.
			shape:     []int{4, 2, 2, 2},
			shouldErr: true,
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			// shape cannot have multiple -1s.
			shape:     []int{4, -1, -1, 2},
			shouldErr: true,
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			// 5 does not wholly divide 4*3*2*2.
			shape:     []int{5, -1, 1, 1},
			shouldErr: true,
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			b, err := reshapeWithError(test.a, test.shape)
			switch test.shouldErr {
			case true:
				if err == nil {
					t.Fatalf("should error")
				}
			default:
				if err != nil {
					t.Fatalf("%+v", err)
				}
				if err := b.Equal(test.b, 0); err != nil {
					t.Fatalf("%#v %#v", b, test.b)
				}
			}
		})
	}
}

func TestMul(t *testing.T) {
	t.Parallel()
	type testcase struct {
		fullX     *Dense
		x         *Dense
		c         complex64
		cMulX     *Dense
		fullCMulX *Dense
	}
	tests := []testcase{}

	var tc testcase
	tc.fullX = T2([][]complex64{{99, 99}, {1, 2i}, {3, 4}, {99, 99}})
	tc.x = tc.fullX.Conj().Transpose(1, 0).Slice([][2]int{{0, 2}, {1, 3}})
	tc.c = 2i
	tc.cMulX = T2([][]complex64{{2i, 6i}, {4, 8i}})
	tc.fullCMulX = T2([][]complex64{{99, 99}, {-2i, 4}, {-6i, -8i}, {99, 99}})
	tests = append(tests, tc)

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			test.x.Mul(test.c)
			if err := test.x.Equal(test.cMulX, 0); err != nil {
				t.Fatalf("%#v", test.x.ToSlice2())
			}
			if err := test.fullX.Equal(test.fullCMulX, 0); err != nil {
				t.Fatalf("%#v", test.fullX.ToSlice2())
			}
		})
	}
}

func addWithError(a *Dense, c complex64, b *Dense) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.Errorf("%#v", r)
		}
	}()

	a.Add(c, b)
	return
}

func TestAdd(t *testing.T) {
	t.Parallel()
	type testcase struct {
		a      *Dense
		c      complex64
		b      *Dense
		aPlusB *Dense
	}
	tests := []testcase{
		{
			a:      T2([][]complex64{{1i, 1, 4}, {1i, 2, 5}, {1i, 3, -6i}, {1i, 1i, 1i}}).Conj().Slice([][2]int{{0, 3}, {1, 3}}).Transpose(1, 0),
			c:      1,
			b:      T2([][]complex64{{-7i, 8, 9}, {10, 11, -12i}}).Conj(),
			aPlusB: T2([][]complex64{{1 + 7i, 10, 12}, {14, 16, 18i}}),
		},
		{
			a:      T2([][]complex64{{1, 2}, {3, -4i}}),
			c:      2 - 3i,
			b:      T2([][]complex64{{5, 6}, {7, 8}}),
			aPlusB: T2([][]complex64{{11 - 15i, 14 - 18i}, {17 - 21i, 16 - 28i}}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			err := addWithError(test.a, test.c, test.b)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			if err := test.a.Equal(test.aPlusB, 0); err != nil {
				t.Fatalf("%#v %#v", test.c, test.aPlusB)
			}
		})
	}
}

func TestAddSlice(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a          *Dense
		c          complex64
		b          *Dense
		ax         [][2]int
		aFullAdded [][]complex64
	}{
		{
			a:          T2([][]complex64{{99, 99, 99, 99}, {99, 1, -2i, 99}, {99, 3, 4, 99}, {99, 99, 99, 99}}),
			c:          2 + 3i,
			b:          T2([][]complex64{{1i, 2i}, {3i, 4i}}),
			ax:         [][2]int{{1, 3}, {1, 3}},
			aFullAdded: [][]complex64{{99, 99, 99, 99}, {99, -2 + 2i, -6 + 2i, 99}, {99, -6 + 6i, -8 + 8i, 99}, {99, 99, 99, 99}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			test.a.Slice(test.ax).Add(test.c, test.b)
			if err := equal2(test.a, test.aFullAdded, 0); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func TestContract(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a    *Dense
		b    *Dense
		axes [][2]int
		c    *Dense
	}{
		{
			a: T2([][]complex64{
				{1i, 2, 3i},
				{4, 5i, 6},
			}),
			b: T2([][]complex64{
				{7, 8i, 9},
				{10i, 11, 12i},
			}).Conj().Transpose(1, 0),
			axes: [][2]int{{0, 1}, {1, 0}},
			c:    Scalar(-39i),
		},
		{
			a: T3([][][]complex64{
				{{0, 1, 2, 3, 4},
					{5, 6, 7, 8, 9},
					{10, 11, 12, 13, 14},
					{15, 16, 17, 18, 19}},
				{{20, 21, 22, 23, 24},
					{25, 26, 27, 28, 29},
					{30, 31, 32, 33, 34},
					{35, 36, 37, 38, 39}},
				{{40, 41, 42, 43, 44},
					{45, 46, 47, 48, 49},
					{50, 51, 52, 53, 54},
					{55, 56, 57, 58, 59}}}),
			b: T3([][][]complex64{
				{{0, 1}, {2, 3}, {4, 5}},
				{{6, 7}, {8, 9}, {10, 11}},
				{{12, 13}, {14, 15}, {16, 17}},
				{{18, 19}, {20, 21}, {22, 23}}}),
			axes: [][2]int{{1, 0}},
			c: T4([][][][]complex64{
				{
					{{420, 450}, {480, 510}, {540, 570}},
					{{456, 490}, {524, 558}, {592, 626}},
					{{492, 530}, {568, 606}, {644, 682}},
					{{528, 570}, {612, 654}, {696, 738}},
					{{564, 610}, {656, 702}, {748, 794}}},
				{
					{{1140, 1250}, {1360, 1470}, {1580, 1690}},
					{{1176, 1290}, {1404, 1518}, {1632, 1746}},
					{{1212, 1330}, {1448, 1566}, {1684, 1802}},
					{{1248, 1370}, {1492, 1614}, {1736, 1858}},
					{{1284, 1410}, {1536, 1662}, {1788, 1914}}},
				{
					{{1860, 2050}, {2240, 2430}, {2620, 2810}},
					{{1896, 2090}, {2284, 2478}, {2672, 2866}},
					{{1932, 2130}, {2328, 2526}, {2724, 2922}},
					{{1968, 2170}, {2372, 2574}, {2776, 2978}},
					{{2004, 2210}, {2416, 2622}, {2828, 3034}}}}),
		},
		{
			a:    trange(1, 1+5*5*5, 1).Mul(-1i).Conj().Reshape(5, 5, 5).Slice([][2]int{{1, 4}, {2, 4}, {1, 3}}).Transpose(1, 2, 0),
			b:    trange(1, 1+4*2*3, 1).Reshape(4, 2, 3).Transpose(2, 0, 1).Slice([][2]int{{0, 2}, {1, 4}, {0, 2}}),
			axes: [][2]int{{0, 2}, {2, 1}},
			c:    T2([][]complex64{{6234i, 6621i}, {6321i, 6714i}}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			c := Zeros(1)
			Contract(c, test.a, test.b, test.axes)
			if err := c.Equal(test.c, 0); err != nil {
				t.Fatalf("%#v %#v", c, test.c)
			}
		})
	}
}

func TestMatMul(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a *Dense
		b *Dense
		c *Dense
	}{
		{
			a: T2([][]complex64{{1 + 2i, 2 + 3i}, {3, -4}}),
			b: T2([][]complex64{{5 - 1i, 6 + 2i}, {-7i, 8}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 2 + 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 - 1i, 6 + 2i}, {-7i, 8}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 2 + 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 + 1i, 6 - 2i}, {7i, 8}}).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 2 - 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Conj().Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 + 1i, 6 - 2i}, {7i, 8}}).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 2 - 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Conj().Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 - 1i, 6 + 2i}, {-7i, 8}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 3, 99}, {99, 2 + 3i, -4, 99}, {99, 99, 99, 99}}).Transpose(1, 0).Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 - 1i, 6 + 2i}, {-7i, 8}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 3, 99}, {99, 2 - 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).H(),
			b: T2([][]complex64{{5 - 1i, 6 + 2i}, {-7i, 8}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 3, 99}, {99, 2 - 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).H(),
			b: T2([][]complex64{{5 + 1i, 6 - 2i}, {7i, 8}}).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 3, 99}, {99, 2 + 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).Transpose(1, 0),
			b: T2([][]complex64{{5 + 1i, 6 - 2i}, {7i, 8}}).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 2 - 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Conj().Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 + 1i, 7i}, {6 - 2i, 8}}).Transpose(1, 0).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 2 - 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Conj().Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 - 1i, -7i}, {6 + 2i, 8}}).Transpose(1, 0),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 2 + 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 + 1i, 7i}, {6 - 2i, 8}}).Transpose(1, 0).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 2 + 3i, 99}, {99, 3, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}),
			b: T2([][]complex64{{5 - 1i, -7i}, {6 + 2i, 8}}).Transpose(1, 0),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 3, 99}, {99, 2 - 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).H(),
			b: T2([][]complex64{{99, 99, 99, 99}, {99, 5 - 1i, 6 + 2i, 99}, {99, -7i, 8, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 3, 99}, {99, 2 - 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).H(),
			b: T2([][]complex64{{99, 99, 99, 99}, {99, 5 - 1i, -7i, 99}, {99, 6 + 2i, 8, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).Transpose(1, 0),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 - 2i, 3, 99}, {99, 2 - 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).H(),
			b: T2([][]complex64{{99, 99, 99, 99}, {99, 5 + 1i, 7i, 99}, {99, 6 - 2i, 8, 99}, {99, 99, 99, 99}}).H().Slice([][2]int{{1, 3}, {1, 3}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 3, 99}, {99, 2 + 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).Transpose(1, 0),
			b: T2([][]complex64{{99, 99, 99, 99}, {99, 5 + 1i, 7i, 99}, {99, 6 - 2i, 8, 99}, {99, 99, 99, 99}}).Transpose(1, 0).Slice([][2]int{{1, 3}, {1, 3}}).Conj(),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
		{
			a: T2([][]complex64{{99, 99, 99, 99}, {99, 1 + 2i, 3, 99}, {99, 2 + 3i, -4, 99}, {99, 99, 99, 99}}).Slice([][2]int{{1, 3}, {1, 3}}).Transpose(1, 0),
			b: T2([][]complex64{{99, 99, 99, 99}, {99, 5 - 1i, -7i, 99}, {99, 6 + 2i, 8, 99}, {99, 99, 99, 99}}).Transpose(1, 0).Slice([][2]int{{1, 3}, {1, 3}}),
			c: T2([][]complex64{{28 - 5i, 18 + 38i}, {15 + 25i, -14 + 6i}}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			c := Zeros(1)
			MatMul(c, test.a, test.b)
			if err := c.Equal(test.c, 0); err != nil {
				t.Fatalf("%#v %#v", c.ToSlice2(), test.c.ToSlice2())
			}
		})
	}
}

func TestFrobeniusNorm(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a    *Dense
		norm float32
	}{
		{
			a:    T2([][]complex64{{999, 1 + 1i, 2, 999}, {999, 3 - 2i, 4, 999}}).Conj().Slice([][2]int{{0, 2}, {1, 3}}),
			norm: 5.916079783099616,
		},
		{
			a:    T1([]complex64{3e-33, 4e-33}),
			norm: 5e-33,
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			if norm := test.a.FrobeniusNorm(); norm != test.norm {
				t.Fatalf("%f %f", norm, test.norm)
			}
		})
	}
}

func TestLapy(t *testing.T) {
	t.Parallel()
	tests := []struct {
		xs []complex64
		y  float32
	}{
		{
			xs: []complex64{3e-33, 4e-33},
			y:  5e-33,
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			if y := lapy(test.xs...); y != test.y {
				t.Fatalf("%v %v", y, test.y)
			}
		})
	}
}

func equal1(a *Dense, b []complex64, tol float32) error {
	shape := []int{len(b)}
	if len(a.Shape()) != len(shape) {
		return errors.Errorf("different shapes %#v %#v", a.Shape(), shape)
	}
	for i := range a.Shape() {
		if a.Shape()[i] != shape[i] {
			return errors.Errorf("different shapes %#v %#v", a.Shape(), shape)
		}
	}
	for i := range b {
		if cmplx.IsNaN(complex128(a.At(i))) {
			return errors.Errorf("NaN %d", i)
		}
		if diff := abs(a.At(i) - b[i]); diff > tol*max(1, abs(b[i])) {
			return errors.Errorf("i %d diff %f a %v b %v", i, diff, a.At(i), b[i])
		}
	}
	return nil
}

func equal2(a *Dense, b [][]complex64, tol float32) error {
	shape := []int{len(b), len(b[0])}
	if len(a.Shape()) != len(shape) {
		return errors.Errorf("different shapes %#v %#v", a.Shape(), shape)
	}
	for i := range a.Shape() {
		if a.Shape()[i] != shape[i] {
			return errors.Errorf("different shapes %#v %#v", a.Shape(), shape)
		}
	}
	for i := range b {
		for j := range b[0] {
			if cmplx.IsNaN(complex128(a.At(i, j))) {
				return errors.Errorf("NaN %d %d", i, j)
			}
			target := b[i][j]
			if diff := abs(a.At(i, j) - target); diff > tol*max(1, abs(target)) {
				return errors.Errorf("i %d j %d diff %f a %v b %v", i, j, diff, a.At(i, j), b[i][j])
			}
		}
	}
	return nil
}

func equal3(a *Dense, b [][][]complex64, tol float32) error {
	shape := []int{len(b), len(b[0]), len(b[0][0])}
	if len(a.Shape()) != len(shape) {
		return errors.Errorf("%#v", a.Shape())
	}
	for i := range a.Shape() {
		if a.Shape()[i] != shape[i] {
			return errors.Errorf("different shapes %#v %#v", a.Shape(), shape)
		}
	}
	for i := range b {
		for j := range b[0] {
			for k := range b[0][0] {
				if cmplx.IsNaN(complex128(a.At(i, j, k))) {
					return errors.Errorf("NaN %d %d %d", i, j, k)
				}
				target := b[i][j][k]
				if diff := abs(a.At(i, j, k) - target); diff > tol*max(1, abs(target)) {
					return errors.Errorf("%d %d %d %f", i, j, k, diff)
				}
			}
		}
	}
	return nil
}

func equal4(a *Dense, b [][][][]complex64, tol float32) error {
	shape := []int{len(b), len(b[0]), len(b[0][0]), len(b[0][0][0])}
	if len(a.Shape()) != len(shape) {
		return errors.Errorf("%#v", a.Shape())
	}
	for i := range a.Shape() {
		if a.Shape()[i] != shape[i] {
			return errors.Errorf("different shapes %#v %#v", a.Shape(), shape)
		}
	}
	for i := range b {
		for j := range b[0] {
			for k := range b[0][0] {
				for l := range b[0][0][0] {
					if cmplx.IsNaN(complex128(a.At(i, j, k, l))) {
						return errors.Errorf("NaN %d %d %d %d", i, j, k, l)
					}
					target := b[i][j][k][l]
					if diff := abs(a.At(i, j, k, l) - target); diff > tol*max(1, abs(target)) {
						return errors.Errorf("%d %d %d %d %f", i, j, k, l, diff)
					}
				}
			}
		}
	}
	return nil
}

func trange(start, end, diff int) *Dense {
	slice := make([]complex64, 0, end-start)
	for i := start; i < end; i += diff {
		slice = append(slice, complex(float32(i), 0))
	}
	return T1(slice)
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}
