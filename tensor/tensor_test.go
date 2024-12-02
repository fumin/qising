package tensor

import (
	"flag"
	"fmt"
	"log"
	"math/cmplx"
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
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.slice2), func(t *testing.T) {
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

func TestEye(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n int
		k int
		b *Dense
	}{
		{n: 4, k: 0, b: T2([][]complex64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})},
		{n: 4, k: 2, b: T2([][]complex64{{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}})},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			b := Zeros(1).Eye(test.n, test.k)
			if !b.Equal(test.b) {
				t.Fatalf("%#v %#v", b, test.b)
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
			if !test.b.Equal(test.bSetA) {
				t.Fatalf("%#v %#v", test.b.ToSlice2(), test.bSetA.ToSlice2())
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

			if !b.Equal(test.b) {
				t.Fatalf("not equal")
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
			shape: []int{4, 12},
			b: T2([][]complex64{
				{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
				{25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
				{37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
			}),
		},
		{
			a: T3([][][]complex64{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
				{{25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}},
				{{37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
			}),
			shape:     []int{4, 2, 2, 2},
			shouldErr: true,
		},
		{
			a:     T2([][]complex64{{-2, -1}, {1, 2}, {3, 4}, {5, 6}}).Transpose(1, 0).Slice([][2]int{{0, 2}, {1, 3}}).Transpose(1, 0),
			shape: []int{4},
			b:     T1([]complex64{1, 2, 3, 4}),
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
				if !b.Equal(test.b) {
					t.Fatalf("%#v %#v", b, test.b)
				}
			}
		})
	}
}

func TestMul(t *testing.T) {
	t.Parallel()
	type testcase struct {
		ca     complex64
		a      *Dense
		b      *Dense
		caMulA *Dense
	}
	tests := []testcase{
		{
			ca:     2i,
			a:      T2([][]complex64{{99, 99}, {1, 2i}, {3, 4}, {99, 99}}).Conj().Transpose(1, 0).Slice([][2]int{{0, 2}, {1, 3}}),
			b:      Zeros(2, 2),
			caMulA: T2([][]complex64{{2i, 6i}, {4, 8i}}),
		},
	}

	var tc testcase
	tc.ca = 2i
	tc.a = T2([][]complex64{{99, 99}, {1, 2i}, {3, 4}, {99, 99}}).Conj().Transpose(1, 0).Slice([][2]int{{0, 2}, {1, 3}})
	tc.b = tc.a
	tc.caMulA = T2([][]complex64{{2i, 6i}, {4, 8i}})
	tests = append(tests, tc)

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			Mul(test.b, test.ca, test.a)
			if !test.b.Equal(test.caMulA) {
				t.Fatalf("%#v", test.b.ToSlice2())
			}
		})
	}
}

func addWithError(c, a, b *Dense) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.Errorf("%#v", r)
		}
	}()

	Add(c, a, b)
	return
}

func TestAdd(t *testing.T) {
	t.Parallel()
	type testcase struct {
		a         *Dense
		b         *Dense
		c         *Dense
		shouldErr bool
		aPlusB    *Dense
	}
	tests := []testcase{
		{
			a:      T2([][]complex64{{1i, 1, 4}, {1i, 2, 5}, {1i, 3, -6i}, {1i, 1i, 1i}}).Conj().Slice([][2]int{{0, 3}, {1, 3}}).Transpose(1, 0),
			b:      T2([][]complex64{{-7i, 8, 9}, {10, 11, -12i}}).Conj(),
			c:      Zeros(2, 3),
			aPlusB: T2([][]complex64{{1 + 7i, 10, 12}, {14, 16, 18i}}),
		},
	}

	var tc testcase
	tc.a = T2([][]complex64{{1i, 2}, {3, 4}}).Conj()
	tc.b = T2([][]complex64{{5i, 6}, {7, 8}})
	tc.c = tc.a
	tc.aPlusB = T2([][]complex64{{4i, 8}, {10, 12}})
	tests = append(tests, tc)

	tc = tc
	tc.a = T2([][]complex64{{1, 2}, {3, 4}})
	tc.b = T2([][]complex64{{5, 6}, {7, 8}})
	tc.c = tc.a.Transpose(1, 0)
	tc.shouldErr = true
	tests = append(tests, tc)

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			err := addWithError(test.c, test.a, test.b)
			switch test.shouldErr {
			case true:
				if err == nil {
					t.Fatalf("should error")
				}
			default:
				if err != nil {
					t.Fatalf("%+v", err)
				}
				if !test.c.Equal(test.aPlusB) {
					t.Fatalf("%#v %#v", test.c, test.aPlusB)
				}
			}
		})
	}
}

func TestAddSlice(t *testing.T) {
	t.Parallel()
	tests := []struct {
		a          *Dense
		b          *Dense
		cFull      *Dense
		ax         [][2]int
		cFullAdded [][]complex64
	}{
		{
			a:          T2([][]complex64{{1, 2}, {3, 4}}),
			b:          T2([][]complex64{{1i, 2i}, {3i, 4i}}),
			cFull:      T2([][]complex64{{99, 99, 99, 99}, {99, -1, -1, 99}, {99, -1, -1, 99}, {99, 99, 99, 99}}),
			ax:         [][2]int{{1, 3}, {1, 3}},
			cFullAdded: [][]complex64{{99, 99, 99, 99}, {99, 1 + 1i, 2 + 2i, 99}, {99, 3 + 3i, 4 + 4i, 99}, {99, 99, 99, 99}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			Add(test.cFull.Slice(test.ax), test.a, test.b)
			if err := equal2(test.cFull, test.cFullAdded, 0); err != nil {
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
			a:    mul(-1i, trange(1, 1+5*5*5, 1)).Conj().Reshape(5, 5, 5).Slice([][2]int{{1, 4}, {2, 4}, {1, 3}}).Transpose(1, 2, 0),
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
			if !c.Equal(test.c) {
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
			a: T2([][]complex64{{1, 2}, {3, 4}}),
			b: T2([][]complex64{{5, 6}, {7, 8}}),
			c: T2([][]complex64{{19, 22}, {43, 50}}),
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			c := Zeros(1)
			MatMul(c, test.a, test.b)
			if !c.Equal(test.c) {
				t.Fatalf("%#v %#v", c, test.c)
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
		if diff := abs(a.At(i) - b[i]); diff > tol {
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
			if diff := abs(a.At(i, j) - b[i][j]); diff > tol {
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
				if diff := abs(a.At(i, j, k) - b[i][j][k]); diff > tol {
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
					if diff := abs(a.At(i, j, k, l) - b[i][j][k][l]); diff > tol {
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

func mul(c complex64, a *Dense) *Dense {
	b := Zeros(a.Shape()...)
	Mul(b, c, a)
	return b
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	m.Run()
}
