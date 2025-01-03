package tensor

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"github.com/pkg/errors"
)

const (
	maxDimension = 16
)

type Tensor interface {
	Shape() []int
	SetAt([]int, complex64)
	At(...int) complex64

	Digits() []int
	Data() []complex64
}

type axis struct {
	// size is the axis length in the underlying data buffer.
	size int
	// start and end are the boundaries of the current axis view.
	start int
	end   int
}

type Dense struct {
	dimension int

	// axis holds information for interpreting the underlying data.
	axis [maxDimension]axis
	data []complex64

	// viewToAxis maps user facing views to the underlying axes.
	viewToAxis [maxDimension]int
	axisToView [maxDimension]int

	// conj indicates whether components have to be conjugated.
	conj bool

	// Derived fields.
	digits [maxDimension]int
	shape  [maxDimension]int
}

func Zeros(shape ...int) *Dense {
	return (&Dense{}).Reset(shape...)
}

func T1(slice []complex64) *Dense {
	return (&Dense{}).T1(slice)
}

func T2(slice [][]complex64) *Dense {
	return (&Dense{}).T2(slice)
}

func T3(slice [][][]complex64) *Dense {
	return (&Dense{}).T3(slice)
}

func T4(slice [][][][]complex64) *Dense {
	return (&Dense{}).T4(slice)
}

func Scalar(c complex64) *Dense {
	return &Dense{data: []complex64{c}}
}

func (t *Dense) Reset(shape ...int) *Dense {
	// Configure axes.
	t.dimension = len(shape)
	for i := range t.dimension {
		t.axis[i].size = shape[i]
		t.axis[i].start = 0
		t.axis[i].end = t.axis[i].size

		t.viewToAxis[i] = i
	}
	t.updateShape()
	t.conj = false

	// Allocate data.
	var volume int = 1
	for i := range t.dimension {
		volume *= t.axis[i].size
	}
	t.data = t.data[:0]
	t.data = append(t.data, make([]complex64, volume)...)

	return t
}

func (t *Dense) T1(slice []complex64) *Dense {
	t.Reset(len(slice))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]]
	}
	return t
}

func (t *Dense) T2(slice [][]complex64) *Dense {
	t.Reset(len(slice), len(slice[0]))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]][t.digits[1]]
	}
	return t
}

func (t *Dense) T3(slice [][][]complex64) *Dense {
	t.Reset(len(slice), len(slice[0]), len(slice[0][0]))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]][t.digits[1]][t.digits[2]]
	}
	return t
}

func (t *Dense) T4(slice [][][][]complex64) *Dense {
	t.Reset(len(slice), len(slice[0]), len(slice[0][0]), len(slice[0][0][0]))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]][t.digits[1]][t.digits[2]][t.digits[3]]
	}
	return t
}

func (t *Dense) Shape() []int {
	return t.shape[:t.dimension]
}

func (t *Dense) SetAt(digits []int, c complex64) {
	if t.conj {
		c = conj(c)
	}
	switch t.dimension {
	case 0:
		t.data[0] = c
	default:
		t.data[t.at(digits)] = c
	}
}

func (t *Dense) At(digits ...int) complex64 {
	var c complex64
	switch t.dimension {
	case 0:
		c = t.data[0]
	default:
		c = t.data[t.at(digits)]
	}

	if t.conj {
		c = conj(c)
	}

	return c
}

func (t *Dense) Digits() []int {
	return t.digits[:t.dimension]
}

func (t *Dense) Data() []complex64 {
	return t.data
}

func (a *Dense) Equal(b *Dense, tol float32) error {
	if len(a.Shape()) != len(b.Shape()) {
		return errors.Errorf("different shapes %d %d", len(a.Shape()), len(b.Shape()))
	}
	for i := range a.Shape() {
		if a.Shape()[i] != b.Shape()[i] {
			return errors.Errorf("different shape at %d %d %d", i, a.Shape()[i], b.Shape()[i])
		}
	}

	digits := a.digits[:a.dimension]
	a.initDigits()
	for a.incrDigits() {
		av := a.At(digits...)
		bv := b.At(digits...)
		if diff := abs(av - bv); diff > tol {
			return errors.Errorf("different values %f at %#v %v %v", diff, digits, av, bv)
		}
	}
	return nil
}

func (t *Dense) All() func(yield func([]int, complex64) bool) {
	digits := t.digits[:t.dimension]
	return func(yield func([]int, complex64) bool) {
		t.initDigits()
		for t.incrDigits() {
			v := t.At(digits...)
			if !yield(digits, v) {
				return
			}
		}
	}
}

func (t *Dense) Set(start []int, a *Dense) *Dense {
	if len(start) != t.dimension || a.dimension != t.dimension {
		panic(fmt.Sprintf("wrong dimension %d %d %d", t.dimension, len(start), a.dimension))
	}
	for i := range t.dimension {
		if start[i]+a.shape[i] > t.shape[i] {
			panic(fmt.Sprintf("%d %d + %d > %d", i, start[i], a.shape[i], t.shape[i]))
		}
	}

	aDigits := a.digits[:a.dimension]
	tDigits := t.digits[:t.dimension]
	a.initDigits()
	for a.incrDigits() {
		av := a.At(aDigits...)

		for i := range tDigits {
			tDigits[i] = start[i] + aDigits[i]
		}
		cv := av
		if t.conj {
			cv = conj(cv)
		}
		t.data[t.at(tDigits)] = cv
	}
	return t
}

func (a *Dense) Slice(boundary [][2]int) *Dense {
	for i := range a.dimension {
		if !(boundary[i][0] >= 0 && boundary[i][0] <= a.shape[i]) {
			panic(fmt.Sprintf("At dim %d boundary %d shape %d", i, boundary[i][0], a.shape[i]))
		}
		if !(boundary[i][1] >= boundary[i][0] && boundary[i][1] <= a.shape[i]) {
			panic(fmt.Sprintf("At dim %d boundary %d %d shape %d", i, boundary[i][0], boundary[i][1], a.shape[i]))
		}
	}

	var outerStride int = 1
	for i := a.dimension - 1; i >= 1; i-- {
		outerStride *= a.axis[i].size
	}

	b := &Dense{dimension: a.dimension, viewToAxis: a.viewToAxis, axis: a.axis, conj: a.conj, data: a.data}
	for i := range b.dimension {
		ax := b.axis[b.viewToAxis[i]]
		b.axis[b.viewToAxis[i]].start = ax.start + boundary[i][0]
		b.axis[b.viewToAxis[i]].end = ax.start + boundary[i][1]

		// We can normalize for the outer most axis.
		if b.viewToAxis[i] == 0 {
			ax = b.axis[b.viewToAxis[i]]
			var newax axis
			newax.size = ax.end - ax.start
			newax.start = 0
			newax.end = newax.size
			b.axis[b.viewToAxis[i]] = newax
			b.data = b.data[ax.start*outerStride : ax.end*outerStride]
		}
	}
	b.updateShape()
	return b
}

func (a *Dense) Transpose(axis ...int) *Dense {
	// Check if axis is {0, 1, 2,...}
	if len(axis) != a.dimension {
		panic(fmt.Sprintf("wrong dimension %d %d", len(axis), a.dimension))
	}
	digits := a.digits[:len(axis)]
	copy(digits, axis)
	slices.Sort(digits)
	if digits[0] != 0 {
		panic(fmt.Sprintf("%d", digits[0]))
	}
	for i := range len(digits) - 1 {
		if digits[i+1] != digits[i]+1 {
			panic(fmt.Sprintf("%d %d %d", i, digits[i+1], digits[i]))
		}
	}

	b := &Dense{dimension: a.dimension, axis: a.axis, conj: a.conj, data: a.data}
	for i := range b.dimension {
		b.viewToAxis[i] = a.viewToAxis[axis[i]]
	}
	b.updateShape()
	return b
}

func (a *Dense) Reshape(shape ...int) *Dense {
	// Transposed tensor cannot be reshaped.
	for i := range a.dimension {
		if a.viewToAxis[i] != i {
			panic(fmt.Sprintf("%d", i))
		}
	}
	// Sliced tensor cannot be reshaped.
	for i := range a.dimension {
		ax := a.axis[i]
		if !(ax.start == 0 && ax.end == ax.size) {
			panic(fmt.Sprintf("axis has been sliced %d", i))
		}
	}

	inferIdx := slices.Index(shape, -1)
	if inferIdx == -1 {
		// User specified all dimensions.
		var newVolume int = 1
		for _, s := range shape {
			newVolume *= s
		}
		if newVolume != len(a.data) {
			panic(fmt.Sprintf("wrong volume %d %d", newVolume, len(a.data)))
		}
	} else {
		var volume int = 1
		for _, s := range a.Shape() {
			volume *= s
		}

		var usedVolume int = 1
		for i, s := range shape {
			if i == inferIdx {
				continue
			}
			usedVolume *= s
		}
		if usedVolume <= 0 {
			panic(fmt.Sprintf("wrong volume %d", usedVolume))
		}

		if volume%usedVolume != 0 {
			panic(fmt.Sprintf("wrong volume %d %d", volume, usedVolume))
		}
		shape[inferIdx] = volume / usedVolume
	}

	b := &Dense{dimension: len(shape), conj: a.conj, data: a.data}
	for i := range b.dimension {
		b.axis[i].size = shape[i]
		b.axis[i].start = 0
		b.axis[i].end = b.axis[i].size

		b.viewToAxis[i] = i
	}
	b.updateShape()

	return b
}

func (a *Dense) Conj() *Dense {
	b := &Dense{dimension: a.dimension, axis: a.axis, viewToAxis: a.viewToAxis, data: a.data}
	b.updateShape()
	b.conj = !a.conj
	return b
}

func (x *Dense) Mul(c complex64) *Dense {
	digits := x.digits[:x.dimension]
	x.initDigits()
	for x.incrDigits() {
		v := x.At(digits...)

		v *= c
		if x.conj {
			v = conj(v)
		}
		x.data[x.at(digits)] = v
	}
	return x
}

func (a *Dense) Add(c complex64, b *Dense) *Dense {
	if !slices.Equal(a.Shape(), b.Shape()) {
		panic(fmt.Sprintf("wrong shape %#v %#v", a.Shape(), b.Shape()))
	}

	aDigits := a.digits[:a.dimension]
	a.initDigits()
	for a.incrDigits() {
		av := a.At(aDigits...)
		bv := b.At(aDigits...)

		av += c * bv
		if a.conj {
			av = conj(av)
		}
		a.data[a.at(aDigits)] = av
	}
	return a
}

func Contract(c *Dense, a, b Tensor, axes [][2]int) *Dense {
	if len(Overlap(c.Data(), a.Data())) > 0 || len(Overlap(c.Data(), b.Data())) > 0 {
		panic("same array")
	}
	// Check shapes match.
	axShapes := make([]int, 0, len(axes))
	for _, axs := range axes {
		if a.Shape()[axs[0]] != b.Shape()[axs[1]] {
			panic(fmt.Sprintf("different axis dimensions %d %d axs %#v %#v %#v", a.Shape()[axs[0]], b.Shape()[axs[1]], axs, a.Shape(), b.Shape()))
		}
		axShapes = append(axShapes, a.Shape()[axs[0]])
	}

	// Find the dimensions of C.
	cAxis := c.axis[:0]
	var cLen int = 1
	cToA := make([][2]int, 0, len(a.Shape()))
	for i := range len(a.Shape()) {
		if !slices.ContainsFunc(axes, func(axs [2]int) bool { return axs[0] == i }) {
			cax := axis{size: a.Shape()[i]}
			cax.start, cax.end = 0, cax.size
			cAxis = append(cAxis, cax)

			cLen *= cax.size
			cToA = append(cToA, [2]int{len(cAxis) - 1, i})
		}
	}
	cToB := make([][2]int, 0, len(b.Shape()))
	for i := range len(b.Shape()) {
		if !slices.ContainsFunc(axes, func(axs [2]int) bool { return axs[1] == i }) {
			cax := axis{size: b.Shape()[i]}
			cax.start, cax.end = 0, cax.size
			cAxis = append(cAxis, cax)

			cLen *= cax.size
			cToB = append(cToB, [2]int{len(cAxis) - 1, i})
		}
	}
	c.dimension = len(cAxis)
	for i := range c.dimension {
		c.viewToAxis[i] = i
	}
	c.updateShape()

	c.data = c.data[:0]
	c.data = append(c.data, make([]complex64, cLen)...)
	if c.dimension == 0 {
		c.data = append(c.data, 0)
	}

	// Do the contraction.
	cntrct := make([]int, len(axShapes))
	var ptr int = -1
	c.initDigits()
	for c.incrDigits() {
		ptr++
		cDigits := c.digits[:c.dimension]

		var v complex64
		initDigits(cntrct)
		for incrDigits(cntrct, axShapes) {
			// Get A component.
			for _, d := range cToA {
				a.Digits()[d[1]] = cDigits[d[0]]
			}
			for i, ctt := range cntrct {
				a.Digits()[axes[i][0]] = ctt
			}
			av := a.At(a.Digits()...)

			// Get B component.
			for _, d := range cToB {
				b.Digits()[d[1]] = cDigits[d[0]]
			}
			for i, ctt := range cntrct {
				b.Digits()[axes[i][1]] = ctt
			}
			bv := b.At(b.Digits()...)

			v += av * bv
		}

		c.data[ptr] = v
	}
	return c
}

func MatMul(c *Dense, a, b Tensor) *Dense {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	if adok && bdok && len(a.Shape()) == 2 && len(b.Shape()) == 2 {
		return matmul(c, ad, bd)
	}
	if len(b.Shape()) == 1 {
		return Contract(c, a, b, [][2]int{{len(a.Shape()) - 1, len(b.Shape()) - 1}})
	}
	return Contract(c, a, b, [][2]int{{len(a.Shape()) - 1, len(b.Shape()) - 2}})
}

func (t *Dense) H() *Dense {
	ax := make([]int, len(t.Shape()))
	for i := range len(t.Shape()) {
		ax[i] = i
	}
	ax[len(ax)-2], ax[len(ax)-1] = ax[len(ax)-1], ax[len(ax)-2]
	return t.Transpose(ax...).Conj()
}

func (t *Dense) FrobeniusNorm() float32 {
	var scale float32
	var sumSquares float32 = 1
	digits := t.digits[:t.dimension]
	t.initDigits()
	for t.incrDigits() {
		v := t.At(digits...)
		if v == 0 {
			continue
		}
		absxi := abs(v)
		if scale < absxi {
			s := scale / absxi
			sumSquares = 1 + sumSquares*s*s
			scale = absxi
		} else {
			s := absxi / scale
			sumSquares += s * s
		}
	}
	return scale * sqrtf(sumSquares)
}

func (t *Dense) ToSlice1() []complex64 {
	if len(t.Shape()) != 1 {
		panic(fmt.Sprintf("%#v", t.Shape()))
	}
	slice := make([]complex64, t.shape[0])
	for i := range len(slice) {
		slice[i] = t.At(i)
	}
	return slice
}

func (t *Dense) ToSlice2() [][]complex64 {
	if len(t.Shape()) != 2 {
		panic(fmt.Sprintf("%#v", t.Shape()))
	}
	slice := make([][]complex64, t.shape[0])
	for i := range len(slice) {
		slice[i] = make([]complex64, t.shape[1])
		for j := range len(slice[i]) {
			slice[i][j] = t.At(i, j)
		}
	}
	return slice
}

func (t *Dense) at(digits []int) int {
	var ptr int
	var power int = 1
	for i := t.dimension - 1; i >= 0; i-- {
		ptr += (t.axis[i].start + digits[t.axisToView[i]]) * power
		power *= t.axis[i].size
	}
	return ptr
}

func (t *Dense) initDigits() {
	if t.dimension == 0 {
		t.digits[0] = -1
		return
	}

	initDigits(t.digits[:t.dimension])
}

func (t *Dense) incrDigits() bool {
	if t.dimension == 0 {
		t.digits[0]++
		return t.digits[0] <= 0
	}

	return incrDigits(t.digits[:t.dimension], t.shape[:t.dimension])
}

func (t *Dense) updateShape() {
	// Update axisToView.
	for i := range t.dimension {
		t.axisToView[t.viewToAxis[i]] = i
	}

	// Update shape.
	for i := range t.dimension {
		ax := t.axis[t.viewToAxis[i]]
		t.shape[i] = ax.end - ax.start
	}
}

func initDigits(digits []int) {
	for i := range digits {
		digits[i] = 0
	}
	digits[len(digits)-1] = -1
}

func incrDigits(digits, base []int) bool {
	digits[len(digits)-1]++

	for i := len(digits) - 1; i >= 1; i-- {
		if digits[i] < base[i] {
			break
		}
		digits[i] = 0
		digits[i-1]++
	}

	return digits[0] < base[0]
}

func sqrtf(v float32) float32 {
	return float32(math.Sqrt(float64(v)))
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func sign(a, b float32) float32 {
	if b >= 0 {
		return absf(a)
	}
	return -absf(a)
}

func sqrt(v complex64) complex64 {
	return complex64(cmplx.Sqrt(complex128(v)))
}

func abs(v complex64) float32 {
	return float32(cmplx.Abs(complex128(v)))
}

func conj(v complex64) complex64 {
	return complex(real(v), -imag(v))
}

// lapy performs sqrt(x*conj(x) + y*conj(y) + ...) similar to the LAPACK routine.
func lapy(vs ...complex64) float32 {
	var scale float32
	var sumSquares float32 = 1
	for _, v := range vs {
		if v == 0 {
			continue
		}
		absxi := abs(v)
		if scale < absxi {
			s := scale / absxi
			sumSquares = 1 + sumSquares*s*s
			scale = absxi
		} else {
			s := absxi / scale
			sumSquares += s * s
		}
	}
	return scale * sqrtf(sumSquares)
}
