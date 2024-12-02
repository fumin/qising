package tensor

import (
	"fmt"
	"slices"
)

const (
	maxDimension = 16
)

type Tensor interface {
	Shape() []int
	At(...int) complex64
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
	return (&Dense{}).Zeros(shape...)
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

func (t *Dense) Zeros(shape ...int) *Dense {
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
	t.Zeros(len(slice))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]]
	}
	return t
}

func (t *Dense) T2(slice [][]complex64) *Dense {
	t.Zeros(len(slice), len(slice[0]))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]][t.digits[1]]
	}
	return t
}

func (t *Dense) T3(slice [][][]complex64) *Dense {
	t.Zeros(len(slice), len(slice[0]), len(slice[0][0]))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]][t.digits[1]][t.digits[2]]
	}
	return t
}

func (t *Dense) T4(slice [][][][]complex64) *Dense {
	t.Zeros(len(slice), len(slice[0]), len(slice[0][0]), len(slice[0][0][0]))

	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = slice[t.digits[0]][t.digits[1]][t.digits[2]][t.digits[3]]
	}
	return t
}

func (t *Dense) Copy(a *Dense) *Dense {
	t.Zeros(a.Shape()...)

	digits := t.digits[:t.dimension]
	var ptr int = -1
	t.initDigits()
	for t.incrDigits() {
		ptr++
		t.data[ptr] = a.At(digits...)
	}
	return t
}

func (t *Dense) Set(start []int, a *Dense) *Dense {
	if len(start) != t.dimension || a.dimension != t.dimension {
		panic("wrong dimension")
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

func (t *Dense) Eye(n, k int) *Dense {
	t.Zeros(n, n)
	for i := range n {
		j := i + k
		if !(j >= 0 && j < n) {
			continue
		}

		ptr := i*t.shape[1] + j
		t.data[ptr] = 1
	}
	return t
}

func (t *Dense) Shape() []int {
	return t.shape[:t.dimension]
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

func (a *Dense) Equal(b *Dense) bool {
	if len(a.Shape()) != len(b.Shape()) {
		return false
	}
	for i := range a.Shape() {
		if a.Shape()[i] != b.Shape()[i] {
			return false
		}
	}

	digits := a.digits[:a.dimension]
	a.initDigits()
	for a.incrDigits() {
		if a.At(digits...) != b.At(digits...) {
			return false
		}
	}
	return true
}

func (a *Dense) Slice(boundary [][2]int) *Dense {
	for i := range a.dimension {
		if !(boundary[i][0] >= 0 && boundary[i][0] <= a.shape[i]) {
			panic(fmt.Sprintf("%d %d", boundary[i][0], a.shape[i]))
		}
		if !(boundary[i][1] >= boundary[i][0] && boundary[i][1] <= a.shape[i]) {
			panic(fmt.Sprintf("%d %d", boundary[i][1], a.shape[i]))
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
			panic(fmt.Sprintf("%d", i))
		}
	}

	var newVolume int = 1
	for _, s := range shape {
		newVolume *= s
	}
	if newVolume != len(a.data) {
		panic(fmt.Sprintf("%d %d", newVolume, len(a.data)))
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

func Mul(b *Dense, ca complex64, a *Dense) *Dense {
	if b != a {
		if len(Overlap(b.data, a.data)) > 0 {
			panic("same array")
		}

		// Find the dimensions of C.
		var bLen int = 1
		bAxis := b.axis[:0]
		for i := range a.dimension {
			aax := a.axis[a.viewToAxis[i]]
			bax := axis{size: aax.end - aax.start}
			bax.start, bax.end = 0, bax.size
			bAxis = append(bAxis, bax)
			bLen *= bax.size
		}
		b.dimension = len(bAxis)
		for i := range b.dimension {
			b.viewToAxis[i] = i
		}
		b.updateShape()

		b.data = b.data[:0]
		b.data = append(b.data, make([]complex64, bLen)...)
	}

	bDigits := b.digits[:b.dimension]
	b.initDigits()
	for b.incrDigits() {
		av := a.At(bDigits...)

		bv := ca * av
		if b.conj {
			bv = conj(bv)
		}
		b.data[b.at(bDigits)] = bv
	}
	return b
}

func Add(c, a, b *Dense) *Dense {
	if !slices.Equal(a.Shape(), b.Shape()) {
		panic("wrong shape")
	}

	if c != a && c != b {
		if len(Overlap(c.data, a.data)) > 0 {
			panic("same array")
		}
		if len(Overlap(c.data, b.data)) > 0 {
			panic("same array")
		}

		// Find the dimensions of C.
		var cLen int = 1
		cAxis := c.axis[:0]
		for i := range a.dimension {
			aax := a.axis[a.viewToAxis[i]]
			cax := axis{size: aax.end - aax.start}
			cax.start, cax.end = 0, cax.size
			cAxis = append(cAxis, cax)
			cLen *= cax.size
		}
		c.dimension = len(cAxis)
		for i := range c.dimension {
			c.viewToAxis[i] = i
		}
		c.updateShape()

		c.data = c.data[:0]
		c.data = append(c.data, make([]complex64, cLen)...)
	}

	cDigits := c.digits[:c.dimension]
	c.initDigits()
	for c.incrDigits() {
		av := a.At(cDigits...)
		bv := b.At(cDigits...)

		cv := av + bv
		if c.conj {
			cv = conj(cv)
		}
		c.data[c.at(cDigits)] = cv
	}
	return c
}

func Contract(c, a, b *Dense, axes [][2]int) *Dense {
	if len(Overlap(c.data, a.data)) > 0 || len(Overlap(c.data, b.data)) > 0 {
		panic("same array")
	}
	// Check shapes match.
	axShapes := make([]int, 0, len(axes))
	for _, axs := range axes {
		if a.shape[axs[0]] != b.shape[axs[1]] {
			panic(fmt.Sprintf("different axis dimensions %d %d", a.shape[axs[0]], b.shape[axs[1]]))
		}
		axShapes = append(axShapes, a.shape[axs[0]])
	}

	// Find the dimensions of C.
	cAxis := c.axis[:0]
	var cLen int = 1
	cToA := make([][2]int, 0, a.dimension)
	for i := range a.dimension {
		if !slices.ContainsFunc(axes, func(axs [2]int) bool { return axs[0] == i }) {
			aax := a.axis[a.viewToAxis[i]]

			cax := axis{size: aax.end - aax.start}
			cax.start, cax.end = 0, cax.size
			cAxis = append(cAxis, cax)

			cLen *= cax.size
			cToA = append(cToA, [2]int{len(cAxis) - 1, i})
		}
	}
	cToB := make([][2]int, 0, b.dimension)
	for i := range b.dimension {
		if !slices.ContainsFunc(axes, func(axs [2]int) bool { return axs[1] == i }) {
			bax := b.axis[b.viewToAxis[i]]

			cax := axis{size: bax.end - bax.start}
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

	// Do the contraction.
	aDigits := a.digits[:a.dimension]
	bDigits := b.digits[:b.dimension]
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
				aDigits[d[1]] = cDigits[d[0]]
			}
			for i, ctt := range cntrct {
				aDigits[axes[i][0]] = ctt
			}
			av := a.At(aDigits...)

			// Get B component.
			for _, d := range cToB {
				bDigits[d[1]] = cDigits[d[0]]
			}
			for i, ctt := range cntrct {
				bDigits[axes[i][1]] = ctt
			}
			bv := b.At(bDigits...)

			v += av * bv
		}

		c.data[ptr] = v
	}
	return c
}

func MatMul(c, a, b *Dense) *Dense {
	return Contract(c, a, b, [][2]int{{a.dimension - 1, b.dimension - 2}})
}

func (t *Dense) H() *Dense {
	ax := make([]int, t.dimension)
	for i := range t.dimension {
		ax[i] = i
	}
	ax[len(ax)-2], ax[len(ax)-1] = ax[len(ax)-1], ax[len(ax)-2]
	return t.Transpose(ax...).Conj()
}

func (t *Dense) FrobeniusNorm() float32 {
	var norm float32
	digits := t.digits[:t.dimension]
	t.initDigits()
	for t.incrDigits() {
		v := t.At(digits...)
		norm += real(v)*real(v) + imag(v)*imag(v)
	}
	return sqrt(norm)
}

func (t *Dense) ToSlice2() [][]complex64 {
	slice := make([][]complex64, t.shape[0])
	for i := range len(slice) {
		slice[i] = make([]complex64, t.shape[1])
	}
	for i := range len(slice) {
		for j := range len(slice[0]) {
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
	initDigits(t.digits[:t.dimension])
}

func (t *Dense) incrDigits() bool {
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
