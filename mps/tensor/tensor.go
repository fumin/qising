package tensor

import (
	"fmt"
	"math/rand"
	"slices"
)

type Tensor struct {
	Shape  []int
	Data   []complex64
	digits []int
}

func T() *Tensor {
	return &Tensor{}
}

func (t *Tensor) Zero(shape []int) *Tensor {
	t.Shape = t.Shape[:0]
	t.Shape = append(t.Shape, shape...)

	var tLen int = 1
	for _, d := range t.Shape {
		tLen *= d
	}
	t.Data = t.Data[:0]
	t.Data = append(t.Data, make([]complex64, tLen)...)

	return t
}

func (a *Tensor) Set(b *Tensor) *Tensor {
	a.Shape = a.Shape[:0]
	a.Shape = append(a.Shape, b.Shape...)
	a.Data = a.Data[:0]
	a.Data = append(a.Data, b.Data...)
	return a
}

func (t *Tensor) T2(slice [][]complex64) *Tensor {
	t.Shape = t.Shape[:0]
	t.Shape = append(t.Shape, len(slice))
	t.Shape = append(t.Shape, len(slice[0]))
	t.Data = t.Data[:0]
	for i := range len(slice) {
		for j := range len(slice[0]) {
			t.Data = append(t.Data, slice[i][j])
		}
	}
	return t
}

func (t *Tensor) T3(slice [][][]complex64) *Tensor {
	t.Shape = t.Shape[:0]
	t.Shape = append(t.Shape, len(slice))
	t.Shape = append(t.Shape, len(slice[0]))
	t.Shape = append(t.Shape, len(slice[0][0]))
	t.Data = t.Data[:0]
	for i := range len(slice) {
		for j := range len(slice[0]) {
			for k := range len(slice[0][0]) {
				t.Data = append(t.Data, slice[i][j][k])
			}
		}
	}
	return t
}

func (t *Tensor) T4(slice [][][][]complex64) *Tensor {
	t.Shape = t.Shape[:0]
	t.Shape = append(t.Shape, len(slice))
	t.Shape = append(t.Shape, len(slice[0]))
	t.Shape = append(t.Shape, len(slice[0][0]))
	t.Shape = append(t.Shape, len(slice[0][0][0]))
	t.Data = t.Data[:0]
	for i := range len(slice) {
		for j := range len(slice[0]) {
			for k := range len(slice[0][0]) {
				for l := range len(slice[0][0][0]) {
					t.Data = append(t.Data, slice[i][j][k][l])
				}
			}
		}
	}
	return t
}

func (t *Tensor) Rand(shape []int) *Tensor {
	t.Shape = t.Shape[:0]
	t.Shape = append(t.Shape, shape...)

	var size int = 1
	for _, d := range t.Shape {
		size *= d
	}
	t.Data = t.Data[:0]
	t.Data = append(t.Data, make([]complex64, size)...)
	for i := range t.Data {
		t.Data[i] = complex(rand.Float32(), rand.Float32())
	}

	return t
}

func (a *Tensor) Equal(b *Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

func (a *Tensor) Slice(start, end []int) *Tensor {
	// Prepare bShape.
	if len(start) != len(a.Shape) {
		panic(fmt.Sprintf("%d %d", len(start), len(a.Shape)))
	}
	if len(start) != len(end) {
		panic(fmt.Sprintf("%d %d", len(start), len(end)))
	}
	a.Shape = append(a.Shape, make([]int, len(start))...)
	bShape := a.Shape[len(start):]
	a.Shape = a.Shape[:len(start)]
	var bLen int = 1
	for i := range start {
		bShape[i] = end[i] - start[i]
		bLen *= bShape[i]
	}

	// Prepare bData.
	aLen := len(a.Data)
	a.Data = append(a.Data, make([]complex64, bLen)...)
	bData := a.Data[aLen:]
	a.Data = a.Data[:aLen]

	a.digits = a.digits[:0]
	a.digits = append(a.digits, make([]int, len(a.Shape)+len(bShape))...)
	bDigits := a.digits[len(a.Shape):]
	a.digits = a.digits[:len(a.Shape)]
	bDigits[len(bDigits)-1] = -1
	var bIdx int = -1
	for incr(bDigits, bShape) {
		bIdx++

		for i := range bDigits {
			a.digits[i] = start[i] + bDigits[i]
		}

		bData[bIdx] = a.Data[at(a.digits, a.Shape)]
	}

	copy(a.Shape, bShape)
	copy(a.Data, bData)
	a.Data = a.Data[:len(bData)]
	return a
}

func (t *Tensor) Reshape(newShape []int) *Tensor {
	// Check alignment.
	pShape, pNewShape := len(t.Shape), len(newShape)
	var prevElems, prevNewElems int
	elems, newElems := 1, 1

	pShape--
	prevElems = elems
	elems *= t.Shape[pShape]

	pNewShape--
	prevNewElems = newElems
	newElems *= newShape[pNewShape]
Loop:
	for {
		switch {
		case elems < newElems:
			if pShape == 0 {
				break Loop
			}
			pShape--
			prevElems = elems
			elems *= t.Shape[pShape]
		case elems == newElems:
			if pShape == 0 {
				break Loop
			}
			pShape--
			prevElems = elems
			elems *= t.Shape[pShape]

			if pNewShape == 0 {
				break Loop
			}
			pNewShape--
			prevNewElems = newElems
			newElems *= newShape[pNewShape]
		default:
			if pNewShape == 0 {
				break Loop
			}
			pNewShape--
			prevNewElems = newElems
			newElems *= newShape[pNewShape]
		}

		overtake := elems > newElems && prevElems < prevNewElems
		newOvertake := elems < newElems && prevElems > prevNewElems
		if overtake || newOvertake {
			panic(fmt.Sprintf("current %d %d %d new %d %d %d", pShape, prevElems, elems, pNewShape, prevNewElems, newElems))
		}
	}
	if elems != newElems {
		panic(fmt.Sprintf("%d %d", elems, newElems))
	}

	t.Shape = t.Shape[:0]
	t.Shape = append(t.Shape, newShape...)
	return t
}

func (t *Tensor) Conj() *Tensor {
	for i := range t.Data {
		t.Data[i] = complex(real(t.Data[i]), -imag(t.Data[i]))
	}
	return t
}

func (b *Tensor) Transpose(axis []int) *Tensor {
	// Check if axis is {0, 1, 2,...}
	b.digits = b.digits[:0]
	b.digits = append(b.digits, axis...)
	slices.Sort(b.digits)
	if b.digits[0] != 0 {
		panic(fmt.Sprintf("%d", b.digits[0]))
	}
	for i := range len(b.digits) - 1 {
		if b.digits[i+1] != b.digits[i]+1 {
			panic(fmt.Sprintf("%d %d %d", i, b.digits[i+1], b.digits[i]))
		}
	}
	if b.digits[len(b.digits)-1] != len(b.Shape)-1 {
		panic(fmt.Sprintf("%d %d", b.digits[len(b.digits)-1], len(b.Shape)))
	}

	// Prepare aShape, aData, aDigits.
	bShapeLen := len(b.Shape)
	b.Shape = append(b.Shape, make([]int, bShapeLen)...)
	aShape := b.Shape[bShapeLen:]
	b.Shape = b.Shape[:bShapeLen]
	copy(aShape, b.Shape)
	for i := range aShape {
		b.Shape[i] = aShape[axis[i]]
	}

	b.digits = b.digits[:0]
	b.digits = append(b.digits, make([]int, 2*len(b.Shape))...)
	aDigits := b.digits[len(b.Shape):]
	aLen := len(b.Data)
	b.Data = append(b.Data, make([]complex64, aLen)...)
	aData := b.Data[aLen:]
	copy(aData, b.Data[:aLen])
	b.Data = b.Data[:aLen]

	b.digits = b.digits[:len(b.Shape)]
	for i := range b.digits {
		b.digits[i] = 0
	}
	b.digits[len(b.digits)-1] = -1
	var bIdx int = -1
	for incr(b.digits, b.Shape) {
		bIdx++

		for i := range b.digits {
			aDigits[axis[i]] = b.digits[i]
		}

		b.Data[bIdx] = aData[at(aDigits, aShape)]
	}

	return b
}

func (c *Tensor) Contract(b *Tensor, aAxis, bAxis int) *Tensor {
	if !(aAxis >= 0 && aAxis < len(c.Shape)) {
		panic(fmt.Sprintf("%d %d", aAxis, len(c.Shape)))
	}
	if !(bAxis >= 0 && bAxis < len(b.Shape)) {
		panic(fmt.Sprintf("%d %d", bAxis, len(b.Shape)))
	}
	if c.Shape[aAxis] != b.Shape[bAxis] {
		panic(fmt.Sprintf("different axis dimensions %d %d", c.Shape[aAxis], b.Shape[bAxis]))
	}

	// Let c = a*b, prepare aShape, aData, c.Shape, and c.Data.
	aShapeLen := len(c.Shape)
	// cShapeLen is len(c.Shape).
	cShapeLen := aShapeLen + len(b.Shape) - 2
	c.Shape = append(c.Shape, make([]int, cShapeLen)...)
	aShape := c.Shape[cShapeLen:]
	copy(aShape, c.Shape[:aShapeLen])
	c.Shape = c.Shape[:0]
	var cLen, aLen int = 1, 1
	for i, d := range aShape {
		aLen *= d
		if i != aAxis {
			c.Shape = append(c.Shape, d)
			cLen *= d
		}
	}
	for i, d := range b.Shape {
		if i != bAxis {
			c.Shape = append(c.Shape, d)
			cLen *= d
		}
	}
	c.Data = append(c.Data, make([]complex64, cLen)...)
	aData := c.Data[cLen:]
	copy(aData, c.Data[:aLen])
	c.Data = c.Data[:cLen]

	var aGap int = 1
	for i := aAxis + 1; i < len(aShape); i++ {
		aGap *= aShape[i]
	}
	var bGap int = 1
	for i := bAxis + 1; i < len(b.Shape); i++ {
		bGap *= b.Shape[i]
	}

	c.digits = c.digits[:0]
	c.digits = append(c.digits, make([]int, len(c.Shape))...)
	c.digits[len(c.digits)-1] = -1
	var cIdx int = -1
	for incr(c.digits, c.Shape) {
		cIdx++

		// Find the start of a.Data.
		aDigits := b.digits[:0]
		aDigits = append(aDigits, make([]int, len(aShape))...)
		copy(aDigits[:aAxis], c.digits[:aAxis])
		aDigits[aAxis] = 0
		copy(aDigits[aAxis+1:], c.digits[aAxis:len(aShape)-1])
		aIdx := at(aDigits, aShape)

		// Find the start of b.Data.
		b.digits = b.digits[:0]
		b.digits = append(b.digits, make([]int, len(b.Shape))...)
		copy(b.digits[:bAxis], c.digits[len(aShape)-1:len(aShape)-1+bAxis])
		b.digits[bAxis] = 0
		copy(b.digits[bAxis+1:], c.digits[len(aShape)-1+bAxis:])
		bIdx := at(b.digits, b.Shape)

		var v complex64
		for j := range aShape[aAxis] {
			av := aData[aIdx+j*aGap]
			bv := b.Data[bIdx+j*bGap]
			v += av * bv
		}

		c.Data[cIdx] = v
	}
	return c
}

func at(digits, shape []int) int {
	var idx int = 0
	var power int = 1
	for i := len(shape) - 1; i >= 0; i-- {
		idx += digits[i] * power
		power *= shape[i]
	}
	return idx
}

func incr(digits, shape []int) bool {
	digits[len(digits)-1]++

	for i := len(digits) - 1; i >= 1; i-- {
		if digits[i] < shape[i] {
			break
		}
		digits[i] = 0
		digits[i-1]++
	}

	return digits[0] < shape[0]
}
