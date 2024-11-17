package qising

import (
	"cmp"
	"container/ring"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"

	"github.com/pkg/errors"
)

var (
	identity = COOIdentity(2)
)

func TransverseFieldIsing(hamiltonian, buf Matrix, n [2]int, h complex64) {
	numSpins := n[0] * n[1]
	hamiltonian.Zeros(1<<numSpins, 1<<numSpins)

	for y := 0; y < n[0]; y++ {
		for x := 0; x < n[1]; x++ {
			up := y - 1
			if up >= 0 {
				coupling(hamiltonian, n, [2]int{up, x}, [2]int{y, x}, buf)
			}

			left := x - 1
			if left >= 0 {
				coupling(hamiltonian, n, [2]int{y, left}, [2]int{y, x}, buf)
			}

			magnetic(hamiltonian, n, [2]int{y, x}, h, buf)
		}
	}
}

func TransverseFieldIsingExplicit(dir string, n [2]int, h complex64) error {
	numSpins := n[0] * n[1]
	shapePath := filepath.Join(dir, fnameShape)
	if err := os.WriteFile(shapePath, []byte(fmt.Sprintf("%d,%d", 1<<numSpins, 1<<numSpins)), 0644); err != nil {
		return errors.Wrap(err, "")
	}

	cooPath := filepath.Join(dir, fnameCOO)
	f, err := os.Create(cooPath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	w := csv.NewWriter(f)

	// bonds is a reusable buffer for recording coupling bonds.
	bonds := make([][2]int, 0, 2)
	// flipped is a reusable buffer for the flipped state.
	flipped := make([]byte, numSpins)
	// prev is the previously written value for compression.
	prev := vRowCol{v: complex64(cmplx.NaN()), row: -1, col: -1}
	vrcs := make([]vRowCol, 0)
Loop:
	for i, state := range bits(numSpins) {
		vrcs = vrcs[:0]
		vrcs = couplingExplicit(vrcs, n, i, state, bonds)
		vrcs = magneticExplicit(vrcs, n, h, i, state, flipped)

		slices.SortFunc(vrcs, rowMajor)
		for _, v := range vrcs {
			var vStr string
			if v.v != prev.v {
				vStr = formatNumpy(v.v)
			}
			var rowStr string
			if v.row != prev.row {
				rowStr = strconv.Itoa(v.row)
			}
			colStr := strconv.Itoa(v.col)

			if err1 := w.Write([]string{vStr, rowStr, colStr}); err1 != nil && err == nil {
				err = errors.Wrap(err1, "")
				break Loop
			}
			prev = v
		}

		if i%1e6 == 0 {
			// log.Printf("%d/%d %.2f", i, int(1)<<numSpins, float64(i)/float64(int(1)<<numSpins))
		}
	}

	w.Flush()
	if err1 := w.Error(); err1 != nil && err == nil {
		err = errors.Wrap(err1, "")
	}
	if err1 := f.Close(); err1 != nil && err == nil {
		err = errors.Wrap(err1, "")
	}
	return err
}

func pickSpinUp(upState []int8, state []byte) {
	ups := 0
	for _, b := range state {
		if b == 1 {
			ups++
		}
	}

	downs := len(state) - ups
	switch {
	case ups < downs:
		for i, b := range state {
			switch b {
			case 0:
				upState[i] = 1
			default:
				upState[i] = -1
			}
		}
	default:
		for i, b := range state {
			switch b {
			case 0:
				upState[i] = -1
			default:
				upState[i] = 1
			}
		}
	}
}

type Statistics struct {
	EigenValue     []float64
	Magnetization  float64
	BinderCumulant float64
}

func GetStatistics(n [2]int, vvs []ValVec) (Statistics, error) {
	var stats Statistics
	for _, vv := range vvs {
		stats.EigenValue = append(stats.EigenValue, real(vv.Val))
	}
	ground := vvs[0]
	numSpins := n[0] * n[1]
	if len(ground.Vec) != 1<<numSpins {
		return Statistics{}, errors.Errorf("%d %d", len(ground.Vec), 1<<numSpins)
	}
	// spinUpBasis is the basis where the majority of spins are up.
	spinUpBasis := make([]int8, numSpins)
	var totalProb float64
	var m2 float64
	for i, fullBasis := range bits(numSpins) {
		pickSpinUp(spinUpBasis, fullBasis)
		amplitude := ground.Vec[i]
		probability := real(amplitude)*real(amplitude) + imag(amplitude)*imag(amplitude)

		var basisM float64
		for _, spin := range spinUpBasis {
			basisM += float64(spin)
		}

		totalProb += probability
		stats.Magnetization += probability * basisM
		stats.BinderCumulant += probability * math.Pow(basisM, 4)
		m2 += probability * math.Pow(basisM, 2)
	}
	if math.Abs(totalProb-1) > 1e-3 {
		return Statistics{}, errors.Errorf("%f", totalProb)
	}

	stats.Magnetization /= float64(numSpins)
	stats.BinderCumulant /= (m2 * m2)
	stats.BinderCumulant = 1 - stats.BinderCumulant/3
	return stats, nil
}

func GradientDescent(m *COO) (float32, []complex64) {
	floor := gerschgorin(m)
	return gradientDescent(m, floor)
}

func gradientDescent(m *COO, floor float32) (float32, []complex64) {
	var lambda float64 = float64(floor)
	vecRe := make([]float64, m.cols)
	vecIm := make([]float64, m.cols)
	for i := range vecRe {
		vecRe[i] = rand.Float64()
		vecIm[i] = rand.Float64()
	}
	var lambdaGrad float64
	vecReGrad := make([]float64, len(vecRe))
	vecImGrad := make([]float64, len(vecIm))

	byRow := make(map[int][]vRowCol)
	for _, v := range m.Data {
		byRow[v.row] = append(byRow[v.row], v)
	}
	batchSize := 256
	data := newDataloader(m.cols, batchSize)

	var lossSEWeight float64 = 0 * float64(len(m.Data))
	lossFn := func() (float64, float64, float64) {
		lambdaGrad = 0
		for i := range vecReGrad {
			vecReGrad[i] = 0
		}
		for i := range vecImGrad {
			vecImGrad[i] = 0
		}

		// Diagonalization loss.
		var lossDiag float64
		iBatch := data.get()
		for _, i := range iBatch {
			reVi, imVi := vecRe[i], vecIm[i]
			var reAvLv, imAvLv float64
			for _, aijV := range byRow[i] {
				j, aij := aijV.col, aijV.v
				reVj, imVj := vecRe[j], vecIm[j]

				reAvLv += float64(real(aij))*reVj - float64(imag(aij))*imVj
				imAvLv += float64(real(aij))*imVj + float64(imag(aij))*reVj
			}
			reAvLv += -lambda * reVi
			imAvLv += -lambda * imVi

			if reAvLv > 0 {
				lossDiag += reAvLv
				lambdaGrad += -reVi
				vecReGrad[i] += -lambda
				for _, aijV := range byRow[i] {
					j, aij := aijV.col, aijV.v
					vecReGrad[j] += float64(real(aij))
					vecImGrad[j] += -float64(imag(aij))
				}
			} else {
				lossDiag += -reAvLv
				lambdaGrad += reVi
				vecReGrad[i] += lambda
				for _, aijV := range byRow[i] {
					j, aij := aijV.col, aijV.v
					vecReGrad[j] += -float64(real(aij))
					vecImGrad[j] += float64(imag(aij))
				}
			}
			if imAvLv > 0 {
				lossDiag += imAvLv
				lambdaGrad += -imVi
				vecImGrad[i] += -lambda
				for _, aijV := range byRow[i] {
					j, aij := aijV.col, aijV.v
					vecReGrad[j] += float64(imag(aij))
					vecImGrad[j] += float64(real(aij))
				}
			} else {
				lossDiag += -imAvLv
				lambdaGrad += imVi
				vecImGrad[i] += lambda
				for _, aijV := range byRow[i] {
					j, aij := aijV.col, aijV.v
					vecReGrad[j] += -float64(imag(aij))
					vecImGrad[j] += -float64(real(aij))
				}
			}
		}

		// Smallest eigenvalue loss.
		lossSE := lossSEWeight * (lambda*lambda - 2*lambda*float64(floor) + float64(floor*floor))
		lambdaGrad += lossSEWeight * (2*lambda - 2*float64(floor))

		loss := lossDiag + lossSE
		return loss, lossDiag, lossSE
	}

	throttler := newSkipThrottler(60 * time.Second)
	epochIters := (m.rows / len(data.batch)) + 1
	learningRate := newLearningRateAdjuster()
	for epoch := 0; epoch < math.MaxInt; epoch++ {
		var diagDiff float64
		for i := 0; i < epochIters; i++ {
			loss, lossDiag, lossSE := lossFn()
			lambda -= learningRate.v * lambdaGrad
			for j := range vecReGrad {
				vecRe[j] -= learningRate.v * vecReGrad[j]
				vecIm[j] -= learningRate.v * vecImGrad[j]
			}
			if i%1000 == 0 {
				// normalize(vecRe, vecIm)
			}

			diagDiff += lossDiag / float64(len(data.batch))
			if false {
				log.Printf("%f %f %f", loss, lossDiag, lossSE)
			}
		}

		diagDiff /= float64(epochIters)
		learningRate.adjust(epoch, diagDiff)
		lossOK := diagDiff < 1e-3
		if true && (throttler.Ok() || lossOK) {
			log.Printf("%d %f %f", epoch, diagDiff, lambda)
		}
		if lossOK {
			break
		}
	}

	vec := make([]complex64, 0, len(vecRe))
	for i, reVi := range vecRe {
		vec = append(vec, complex64(complex(reVi, vecIm[i])))
	}
	// Make the first zero entry real.
	var c complex64 = complex(1, 0)
	for _, v := range vec {
		if abs(v) > 1e-6 {
			c = v
			break
		}
	}
	for i := range vec {
		vec[i] /= c
	}
	// Normalize.
	var norm float32
	for _, v := range vec {
		norm += real(v)*real(v) + imag(v)*imag(v)
	}
	norm = float32(math.Sqrt(float64(norm)))
	for i := range vec {
		vec[i] /= complex(norm, 0)
	}
	return float32(lambda), vec
}

type learningRateAdjuster struct {
	v    float64
	loss *ring.Ring
}

func newLearningRateAdjuster() *learningRateAdjuster {
	a := &learningRateAdjuster{loss: ring.New(100)}
	a.adjust(-1, math.MaxFloat64)

	for i := 0; i < a.loss.Len(); i++ {
		a.loss.Value = math.MaxFloat64 / float64(a.loss.Len()) / 10
		a.loss = a.loss.Next()
	}

	return a
}

func (a *learningRateAdjuster) adjust(itrtn int, loss float64) {
	// var avg float64
	// a.loss.Do(func(l any) {
	// 	avg += l.(float64)
	// })
	// avg /= float64(a.loss.Len())

	a.loss.Value = loss
	a.loss = a.loss.Next()

	switch {
	case loss < 0.0007:
		a.v = 1e-7
	case loss < 0.003:
		a.v = 1e-6
	case loss < 0.007:
		a.v = 7e-6
	case loss < 0.07:
		a.v = 1e-5
	case loss < 0.3:
		a.v = 1e-4
	default:
		a.v = 1e-3
	}
}

type dataloader struct {
	indices []int
	ptr     int

	batch []int
}

func newDataloader(n, batchSize int) *dataloader {
	dl := &dataloader{
		indices: make([]int, n),
		batch:   make([]int, batchSize),
	}

	for i := 0; i < n; i++ {
		dl.indices[i] = i
	}
	dl.shuffle()
	dl.ptr = -1

	return dl
}

func (dl *dataloader) get() []int {
	for b := range dl.batch {
		dl.ptr++
		if dl.ptr >= len(dl.indices) {
			dl.shuffle()
			dl.ptr = 0
		}
		dl.batch[b] = dl.indices[dl.ptr]
	}
	return dl.batch
}

func (dl *dataloader) shuffle() {
	rand.Shuffle(len(dl.indices), func(i, j int) {
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
	})
}

// Theorem A3, Bounds for the eigenvalues of a matrix, Kenneth R. Garren.
func gerschgorin(m *COO) float32 {
	type circle struct {
		center complex64
		radius float32
	}
	circles := make([]circle, 0, m.rows)

	var curRow int = m.Data[0].row
	var curCenter complex64
	var curRadius float32
	for _, v := range m.Data {
		if v.row != curRow {
			c := circle{center: curCenter, radius: curRadius}
			circles = append(circles, c)

			curRow = v.row
			curCenter = 0
			curRadius = 0
		}

		if v.row == v.col {
			curCenter = v.v
		} else {
			curRadius += abs(v.v)
		}
	}
	// Last current row.
	c := circle{center: curCenter, radius: curRadius}
	circles = append(circles, c)

	// Find the circle with the minimum circumference.
	cMin := func(c circle) float32 {
		return real(c.center) - c.radius
	}
	slices.SortFunc(circles, func(a, b circle) int {
		return cmp.Compare(cMin(a), cMin(b))
	})
	return cMin(circles[0])
}

func normalize(re, im []float64) {
	var norm float64
	for i, reI := range re {
		imI := im[i]
		norm += reI*reI + imI*imI
	}
	norm = math.Sqrt(norm)

	for i := range re {
		re[i], im[i] = re[i]/norm, im[i]/norm
	}
}

func coupling(hamiltonian Matrix, n [2]int, i [2]int, j [2]int, system Matrix) {
	system.Scalar(1)
	for y := 0; y < n[0]; y++ {
		for x := 0; x < n[1]; x++ {
			yx := [2]int{y, x}

			switch {
			case yx == i || yx == j:
				system.Kron(PauliZ)
			default:
				system.Kron(identity)
			}
		}
	}

	hamiltonian.Add(-1, system)
}

func magnetic(hamiltonian Matrix, n [2]int, i [2]int, h complex64, system Matrix) {
	system.Scalar(1)
	for y := 0; y < n[0]; y++ {
		for x := 0; x < n[1]; x++ {
			yx := [2]int{y, x}
			switch {
			case yx == i:
				system.Kron(PauliX)
			default:
				system.Kron(identity)
			}
		}
	}

	hamiltonian.Add(-h, system)
}

func couplingExplicit(vrcs []vRowCol, n [2]int, i int, state []byte, bonds [][2]int) []vRowCol {
	var diag complex64
	for y := range n[0] {
		for x := range n[1] {
			spin := state[y*n[1]+x]

			bonds = bonds[:0]
			up := y - 1
			if up >= 0 {
				bonds = append(bonds, [2]int{up, x})
			}
			left := x - 1
			if left >= 0 {
				bonds = append(bonds, [2]int{y, left})
			}

			for _, b := range bonds {
				spinOther := state[b[0]*n[1]+b[1]]
				switch {
				case spinOther == spin:
					diag -= 1
				default:
					diag += 1
				}
			}
		}
	}
	if diag != 0 {
		vrcs = append(vrcs, vRowCol{v: diag, row: i, col: i})
	}
	return vrcs
}

func magneticExplicit(vrcs []vRowCol, n [2]int, h complex64, i int, state []byte, flipped []byte) []vRowCol {
	for y := range n[0] {
		for x := range n[1] {
			copy(flipped, state)
			idx := y*n[1] + x
			switch flipped[idx] {
			case 1:
				flipped[idx] = 0
			default:
				flipped[idx] = 1
			}

			col := bitIndex(flipped)
			vrcs = append(vrcs, vRowCol{v: -h, row: i, col: col})
		}
	}
	return vrcs
}

func indexBit(state []byte, n, i int) {
	stateStr := strconv.FormatInt(int64(i), 2)

	state = state[:0]
	// Pad zeros in front.
	for j := 0; j < n-len(stateStr); j++ {
		state = append(state, 0)
	}
	for _, bit := range []byte(stateStr) {
		state = append(state, bit-'0')
	}
}

func bits(n int) func(yield func(int, []byte) bool) {
	state := make([]byte, n)
	return func(yield func(int, []byte) bool) {
		numStates := 1 << n
		for i := range numStates {
			indexBit(state, n, i)
			if !yield(i, state) {
				return
			}
		}
	}
}

func bitIndex(state []byte) int {
	idx := 0
	for i := len(state) - 1; i >= 0; i-- {
		if state[i] == 1 {
			idx += 1 << (len(state) - 1 - i)
		}
	}
	return idx
}

func abs(c complex64) float32 {
	return float32(cmplx.Abs(complex128(c)))
}
