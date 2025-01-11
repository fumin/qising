package mat

import (
	"cmp"
	"container/ring"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"slices"
	"time"

	"qising/exactdiag/mat/util"
)

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

	throttler := util.NewSkipThrottler(60 * time.Second)
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

func abs(c complex64) float32 {
	return float32(cmplx.Abs(complex128(c)))
}
