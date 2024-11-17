package qising

import (
	"cmp"
	"encoding/csv"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/pkg/errors"

	"qising/mat"
)

var (
	identity = mat.COOIdentity(2)
)

func TransverseFieldIsing(hamiltonian, buf mat.Matrix, n [2]int, h complex64) {
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
	shapePath := filepath.Join(dir, mat.FnameShape)
	if err := os.WriteFile(shapePath, []byte(fmt.Sprintf("%d,%d", 1<<numSpins, 1<<numSpins)), 0644); err != nil {
		return errors.Wrap(err, "")
	}

	cooPath := filepath.Join(dir, mat.FnameCOO)
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
				vStr = mat.FormatNumpy(v.v)
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

func GetStatistics(n [2]int, vvs []mat.ValVec) (Statistics, error) {
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

func coupling(hamiltonian mat.Matrix, n [2]int, i [2]int, j [2]int, system mat.Matrix) {
	system.Scalar(1)
	for y := 0; y < n[0]; y++ {
		for x := 0; x < n[1]; x++ {
			yx := [2]int{y, x}

			switch {
			case yx == i || yx == j:
				system.Kron(mat.M(mat.PauliZ))
			default:
				system.Kron(identity)
			}
		}
	}

	hamiltonian.Add(-1, system)
}

func magnetic(hamiltonian mat.Matrix, n [2]int, i [2]int, h complex64, system mat.Matrix) {
	system.Scalar(1)
	for y := 0; y < n[0]; y++ {
		for x := 0; x < n[1]; x++ {
			yx := [2]int{y, x}
			switch {
			case yx == i:
				system.Kron(mat.M(mat.PauliX))
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

type vRowCol struct {
	v   complex64
	row int
	col int
}

func rowMajor(a, b vRowCol) int {
	if c := cmp.Compare(a.row, b.row); c != 0 {
		return c
	}
	return cmp.Compare(a.col, b.col)
}
