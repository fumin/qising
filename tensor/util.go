package tensor

import (
	"bufio"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/pkg/errors"
)

// SameArray returns true if x and y share the same underlying array. Sharing
// the same underlying array does not imply overlap, but rather that it is
// possible to reslice one or the other such that both point to the same memory
// region.
// For more details, please see https://groups.google.com/g/golang-nuts/c/ks1jvoyMYuc.
func SameArray(x, y []complex64) bool {
	return cap(x) > 0 && cap(y) > 0 && &(x[:cap(x)][cap(x)-1]) == &(y[:cap(y)][cap(y)-1])
}

// Overlap returns a slice pointing to the overlapping memory between x and y.
// Nil is returned if x and y do not share the same underlying array or if they
// do not overlap.
func Overlap(x, y []complex64) (z []complex64) {
	if len(x) == 0 || len(y) == 0 || !SameArray(x, y) {
		return
	} else if cap(x) < cap(y) {
		x, y = y, x
	}
	if cap(x)-len(x) < cap(y) {
		z = y
		if lxy := len(x) - (cap(x) - cap(y)); lxy < len(y) {
			z = z[:lxy]
		}
	}
	return
}

func ReadMatrixMarket(r io.Reader) (*Dense, error) {
	const general = "general"
	const symmetric = "symmetric"
	const hermitian = "hermitian"

	s := bufio.NewScanner(r)
	s.Scan()
	if err := s.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}
	header := strings.Fields(s.Text())
	if header[0] != "%%MatrixMarket" {
		return nil, errors.Errorf("unknown format")
	}
	if header[1] != "matrix" {
		return nil, errors.Errorf("unknown format")
	}
	if header[2] != "coordinate" {
		return nil, errors.Errorf("unknown format")
	}
	mtype := header[3]
	if !slices.Contains([]string{"real", "complex"}, mtype) {
		return nil, errors.Errorf("unsupported")
	}
	structure := header[4]
	if !slices.Contains([]string{general, symmetric, hermitian}, structure) {
		return nil, errors.Errorf("unsupported")
	}

	var nr, nc, nnz int
	for s.Scan() {
		line := s.Text()
		if line[0] == '%' {
			continue
		}
		n, err := fmt.Sscan(line, &nr, &nc, &nnz)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		if n != 3 {
			return nil, errors.Errorf("unknown format")
		}
		break
	}
	if err := s.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}

	if slices.Contains([]string{symmetric, hermitian}, structure) && nr != nc {
		return nil, errors.Errorf("unknown format")
	}

	t := Zeros(nr, nc)
	for i := 0; i < nnz; i++ {
		if !s.Scan() {
			return nil, errors.Errorf("unknown format")
		}
		var (
			i, j   int
			vr, vi float64
		)
		switch mtype {
		case "real":
			n, err := fmt.Sscan(s.Text(), &i, &j, &vr)
			if err != nil {
				return nil, errors.Wrap(err, "")
			}
			if n != 3 {
				return nil, errors.Errorf("unknown format")
			}
		default:
			n, err := fmt.Sscan(s.Text(), &i, &j, &vr, &vi)
			if err != nil {
				return nil, errors.Wrap(err, "")
			}
			if n != 4 {
				return nil, errors.Errorf("unknown format")
			}
		}
		if i < 1 || nr < i {
			return nil, errors.Errorf("unknown format")
		}
		if j < 1 || nc < j {
			return nil, errors.Errorf("unknown format")
		}

		t.SetAt([]int{i - 1, j - 1}, complex(float32(vr), float32(vi)))
		if structure == symmetric && i != j {
			t.SetAt([]int{j - 1, i - 1}, complex(float32(vr), float32(vi)))
		}
		if structure == hermitian && i != j {
			t.SetAt([]int{j - 1, i - 1}, complex(float32(vr), -float32(vi)))
		}
	}

	return t, nil
}
