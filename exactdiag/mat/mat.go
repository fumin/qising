package mat

import (
	"cmp"
	_ "embed"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

const (
	FnameShape = "shape.csv"
	FnameCOO   = "coo.csv"
)

var (
	PauliX = [][]complex64{
		{0, 1},
		{1, 0},
	}
	PauliY = [][]complex64{
		{0, -1i},
		{1i, 0},
	}
	PauliZ = [][]complex64{
		{1, 0},
		{0, -1},
	}
)

type Matrix interface {
	Zeros(int, int)
	Scalar(complex64)
	Rows() int
	Cols() int

	Add(complex64, Matrix)
	Kron(*COO)
	COO() *COO

	WriteCOO(string) error
}

type vRowCol struct {
	v   complex64
	row int
	col int
}

type COO struct {
	rows int
	cols int
	Data []vRowCol

	m map[[2]int]complex64
}

func M(dense [][]complex64) *COO {
	m := &COO{rows: len(dense), cols: len(dense[0]), Data: make([]vRowCol, 0), m: make(map[[2]int]complex64)}
	for i, row := range dense {
		for j, v := range row {
			if v == 0 {
				continue
			}
			m.Data = append(m.Data, vRowCol{v: v, row: i, col: j})
		}
	}
	return m
}

func COOZeros(rows, cols int) *COO {
	m := M([][]complex64{{0}})
	m.Zeros(rows, cols)
	return m
}

func COOIdentity(rows int) *COO {
	m := M([][]complex64{{0}})
	m.Zeros(rows, rows)
	for i := 0; i < rows; i++ {
		m.Data = append(m.Data, vRowCol{v: 1, row: i, col: i})
	}
	return m
}

func (m *COO) Rows() int { return m.rows }
func (m *COO) Cols() int { return m.cols }

func (m *COO) Zeros(rows, cols int) {
	m.rows, m.cols = rows, cols
	m.Data = m.Data[:0]
}

func (m *COO) Scalar(v complex64) {
	m.rows, m.cols = 1, 1
	m.Data = m.Data[:0]
	m.Data = append(m.Data, vRowCol{v: v, row: 0, col: 0})
}

func (a *COO) Equal(b *COO) bool {
	if a.rows != b.rows {
		return false
	}
	if a.cols != b.cols {
		return false
	}
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i, av := range a.Data {
		bv := b.Data[i]
		if av != bv {
			return false
		}
	}
	return true
}

func (m *COO) Slice(yBoundN, xBoundN [2]int) *COO {
	yBound, xBound := yBoundN, xBoundN
	for i := 0; i < 2; i++ {
		if yBound[i] < 0 {
			yBound[i] += m.rows
		}
		if xBound[i] < 0 {
			xBound[i] += m.cols
		}
	}

	s := &COO{rows: yBound[1] - yBound[0], cols: xBound[1] - xBound[0], Data: make([]vRowCol, 0)}
	for _, v := range m.Data {
		if v.row < yBound[0] {
			continue
		}
		if v.row >= yBound[1] {
			break
		}
		if v.col < xBound[0] || v.col >= xBound[1] {
			continue
		}
		s.Data = append(s.Data, vRowCol{v: v.v, row: v.row - yBound[0], col: v.col - xBound[0]})
	}
	return s
}

func (a *COO) Add(c complex64, bMatrix Matrix) {
	b := bMatrix.COO()
	clear(b.m)
	for _, v := range b.Data {
		b.m[[2]int{v.row, v.col}] = v.v
	}

	for i, av := range a.Data {
		var byx [2]int
		switch {
		case b.rows == 1 && b.cols == 1:
		case b.rows == a.rows && b.cols == 1:
			byx[0] = av.row
		case b.rows == a.rows && b.cols == a.cols:
			byx[0], byx[1] = av.row, av.col
		default:
			panic(fmt.Sprintf("wrong dimensions"))
		}
		bv := b.m[byx]
		delete(b.m, byx)

		a.Data[i].v = av.v + c*bv
	}

	a.Data = slices.DeleteFunc(a.Data, func(v vRowCol) bool {
		return v.v == 0
	})
	for yx, bv := range b.m {
		a.Data = append(a.Data, vRowCol{v: c * bv, row: yx[0], col: yx[1]})
	}
	slices.SortFunc(a.Data, rowMajor)
	clear(b.m)
}

func (a *COO) Mul(b *COO) {
	clear(b.m)
	for _, v := range b.Data {
		b.m[[2]int{v.row, v.col}] = v.v
	}

	for i, av := range a.Data {
		var byx [2]int
		switch {
		case b.rows == 1 && b.cols == 1:
		case b.rows == a.rows && b.cols == 1:
			byx[0] = av.row
		case b.rows == a.rows && b.cols == a.cols:
			byx[0], byx[1] = av.row, av.col
		default:
			panic(fmt.Sprintf("wrong dimensions"))
		}
		bv := b.m[byx]

		a.Data[i].v = av.v * bv
	}

	a.Data = slices.DeleteFunc(a.Data, func(v vRowCol) bool {
		return v.v == 0
	})
	clear(b.m)
}

func (a *COO) Kron(b *COO) {
	rows := a.rows * b.rows
	cols := a.cols * b.cols
	a.rows, a.cols = rows, cols

	prevElemNum := len(a.Data)
	for i := prevElemNum - 1; i >= 0; i-- {
		av := a.Data[i]
		a.Data[i].v = 0
		for _, bv := range b.Data {
			ky := av.row*b.rows + bv.row
			kx := av.col*b.cols + bv.col
			a.Data = append(a.Data, vRowCol{v: av.v * bv.v, row: ky, col: kx})
		}
	}

	a.Data = slices.DeleteFunc(a.Data, func(v vRowCol) bool {
		return v.v == 0
	})
	slices.SortFunc(a.Data, rowMajor)
}

func (m *COO) COO() *COO {
	return m
}

func (m *COO) Dense() [][]complex64 {
	dense := make([][]complex64, m.rows)
	for i := range dense {
		dense[i] = make([]complex64, m.cols)
	}

	for _, v := range m.Data {
		dense[v.row][v.col] = v.v
	}

	return dense
}

func (m *COO) WriteCOO(dir string) error {
	shapePath := filepath.Join(dir, FnameShape)
	if err := os.WriteFile(shapePath, []byte(fmt.Sprintf("%d,%d", m.rows, m.cols)), 0644); err != nil {
		return errors.Wrap(err, "")
	}

	cooPath := filepath.Join(dir, FnameCOO)
	cooF, err := os.Create(cooPath)
	if err != nil {
		return errors.Wrap(err, "")
	}

	w := csv.NewWriter(cooF)
	for _, v := range m.Data {
		if err1 := w.Write([]string{FormatNumpy(v.v), strconv.Itoa(v.row), strconv.Itoa(v.col)}); err1 != nil && err == nil {
			err = errors.Wrap(err1, "")
			break
		}
	}
	w.Flush()
	if err1 := w.Error(); err1 != nil && err == nil {
		err = errors.Wrap(err1, "")
	}

	if err1 := cooF.Close(); err1 != nil && err == nil {
		err = errors.Wrap(err1, "")
	}
	return err
}

type COOReader struct {
	f *os.File
	r *csv.Reader
	i int

	prev vRowCol
}

func NewCOOReader(dir string) (*COOReader, error) {
	r := &COOReader{i: -1}

	cooPath := filepath.Join(dir, "coo.csv")
	var err error
	r.f, err = os.Open(cooPath)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	r.r = csv.NewReader(r.f)
	return r, nil
}

func (r *COOReader) Close() error {
	return r.f.Close()
}

func (r *COOReader) Read() (vRowCol, error) {
	r.i++
	record, err := r.r.Read()
	if err == io.EOF {
		return vRowCol{}, io.EOF
	}
	if err != nil {
		return vRowCol{}, errors.Wrap(err, fmt.Sprintf("%d", r.i))
	}
	if len(record) != 3 {
		return vRowCol{}, errors.Errorf("%d %#v", r.i, record)
	}

	var vrc vRowCol
	switch {
	case record[0] == "":
		vrc.v = r.prev.v
	default:
		s := strings.ReplaceAll(record[0], "j", "i")
		v, err := strconv.ParseComplex(s, 128)
		if err != nil {
			return vRowCol{}, errors.Wrap(err, fmt.Sprintf("%d %#v", r.i, record))
		}
		vrc.v = complex64(v)
	}

	switch {
	case record[1] == "":
		vrc.row = r.prev.row
	default:
		vrc.row, err = strconv.Atoi(record[1])
		if err != nil {
			return vRowCol{}, errors.Wrap(err, fmt.Sprintf("%d %#v", r.i, record))
		}
	}

	vrc.col, err = strconv.Atoi(record[2])
	if err != nil {
		return vRowCol{}, errors.Wrap(err, fmt.Sprintf("%d %#v", r.i, record))
	}

	r.prev = vrc
	return vrc, nil
}

func ReadCOO(dir string) (*COO, error) {
	m := M([][]complex64{{0}})
	m.Data = m.Data[:0]
	var err error
	m.rows, m.cols, err = readShape(dir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	r, err := NewCOOReader(dir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer r.Close()
	for {
		v, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.Wrap(err, "")
		}

		m.Data = append(m.Data, v)
	}

	return m, nil
}

func readShape(dir string) (int, int, error) {
	f, err := os.Open(filepath.Join(dir, "shape.csv"))
	if err != nil {
		return -1, -1, errors.Wrap(err, "")
	}
	defer f.Close()

	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return -1, -1, errors.Wrap(err, "")
	}
	if len(records) == 0 {
		return -1, -1, errors.Errorf("empty")
	}
	row := records[0]

	if len(row) != 2 {
		return -1, -1, errors.Errorf("%#v", row)
	}
	i, err := strconv.Atoi(row[0])
	if err != nil {
		return -1, -1, errors.Wrap(err, fmt.Sprintf("%#v", row))
	}
	j, err := strconv.Atoi(row[1])
	if err != nil {
		return -1, -1, errors.Wrap(err, fmt.Sprintf("%#v", row))
	}

	return i, j, nil
}

func (m *COO) String() string {
	clear(m.m)
	for _, v := range m.Data {
		m.m[[2]int{v.row, v.col}] = v.v
	}

	lines := []string{}
	for i := 0; i < m.rows; i++ {
		cs := []string{}
		for j := 0; j < m.cols; j++ {
			v := m.m[[2]int{i, j}]
			switch {
			case imag(v) == 0:
				cs = append(cs, format(real(v)))
			case real(v) == 0:
				cs = append(cs, format(imag(v))+"i")
			default:
				cs = append(cs, format(real(v))+"+"+format(imag(v))+"i")
			}
		}
		l := strings.Join(cs, "\t")
		lines = append(lines, l)
	}

	clear(m.m)
	return strings.Join(lines, "\n")
}

type ValVec struct {
	Val complex128
	Vec []complex128
}

func (m *COO) Eigen() []ValVec {
	gnm := mat.NewDense(m.rows, m.cols, nil)
	gnm.Zero()
	for _, v := range m.Data {
		if imag(v.v) != 0 {
			panic("not real")
		}
		gnm.Set(v.row, v.col, float64(real(v.v)))
	}

	var eig mat.Eigen
	ok := eig.Factorize(gnm, mat.EigenRight)
	if !ok {
		panic("eig.Factorize failed")
	}
	vals := eig.Values(nil)
	vecs := mat.NewCDense(m.rows, m.cols, nil)
	eig.VectorsTo(vecs)

	vecsR, _ := vecs.Caps()
	vvs := make([]ValVec, 0, len(vals))
	for i, v := range vals {
		vec := make([]complex128, 0, vecsR)
		for j := 0; j < vecsR; j++ {
			vec = append(vec, vecs.At(j, i))
		}
		vvs = append(vvs, ValVec{Val: v, Vec: vec})
	}
	slices.SortFunc(vvs, func(a, b ValVec) int { return cmp.Compare(real(a.Val), real(b.Val)) })

	return vvs
}

func Eigs(m Matrix) []ValVec {
	vv, err := eigs(m)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return vv
}

func eigs(m Matrix) ([]ValVec, error) {
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer os.RemoveAll(dir)

	if err := m.WriteCOO(dir); err != nil {
		return nil, errors.Wrap(err, "")
	}

	vv, err := eigsDir(dir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	return vv, nil
}

func EigsDir(dir string) []ValVec {
	vv, err := eigsDir(dir)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return vv
}

//go:embed eigs.py
var eigsPy []byte

func eigsDir(mDir string) ([]ValVec, error) {
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer os.RemoveAll(dir)

	eigsPyPath := filepath.Join(dir, "eigs.py")
	if err := os.WriteFile(eigsPyPath, eigsPy, 0644); err != nil {
		return nil, errors.Wrap(err, "")
	}

	eigCsvPath := filepath.Join(dir, "eig.csv")
	cmd := exec.Command("python", eigsPyPath, fmt.Sprintf("-coo=%s", mDir), fmt.Sprintf("-eig=%s", eigCsvPath))
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("%s", stdoutStderr))
	}

	f, err := os.Open(eigCsvPath)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer f.Close()
	r := csv.NewReader(f)
	rowI := -1

	rec, err := r.Read()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	buf := make([]complex128, len(rec))
	parseRec := func(rec []string) ([]complex128, error) {
		for j, vStr := range rec {
			s := strings.TrimSpace(vStr)
			s = strings.ReplaceAll(s, "j", "i")
			buf[j], err = strconv.ParseComplex(s, 128)
			if err != nil {
				return nil, errors.Errorf("%d %d %#v", rowI, j, rec)
			}
		}
		return buf, nil
	}
	vs, err := parseRec(rec)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	vvs := make([]ValVec, len(rec))
	for j, v := range vs {
		vvs[j].Val = v
	}

	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		rowI++

		vs, err := parseRec(rec)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		for j, v := range vs {
			vvs[j].Vec = append(vvs[j].Vec, v)
		}
	}

	return vvs, nil
}

func rowMajor(a, b vRowCol) int {
	if c := cmp.Compare(a.row, b.row); c != 0 {
		return c
	}
	return cmp.Compare(a.col, b.col)
}

func format(v float32) string {
	// If v is 0 or -0, return "0" immediately to avoid returning "-0".
	if v == 0 {
		return " 0"
	}

	s := fmt.Sprintf("%v", v)

	// Add a space before non-negative numbers to align with other negative numbers in the same column.
	if v >= 0 {
		s = " " + s
	}

	return s
}

func FormatNumpy(v complex64) string {
	switch {
	case imag(v) == 0:
		return strconv.FormatFloat(float64(real(v)), 'g', -1, 32)
	default:
		s := fmt.Sprintf("%v", v)
		s = strings.ReplaceAll(s, "i", "j")
		return s
	}
}
