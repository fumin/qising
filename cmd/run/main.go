package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/pkg/errors"

	"qising"
	"qising/mat"
)

const (
	fnameEigen      = "eig.csv"
	fnameDone       = "done.txt"
	fnameStatistics = "statistics.txt"
)

var (
	runDir = flag.String("d", filepath.Join("runs", "qising"), "run directory")
)

type Statistics struct {
	n [2]int
	h complex64
	qising.Statistics
}

func getStatistics(dir string, n [2]int) error {
	vvs, err := readEig(dir)
	if err != nil {
		return errors.Wrap(err, "")
	}

	stats, err := qising.GetStatistics(n, vvs)
	if err != nil {
		return errors.Wrap(err, "")
	}

	b, err := json.Marshal(stats)
	if err != nil {
		return errors.Wrap(err, "")
	}
	mPath := filepath.Join(dir, fnameStatistics)
	if err := os.WriteFile(mPath, b, 0644); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func solveGround(dir string, n [2]int, h complex64) error {
	tmpDir, err := os.MkdirTemp("", "")
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer os.RemoveAll(tmpDir)

	qising.TransverseFieldIsingExplicit(tmpDir, n, h)
	vv := mat.EigsDir(tmpDir)

	if err := writeEig(dir, vv); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func solve(dir string, n [2]int, h complex64) error {
	donePath := filepath.Join(dir, fnameDone)
	if _, err := os.Stat(donePath); err == nil {
		return nil
	}
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	if err := solveGround(dir, n, h); err != nil {
		return errors.Wrap(err, "")
	}
	if err := getStatistics(dir, n); err != nil {
		return errors.Wrap(err, "")
	}

	if err := os.WriteFile(donePath, nil, 0644); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func gather(dir string) ([]Statistics, error) {
	stats := make([]Statistics, 0)
	nEntries, err := os.ReadDir(dir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	for _, nent := range nEntries {
		// Parse for lattice size.
		nstr := strings.Split(nent.Name(), "x")
		var n [2]int
		for i, s := range nstr {
			n[i], err = strconv.Atoi(s)
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%#v", nent))
			}
		}

		ndir := filepath.Join(dir, nent.Name())
		hEntries, err := os.ReadDir(ndir)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("%#v", nent))
		}
		for _, hent := range hEntries {
			hf, err := strconv.ParseFloat(hent.Name(), 64)
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%#v %#v", nent, hent))
			}
			h := complex(float32(hf), 0)

			hdir := filepath.Join(ndir, hent.Name())
			sb, err := os.ReadFile(filepath.Join(hdir, fnameStatistics))
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%#v %#v", nent, hent))
			}
			s := Statistics{n: n, h: h}
			if err := json.Unmarshal(sb, &s); err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%#v %#v", nent, hent))
			}
			stats = append(stats, s)
		}
	}
	return stats, nil
}

func readEig(dir string) ([]mat.ValVec, error) {
	fpath := filepath.Join(dir, fnameEigen)
	f, err := os.Open(fpath)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer f.Close()
	r := csv.NewReader(f)
	rowI := -1

	record, err := r.Read()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	vvs := make([]mat.ValVec, len(record))
	for j, s := range record {
		v, err := strconv.ParseComplex(s, 128)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		vvs[j].Val = v
	}

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		rowI++

		for j, s := range record {
			v, err := strconv.ParseComplex(s, 128)
			if err != nil {
				return nil, errors.Wrap(err, "")
			}
			vvs[j].Vec = append(vvs[j].Vec, v)
		}
	}

	return vvs, nil
}

func writeEig(dir string, vvs []mat.ValVec) error {
	fpath := filepath.Join(dir, fnameEigen)
	f, err := os.Create(fpath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	w := csv.NewWriter(f)

	row := make([]string, len(vvs))
	for j, vv := range vvs {
		row[j] = strconv.FormatComplex(vv.Val, 'f', -1, 128)
	}
	if err1 := w.Write(row); err1 != nil && err == nil {
		err = errors.Wrap(err1, "")
	}
	for i := range len(vvs[0].Vec) {
		for j, vv := range vvs {
			row[j] = strconv.FormatComplex(vv.Vec[i], 'f', -1, 128)
		}
		if err1 := w.Write(row); err1 != nil && err == nil {
			err = errors.Wrap(err1, "")
			break
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

func main() {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds | log.Llongfile | log.LstdFlags)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	if err := os.MkdirAll(*runDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	// Maximum lattice size.
	const maxL = 5
	type dimtc struct {
		dimension int
		tcGuess   float64
	}
	appendConfigs := func(configs []Statistics, c dimtc) []Statistics {
		tcLog := math.Log10(c.tcGuess)
		for i := 2; i <= maxL; i++ {
			n := [2]int{i * i, 1}
			if c.dimension == 2 {
				n = [2]int{i, i}
			}

			hLogs := []float64{-2, -1.5, -1, 1, 1.5, 2}
			for _, hl := range []float64{0.05, 0.1, 0.2, 0.3, 0.4, 0.5} {
				hLogs = append(hLogs, tcLog+hl)
				hLogs = append(hLogs, tcLog-hl)
			}

			for _, hl := range hLogs {
				h := complex(float32(math.Pow(10, hl)), 0)
				configs = append(configs, Statistics{n: n, h: h})
			}
		}
		return configs
	}

	configs := make([]Statistics, 0)
	configs = appendConfigs(configs, dimtc{dimension: 1, tcGuess: 1})
	configs = appendConfigs(configs, dimtc{dimension: 2, tcGuess: 2})

	// Solve for the hamiltonian.
	for _, c := range configs {
		nstr := fmt.Sprintf("%dx%d", c.n[0], c.n[1])
		hstr := fmt.Sprintf("%f", real(c.h))
		dir := filepath.Join(*runDir, nstr, hstr)

		if err := solve(dir, c.n, c.h); err != nil {
			return errors.Wrap(err, fmt.Sprintf("%d %f", c.n, c.h))
		}
		log.Printf("%v %f", c.n, real(c.h))
	}

	// Gather results and print them.
	stats, err := gather(*runDir)
	if err != nil {
		return errors.Wrap(err, "")
	}
	fmt.Printf("n0,n1,h,e0,e1,e2,m,binder\n")
	for _, s := range stats {
		fmt.Printf("%d,%d,%f,%f,%f,%f,%f,%f\n", s.n[0], s.n[1], real(s.h), s.EigenValue[0], s.EigenValue[1], s.EigenValue[2], s.Magnetization, s.BinderCumulant)
	}
	return nil
}
