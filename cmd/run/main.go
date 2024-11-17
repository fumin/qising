package main

import (
	"encoding/csv"
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
)

const (
	fnameEigen         = "eig.csv"
	fnameDone          = "done.txt"
	fnameMagnetization = "magnetization.txt"
)

var (
	runDir = flag.String("d", filepath.Join("runs", "qising"), "run directory")
)

func magnetize(dir string, n [2]int) error {
	vvs, err := readEig(dir)
	if err != nil {
		return errors.Wrap(err, "")
	}
	ground := vvs[0]

	m, err := qising.Magnetization(n, ground.Vec)
	if err != nil {
		return errors.Wrap(err, "")
	}

	mPath := filepath.Join(dir, fnameMagnetization)
	if err := os.WriteFile(mPath, []byte(strconv.FormatFloat(m, 'f', -1, 64)), 0644); err != nil {
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
	vv := qising.EigsDir(tmpDir)

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
	if err := magnetize(dir, n); err != nil {
		return errors.Wrap(err, "")
	}

	if err := os.WriteFile(donePath, nil, 0644); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

type nhMagnetization struct {
	n [2]int
	h complex64
	m float32
}

func gather(dir string) ([]nhMagnetization, error) {
	nhms := make([]nhMagnetization, 0)
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
			mstr, err := os.ReadFile(filepath.Join(hdir, fnameMagnetization))
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%#v %#v", nent, hent))
			}
			m, err := strconv.ParseFloat(string(mstr), 64)
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%#v %#v", nent, hent))
			}
			nhms = append(nhms, nhMagnetization{n: n, h: h, m: float32(m)})
		}
	}
	return nhms, nil
}

func readEig(dir string) ([]qising.ValVec, error) {
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
	vvs := make([]qising.ValVec, len(record))
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

func writeEig(dir string, vvs []qising.ValVec) error {
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
	appendConfigs := func(configs []nhMagnetization, c dimtc) []nhMagnetization {
		tcLog := math.Log10(c.tcGuess)
		for i := 2; i <= maxL; i++ {
			n := [2]int{i * i, 1}
			if c.dimension == 2 {
				n = [2]int{i, i}
			}

			hLogs := []float64{-2, -1.5, -1, 1, 1.5, 2}
			for _, hl := range []float64{0.1, 0.2, 0.3, 0.4, 0.5} {
				hLogs = append(hLogs, tcLog+hl)
				hLogs = append(hLogs, tcLog-hl)
			}

			for _, hl := range hLogs {
				h := complex(float32(math.Pow(10, hl)), 0)
				configs = append(configs, nhMagnetization{n: n, h: h})
			}
		}
		return configs
	}

	configs := make([]nhMagnetization, 0)
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
	magnetizations, err := gather(*runDir)
	if err != nil {
		return errors.Wrap(err, "")
	}
	fmt.Printf("n0,n1,h,m\n")
	for _, m := range magnetizations {
		fmt.Printf("%d,%d,%f,%f\n", m.n[0], m.n[1], real(m.h), m.m)
	}
	return nil
}
