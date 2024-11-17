package mat

import (
	"context"
	"database/sql"
	"encoding/csv"
	"fmt"
	"io"
	"math/cmplx"
	"os"
	"path/filepath"
	"strconv"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

const (
	tableMatrix = "m"
)

type DiskMatrix struct {
	Path string
	rows int
	cols int

	db *sql.DB
}

func DiskM(dbPath string, dense [][]complex64) *DiskMatrix {
	m, err := diskM(dbPath, dense)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return m
}

func diskM(dbPath string, dense [][]complex64) (*DiskMatrix, error) {
	m := &DiskMatrix{Path: dbPath, rows: len(dense), cols: len(dense[0])}
	var err error
	m.db, err = newDB(m.Path)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	for i, row := range dense {
		for j, v := range row {
			if err := setItem(ctx, m.db, i, j, v); err != nil {
				return nil, errors.Wrap(err, "")
			}
		}
	}

	return m, nil
}

func (m *DiskMatrix) Close() error {
	var err error
	if err1 := m.db.Close(); err1 != nil && err == nil {
		err = err1
	}
	if err1 := os.Remove(m.Path); err1 != nil && err == nil {
		err = err1
	}
	return err
}

func (m *DiskMatrix) Zeros(rows, cols int) {
	m.rows, m.cols = rows, cols
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := deleteAll(ctx, m.db); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
}

func (m *DiskMatrix) Scalar(v complex64) {
	m.rows, m.cols = 1, 1
	if err := m.scalar(v); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
}

func (m *DiskMatrix) scalar(v complex64) error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := deleteAll(ctx, m.db); err != nil {
		return errors.Wrap(err, "")
	}
	if err := setItem(ctx, m.db, 0, 0, v); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (m *DiskMatrix) Rows() int { return m.rows }
func (m *DiskMatrix) Cols() int { return m.cols }

func (m *DiskMatrix) At(i, j int) complex64 {
	v, err := m.at(i, j)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return v
}

func (a *DiskMatrix) COO() *COO {
	b, err := a.coo()
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return b
}

func (a *DiskMatrix) coo() (*COO, error) {
	b := &COO{rows: a.rows, cols: a.cols, Data: make([]vRowCol, 0), m: make(map[[2]int]complex64)}

	ctx, cancel := context.WithTimeout(context.Background(), 48*time.Hour)
	defer cancel()
	sqlStr := fmt.Sprintf(`SELECT i, j, re, im FROM %s ORDER BY i, j`, tableMatrix)
	rows, err := a.db.QueryContext(ctx, sqlStr)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer rows.Close()

	for rows.Next() {
		var i, j int
		var re, im float32
		if err := rows.Scan(&i, &j, &re, &im); err != nil {
			return nil, errors.Wrap(err, "")
		}
		v := complex(re, im)

		b.Data = append(b.Data, vRowCol{v: v, row: i, col: j})
	}
	if err := rows.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}

	return b, nil
}

func (m *DiskMatrix) at(i, j int) (complex64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	sqlStr := fmt.Sprintf(`SELECT re, im FROM %s WHERE i=? AND j=?`, tableMatrix)
	var re, im float32
	err := m.db.QueryRowContext(ctx, sqlStr, i, j).Scan(&re, &im)
	switch {
	case err == sql.ErrNoRows:
		return 0, nil
	case err != nil:
		return complex64(cmplx.NaN()), errors.Wrap(err, "")
	default:
		return complex(re, im), nil
	}
}

func (a *DiskMatrix) Add(c complex64, b Matrix) {
	if err := a.add(c, b); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
}

func (a *DiskMatrix) add(c complex64, bMatrix Matrix) error {
	b := bMatrix.(*DiskMatrix)
	ctx, cancel := context.WithTimeout(context.Background(), 48*time.Hour)
	defer cancel()
	sqlStr := fmt.Sprintf(`SELECT i, j, re, im FROM %s ORDER BY i, j`, tableMatrix)
	rows, err := b.db.QueryContext(ctx, sqlStr)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer rows.Close()

	for rows.Next() {
		var i, j int
		var re, im float32
		if err := rows.Scan(&i, &j, &re, &im); err != nil {
			return errors.Wrap(err, "")
		}
		bv := complex(re, im)
		av := a.At(i, j)
		if err := setItem(ctx, a.db, i, j, av+c*bv); err != nil {
			return errors.Wrap(err, "")
		}
	}
	if err := rows.Err(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (a *DiskMatrix) Kron(b *COO) {
	if err := a.kron(b); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
}

func (a *DiskMatrix) kron(b *COO) error {
	rows := a.rows * b.rows
	cols := a.cols * b.cols
	a.rows, a.cols = rows, cols

	dir, err := os.MkdirTemp("", "")
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer os.RemoveAll(dir)

	if err := a.WriteCOO(dir); err != nil {
		return errors.Wrap(err, "")
	}
	cooReader, err := NewCOOReader(dir)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer cooReader.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 48*time.Hour)
	defer cancel()
	sqlStr := fmt.Sprintf(`DELETE FROM %s`, tableMatrix)
	if _, err := a.db.ExecContext(ctx, sqlStr); err != nil {
		return errors.Wrap(err, fmt.Sprintf("db %s", a.Path))
	}

	for {
		av, err := cooReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return errors.Wrap(err, "")
		}

		for _, bv := range b.Data {
			ky := av.row*b.rows + bv.row
			kx := av.col*b.cols + bv.col
			if err := setItem(ctx, a.db, ky, kx, av.v*bv.v); err != nil {
				return errors.Wrap(err, "")
			}
		}
	}
	return nil
}

func (m *DiskMatrix) NumNonZero() int {
	n, err := m.numNonZero()
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return n
}

func (m *DiskMatrix) numNonZero() (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	sqlStr := fmt.Sprintf("SELECT count(1) FROM %s", tableMatrix)
	var n int
	if err := m.db.QueryRowContext(ctx, sqlStr).Scan(&n); err != nil {
		return -1, errors.Wrap(err, "")
	}
	return n, nil
}

func (m *DiskMatrix) WriteCOO(dir string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 48*time.Hour)
	defer cancel()

	shapePath := filepath.Join(dir, FnameShape)
	if err := os.WriteFile(shapePath, []byte(fmt.Sprintf("%d,%d", m.rows, m.cols)), 0644); err != nil {
		return errors.Wrap(err, "")
	}

	sqlStr := fmt.Sprintf(`SELECT i, j, re, im FROM %s ORDER BY i, j`, tableMatrix)
	rows, err := m.db.QueryContext(ctx, sqlStr)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer rows.Close()

	cooPath := filepath.Join(dir, FnameCOO)
	cooF, err := os.Create(cooPath)
	if err != nil {
		return errors.Wrap(err, "")
	}
	w := csv.NewWriter(cooF)

	for rows.Next() {
		var i, j int
		var re, im float32
		if err1 := rows.Scan(&i, &j, &re, &im); err1 != nil && err == nil {
			err = errors.Wrap(err1, "")
			break
		}
		v := complex(re, im)

		if err1 := w.Write([]string{FormatNumpy(v), strconv.Itoa(i), strconv.Itoa(j)}); err1 != nil && err == nil {
			err = errors.Wrap(err1, "")
			break
		}
	}
	if err1 := rows.Err(); err1 != nil && err == nil {
		err = errors.Wrap(err1, "")
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

func setItemMust(ctx context.Context, db *sql.DB, i, j int, v complex64) {
	if err := setItem(ctx, db, i, j, v); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
}

func setItem(ctx context.Context, db *sql.DB, i, j int, v complex64) error {
	sqlStr := fmt.Sprintf(`INSERT OR REPLACE INTO %s (i, j, re, im) VALUES (?, ?, ?, ?)`, tableMatrix)
	args := []any{i, j, real(v), imag(v)}
	if v == 0 {
		sqlStr = fmt.Sprintf(`DELETE FROM %s WHERE i=? AND j=?`, tableMatrix)
		args = []any{i, j}
	}
	if _, err := db.ExecContext(ctx, sqlStr, args...); err != nil {
		return errors.Wrap(err, fmt.Sprintf("%s %#v", sqlStr, args))
	}
	return nil
}

func newDBMust(dbPath string) *sql.DB {
	db, err := newDB(dbPath)
	if err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	return db
}

func newDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", fmt.Sprintf("file:%s", dbPath))
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	if err := prepareDB(db); err != nil {
		db.Close()
		return nil, errors.Wrap(err, "")
	}

	return db, nil
}

func prepareDB(db *sql.DB) error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	sqlStr := fmt.Sprintf(`DROP TABLE IF EXISTS %s`, tableMatrix)
	if _, err := db.ExecContext(ctx, sqlStr); err != nil {
		return errors.Wrap(err, "")
	}
	sqlStr = fmt.Sprintf(`CREATE TABLE %s (i INTEGER, j INTEGER, re REAL, im REAL, PRIMARY KEY (i, j)) STRICT`, tableMatrix)
	if _, err := db.ExecContext(ctx, sqlStr); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func deleteAll(ctx context.Context, db *sql.DB) error {
	sqlStr := fmt.Sprintf(`DELETE FROM %s`, tableMatrix)
	if _, err := db.ExecContext(ctx, sqlStr); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
