// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a float32 matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []float32
	States [][]float32
}

// NewMatrix32 creates a new float32 matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float32, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]float32, states)
		for i := range m.States {
			m.States[i] = make([]float32, cols*rows)
		}
	}
	return m
}

// Size is the size of the float32 matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

// MulT multiplies two matrices and computes the transpose
func MulT(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float32, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float32 matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two float32 matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func Sigmoid(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, float32(1/(1+math.Exp(-float64(value)))))
	}
	return o
}

// Step computes the step function of a float32 matrix
func Step(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		if value > 0 {
			value = 1
		} else {
			value = -1
		}
		o.Data = append(o.Data, value)
	}
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// Normalize normalizes a matrix to the unit vector
func Normalize(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := float32(0.0)
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := float32(math.Sqrt(float64(sum)))
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Rows,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		softmax(outputs)
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) ([]Matrix, []float32) {
	entropies, values, results := make([]float32, V.Cols), make([]float32, K.Rows), make([]float32, 0, K.Rows)
	outputs := make([]Matrix, 0, 8)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		output := NewMatrix(0, V.Cols, 1)
		output.Data = append(output.Data, entropies...)
		outputs = append(outputs, output)
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += float64(e) * math.Log(float64(e))
		}
		results = append(results, float32(-entropy))
	}
	return outputs, results
}

// EverettActivation is the everett complex activation function
func EverettActivation(m Matrix) Matrix {
	o := Matrix{
		Cols: 2 * m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, 2*m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		min, max := value, value
		if min > 0 {
			min = 0
		}
		if max < 0 {
			max = 0
		}
		o.Data = append(o.Data, min, max)
	}
	return o
}

// TaylorSoftmax is the taylor softmax
// https://arxiv.org/abs/1511.05042
func TaylorSoftmax(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	var sum float32
	columns, lenm := m.Cols, len(m.Data)
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			sum += 1 + v + v*v/2
		}
	}
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			o.Data = append(o.Data, (1+v+v*v/2)/sum)
		}
	}
	return o
}

func InBetween(i, min, max int) bool {
	if (i >= min) && (i <= max) {
		return true
	} else {
		return false
	}
}

type Mat [][]float64

func (m Mat) Print() {
	for _, row := range m {
		for _, el := range row {
			fmt.Printf("%f ", el)
		}
		fmt.Println("")
	}
}

func (m Mat) Rows() int {
	return len(m)
}

func (m Mat) Columns() int {
	return len(m[0])
}

func (m Mat) ExcludeColumn(column int) (Mat, error) {

	if !InBetween(column, 1, m.Columns()) {
		return Mat{}, fmt.Errorf("input not in range")
	}

	result := make(Mat, m.Rows())
	for i, row := range m {
		for j, el := range row {
			if j == column-1 {
				continue
			}
			result[i] = append(result[i], el)
		}
	}
	return result, nil
}

func (m Mat) ExcludeRow(row int) (Mat, error) {
	if !InBetween(row, 1, m.Rows()) {
		return Mat{}, fmt.Errorf("input not in range")
	}

	var result Mat
	for i, r := range m {
		if i == row-1 {
			continue
		}
		result = append(result, r)
	}
	return result, nil
}

func (m Mat) SubMatrix(column_start int, column_end int, row_start int, row_end int) (Mat, error) {

	if !InBetween(column_start, 1, column_end) ||
		!InBetween(column_end, column_start, m.Columns()) ||
		!InBetween(row_start, 1, row_end) ||
		!InBetween(row_end, row_start, m.Rows()) {
		return Mat{}, fmt.Errorf("inputs not in range")
	}

	partial_matrix := m[row_start+1 : row_end+1]
	for i, row := range partial_matrix {
		partial_matrix[i] = row[column_start+1 : column_end+1]
	}

	return partial_matrix, nil
}

func (m Mat) IsMatrix() bool {

	if m.Rows() == 0 {
		return false
	}

	for _, row := range m {

		if len(m[0]) != len(row) {
			return false
		}

	}

	return true
}

func (m Mat) IsSquare() bool {
	return m.Columns() == m.Rows()
}

func (m Mat) Det() (float64, error) {

	if !m.IsMatrix() || !m.IsSquare() {
		return -1, fmt.Errorf("determinant is not defined for the input [Matrix: %t][Square: %t]",
			m.IsMatrix(), m.IsSquare())
	}

	if m.Rows() == 2 {
		return m[0][0]*m[1][1] - m[0][1]*m[1][0], nil
	}

	// if rows are more than 2
	partial_matrix, err := m.ExcludeRow(1)
	if err != nil {
		return -1, err
	}

	var temp float64 = 0
	for i, el := range m[0] {

		reduced_matrix, err := partial_matrix.ExcludeColumn(i + 1)
		if err != nil {
			return -1, err
		}

		partial_det, err := reduced_matrix.Det()
		if err != nil {
			return -1, err
		}
		temp = temp + partial_det*el*math.Pow(-1, float64(i))
	}

	return temp, nil
}

func Determinant(a Matrix) (float32, error) {
	b := make(Mat, a.Rows)
	for i := range b {
		c := make([]float64, a.Cols)
		for j := range c {
			c[j] = float64(a.Data[i*a.Cols+j])
		}
		b[i] = c
	}
	det, err := b.Det()
	return float32(det), err
}

// Multi is a multivariate distribution
type Multi struct {
	A Matrix
	U []float32
}

// NewMulti make a new multi
func NewMulti(vars int) Multi {
	factor := float32(math.Sqrt(2.0 / float64(vars)))
	a := NewMatrix(0, vars, vars)
	a.Data = a.Data[:cap(a.Data)]
	for i := 0; i < vars; i++ {
		for j := 0; j < vars; j++ {
			if i == j {
				a.Data[i*vars+j] = factor
			}
		}
	}
	u := make([]float32, vars)
	return Multi{
		A: a,
		U: u,
	}
}

// Factor factores a matrix into AA^T
func Factor(vars [][]float32, debug bool) Multi {
	rng := rand.New(rand.NewSource(1))
	length := len(vars)
	set := tf32.NewSet()
	set.Add("A", length, length)
	set.Add("E", length, length)

	e := set.Weights[1]
	e.X = e.X[:cap(e.X)]
	mu := make([]float32, length)
	for i, v := range vars {
		for _, vv := range v {
			mu[i] += vv
		}
	}
	size := len(vars[0])
	for i := range mu {
		mu[i] /= float32(size)
	}
	for i := 0; i < length; i++ {
		for j := i; j < length; j++ {
			for k := 0; k < size; k++ {
				e.X[i*length+j] += (vars[i][k] - mu[i]) * (vars[j][k] - mu[j])
			}
			e.X[i*length+j] /= float32(size)
		}
	}
	for i := 0; i < length; i++ {
		for j := i + 1; j < length; j++ {
			e.X[j*length+i] = e.X[i*length+j]
		}
	}

	for _, w := range set.Weights[:1] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	cost := tf32.Avg(tf32.Quadratic(tf32.Mul(set.Get("A"), tf32.T(set.Get("A"))), set.Get("E")))
	alpha, eta, iterations := float32(.01), float32(.01), 8*2048
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}

		w := set.Weights[0]
		for k, d := range w.D {
			deltas[0][k] = alpha*deltas[0][k] - eta*d*scaling
			set.Weights[0].X[k] += deltas[0][k]
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if debug {
			fmt.Println(i, total)
		}
		i++
	}

	if debug {
		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
		if err != nil {
			panic(err)
		}
	}

	a := NewMatrix(0, set.Weights[0].S[0], set.Weights[0].S[1])
	for _, v := range set.Weights[0].X {
		a.Data = append(a.Data, v)
	}

	return Multi{
		A: a,
		U: mu,
	}
}

// Sample samples from the multivariate distribution
func (m Multi) Sample(rng *rand.Rand) []float32 {
	s := NewMatrix(0, len(m.U), 1)
	for i := 0; i < len(m.U); i++ {
		s.Data = append(s.Data, float32(rng.NormFloat64()))
	}
	s = MulT(m.A, s)
	for i := range s.Data {
		s.Data[i] += m.U[i]
	}
	return s.Data
}
