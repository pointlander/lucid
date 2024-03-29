// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark2

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/matrix"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// ModelWindow is the window size
	ModelWindow = 32
	// GaussianWindow is the gaussian window
	GaussianWindow = 8
	// ModelSamples is the number of samples
	ModelSamples = 256
	// Inputs is the number of inputs
	Inputs = 4
	// Outputs is the number of outputs
	Outputs = 4
	// Embedding is the embedding size
	Embedding = 3 * 4
	// Clusters is the number of clusters
	Clusters = 3
)

var colors = [...]color.RGBA{
	{R: 0xff, G: 0x00, B: 0x00, A: 255},
	{R: 0x00, G: 0xff, B: 0x00, A: 255},
	{R: 0x00, G: 0x00, B: 0xff, A: 255},
}

// Random is a random variable
type Rand struct {
	Mean   float32
	StdDev float32
	Count  float32
}

// XSet is a set of statistics
type XSet [][]Random

// XNewStatistics generates a new statistics model
func XNewStatistics(inputs, outputs int) XSet {
	statistics := make(XSet, outputs)
	for i := range statistics {
		for j := 0; j < inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 1,
			})
		}
	}
	return statistics
}

// Sample samples from the statistics
func (s XSet) Sample(rng *rand.Rand, inputs, outputs int) []Matrix {
	neurons := make([]Matrix, outputs)
	for j := range neurons {
		neurons[j] = NewMatrix(0, inputs, 1)
		for k := 0; k < inputs; k++ {
			v := float32(rng.NormFloat64())*s[j][k].StdDev + s[j][k].Mean
			if v > 0 {
				v = 1
			} else {
				v = -1
			}
			neurons[j].Data = append(neurons[j].Data, v)
		}
	}
	return neurons
}

// XNet is a net
type XNet struct {
	Inputs  int
	Outputs int
	Rng     *rand.Rand
	Q       XSet
	K       XSet
	V       XSet
}

// XNewNet makes a new network
func XNewNet(seed int64, inputs, outputs int) XNet {
	rng := rand.New(rand.NewSource(seed))
	return XNet{
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		Q:       XNewStatistics(inputs, outputs),
		K:       XNewStatistics(inputs, outputs),
		V:       XNewStatistics(inputs, outputs),
	}
}

// XSample is a sample of a random neural network
type XSample struct {
	Entropy float32
	Neurons []Matrix
	Outputs Matrix
	Out     Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n XNet) CalculateStatistics(systems []XSample) XSet {
	statistics := make(XSet, n.Outputs)
	for i := range statistics {
		for j := 0; j < n.Inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 0,
			})
		}
	}
	weights, sum := make([]float32, ModelWindow), float32(0)
	for i := range weights {
		value := math.Exp(-float64(systems[i].Entropy))
		sum += float32(value)
		weights[i] = float32(value)
	}
	for i := range weights {
		weights[i] /= sum
	}

	for i := range systems[:ModelWindow] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				statistics[j][k].Mean += weights[i] * value
			}
		}
	}
	for i := range systems[:ModelWindow] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := statistics[j][k].Mean - value
				statistics[j][k].StdDev += weights[i] * diff * diff
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].StdDev /= (ModelWindow - 1.0) / ModelWindow
			statistics[i][j].StdDev = float32(math.Sqrt(float64(statistics[i][j].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *XNet) Fire(query, key, value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewMatrix(0, n.Outputs, ModelSamples)
	k := NewMatrix(0, n.Outputs, ModelSamples)
	v := NewMatrix(0, n.Outputs, ModelSamples)
	systemsQ := make([]XSample, 0, 8)
	systemsK := make([]XSample, 0, 8)
	systemsV := make([]XSample, 0, 8)
	for i := 0; i < ModelSamples; i++ {
		neurons := n.Q.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], query)
			q.Data = append(q.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsQ = append(systemsQ, XSample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.K.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], key)
			k.Data = append(k.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsK = append(systemsK, XSample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < ModelSamples; i++ {
		neurons := n.V.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], value)
			v.Data = append(v.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsV = append(systemsV, XSample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	outputs, entropies := SelfEntropy(q, k, v)
	for i, entropy := range entropies {
		systemsQ[i].Entropy = entropy
		systemsQ[i].Out = outputs[i]
		systemsK[i].Entropy = entropy
		systemsK[i].Out = outputs[i]
		systemsV[i].Entropy = entropy
		systemsV[i].Out = outputs[i]
	}
	sort.Slice(systemsQ, func(i, j int) bool {
		return systemsQ[i].Entropy < systemsQ[j].Entropy
	})
	sort.Slice(systemsK, func(i, j int) bool {
		return systemsK[i].Entropy < systemsK[j].Entropy
	})
	sort.Slice(systemsV, func(i, j int) bool {
		return systemsV[i].Entropy < systemsV[j].Entropy
	})

	n.Q = n.CalculateStatistics(systemsQ)
	n.K = n.CalculateStatistics(systemsK)
	n.V = n.CalculateStatistics(systemsV)

	return systemsV[0].Entropy, systemsQ[0].Outputs, systemsK[0].Outputs, systemsV[0].Outputs
}

// Iris is a iris data point
type Iris struct {
	iris.Iris
	I         int
	Embedding []float32
	Cluster   int
}

// XGMM is a gaussian mixture model clustering algorithm
// https://github.com/Ransaka/GMM-from-scratch
// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
func XGMM(flowers []Iris) {
	rng := rand.New(rand.NewSource(3))
	type Cluster struct {
		E XSet
		U []Random
	}
	var Pi []Random
	factor := float32(math.Sqrt(2.0 / float64(Embedding)))
	clusters := [Clusters]Cluster{}
	for i := range clusters {
		clusters[i].E = make(XSet, Embedding)
		for j := range clusters[i].E {
			row := make([]Random, Embedding)
			for k := range row {
				row[k].Mean = factor * float32(rng.NormFloat64())
				row[k].StdDev = factor * float32(rng.NormFloat64())
			}
			clusters[i].E[j] = row
		}
		clusters[i].U = make([]Random, Embedding, Embedding)
		for j := range clusters[i].U {
			clusters[i].U[j].Mean = factor * float32(rng.NormFloat64())
			clusters[i].U[j].StdDev = factor * float32(rng.NormFloat64())
		}
	}
	Pi = make([]Random, Clusters*len(flowers), Clusters*len(flowers))
	for j := range Pi {
		Pi[j].StdDev = 1
	}
	type Sample struct {
		E  [Clusters]Matrix
		U  [Clusters]Matrix
		C  float64
		Pi []float32
	}
	samples := make([]Sample, ModelSamples, ModelSamples)
	for i := 0; i < 128; i++ {
		for j := range samples {
			samples[j].Pi = make([]float32, Clusters*len(flowers), Clusters*len(flowers))
			for l := 0; l < len(samples[j].Pi); l += Clusters {
				sum := float32(0.0)
				for k := range clusters {
					r := Pi[l+k]
					samples[j].Pi[l+k] = r.StdDev*float32(rng.NormFloat64()) + r.Mean
					if samples[j].Pi[l+k] < 0 {
						samples[j].Pi[l+k] = -samples[j].Pi[l+k]
					}
					sum += samples[j].Pi[l+k]
				}
				for k := 0; k < Clusters; k++ {
					samples[j].Pi[l+k] /= sum
				}
			}
			var cs [Clusters][]float64
			for k := range cs {
				cs[k] = make([]float64, len(flowers), len(flowers))
			}
			samples[j].C = 0
			for k := range clusters {
				samples[j].E[k] = NewMatrix(0, Embedding, Embedding)
				samples[j].U[k] = NewMatrix(0, Embedding, 1)
				for l := range clusters[k].E {
					for m := range clusters[k].E[l] {
						r := clusters[k].E[l][m]
						samples[j].E[k].Data = append(samples[j].E[k].Data,
							r.StdDev*float32(rng.NormFloat64())+r.Mean)
					}
				}
				for l := range clusters[k].U {
					r := clusters[k].U[l]
					samples[j].U[k].Data = append(samples[j].U[k].Data,
						r.StdDev*float32(rng.NormFloat64())+r.Mean)
				}

				d, _ := Determinant(samples[j].E[k])
				det := float32(d)
				for f := range flowers {
					x := NewMatrix(0, Embedding, 1)
					x.Data = append(x.Data, flowers[f].Embedding...)
					y := MulT(T(MulT(Sub(x, samples[j].U[k]), samples[j].E[k])), Sub(x, samples[j].U[k]))
					pdf := math.Pow(2*math.Pi, -Embedding/2) *
						math.Pow(float64(det), 1/2) *
						math.Exp(float64(-y.Data[0]/2))
					cs[k][f] = float64(samples[j].Pi[f*Clusters+k]) * pdf
				}
			}
			for f := range flowers {
				sum := 0.0
				for k := range clusters {
					sum += cs[k][f]
				}
				for k := range clusters {
					cs[k][f] /= sum
				}
			}
			for k := range clusters {
				mean := 0.0
				for _, value := range cs[k] {
					mean += value
				}
				mean /= float64(len(flowers))
				stddev := 0.0
				for _, value := range cs[k] {
					diff := value - mean
					stddev += diff * diff
				}
				stddev /= float64(len(flowers))
				stddev = math.Sqrt(stddev)
				samples[j].C += math.Exp(-stddev)
			}
		}

		aa := [Clusters]Cluster{}
		for i := range aa {
			aa[i].E = make(XSet, Embedding)
			for j := range aa[i].E {
				aa[i].E[j] = make([]Random, Embedding)
			}
			aa[i].U = make([]Random, Embedding, Embedding)
		}

		sort.Slice(samples, func(i, j int) bool {
			return samples[i].C < samples[j].C
		})
		fmt.Println(samples[0].C)

		weights, sum := make([]float64, GaussianWindow), 0.0
		for i := range weights {
			sum += samples[i].C
			weights[i] = samples[i].C
		}
		for i := range weights {
			weights[i] /= sum
		}

		for k := range clusters {
			for i := range samples[:GaussianWindow] {
				index := 0
				for x := range aa[k].E {
					for y := range aa[k].E[x] {
						value := samples[i].E[k].Data[index]
						aa[k].E[x][y].Mean += float32(weights[i]) * value
						index++
					}
				}
			}
			for i := range samples[:GaussianWindow] {
				index := 0
				for x := range aa[k].E {
					for y := range aa[k].E[x] {
						diff := aa[k].E[x][y].Mean - samples[i].E[k].Data[index]
						aa[k].E[x][y].StdDev += float32(weights[i]) * diff * diff
						index++
					}
				}
			}
			for x := range aa[k].E {
				for y := range aa[k].E[x] {
					aa[k].E[x][y].StdDev /= (float32(GaussianWindow) - 1) / float32(GaussianWindow)
					aa[k].E[x][y].StdDev = float32(math.Sqrt(float64(aa[k].E[x][y].StdDev)))
				}
			}

			for i := range samples[:GaussianWindow] {
				index := 0
				for x := range aa[k].U {
					value := samples[i].U[k].Data[index]
					aa[k].U[x].Mean += float32(weights[i]) * value
					index++
				}
			}
			for i := range samples[:GaussianWindow] {
				index := 0
				for x := range aa[k].U {
					diff := aa[k].U[x].Mean - samples[i].U[k].Data[index]
					aa[k].U[x].StdDev += float32(weights[i]) * diff * diff
					index++
				}
			}
			for x := range aa[k].U {
				aa[k].U[x].StdDev /= (float32(GaussianWindow) - 1) / float32(GaussianWindow)
				aa[k].U[x].StdDev = float32(math.Sqrt(float64(aa[k].U[x].StdDev)))
			}

			for i := range samples[:GaussianWindow] {
				for x := range Pi {
					value := samples[i].Pi[x]
					Pi[x].Mean += float32(weights[i]) * value
				}
			}
			for i := range samples[:GaussianWindow] {
				for x := range Pi {
					diff := Pi[x].Mean - samples[i].Pi[x]
					Pi[x].StdDev += float32(weights[i]) * diff * diff
				}
			}
			for x := range Pi {
				Pi[x].StdDev /= (float32(GaussianWindow) - 1) / float32(GaussianWindow)
				Pi[x].StdDev = float32(math.Sqrt(float64(Pi[x].StdDev)))
			}
		}

		clusters = aa
	}

	for i := range flowers {
		x := NewMatrix(0, Embedding, 1)
		x.Data = append(x.Data, flowers[i].Embedding...)

		index, max := 0, 0.0
		for j := 0; j < Clusters; j++ {
			sort.Slice(samples, func(i, j int) bool {
				return samples[i].C < samples[j].C
			})
			sample := samples[0]

			d, _ := Determinant(sample.E[j])
			det := float32(d)
			y := MulT(T(MulT(Sub(x, sample.U[j]), sample.E[j])), Sub(x, sample.U[j]))
			pdf := math.Pow(2*math.Pi, -Embedding/2) *
				math.Pow(float64(det), 1/2) *
				math.Exp(float64(-y.Data[0]/2))
			pdf *= float64(samples[j].Pi[i*Clusters+j])
			if pdf > max {
				index, max = j, pdf
			}
		}
		flowers[i].Cluster = index
		fmt.Println(index, flowers[i].Label, max)
	}
}

// Mark2 is the mark2 model
func Mark2() {
	rng := rand.New(rand.NewSource(1))
	data, err := iris.Load()
	if err != nil {
		panic(err)
	}

	flowers := make([]Iris, len(data.Fisher))

	for i, value := range data.Fisher {
		sum := 0.0
		for _, v := range value.Measures {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= length
		}
		flowers[i].Iris = value
		flowers[i].I = i
	}
	net := XNewNet(1, Inputs+1, Outputs)
	projection := XNewNet(2, Outputs, 2)
	length := len(data.Fisher)
	const epochs = 5
	points := make(plotter.XYs, len(flowers))
	for i := 0; i < epochs; i++ {
		perm := rng.Perm(len(flowers))
		for epoch := 0; epoch < length; epoch++ {
			index := perm[epoch]
			query := NewMatrix(0, Inputs+1, 1)
			for _, value := range flowers[index].Measures {
				query.Data = append(query.Data, float32(value))
			}
			query.Data = append(query.Data, 0)
			key := NewMatrix(0, Inputs+1, 1)
			for _, value := range flowers[index].Measures {
				key.Data = append(key.Data, float32(value))
			}
			key.Data = append(key.Data, 0)
			value := NewMatrix(0, Inputs+1, 1)
			for _, v := range flowers[index].Measures {
				value.Data = append(value.Data, float32(v))
			}
			value.Data = append(value.Data, 0)
			label := flowers[index].Label
			entropy, q, k, v := net.Fire(query, key, value)
			fmt.Println(label, entropy, v.Data)
			copy(query.Data, q.Data)
			query.Data[4] = 1
			copy(key.Data, k.Data)
			key.Data[4] = 1
			copy(value.Data, v.Data)
			value.Data[4] = 1
			if i == epochs-1 {
				entropy, q, k, v = net.Fire(query, key, value)
				flowers[index].Embedding = append(flowers[index].Embedding, q.Data...)
				flowers[index].Embedding = append(flowers[index].Embedding, k.Data...)
				flowers[index].Embedding = append(flowers[index].Embedding, v.Data...)
				_, _, _, point := projection.Fire(q, k, v)
				points[index] = plotter.XY{X: float64(point.Data[0]), Y: float64(point.Data[1])}
				fmt.Println(label, entropy, v.Data)
			} else {
				entropy, q, k, v = net.Fire(query, key, value)
				projection.Fire(q, k, v)
				fmt.Println(label, entropy, v.Data)
			}
		}
	}
	XGMM(flowers)

	p := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Legend.Top = true

	c := 0
	sets := make([]plotter.XYs, Clusters, Clusters)
	names := make([]string, Clusters)
	for j := range sets {
		sets[j] = make(plotter.XYs, 0, 50)
	}
	for key, value := range flowers {
		if names[value.Cluster] == "" {
			names[value.Cluster] = value.Label
		}
		sets[value.Cluster] = append(sets[value.Cluster], points[key])
	}
	for j, set := range sets {
		scatter, err := plotter.NewScatter(set)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[c]
		p.Add(scatter)
		p.Legend.Add(names[j], scatter)
		c++
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, "projection.png")
	if err != nil {
		panic(err)
	}

	/*graph := pagerank.NewGraph64()
	for i := 0; i < len(vectors); i++ {
		for j := 0; j < len(vectors); j++ {
			a, b := vectors[i], vectors[j]
			distance := 0.0
			for k, value := range a.Data {
				diff := float64(b.Data[k] - value)
				distance += diff * diff
			}
			distance = math.Sqrt(distance)
			graph.Link(uint64(i), uint64(j), distance)
		}
	}
	type Rank struct {
		Rank  float64
		Index uint64
	}
	ranks := make([]Rank, 150)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = Rank{
			Rank:  rank,
			Index: node,
		}
	})
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].Rank < ranks[j].Rank
	})
	for _, v := range ranks {
		fmt.Println(v.Index, v.Rank)
	}*/
}
