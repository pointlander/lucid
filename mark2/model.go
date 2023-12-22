// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark2

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/lucid/matrix"
)

const (
	// Window is the window size
	Window = 8
	// Samples is the number of samples
	Samples = 256
	// Inputs is the number of inputs
	Inputs = 4
	// Outputs is the number of outputs
	Outputs = 4
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// Set is a set of statistics
type Set []Random

// NewStatistics generates a new statistics model
func NewStatistics(inputs, outputs int) Set {
	statistics := make(Set, 0, outputs)
	factor := float32(math.Sqrt(2.0 / float64(inputs)))
	for j := 0; j < inputs; j++ {
		statistics = append(statistics, Random{
			Mean:   0,
			StdDev: factor,
		})
	}
	return statistics
}

// Sample samples from the statistics
func (s Set) Sample(rng *rand.Rand, inputs, outputs int) (Matrix, [][]Matrix) {
	neurons := make([][]Matrix, 3)
	factor := float32(math.Sqrt(2.0 / float64(inputs)))
	model := NewMatrix(0, inputs, inputs)
	for i := range s {
		model.Data = append(model.Data, s[i].StdDev*float32(rng.NormFloat64())+s[i].Mean)
	}
	input := NewMatrix(0, inputs, 1)
	for i := 0; i < inputs; i++ {
		input.Data = append(input.Data, factor*float32(rng.NormFloat64()))
	}
	for i := range neurons {
		neurons[i] = make([]Matrix, outputs)
	}
	for j := 0; j < outputs; j++ {
		for i := range neurons {
			neurons[i][j] = NewMatrix(0, inputs, 1)
			output := Sigmoid(MulT(model, Normalize(input)))
			for _, v := range output.Data {
				if v > .5 {
					v = 1
				} else {
					v = -1
				}
				neurons[i][j].Data = append(neurons[i][j].Data, v)
			}
			copy(input.Data, output.Data)
			input.Data[0] = factor * float32(rng.NormFloat64())
		}
	}
	return model, neurons
}

// Net is a net
type Net struct {
	Inputs  int
	Outputs int
	Rng     *rand.Rand
	D       Set
}

// NewNet makes a new network
func NewNet(seed int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		D:       NewStatistics(inputs, outputs),
	}
}

// Sample is a sample of a random neural network
type Sample struct {
	Entropy float32
	Neurons [][]Matrix
	Model   Matrix
	Outputs Matrix
	Out     Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n Net) CalculateStatistics(systems [][]Sample) Set {
	system := systems[0]
	statistics := make(Set, n.Outputs)
	for i := range system[:Window] {
		for j, value := range system[i].Model.Data {
			statistics[j].Mean += value
		}
	}
	for i := range statistics {
		statistics[i].Mean /= Window
	}
	for i := range system[:Window] {
		for j, value := range system[i].Model.Data {
			diff := statistics[j].Mean - value
			statistics[j].StdDev += diff * diff
		}
	}
	for i := range statistics {
		statistics[i].StdDev /= Window
		statistics[i].StdDev = float32(math.Sqrt(float64(statistics[i].StdDev)))
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(input Matrix) (float32, Matrix) {
	x := make([]Matrix, 3)
	for i := range x {
		x[i] = NewMatrix(0, n.Outputs, Samples)
	}
	systems := make([][]Sample, 3)
	for i := range systems {
		systems[i] = make([]Sample, 0, 8)
	}
	for i := 0; i < Samples; i++ {
		model, neurons := n.D.Sample(n.Rng, n.Inputs, n.Outputs)
		for j := range neurons {
			outputs := NewMatrix(0, n.Outputs, 1)
			for k := range neurons[j] {
				out := MulT(neurons[j][k], input)
				x[j].Data = append(x[j].Data, out.Data[0])
				outputs.Data = append(outputs.Data, out.Data[0])
			}
			systems[j] = append(systems[j], Sample{
				Neurons: neurons,
				Outputs: outputs,
				Model:   model,
			})
		}
	}
	outputs, entropies := SelfEntropy(x[0], x[1], x[2])
	for i, entropy := range entropies {
		for j := range systems {
			systems[j][i].Entropy = entropy
			systems[j][i].Out = outputs[i]
		}
	}
	for s := range systems {
		sort.Slice(systems[s], func(i, j int) bool {
			return systems[s][i].Entropy < systems[s][j].Entropy
		})
	}

	n.D = n.CalculateStatistics(systems)
	vector := NewMatrix(0, 3*n.Outputs, 1)
	for s := range systems {
		vector.Data = append(vector.Data, systems[s][0].Outputs.Data...)
	}
	return systems[2][0].Entropy, vector //systemsV[0].Outputs
}

// Mark2 is the mark2 model
func Mark2() {
	data, err := iris.Load()
	if err != nil {
		panic(err)
	}

	for _, value := range data.Fisher {
		sum := 0.0
		for _, v := range value.Measures {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= length
		}
	}
	//layer := NewNet(2, Inputs, 2*Inputs)
	//net := NewNet(1, 2*Inputs, Outputs)
	net := NewNet(1, Inputs, Outputs)
	length := len(data.Fisher)
	for epoch := 0; epoch < length; epoch++ {
		input := NewMatrix(0, Inputs, 1)
		for _, value := range data.Fisher[epoch].Measures {
			input.Data = append(input.Data, float32(value))
		}
		label := data.Fisher[epoch].Label
		//_, output := layer.Fire(input)
		entropy, output := net.Fire(input)
		fmt.Println(label, entropy, output.Data)
	}
	nn := map[string][]float32{
		"Iris-setosa":     nil,
		"Iris-versicolor": nil,
		"Iris-virginica":  nil,
	}
	for epoch := 0; epoch < length; epoch++ {
		input := NewMatrix(0, Inputs, 1)
		for _, value := range data.Fisher[epoch].Measures {
			input.Data = append(input.Data, float32(value))
		}
		label := data.Fisher[epoch].Label
		//e, output := layer.Fire(input)
		entropy, output := net.Fire(input)
		fmt.Println(label, entropy, output.Data)
		if value := nn[label]; value == nil {
			nn[label] = output.Data
		}
	}
	clusters := map[string][]int{
		"Iris-setosa":     []int{},
		"Iris-versicolor": []int{},
		"Iris-virginica":  []int{},
	}
	vectors := []Matrix{}
	for epoch := 0; epoch < length; epoch++ {
		input := NewMatrix(0, Inputs, 1)
		for _, value := range data.Fisher[epoch].Measures {
			input.Data = append(input.Data, float32(value))
		}
		label := data.Fisher[epoch].Label
		//e, output := layer.Fire(input)
		entropy, output := net.Fire(input)
		vectors = append(vectors, output)
		fmt.Println(label, entropy, output.Data)
		min, index := math.MaxFloat32, ""
		for name, vector := range nn {
			distance := 0.0
			for j, value := range vector {
				diff := float64(output.Data[j] - value)
				distance += diff * diff
			}
			distance = math.Sqrt(distance)
			if distance < min {
				min, index = distance, name
			}
		}
		list := clusters[index]
		list = append(list, epoch)
		clusters[index] = list
	}
	fmt.Println(clusters)

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
