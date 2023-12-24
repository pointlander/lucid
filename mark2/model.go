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
	Window = 4
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
type Set [][]Random

// NewStatistics generates a new statistics model
func NewStatistics(inputs, outputs int) Set {
	statistics := make(Set, outputs)
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
func (s Set) Sample(rng *rand.Rand, inputs, outputs int) []Matrix {
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

// Net is a net
type Net struct {
	Inputs  int
	Outputs int
	Rng     *rand.Rand
	Q       Set
	K       Set
	V       Set
}

// NewNet makes a new network
func NewNet(seed int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		Q:       NewStatistics(inputs, outputs),
		K:       NewStatistics(inputs, outputs),
		V:       NewStatistics(inputs, outputs),
	}
}

// Sample is a sample of a random neural network
type Sample struct {
	Entropy float32
	Neurons []Matrix
	Outputs Matrix
	Out     Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n Net) CalculateStatistics(systems []Sample) Set {
	statistics := make(Set, n.Outputs)
	for i := range statistics {
		for j := 0; j < n.Inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 0,
			})
		}
	}
	for i := range systems[:Window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				statistics[j][k].Mean += value
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].Mean /= Window
		}
	}
	for i := range systems[:Window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := statistics[j][k].Mean - value
				statistics[j][k].StdDev += diff * diff
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].StdDev /= Window
			statistics[i][j].StdDev = float32(math.Sqrt(float64(statistics[i][j].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(query, key, value Matrix) (float32, Matrix, Matrix, Matrix) {
	q := NewMatrix(0, n.Outputs, Samples)
	k := NewMatrix(0, n.Outputs, Samples)
	v := NewMatrix(0, n.Outputs, Samples)
	systemsQ := make([]Sample, 0, 8)
	systemsK := make([]Sample, 0, 8)
	systemsV := make([]Sample, 0, 8)
	for i := 0; i < Samples; i++ {
		neurons := n.Q.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], query)
			q.Data = append(q.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsQ = append(systemsQ, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < Samples; i++ {
		neurons := n.K.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], key)
			k.Data = append(k.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsK = append(systemsK, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < Samples; i++ {
		neurons := n.V.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], value)
			v.Data = append(v.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsV = append(systemsV, Sample{
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
	net := NewNet(1, Inputs+1, Outputs)
	length := len(data.Fisher)
	for epoch := 0; epoch < length; epoch++ {
		query := NewMatrix(0, Inputs+1, 1)
		for _, value := range data.Fisher[epoch].Measures {
			query.Data = append(query.Data, float32(value))
		}
		query.Data = append(query.Data, 0)
		key := NewMatrix(0, Inputs+1, 1)
		for _, value := range data.Fisher[epoch].Measures {
			key.Data = append(key.Data, float32(value))
		}
		key.Data = append(key.Data, 0)
		value := NewMatrix(0, Inputs+1, 1)
		for _, v := range data.Fisher[epoch].Measures {
			value.Data = append(value.Data, float32(v))
		}
		value.Data = append(value.Data, 0)
		label := data.Fisher[epoch].Label
		//_, output := layer.Fire(input)
		entropy, q, k, v := net.Fire(query, key, value)
		fmt.Println(label, entropy, v.Data)
		copy(query.Data, q.Data)
		query.Data[4] = 1
		copy(key.Data, k.Data)
		key.Data[4] = 1
		copy(value.Data, v.Data)
		value.Data[4] = 1
		entropy, q, k, v = net.Fire(query, key, value)
		fmt.Println(label, entropy, v.Data)
	}
	nn := map[string][6][]float32{
		"Iris-setosa":     [6][]float32{},
		"Iris-versicolor": [6][]float32{},
		"Iris-virginica":  [6][]float32{},
	}
	for epoch := 0; epoch < length; epoch++ {
		query := NewMatrix(0, Inputs+1, 1)
		for _, value := range data.Fisher[epoch].Measures {
			query.Data = append(query.Data, float32(value))
		}
		query.Data = append(query.Data, 0)
		key := NewMatrix(0, Inputs+1, 1)
		for _, value := range data.Fisher[epoch].Measures {
			key.Data = append(key.Data, float32(value))
		}
		key.Data = append(key.Data, 0)
		value := NewMatrix(0, Inputs+1, 1)
		for _, v := range data.Fisher[epoch].Measures {
			value.Data = append(value.Data, float32(v))
		}
		value.Data = append(value.Data, 0)
		label := data.Fisher[epoch].Label
		//e, output := layer.Fire(input)
		entropy, q, k, v := net.Fire(query, key, value)
		if value := nn[label]; value[0] == nil {
			value[0] = q.Data
			value[1] = k.Data
			value[2] = v.Data
			nn[label] = value
		}
		fmt.Println(label, entropy, v.Data)
		copy(query.Data, q.Data)
		query.Data[4] = 1
		copy(key.Data, k.Data)
		key.Data[4] = 1
		copy(value.Data, v.Data)
		value.Data[4] = 1
		entropy, q, k, v = net.Fire(query, key, value)
		fmt.Println(label, entropy, v.Data)
		if value := nn[label]; value[3] == nil {
			value[3] = q.Data
			value[4] = k.Data
			value[5] = v.Data
			nn[label] = value
		}
	}
	clusters := map[string][150]int{
		"Iris-setosa":     [150]int{},
		"Iris-versicolor": [150]int{},
		"Iris-virginica":  [150]int{},
	}
	//vectors := []Matrix{}
	search := func(query Matrix, epoch, a int) {
		min, index := math.MaxFloat32, ""
		for name, set := range nn {
			distance := 0.0
			for j, value := range set[a] {
				diff := float64(query.Data[j] - value)
				distance += diff * diff
			}
			distance = math.Sqrt(distance)
			if distance < min {
				min, index = distance, name
			}
		}
		list := clusters[index]
		list[epoch]++
		clusters[index] = list
	}
	for epoch := 0; epoch < length; epoch++ {
		query := NewMatrix(0, Inputs+1, 1)
		for _, value := range data.Fisher[epoch].Measures {
			query.Data = append(query.Data, float32(value))
		}
		query.Data = append(query.Data, 0)
		key := NewMatrix(0, Inputs+1, 1)
		for _, value := range data.Fisher[epoch].Measures {
			key.Data = append(key.Data, float32(value))
		}
		key.Data = append(key.Data, 0)
		value := NewMatrix(0, Inputs+1, 1)
		for _, v := range data.Fisher[epoch].Measures {
			value.Data = append(value.Data, float32(v))
		}
		value.Data = append(value.Data, 0)
		label := data.Fisher[epoch].Label
		//e, output := layer.Fire(input)
		entropy, q, k, v := net.Fire(query, key, value)
		search(q, epoch, 0)
		search(k, epoch, 1)
		search(v, epoch, 2)
		copy(query.Data, q.Data)
		query.Data[4] = 1
		copy(key.Data, k.Data)
		key.Data[4] = 1
		copy(value.Data, v.Data)
		value.Data[4] = 1
		entropy, q, k, v = net.Fire(query, key, value)
		//vectors = append(vectors, output)
		fmt.Println(label, entropy, v.Data)
		search(q, epoch, 3)
		search(k, epoch, 4)
		search(v, epoch, 5)
	}
	results := map[string][]int{
		"Iris-setosa":     []int{},
		"Iris-versicolor": []int{},
		"Iris-virginica":  []int{},
	}
	for i := 0; i < 150; i++ {
		max, name := 0, ""
		for k, v := range clusters {
			if v[i] > max {
				max, name = v[i], k
			}
		}
		list := results[name]
		list = append(list, i)
		results[name] = list
	}
	fmt.Println(clusters)
	fmt.Println(results)

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
