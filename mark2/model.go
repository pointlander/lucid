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
	Outputs = 3
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// Mark2 is the mark2 model
func Mark2() {
	rng := rand.New(rand.NewSource(1))

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

	distribution := make([][]Random, Outputs)
	for i := range distribution {
		for j := 0; j < Inputs; j++ {
			distribution[i] = append(distribution[i], Random{
				Mean:   0,
				StdDev: 1,
			})
		}
	}

	for epoch := 0; epoch < 300; epoch++ {
		input := NewMatrix(0, 4, 1)
		for _, value := range data.Fisher[epoch%150].Measures {
			input.Data = append(input.Data, float32(value))
		}
		label := data.Fisher[epoch%150].Label
		output := NewMatrix(0, Outputs, Samples)
		type System struct {
			Entropy float32
			Neurons []Matrix
			Outputs Matrix
		}
		systems := make([]System, 0, 8)
		for i := 0; i < Samples; i++ {
			neurons := make([]Matrix, Outputs)
			for j := range neurons {
				neurons[j] = NewMatrix(0, Inputs, 1)
				for k := 0; k < Inputs; k++ {
					neurons[j].Data = append(neurons[j].Data,
						float32(rng.NormFloat64())*distribution[j][k].StdDev+distribution[j][k].Mean)
				}
			}
			outputs := NewMatrix(0, Outputs, 1)
			for j := range neurons {
				out := MulT(neurons[j], input)
				output.Data = append(output.Data, out.Data[0])
				outputs.Data = append(outputs.Data, out.Data[0])
			}
			systems = append(systems, System{
				Neurons: neurons,
				Outputs: outputs,
			})
		}
		entropies := SelfEntropy(output, output, output)
		for i, entropy := range entropies {
			systems[i].Entropy = entropy
		}
		sort.Slice(entropies, func(i, j int) bool {
			return systems[i].Entropy < systems[j].Entropy
		})
		fmt.Println(label, systems[0].Outputs.Data)
		next := make([][]Random, Outputs)
		for i := range next {
			for j := 0; j < Inputs; j++ {
				next[i] = append(next[i], Random{
					Mean:   0,
					StdDev: 0,
				})
			}
		}
		for i := range systems[:Window] {
			for j := range systems[i].Neurons {
				for k, value := range systems[i].Neurons[j].Data {
					next[j][k].Mean += value
				}
			}
		}
		for i := range next {
			for j := range next[i] {
				next[i][j].Mean /= Window
			}
		}
		for i := range systems[:Window] {
			for j := range systems[i].Neurons {
				for k, value := range systems[i].Neurons[j].Data {
					diff := next[j][k].Mean - value
					next[j][k].StdDev += diff * diff
				}
			}
		}
		for i := range next {
			for j := range next[i] {
				next[i][j].StdDev /= Window
				next[i][j].StdDev = float32(math.Sqrt(float64(next[i][j].StdDev)))
			}
		}
		distribution = next
	}
}
