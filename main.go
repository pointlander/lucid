// Copyright RNN The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/datum/iris"
)

const (
	// Width is the width of the model
	Width = 4
)

// Inputs is the input to the first layer
type Inputs struct {
	Inputs [][4]float32
	Labels []int
	Epoch  int64
}

func neuron1(seed int64, id int, in <-chan Inputs, out [3]chan<- Input) {
	rng := rand.New(rand.NewSource(seed))
	weights := NewMatrix(0, 4, 1)
	bias := NewMatrix(0, 1, 1)
	factor := math.Sqrt(2.0 / float64(4))
	for i := 0; i < 4; i++ {
		weights.Data = append(weights.Data, float32(rng.NormFloat64()*factor))
	}
	bias.Data = append(bias.Data, 0)
	for input := range in {
		o := Input{
			Input:  make([]float32, 3),
			Labels: input.Labels,
			Epoch:  input.Epoch,
		}
		for j := range input.Inputs {
			i := NewMatrix(0, 4, 1)
			i.Data = append(i.Data, input.Inputs[j][:]...)
			output := Step(Add(MulT(weights, i), bias))
			o.Input[j] = output.Data[0]
		}
		for j := range out {
			out[j] <- o
		}
	}
}

// Input is the input into the second layer
type Input struct {
	Input  []float32
	Labels []int
	Epoch  int64
}

func neuron2(seed int64, id int, in [Width]<-chan Input, out chan<- Input) {
	rng := rand.New(rand.NewSource(seed))
	weights := NewMatrix(0, Width, 1)
	bias := NewMatrix(0, 1, 1)
	factor := math.Sqrt(2.0 / float64(Width))
	for i := 0; i < Width; i++ {
		weights.Data = append(weights.Data, float32(rng.NormFloat64()*factor))
	}
	bias.Data = append(bias.Data, 0)
	for {
		inputs := make([]Matrix, 3)
		for i := range inputs {
			inputs[i] = NewMatrix(0, Width, 1)
		}
		labels := []int{}
		var epoch = int64(0)
		for _, in := range in {
			value := <-in
			labels = value.Labels
			epoch = value.Epoch
			for k, v := range value.Input {
				inputs[k].Data = append(inputs[k].Data, v)
			}
		}
		o := Input{
			Input:  make([]float32, 3),
			Labels: labels,
			Epoch:  epoch,
		}
		for i := range inputs {
			output := Add(MulT(weights, inputs[i]), bias)
			o.Input[i] = output.Data[0]
		}
		out <- o
	}
}

func main() {
	rng := rand.New(rand.NewSource(1))

	input := make([]chan Inputs, Width)
	output := make([]chan Input, Width*3)
	for i := range input {
		input[i] = make(chan Inputs, 8)
	}
	for i := range output {
		output[i] = make(chan Input, 8)
	}
	top := make([]chan Input, 3)
	for i := range top {
		top[i] = make(chan Input, 8)
	}

	id := 0
	for i := 0; i < Width; i++ {
		go neuron1(rng.Int63(), id, input[i], [3]chan<- Input{output[i], output[i+Width], output[i+2*Width]})
		id++
	}
	for i := 0; i < 3; i++ {
		input := [Width]<-chan Input{}
		for j := range input {
			input[j] = output[Width*i+j]
		}
		go neuron2(rng.Int63(), id, input, top[i])
		id++
	}

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

	for epoch := 1; epoch < 256; epoch++ {
		in := Inputs{
			Epoch:  int64(epoch),
			Labels: make([]int, 3),
			Inputs: make([][4]float32, 3),
		}
		indexes := [3]int{rng.Intn(50), 50 + rng.Intn(50), 100 + rng.Intn(50)}
		for i := range in.Inputs {
			for j := range in.Inputs[i] {
				in.Inputs[i][j] = float32(data.Fisher[indexes[i]].Measures[j])
			}
			in.Labels[i] = iris.Labels[data.Fisher[indexes[i]].Label]
		}
		for i := range input {
			input[i] <- in
		}
		fmt.Println(epoch)

		outputs := make([]Matrix, 3)
		for i := range outputs {
			outputs[i] = NewMatrix(0, 3, 1)
		}
		labels := []int{}
		for _, in := range top {
			value := <-in
			labels = value.Labels
			for k, v := range value.Input {
				outputs[k].Data = append(outputs[k].Data, v)
			}
		}
		loss := 0.0
		for i := range outputs {
			output := TaylorSoftmax(outputs[i])
			expected := make([]float32, 3)
			expected[labels[i]] = 1

			for i, v := range output.Data {
				diff := float64(float32(v) - expected[i])
				loss += diff * diff
			}
		}
		fmt.Println(loss)
	}
}
