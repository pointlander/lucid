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
	// Window is the size of the window
	Window = 16
)

// Inputs is the input to the first layer
type Inputs struct {
	Inputs [][4]float32
	Labels []int
	Epoch  int64
}

func neuron1(seed int64, id int, in <-chan Inputs, out [3]chan<- Input, done chan bool) {
	//rng := rand.New(rand.NewSource(seed))
	multi := NewMulti(5)
	for input := range in {
		seeds := rand.New(rand.NewSource(input.Epoch))
	sample:
		for {
			select {
			case <-done:
				break sample
			default:
				epoch := seeds.Int63()
				weights := NewMatrix(0, 4, 1)
				bias := NewMatrix(0, 1, 1)
				samples := multi.Sample(rand.New(rand.NewSource(epoch + int64(id))))
				index := 0
				for i := 0; i < 4; i++ {
					weights.Data = append(weights.Data, samples[index])
					index++
				}
				bias.Data = append(bias.Data, samples[index])
				o := Input{
					Input:  make([]float32, 3),
					Labels: input.Labels,
					Epoch:  epoch,
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
	}
}

// Input is the input into the second layer
type Input struct {
	Input  []float32
	Labels []int
	Epoch  int64
}

func neuron2(seed int64, id int, in [Width]<-chan Input, out chan<- Input) {
	//rng := rand.New(rand.NewSource(seed))
	multi := NewMulti(Width + 1)
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
		weights := NewMatrix(0, Width, 1)
		bias := NewMatrix(0, 1, 1)
		samples := multi.Sample(rand.New(rand.NewSource(epoch + int64(id))))
		index := 0
		for i := 0; i < Width; i++ {
			weights.Data = append(weights.Data, samples[index])
			index++
		}
		bias.Data = append(bias.Data, samples[index])
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
	done := make([]chan bool, Width)
	for i := range done {
		done[i] = make(chan bool, 8)
	}

	id := 0
	for i := 0; i < Width; i++ {
		go neuron1(rng.Int63(), id, input[i], [3]chan<- Input{output[i], output[i+Width], output[i+2*Width]}, done[i])
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

	type Loss struct {
		Loss  float32
		Epoch int64
	}
	losses := make([]Loss, Window)
	for i := range losses {
		losses[i].Loss = math.MaxFloat32
	}
	count := 0
	for epoch := 1; epoch < 256; epoch++ {
		in := Inputs{
			Epoch:  int64(rng.Int31()),
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

	search:
		for {
			outputs := make([]Matrix, 3)
			for i := range outputs {
				outputs[i] = NewMatrix(0, 3, 1)
			}
			labels := []int{}
			loss := Loss{}
			for _, in := range top {
				value := <-in
				labels = value.Labels
				loss.Epoch = value.Epoch
				for k, v := range value.Input {
					outputs[k].Data = append(outputs[k].Data, v)
				}
			}
			for i := range outputs {
				output := TaylorSoftmax(outputs[i])
				expected := make([]float32, 3)
				expected[labels[i]] = 1

				for i, v := range output.Data {
					diff := float32(v) - expected[i]
					loss.Loss += diff * diff
				}
			}
			index := 0
			for index < len(losses) {
				if loss.Loss < losses[index].Loss {
					count++
					loss, losses[index] = losses[index], loss
					if index == 0 {
						fmt.Println(losses[0])
						if count > Window {
							for _, done := range done {
								done <- true
							}
							break search
						}
					}
					index++
					for index < len(losses) {
						loss, losses[index] = losses[index], loss
						index++
					}
				}
				index++
			}
		}
	}
}
