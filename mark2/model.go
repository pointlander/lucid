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

	input := NewMatrix(0, 4, 1)
	for _, value := range data.Fisher[0].Measures {
		input.Data = append(input.Data, float32(value))
	}
	output := NewMatrix(0, 3, 256)
	for i := 0; i < 256; i++ {
		neurons := make([]Matrix, 3)
		for j := range neurons {
			neurons[j] = NewMatrix(0, 4, 1)
			for k := 0; k < 4; k++ {
				neurons[j].Data = append(neurons[j].Data, float32(rng.NormFloat64()))
			}
		}
		for j := range neurons {
			out := MulT(neurons[j], input)
			output.Data = append(output.Data, out.Data[0])
		}
	}
	entropies := SelfEntropy(output, output, output)
	sort.Slice(entropies, func(i, j int) bool {
		return entropies[i] < entropies[j]
	})
	fmt.Println(entropies)
}
