// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mark3

import (
	"fmt"
	"math"
	"math/rand"

	. "github.com/pointlander/lucid/matrix"
)

func Mark3() {
	rng := rand.New(rand.NewSource(1))
	a := NewMatrix(0, 8, 8)
	for i := 0; i < 8*8; i++ {
		a.Data = append(a.Data, float32(rng.NormFloat64()))
	}
	type Sample struct {
		A    Matrix
		AA   Matrix
		Loss float32
	}
	min := float32(math.MaxFloat32)
	for {
		b := NewMatrix(0, 8, 8)
		for i := 0; i < 8*8; i++ {
			b.Data = append(b.Data, float32(rng.NormFloat64()))
		}
		ab := MulT(a, b)
		ab = Normalize(ab)
		abab := Sub(a, MulT(ab, T(ab)))
		sum := float32(0.0)
		for _, value := range abab.Data {
			sum += value * value
		}
		loss := float32(math.Sqrt(float64(sum)))
		if loss < min {
			min, a = loss, ab
			fmt.Println(loss)
		}
	}
}
