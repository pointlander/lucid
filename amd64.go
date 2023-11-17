// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64
// +build amd64

package main

import (
	"github.com/ziutek/blas"
)

func dot(X, Y []float32) float32 {
	return blas.Sdot(len(X), X, 1, Y, 1)
}
