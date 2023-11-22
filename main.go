// Copyright 2023 The Lucid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"

	"github.com/pointlander/lucid/mark1"
	"github.com/pointlander/lucid/mark2"
)

var (
	// FlagMark1 is the mark1 model mode
	FlagMark1 = flag.Bool("mark1", false, "mark1 model")
	// FlagMark2 is the mark2 model
	FlagMark2 = flag.Bool("mark2", false, "mark2 model")
)

func main() {
	flag.Parse()

	if *FlagMark1 {
		mark1.Mark1()
		return
	} else if *FlagMark2 {
		mark2.Mark2()
		return
	}
}
