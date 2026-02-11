package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/josephyu/tbf/go/tbf"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <output.tbf>\n", os.Args[0])
		os.Exit(2)
	}
	out := os.Args[1]

	w, err := tbf.NewWriter(out, tbf.DefaultPageSize)
	if err != nil {
		panic(err)
	}

	item1 := tbf.TensorItem{Key: "x", DType: tbf.Int32, Shape: []int64{2, 2}, Data: make([]byte, 16)}
	binary.LittleEndian.PutUint32(item1.Data[0:4], 1)
	binary.LittleEndian.PutUint32(item1.Data[4:8], 2)
	binary.LittleEndian.PutUint32(item1.Data[8:12], 3)
	binary.LittleEndian.PutUint32(item1.Data[12:16], 4)

	item2 := tbf.TensorItem{Key: "y", DType: tbf.UInt8, Shape: []int64{3}, Data: []byte{9, 8, 7}}
	if err := w.AddRecord([]tbf.TensorItem{item1, item2}); err != nil {
		panic(err)
	}

	item3 := tbf.TensorItem{Key: "z", DType: tbf.Float32, Shape: []int64{2}, Data: make([]byte, 8)}
	binary.LittleEndian.PutUint32(item3.Data[0:4], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(item3.Data[4:8], math.Float32bits(-2.0))
	if err := w.AddRecord([]tbf.TensorItem{item3}); err != nil {
		panic(err)
	}
	if err := w.Close(); err != nil {
		panic(err)
	}
}
