package tbf

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestWriterAlignmentAndFooter(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sample.tbf")

	w, err := NewWriter(path, 4096)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.AddRecord([]TensorItem{{
		Key:   "a",
		DType: Int32,
		Shape: []int64{2},
		Data:  []byte{1, 0, 0, 0, 2, 0, 0, 0},
	}}); err != nil {
		t.Fatalf("AddRecord: %v", err)
	}
	if err := w.AddRecord([]TensorItem{{
		Key:   "b",
		DType: UInt8,
		Shape: []int64{3},
		Data:  []byte{3, 4, 5},
	}}); err != nil {
		t.Fatalf("AddRecord #2: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(buf) < 64 {
		t.Fatalf("unexpected file len: %d", len(buf))
	}

	footer := buf[len(buf)-64:]
	if string(footer[0:8]) != "TBFTRLR1" {
		t.Fatalf("bad footer magic: %q", string(footer[0:8]))
	}
	version := binary.LittleEndian.Uint32(footer[8:12])
	if version != Version {
		t.Fatalf("bad footer version: %d", version)
	}
	indexOffset := binary.LittleEndian.Uint64(footer[12:20])
	indexSize := binary.LittleEndian.Uint64(footer[20:28])
	if indexOffset+indexSize > uint64(len(buf)-64) {
		t.Fatalf("bad index range, off=%d size=%d", indexOffset, indexSize)
	}

	idx := buf[indexOffset : indexOffset+indexSize]
	if len(idx) < 28 {
		t.Fatalf("index too short: %d", len(idx))
	}
	if string(idx[0:8]) != "TBFIDX01" {
		t.Fatalf("bad index magic: %q", string(idx[0:8]))
	}
	entryCount := binary.LittleEndian.Uint64(idx[12:20])
	recordCount := binary.LittleEndian.Uint64(idx[20:28])
	if recordCount != 2 || entryCount != 2 {
		t.Fatalf("counts mismatch, record=%d entry=%d", recordCount, entryCount)
	}
}
