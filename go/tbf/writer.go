package tbf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	Version         uint32 = 1
	DefaultPageSize uint32 = 4096
)

var (
	fileMagic   = [8]byte{'T', 'B', 'F', 'D', 'A', 'T', 'A', '1'}
	indexMagic  = [8]byte{'T', 'B', 'F', 'I', 'D', 'X', '0', '1'}
	footerMagic = [8]byte{'T', 'B', 'F', 'T', 'R', 'L', 'R', '1'}
)

type DType uint16

const (
	Float32  DType = 1
	Float64  DType = 2
	Float16  DType = 3
	BFloat16 DType = 4
	Int8     DType = 5
	UInt8    DType = 6
	Int16    DType = 7
	Int32    DType = 8
	Int64    DType = 9
	Bool     DType = 10
)

func dtypeElementSize(dtype DType) (uint64, bool) {
	switch dtype {
	case Float32:
		return 4, true
	case Float64:
		return 8, true
	case Float16, BFloat16, Int16:
		return 2, true
	case Int8, UInt8, Bool:
		return 1, true
	case Int32:
		return 4, true
	case Int64:
		return 8, true
	default:
		return 0, false
	}
}

type TensorItem struct {
	Key   string
	DType DType
	Shape []int64
	Data  []byte
}

type indexEntry struct {
	recordID   uint64
	key        []byte
	dtypeCode  DType
	shape      []int64
	dataOffset uint64
	nbytes     uint64
}

type Writer struct {
	f           *os.File
	pageSize    uint32
	recordCount uint64
	entries     []indexEntry
	closed      bool
}

func NewWriter(path string, pageSize uint32) (*Writer, error) {
	if pageSize == 0 {
		return nil, errors.New("pageSize must be > 0")
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := &Writer{f: f, pageSize: pageSize}
	if err := w.writeFileHeader(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return w, nil
}

func (w *Writer) Close() error {
	if w.closed {
		return nil
	}
	if err := w.writeIndexAndFooter(); err != nil {
		_ = w.f.Close()
		w.closed = true
		return err
	}
	w.closed = true
	return w.f.Close()
}

func (w *Writer) AddRecord(items []TensorItem) error {
	recordID := w.recordCount
	for _, item := range items {
		if err := w.AddTensor(recordID, item); err != nil {
			return err
		}
	}
	w.recordCount++
	return nil
}

func (w *Writer) AddTensor(recordID uint64, item TensorItem) error {
	if w.closed {
		return errors.New("writer already closed")
	}
	if item.Key == "" {
		return errors.New("tensor key cannot be empty")
	}
	elSize, ok := dtypeElementSize(item.DType)
	if !ok {
		return fmt.Errorf("unsupported dtype code: %d", item.DType)
	}
	expected, err := expectedNBytes(item.Shape, elSize)
	if err != nil {
		return err
	}
	if uint64(len(item.Data)) != expected {
		return fmt.Errorf("data length mismatch for key %q: got=%d expect=%d", item.Key, len(item.Data), expected)
	}

	offset, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	aligned := alignUp(uint64(offset), uint64(w.pageSize))
	padLen := aligned - uint64(offset)
	if padLen > 0 {
		if _, err := w.f.Write(make([]byte, padLen)); err != nil {
			return err
		}
	}
	if len(item.Data) > 0 {
		if _, err := w.f.Write(item.Data); err != nil {
			return err
		}
	}
	w.entries = append(w.entries, indexEntry{
		recordID:   recordID,
		key:        []byte(item.Key),
		dtypeCode:  item.DType,
		shape:      append([]int64(nil), item.Shape...),
		dataOffset: aligned,
		nbytes:     uint64(len(item.Data)),
	})
	return nil
}

func (w *Writer) writeFileHeader() error {
	if _, err := w.f.Write(fileMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(w.f, binary.LittleEndian, Version); err != nil {
		return err
	}
	var reserved uint32 = 0
	return binary.Write(w.f, binary.LittleEndian, reserved)
}

func (w *Writer) writeIndexAndFooter() error {
	indexOffsetSigned, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	indexOffset := uint64(indexOffsetSigned)

	if _, err := w.f.Write(indexMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(w.f, binary.LittleEndian, Version); err != nil {
		return err
	}
	entryCount := uint64(len(w.entries))
	if err := binary.Write(w.f, binary.LittleEndian, entryCount); err != nil {
		return err
	}
	if err := binary.Write(w.f, binary.LittleEndian, w.recordCount); err != nil {
		return err
	}

	for _, entry := range w.entries {
		keyLen := uint32(len(entry.key))
		ndim := uint16(len(entry.shape))
		if err := binary.Write(w.f, binary.LittleEndian, entry.recordID); err != nil {
			return err
		}
		if err := binary.Write(w.f, binary.LittleEndian, keyLen); err != nil {
			return err
		}
		dtypeCode := uint16(entry.dtypeCode)
		if err := binary.Write(w.f, binary.LittleEndian, dtypeCode); err != nil {
			return err
		}
		if err := binary.Write(w.f, binary.LittleEndian, ndim); err != nil {
			return err
		}
		if err := binary.Write(w.f, binary.LittleEndian, entry.dataOffset); err != nil {
			return err
		}
		if err := binary.Write(w.f, binary.LittleEndian, entry.nbytes); err != nil {
			return err
		}
		for _, dim := range entry.shape {
			if err := binary.Write(w.f, binary.LittleEndian, dim); err != nil {
				return err
			}
		}
		if _, err := w.f.Write(entry.key); err != nil {
			return err
		}
	}

	curSigned, err := w.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	indexSize := uint64(curSigned) - indexOffset

	if _, err := w.f.Write(footerMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(w.f, binary.LittleEndian, Version); err != nil {
		return err
	}
	if err := binary.Write(w.f, binary.LittleEndian, indexOffset); err != nil {
		return err
	}
	if err := binary.Write(w.f, binary.LittleEndian, indexSize); err != nil {
		return err
	}
	_, err = w.f.Write(make([]byte, 36))
	return err
}

func alignUp(v, alignment uint64) uint64 {
	if alignment == 0 {
		return v
	}
	if v%alignment == 0 {
		return v
	}
	return v + (alignment - (v % alignment))
}

func expectedNBytes(shape []int64, elementSize uint64) (uint64, error) {
	if len(shape) == 0 {
		return elementSize, nil
	}
	prod := uint64(1)
	for _, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("negative shape dim: %d", dim)
		}
		prod *= uint64(dim)
	}
	return prod * elementSize, nil
}
