package tbf

import (
	"bytes"
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPythonWriterFooterAndAlignment(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not found, skip cross-language test")
	}

	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	outPath := filepath.Join(t.TempDir(), "python_fixture.tbf")
	pythonCode := `from tbf import write_tbf; import torch; write_tbf(r"` + outPath + `", [{"x": torch.tensor([1,2,3], dtype=torch.int32)}, {"y": torch.tensor([4,5], dtype=torch.int64)}])`

	cmd := exec.Command("uv", "run", "python", "-c", pythonCode)
	cmd.Dir = repoRoot
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to generate python fixture: %v\n%s", err, string(out))
	}

	buf, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(buf) < 64 {
		t.Fatalf("bad file size: %d", len(buf))
	}
	if !bytes.Equal(buf[:8], []byte("TBFDATA1")) {
		t.Fatalf("bad file magic: %q", string(buf[:8]))
	}

	footer := buf[len(buf)-64:]
	if !bytes.Equal(footer[:8], []byte("TBFTRLR1")) {
		t.Fatalf("bad footer magic: %q", string(footer[:8]))
	}
	if binary.LittleEndian.Uint32(footer[8:12]) != Version {
		t.Fatalf("bad footer version")
	}
	indexOffset := binary.LittleEndian.Uint64(footer[12:20])
	indexSize := binary.LittleEndian.Uint64(footer[20:28])
	if indexOffset+indexSize > uint64(len(buf)-64) {
		t.Fatalf("index out of range: off=%d size=%d", indexOffset, indexSize)
	}

	idx := buf[indexOffset : indexOffset+indexSize]
	if !bytes.Equal(idx[:8], []byte("TBFIDX01")) {
		t.Fatalf("bad index magic: %q", string(idx[:8]))
	}
	if binary.LittleEndian.Uint32(idx[8:12]) != Version {
		t.Fatalf("bad index version")
	}
	entryCount := binary.LittleEndian.Uint64(idx[12:20])
	recordCount := binary.LittleEndian.Uint64(idx[20:28])
	if recordCount != 2 || entryCount != 2 {
		t.Fatalf("count mismatch: records=%d entries=%d", recordCount, entryCount)
	}

	pos := 28
	for i := 0; i < int(entryCount); i++ {
		recordID := binary.LittleEndian.Uint64(idx[pos : pos+8])
		keyLen := binary.LittleEndian.Uint32(idx[pos+8 : pos+12])
		ndim := binary.LittleEndian.Uint16(idx[pos+14 : pos+16])
		dataOffset := binary.LittleEndian.Uint64(idx[pos+16 : pos+24])
		if recordID != uint64(i) {
			t.Fatalf("record id mismatch at entry %d: %d", i, recordID)
		}
		if dataOffset%uint64(DefaultPageSize) != 0 {
			t.Fatalf("unaligned data offset: %d", dataOffset)
		}
		pos += 32
		pos += int(ndim) * 8
		pos += int(keyLen)
	}
	if pos != len(idx) {
		t.Fatalf("index parse mismatch: pos=%d len=%d", pos, len(idx))
	}
}
