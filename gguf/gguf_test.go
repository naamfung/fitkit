package gguf

import "testing"

func TestTensorPaddedSize(t *testing.T) {
	layout := &Layout{
		Alignment: 32,
		TensorMap: map[string]TensorInfo{
			"w": {Name: "w", Shape: []int64{4096, 4096}},
			"x": {Name: "x", Shape: []int64{128, 512}},
			"y": {Name: "y", Shape: []int64{4096, 1}}, // trailing-1 dim trimmed to 1D
		},
	}
	// q8_0: block 32, type size 34 → 4096/32*34*4096
	wantPayload := (4096 / 32) * 34 * 4096
	payload, padded, ok := TensorPaddedSize(layout, "w", "Q8_0")
	if !ok {
		t.Fatal("w/Q8_0: expected ok")
	}
	if payload != wantPayload {
		t.Errorf("w/Q8_0 payload=%d want %d", payload, wantPayload)
	}
	if padded != align(wantPayload, 32) {
		t.Errorf("w/Q8_0 padded=%d want %d", padded, align(wantPayload, 32))
	}
	// iq1_s: block 256, type size 50; ne0=4096 divisible → ok
	if _, _, ok := TensorPaddedSize(layout, "w", "IQ1_S"); !ok {
		t.Error("w/IQ1_S: expected ok (4096 divisible by 256)")
	}
	// ne0=128 not divisible by iq1_s block 256 → not ok
	if _, _, ok := TensorPaddedSize(layout, "x", "IQ1_S"); ok {
		t.Error("x/IQ1_S: expected not ok (128 not divisible by 256)")
	}
	// unknown qtype → not ok
	if _, _, ok := TensorPaddedSize(layout, "w", "NOPE"); ok {
		t.Error("w/NOPE: expected not ok (unknown qtype)")
	}
	// unknown tensor → not ok
	if _, _, ok := TensorPaddedSize(layout, "ghost", "Q8_0"); ok {
		t.Error("ghost/Q8_0: expected not ok (unknown tensor)")
	}
	// 1D tensor (trailing-1 trimmed) still encodable
	if _, _, ok := TensorPaddedSize(layout, "y", "Q8_0"); !ok {
		t.Error("y/Q8_0: expected ok (trailing-1 dims trimmed)")
	}
}

// TestPredictQuantizedSizeUsesTensorPaddedSize guards the refactor: the helper
// and the full prediction must agree on the per-tensor sizes of a tiny recipe.
func TestPredictQuantizedSizeUsesTensorPaddedSize(t *testing.T) {
	layout := &Layout{
		Alignment: 32,
		Fields:    []Field{{Key: "general.file_type", ValueType: UINT32, EncodedSize: 8 + len("general.file_type") + 4 + 4}},
		Tensors:   []TensorInfo{{Name: "w", Shape: []int64{4096, 4096}, RelativeOffset: 0}},
		TensorMap: map[string]TensorInfo{"w": {Name: "w", Shape: []int64{4096, 4096}}},
	}
	meta := &QuantizationMetadata{FileType: 7, QuantizationVersion: 2}
	pred, err := PredictQuantizedSize(layout, map[string]string{"w": "q8_0"}, meta)
	if err != nil {
		t.Fatalf("PredictQuantizedSize: %v", err)
	}
	payload, padded, ok := TensorPaddedSize(layout, "w", "q8_0")
	if !ok {
		t.Fatal("TensorPaddedSize not ok")
	}
	if len(pred.Tensors) != 1 {
		t.Fatalf("pred.Tensors=%d want 1", len(pred.Tensors))
	}
	if pred.Tensors[0].PayloadBytes != payload || pred.Tensors[0].PaddedBytes != padded {
		t.Errorf("pred %+v != helper payload=%d padded=%d", pred.Tensors[0], payload, padded)
	}
}
