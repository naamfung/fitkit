// Command fitpoc is the PoC driver: given the real source GGUF, an oracle
// dry-run log, the corresponding analysis.json, and the quantize-record.json,
// it recomputes the exact output size purely in Go and compares it byte-for-byte
// against what the Python pipeline recorded.
//
// Usage:
//
//	fitpoc <source.bf16.gguf> <oracle-log> <analysis.json> <quantize-record.json>
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"fitting/gguf"
)

type analysisDoc struct {
	Metadata struct {
		FileType            int `json:"file_type"`
		QuantizationVersion int `json:"quantization_version"`
		Imatrix             *struct {
			File         string `json:"file"`
			Dataset      string `json:"dataset,omitempty"`
			EntriesCount int    `json:"entries_count"`
			ChunksCount  int    `json:"chunks_count,omitempty"`
		} `json:"imatrix"`
	} `json:"metadata"`
}

type recordDoc struct {
	SizeBytes    int64 `json:"size_bytes"`
	Refinalized  int64 `json:"refinalized_expected_bytes"`
	ExpectBytes  *int  `json:"expect_bytes,omitempty"`
}

func main() {
	if len(os.Args) != 5 {
		fmt.Fprintf(os.Stderr, "usage: fitpoc <source.gguf> <oracle-log> <analysis.json> <quantize-record.json>\n")
		os.Exit(2)
	}
	source := os.Args[1]
	oracleLog := os.Args[2]
	analysisPath := os.Args[3]
	recordPath := os.Args[4]

	// 1) Read the real GGUF header/layout in Go.
	layout, err := gguf.ReadLayout(source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc: layout:", err)
		os.Exit(1)
	}
	fmt.Printf("layout   : %d tensors, align %d, rawMeta %d bytes\n",
		len(layout.Tensors), layout.Alignment, layout.RawMetadataBytes)

	// 2) Parse the oracle dry-run log into tensor→qtype in Go.
	recipe, err := gguf.ParseDstTypeMapFromFile(oracleLog)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc: dryrun:", err)
		os.Exit(1)
	}
	fmt.Printf("recipe   : %d tensors parsed\n", len(recipe))

	// 3) Load metadata (file_type, quantization_version, imatrix) from analysis.json.
	ab, err := os.ReadFile(analysisPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc:", err)
		os.Exit(1)
	}
	var an analysisDoc
	if err := json.Unmarshal(ab, &an); err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc: analysis json:", err)
		os.Exit(1)
	}
	meta := &gguf.QuantizationMetadata{
		FileType:            an.Metadata.FileType,
		QuantizationVersion: an.Metadata.QuantizationVersion,
	}
	if an.Metadata.Imatrix != nil {
		im := an.Metadata.Imatrix
		meta.Imatrix = &gguf.ImatrixProvenance{
			File:         im.File,
			Dataset:      im.Dataset,
			EntriesCount: im.EntriesCount,
			ChunksCount:  im.ChunksCount,
		}
	}

	// 4) Predict the exact size in Go.
	pred, err := gguf.PredictQuantizedSize(layout, recipe, meta)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc: predict:", err)
		os.Exit(1)
	}

	// 5) Read the recorded expected size from the quantize record.
	rb, err := os.ReadFile(recordPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc:", err)
		os.Exit(1)
	}
	var rec recordDoc
	if err := json.Unmarshal(rb, &rec); err != nil {
		fmt.Fprintln(os.Stderr, "fitpoc: record json:", err)
		os.Exit(1)
	}
	expected := rec.Refinalized
	if expected == 0 {
		expected = rec.SizeBytes
	}

	fmt.Printf("metadata : %d bytes\n", pred.MetadataBytes)
	fmt.Printf("payload  : %d bytes (padding %d)\n", pred.TensorPayload, pred.TensorPadding)
	fmt.Printf("total    : %d bytes\n", pred.TotalBytes)
	fmt.Printf("recorded : %d bytes\n", expected)

	if int64(pred.TotalBytes) == expected {
		fmt.Println("RESULT   : MATCH (byte-for-byte identical)")
	} else {
		fmt.Printf("RESULT   : MISMATCH (delta %d bytes)\n", int64(pred.TotalBytes)-expected)
		os.Exit(1)
	}
}