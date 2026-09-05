package pipeline

import (
	"encoding/json"
	"fmt"
	"os"

	gguf "fitgo/gguf"
)

// ---- analysis.json DTO ----------------------------------------------------

type analysisDoc struct {
	SchemaVersion int `json:"schema_version"`
	// Mode is the optimization direction: "up" (lower→upper, baseline lower)
	// or "down" (upper→lower, baseline upper). Default "up".
	Mode   string `json:"mode"`
	Source struct {
		Path      string `json:"path"`
		SizeBytes int64  `json:"size_bytes"`
		Sha256    string `json:"sha256"`
	} `json:"source"`
	Imatrix struct {
		Path   string `json:"path"`
		Sha256 string `json:"sha256"`
		Arg    string `json:"arg"`
	} `json:"imatrix"`
	Runtime struct {
		Dir           string `json:"dir"`
		LlamaQuantize string `json:"llama_quantize"`
	} `json:"runtime"`
	Presets struct {
		Lower struct {
			Name               string `json:"name"`
			FileType           int    `json:"file_type"`
			PredictedSizeBytes int64  `json:"predicted_size_bytes"`
		} `json:"lower"`
		Upper struct {
			Name               string `json:"name"`
			FileType           int    `json:"file_type"`
			PredictedSizeBytes int64  `json:"predicted_size_bytes"`
		} `json:"upper"`
	} `json:"presets"`
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
	BlockSpanAuto   int       `json:"block_span_auto"`
	Candidates      []candDTO `json:"candidates"`
	Rejected        []rejDTO  `json:"rejected"`
	CandidatesLower int       `json:"candidate_count,omitempty"`
	CandidateTensors int      `json:"candidate_tensors,omitempty"`
	LowerRecipe     recipeDTO `json:"lower_recipe"`
	UpperRecipe     recipeDTO `json:"upper_recipe"`
}

type candDTO struct {
	Tensor         string  `json:"tensor"`
	FromQtype      string  `json:"from_qtype"`
	ToQtype        string  `json:"to_qtype"`
	DeltaBytes     int     `json:"delta_bytes"`
	Importance     float64 `json:"importance"`
	RawImportance  float64 `json:"raw_importance"`
	ExpectedGain   float64 `json:"expected_gain"`
	UtilityPerByte float64 `json:"utility_per_byte"`
	Profiled       bool    `json:"profiled"`
	Block          *int    `json:"block"`
	Role           string  `json:"role"`
	Step           int     `json:"step,omitempty"`
	TotalSteps     int     `json:"total_steps,omitempty"`
}

type rejDTO struct {
	Tensor     string `json:"tensor"`
	FromQtype  string `json:"from_qtype"`
	ToQtype    string `json:"to_qtype"`
	DeltaBytes int    `json:"delta_bytes"`
	Reason     string `json:"reason"`
}

type recipeDTO struct {
	TotalTensors      int             `json:"total_tensors"`
	ReportedOrigBytes int             `json:"reported_orig_bytes"`
	ReportedNewBytes  int             `json:"reported_new_bytes"`
	Tensors           []assignmentDTO `json:"tensors"`
}

type assignmentDTO struct {
	Ordinal      int     `json:"ordinal"`
	TotalTensors int     `json:"total_tensors"`
	Name         string  `json:"name"`
	Shape        []int64 `json:"shape"`
	SrcType      string  `json:"src_type"`
	DstType      string  `json:"dst_type"`
	IsQuantized  bool    `json:"is_quantized"`
	OrigBytes    int     `json:"orig_bytes"`
	NewBytes     int     `json:"new_bytes"`
}

// LoadAnalysis loads analysis.json into an in-memory model for plan/quantize.
type Analysis struct {
	Path           string
	Mode           string
	SourcePath     string
	LowerPreset    string
	UpperPreset    string
	LowerFileType  int
	LowerSizeBytes int
	UpperSizeBytes int
	Candidates     []UpgradeCandidate
	Rejected       []RejectedTransition
	TensorCount    int
	Metadata       *gguf.QuantizationMetadata
	BlockSpanAuto  int
	LowerRecipe    *Recipe
	UpperRecipe    *Recipe
	LlamaQuantize  string
	ImatrixArg     string
}

// Direction returns the optimization direction; absent mode defaults to "up".
func (a *Analysis) Direction() string {
	if a.Mode == "down" {
		return "down"
	}
	return "up"
}

// BaselineRecipe returns the recipe that non-selected tensors keep: the lower
// preset in "up" mode, the upper preset in "down" mode.
func (a *Analysis) BaselineRecipe() *Recipe {
	if a.Direction() == "down" {
		return a.UpperRecipe
	}
	return a.LowerRecipe
}

// RecommendedPreset returns the llama-quantize preset that produces the
// baseline quality (the default ftype the output is quantized to).
func (a *Analysis) RecommendedPreset() string {
	if a.Direction() == "down" {
		return a.UpperPreset
	}
	return a.LowerPreset
}

// FloatBaselineDown reports whether the optimization runs in "down" mode from a
// float (F16/F32/BF16) baseline. llama-quantize only applies per-tensor
// overrides when the output ftype is quantized, so this case needs the nominal
// ftype + full-override trick (see EffectiveFtypePreset).
func (a *Analysis) FloatBaselineDown() bool {
	return a.Direction() == "down" && isFloatPreset(a.UpperPreset)
}

// EffectiveFtypePreset returns the preset actually passed to llama-quantize.
// For a float-baseline "down" run it substitutes a quantized nominal preset
// (Q8_0) — otherwise llama-quantize would ignore the tensor-type file entirely
// and emit every tensor at the float baseline.
func (a *Analysis) EffectiveFtypePreset() string {
	if a.FloatBaselineDown() {
		return "Q8_0"
	}
	return a.RecommendedPreset()
}

func LoadAnalysis(path string) (*Analysis, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("pipeline: read analysis %s: %v", path, err)
	}
	var doc analysisDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("pipeline: parse analysis: %v", err)
	}
	if doc.SchemaVersion != 1 {
		return nil, fmt.Errorf("pipeline: unsupported analysis schema %d", doc.SchemaVersion)
	}

	a := &Analysis{
		Path:           path,
		Mode:           doc.Mode,
		SourcePath:     doc.Source.Path,
		LowerPreset:    doc.Presets.Lower.Name,
		UpperPreset:    doc.Presets.Upper.Name,
		LowerFileType:  doc.Presets.Lower.FileType,
		LowerSizeBytes: int(doc.Presets.Lower.PredictedSizeBytes),
		UpperSizeBytes: int(doc.Presets.Upper.PredictedSizeBytes),
		BlockSpanAuto:  doc.BlockSpanAuto,
		LlamaQuantize:  doc.Runtime.LlamaQuantize,
		ImatrixArg:     doc.Imatrix.Arg,
		Metadata: &gguf.QuantizationMetadata{
			FileType:            doc.Metadata.FileType,
			QuantizationVersion: doc.Metadata.QuantizationVersion,
		},
	}
	if doc.Metadata.Imatrix != nil {
		im := doc.Metadata.Imatrix
		a.Metadata.Imatrix = &gguf.ImatrixProvenance{
			File: im.File, Dataset: im.Dataset,
			EntriesCount: im.EntriesCount, ChunksCount: im.ChunksCount,
		}
	}
	for _, c := range doc.Candidates {
		a.Candidates = append(a.Candidates, UpgradeCandidate{
			Tensor: c.Tensor, FromQtype: c.FromQtype, ToQtype: c.ToQtype,
			DeltaBytes: c.DeltaBytes, Importance: c.Importance, RawImportance: c.RawImportance,
			ExpectedGain: c.ExpectedGain, UtilityPerByte: c.UtilityPerByte,
			Profiled: c.Profiled, Role: c.Role, Block: -1,
			Step: c.Step, TotalSteps: c.TotalSteps,
		})
		if c.Block != nil {
			a.Candidates[len(a.Candidates)-1].Block = *c.Block
		}
	}
	a.TensorCount = doc.CandidateTensors
	if a.TensorCount <= 0 {
		// fallback for analysis files predating candidate_tensors
		seen := map[string]bool{}
		for _, c := range a.Candidates {
			seen[c.Tensor] = true
		}
		a.TensorCount = len(seen)
	}
	for _, r := range doc.Rejected {
		a.Rejected = append(a.Rejected, RejectedTransition{
			Tensor: r.Tensor, FromQtype: r.FromQtype, ToQtype: r.ToQtype,
			DeltaBytes: r.DeltaBytes, Reason: r.Reason,
		})
	}
	rec := &Recipe{TotalTensors: doc.LowerRecipe.TotalTensors}
	for _, t := range doc.LowerRecipe.Tensors {
		rec.Tensors = append(rec.Tensors, Assignment{
			Ordinal: t.Ordinal, TotalTensors: t.TotalTensors, Name: t.Name,
			Shape: t.Shape, SrcType: t.SrcType, DstType: t.DstType,
			IsQuantized: t.IsQuantized, OrigBytes: t.OrigBytes, NewBytes: t.NewBytes,
		})
	}
	rec.ReportedOrigBytes = doc.LowerRecipe.ReportedOrigBytes
	rec.ReportedNewBytes = doc.LowerRecipe.ReportedNewBytes
	a.LowerRecipe = rec

	urec := &Recipe{TotalTensors: doc.UpperRecipe.TotalTensors}
	for _, t := range doc.UpperRecipe.Tensors {
		urec.Tensors = append(urec.Tensors, Assignment{
			Ordinal: t.Ordinal, TotalTensors: t.TotalTensors, Name: t.Name,
			Shape: t.Shape, SrcType: t.SrcType, DstType: t.DstType,
			IsQuantized: t.IsQuantized, OrigBytes: t.OrigBytes, NewBytes: t.NewBytes,
		})
	}
	urec.ReportedOrigBytes = doc.UpperRecipe.ReportedOrigBytes
	urec.ReportedNewBytes = doc.UpperRecipe.ReportedNewBytes
	a.UpperRecipe = urec
	return a, nil
}
