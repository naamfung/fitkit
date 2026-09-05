package pipeline

import (
	"crypto/sha256"
	"fmt"
	"os"
)

// This file emits the structural artifacts that mirror the Python `fit` CLI's
// default outputs (analyze profile.json + plan.json/-recipe.json + the quantize
// record). Raw dry-run/oracle logs are intentionally not persisted.

// PlanRecord carries everything needed to write the schema-v1 plan.json record,
// mirroring the payload produced by Python pipeline.plan().
type PlanRecord struct {
	AnalysisPath       string
	AnalysisSHA256     string
	Policy             string
	TargetBytes        int
	LowerPreset        string
	UpperPreset        string
	LowerSizeBytes     int
	UpperSizeBytes     int
	PredictedSizeBytes int
	MetadataBytes      int
	TensorPayload      int
	TensorPadding      int
	UnusedBytes        int
	SelectedCount      int
	SkippedCount       int
	SelectedCostBytes  int
	OracleIterations   int
	RecipePath         string
	RecipeSHA256       string
	TensorTypesPath    string
	TensorTypesSHA256  string
	ModelName          string
	DominantQtype      string
	QtypeShares        map[string]float64
	SuggestedFilename  string
}

// WritePlanJSON writes the schema-v1 plan record to path.
func WritePlanJSON(r PlanRecord, path string) error {
	payload := map[string]any{
		"schema_version":          1,
		"fit_gguf_version":        "0.2.0",
		"analysis_path":           r.AnalysisPath,
		"analysis_sha256":         r.AnalysisSHA256,
		"policy":                  r.Policy,
		"seed":                    nil,
		"block_span":              nil,
		"fit":                     nil,
		"target_bytes":            r.TargetBytes,
		"lower_preset":            r.LowerPreset,
		"upper_preset":            r.UpperPreset,
		"lower_size_bytes":        r.LowerSizeBytes,
		"upper_size_bytes":        r.UpperSizeBytes,
		"predicted_size_bytes":    r.PredictedSizeBytes,
		"metadata_bytes":          r.MetadataBytes,
		"tensor_payload_bytes":    r.TensorPayload,
		"tensor_padding_bytes":    r.TensorPadding,
		"unused_bytes":            r.UnusedBytes,
		"selected_count":          r.SelectedCount,
		"skipped_count":           r.SkippedCount,
		"selected_cost_bytes":     r.SelectedCostBytes,
		"oracle_iterations":       r.OracleIterations,
		"recipe_path":             r.RecipePath,
		"recipe_sha256":           r.RecipeSHA256,
		"tensor_types_path":       r.TensorTypesPath,
		"tensor_types_sha256":     r.TensorTypesSHA256,
		"model_name":              r.ModelName,
		"dominant_qtype":          r.DominantQtype,
		"qtype_parameter_shares":  r.QtypeShares,
		"refine_profile":          nil,
		"fidelity":                nil,
		"suggested_filename":      r.SuggestedFilename,
	}
	return writeJSON(path, payload)
}

// WriteFitRecipe writes the schema-v1 FIT recipe record, mirroring Python
// optimizer.write_fit_recipe.
func WriteFitRecipe(plan *OptimizationPlan, path, lower, upper string) error {
	overrides := make([]map[string]any, 0, len(plan.Selected))
	for _, c := range plan.Selected {
		ov := map[string]any{
			"tensor":          c.Tensor,
			"from_qtype":      c.FromQtype,
			"to_qtype":        c.ToQtype,
			"delta_bytes":     c.DeltaBytes,
			"importance":      c.Importance,
			"raw_importance":  c.RawImportance,
			"expected_gain":   c.ExpectedGain,
			"utility_per_byte": c.UtilityPerByte,
			"profiled":        c.Profiled,
			"block":           c.Block,
			"role":            c.Role,
		}
		overrides = append(overrides, ov)
	}
	payload := map[string]any{
		"schema_version":       plan.SchemaVersion,
		"target_bytes":         plan.TargetBytes,
		"lower_preset":         lower,
		"upper_preset":         upper,
		"lower_size_bytes":     plan.LowerSizeBytes,
		"predicted_size_bytes": plan.PredictedSizeBytes,
		"unused_bytes":         plan.UnusedBytes,
		"selected_cost_bytes":  plan.SelectedCostBytes(),
		"selected_count":       len(plan.Selected),
		"skipped_count":        plan.SkippedCount,
		"overrides":            overrides,
	}
	return writeJSON(path, payload)
}

// WriteProfileJSON writes the deterministic profile record, mirroring Python
// imatrix.write_profile_json.
func WriteProfileJSON(profile *ImatrixProfile, path string) error {
	entries := make([]map[string]any, 0, len(profile.Entries))
	for _, e := range profile.Entries {
		entries = append(entries, map[string]any{
			"name":                 e.Name,
			"block":                e.Block,
			"role":                 e.Role,
			"width":                e.Width,
			"count_values":         e.CountValues,
			"count_min":            e.CountMin,
			"count_max":            e.CountMax,
			"count_sum":            e.CountSum,
			"mean":                 e.Mean,
			"rms":                  e.RMS,
			"stddev":               e.Stddev,
			"minimum":              e.Minimum,
			"p50":                  e.P50,
			"p95":                  e.P95,
			"p99":                  e.P99,
			"maximum":              e.Maximum,
			"nonzero_fraction":     e.NonzeroFraction,
			"global_relative_mean": e.GlobalRelativeMean,
			"role_relative_mean":   e.RoleRelativeMean,
			"global_percentile":    e.GlobalPercentile,
			"role_percentile":      e.RolePercentile,
		})
	}
	payload := map[string]any{
		"schema_version": profile.SchemaVersion,
		"source_file":    profile.SourceFile,
		"datasets":       profile.Datasets,
		"chunk_count":    profile.ChunkCount,
		"chunk_size":     profile.ChunkSize,
		"entry_count":    len(profile.Entries),
		"entries":        entries,
	}
	return writeJSON(path, payload)
}

// SHA256File returns the lowercase hex SHA-256 of a file's bytes.
func SHA256File(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), nil
}