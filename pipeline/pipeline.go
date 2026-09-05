package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	gguf "fitgo/gguf"
)

// PresetFileTypes is defined in whitelist.go.

const kvOverrideStringMaxBytes = 127

// RuntimeBinary resolves a llama tool name to an existing file (Windows: .exe).
func RuntimeBinary(runtimeDir, name string) (string, error) {
	base := filepath.Join(runtimeDir, name)
	if fileExists(base) {
		return base, nil
	}
	exe := base + ".exe"
	if fileExists(exe) {
		return exe, nil
	}
	return "", fmt.Errorf("pipeline: %s not found in %s", name, runtimeDir)
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// isFloatPreset reports whether the preset is a non-quantized float type.
func isFloatPreset(name string) bool {
	switch strings.ToUpper(name) {
	case "F16", "F32", "BF16":
		return true
	}
	return false
}

func execCapture(argv []string) (string, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	var outBuf, errBuf strings.Builder
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	err := cmd.Run()
	out := outBuf.String() + errBuf.String()
	if err != nil {
		return out, fmt.Errorf("pipeline: %s: %w", argv[0], err)
	}
	return out, nil
}

// AllowRequantize, when true, adds --allow-requantize to every llama-quantize
// invocation so a pre-quantized source (e.g. a Q8_0 model) can be used as the
// parent. Safe for unquantized (BF16/F16/F32) sources. Set it before invoking
// Analyze/Plan/Quantize.
var AllowRequantize bool

// UseLadder, when true (default), generates multi-step ladder candidates so the
// optimizer can downgrade/upgrade each tensor part-way between the presets.
// Set to false (-no-ladder) to restore the binary per-tensor candidate model.
var UseLadder = true

// RunDryRun runs llama-quantize --dry-run and parses the recipe.
func RunDryRun(runtimeDir, source, imatrixArg, preset, tensorTypes string) (*Recipe, error) {
	binary, err := RuntimeBinary(runtimeDir, "llama-quantize")
	if err != nil {
		return nil, err
	}
	argv := []string{binary, "--dry-run"}
	if AllowRequantize {
		argv = append(argv, "--allow-requantize")
	}
	argv = append(argv, "--imatrix", imatrixArg)
	if tensorTypes != "" {
		argv = append(argv, "--tensor-type-file", tensorTypes)
	}
	argv = append(argv, source, preset)
	text, err := execCapture(argv)
	if err != nil {
		return nil, fmt.Errorf("dry-run for %s failed: %v", preset, err)
	}
	return ParseRecipe(text)
}

// ------------------------- analyze -----------------------------------------

type AnalyzeResult struct {
	AnalysisJSON string
}

// Analyze runs the source/imatrix pair through the pinned runtime and freezes
// an analysis.json artifact. mode is "up" (lower→upper) or "down" (upper→lower).
func Analyze(source, imatrix, runtimeDir, outDir, lower, upper, imatrixArg string, hashSources bool, mode string) (*AnalyzeResult, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}
	if imatrixArg == "" {
		imatrixArg = filepath.Base(imatrix)
	}
	if _, ok := PresetFileTypes[lower]; !ok {
		return nil, fmt.Errorf("pipeline: unknown lower preset %s", lower)
	}
	if _, ok := PresetFileTypes[upper]; !ok {
		return nil, fmt.Errorf("pipeline: unknown upper preset %s", upper)
	}
	if mode != "down" {
		mode = "up"
	}

	layout, err := gguf.ReadLayout(source)
	if err != nil {
		return nil, err
	}

	lowerRecipe, err := RunDryRun(runtimeDir, source, imatrixArg, lower, "")
	if err != nil {
		return nil, err
	}
	upperRecipe, err := RunDryRun(runtimeDir, source, imatrixArg, upper, "")
	if err != nil {
		return nil, err
	}

	profile, err := LoadImatrixProfile(imatrix)
	if err != nil {
		return nil, err
	}
	im := DeriveImatrixProvenance(imatrixArg, profile)
	// The output defaults to the baseline quality's ftype: the lower preset in
	// "up" mode, the upper preset in "down" mode. A float (F16/F32/BF16)
	// baseline in "down" mode must substitute a quantized nominal ftype (Q8_0):
	// llama-quantize only honors --tensor-type-file overrides when the output
	// ftype is quantized, so the tensors are overridden individually instead.
	effectivePreset := lower
	if mode == "down" {
		effectivePreset = upper
		if isFloatPreset(upper) {
			effectivePreset = "Q8_0"
		}
	}
	meta := &gguf.QuantizationMetadata{
		FileType:            PresetFileTypes[effectivePreset],
		QuantizationVersion: 2,
		Imatrix:             im,
	}
	lowerSize, err := gguf.PredictQuantizedSize(layout, RecipeDstMap(lowerRecipe), meta)
	if err != nil {
		return nil, err
	}
	upperSize, err := gguf.PredictQuantizedSize(layout, RecipeDstMap(upperRecipe), meta)
	if err != nil {
		return nil, err
	}
	if lowerSize.TotalBytes >= upperSize.TotalBytes {
		return nil, fmt.Errorf("pipeline: lower preset %s not below upper preset %s", lower, upper)
	}
	cs, err := GenerateUpgradeCandidates(lowerRecipe, upperRecipe, lowerSize, upperSize, profile, mode)
	if UseLadder {
		cs, err = GenerateLadderCandidates(lowerRecipe, upperRecipe, lowerSize, upperSize, layout, profile, mode)
	}
	if err != nil {
		return nil, err
	}

	doc := makeAnalysisDoc(source, imatrix, imatrixArg, runtimeDir, lower, upper, mode,
		meta, lowerRecipe, upperRecipe, cs, profile, lowerSize, upperSize, hashSources)
	analysisPath := filepath.Join(outDir, "analysis.json")
	if err := writeJSON(analysisPath, doc); err != nil {
		return nil, err
	}
	return &AnalyzeResult{AnalysisJSON: analysisPath}, nil
}

// ------------------------- plan --------------------------------------------

// Plan selects the size-exact recipe and writes plan records + tensor-types.
func Plan(analysisPath, outPrefix string, targetBytes int, policy string, blockSpan string, modelName string) (*OptimizationPlan, *gguf.Prediction, error) {
	a, err := LoadAnalysis(analysisPath)
	if err != nil {
		return nil, nil, err
	}
	if targetBytes < a.LowerSizeBytes || targetBytes > a.UpperSizeBytes {
		return nil, nil, fmt.Errorf("pipeline: target %d outside preset range [%d, %d]",
			targetBytes, a.LowerSizeBytes, a.UpperSizeBytes)
	}

	cs := &CandidateSet{
		Candidates: a.Candidates, Rejected: a.Rejected,
		LowerSizeBytes: a.LowerSizeBytes, UpperSizeBytes: a.UpperSizeBytes,
		Direction: a.Direction(),
	}

	resolvedSpan := a.BlockSpanAuto
	if blockSpan == "auto" {
		// keep analysis value
	} else {
		fmt.Sscanf(blockSpan, "%d", &resolvedSpan)
	}

	layout, err := gguf.ReadLayout(a.SourcePath)
	if err != nil {
		return nil, nil, err
	}

	selectOpt := func(t int) (*OptimizationPlan, error) {
		switch policy {
		case "original":
			return optimizeGreedy(t, cs)
		case "random":
			return optimizeRandom(t, cs, "random")
		default:
			return optimizeBlockBalanced(t, cs, resolvedSpan)
		}
	}

	optimization, err := selectOpt(targetBytes)
	if err != nil {
		return nil, nil, err
	}

	// oracle loop
	effectiveTarget := targetBytes
	var prediction *gguf.Prediction
	oracleIterations := 0
	for iter := 1; iter <= 8; iter++ {
		oracleIterations = iter
		writePlanTensorTypes(a, optimization, outPrefix+"-tensor-types.txt")
		recipe, err := RunDryRun(filepath.Dir(a.LlamaQuantize), a.SourcePath,
			a.ImatrixArg, a.EffectiveFtypePreset(), outPrefix+"-tensor-types.txt")
		if err != nil {
			return nil, nil, err
		}
		prediction, err = gguf.PredictQuantizedSize(layout, RecipeDstMap(recipe), a.Metadata)
		if err != nil {
			return nil, nil, err
		}
		if prediction.TotalBytes <= targetBytes {
			break
		}
		if iter >= 8 {
			return nil, nil, fmt.Errorf("pipeline: oracle prediction %d still exceeds target %d", prediction.TotalBytes, targetBytes)
		}
		effectiveTarget = effectiveTarget - (prediction.TotalBytes - effectiveTarget) - (1 << 20)
		optimization, err = selectOpt(effectiveTarget)
		if err != nil {
			return nil, nil, fmt.Errorf("oracle loop failed to converge: %v", err)
		}
	}

	// final tensor-types (if the loop already wrote it, rewrite to match)
	writePlanTensorTypes(a, optimization, outPrefix+"-tensor-types.txt")
	optimization.OracleIterations = oracleIterations
	_ = modelName
	return optimization, prediction, nil
}

// ------------------------- quantize ----------------------------------------

// Quantize runs the real quantizer and returns actual bytes + recorded expected.
// When recordPath is non-empty, it writes the schema-v1 .quantize-record.json
// (mirroring the Python fit quantize step); pass "" to skip.
func Quantize(analysisPath, tensorTypesPath, outPath string, expectBytes int, imatrixArg, recordPath string) (int, int, error) {
	a, err := LoadAnalysis(analysisPath)
	if err != nil {
		return 0, 0, err
	}
	binary := a.LlamaQuantize
	if !fileExists(binary) {
		return 0, 0, fmt.Errorf("pipeline: llama-quantize not found at %s", binary)
	}
	if imatrixArg == "" {
		imatrixArg = a.ImatrixArg
	}

	// predict expected from THIS invocation
	layout, err := gguf.ReadLayout(a.SourcePath)
	if err != nil {
		return 0, 0, err
	}
	// re-finalize metadata with the actual imatrix path string used
	meta := a.Metadata

	// effective recipe dry-run with overrides
	oracle, err := RunDryRun(filepath.Dir(binary), a.SourcePath, imatrixArg, a.EffectiveFtypePreset(), tensorTypesPath)
	if err != nil {
		return 0, 0, err
	}
	pred, err := gguf.PredictQuantizedSize(layout, RecipeDstMap(oracle), meta)
	if err != nil {
		return 0, 0, err
	}
	expected := pred.TotalBytes

	argv := []string{binary, "--imatrix", imatrixArg, "--tensor-type-file", tensorTypesPath}
	if AllowRequantize {
		argv = append(argv, "--allow-requantize")
	}
	argv = append(argv, a.SourcePath, outPath, a.EffectiveFtypePreset())
	raw, rc := execCaptureRC(argv)

	st, err := os.Stat(outPath)
	if err != nil {
		return 0, 0, err
	}
	actual := int(st.Size())

	if recordPath != "" {
		if err := writeQuantizeRecord(recordPath, a, tensorTypesPath, argv, rc, outPath, actual, expected, expectBytes, imatrixArg, raw); err != nil {
			return actual, expected, err
		}
	}

	if expectBytes != 0 && actual != expectBytes {
		return actual, expected, fmt.Errorf("pipeline: output %d bytes; expected %d", actual, expectBytes)
	}
	return actual, expected, nil
}

func writeQuantizeRecord(recordPath string, a *Analysis, tensorTypesPath string, argv []string, rc int, outPath string, actual, expected, expectBytes int, imatrixArg, raw string) error {
	tensorTypesSHA, _ := SHA256File(tensorTypesPath)
	outSHA, _ := SHA256File(outPath)
	tail := raw
	if len(tail) > 2000 {
		tail = tail[len(tail)-2000:]
	}
	payload := map[string]any{
		"schema_version":                1,
		"analysis_path":                 a.Path,
		"tensor_types_path":             tensorTypesPath,
		"tensor_types_sha256":           tensorTypesSHA,
		"command":                       argv,
		"returncode":                    rc,
		"output_path":                   outPath,
		"size_bytes":                    actual,
		"imatrix_arg":                   imatrixArg,
		"analysis_imatrix_arg":          a.ImatrixArg,
		"refinalized_expected_bytes":    expected,
		"size_matches_refinalization":   actual == expected,
		"expect_bytes":                  expectBytes,
		"size_matches_expectation":      expectBytes == 0 || actual == expectBytes,
		"sha256":                        outSHA,
		"stderr_tail":                   tail,
	}
	return writeJSON(recordPath, payload)
}

// execCaptureRC runs a command, returning combined stdout+stderr and the exit
// code (or -1 when it could not be started).
func execCaptureRC(argv []string) (string, int) {
	cmd := exec.Command(argv[0], argv[1:]...)
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	rc := 0
	if err := cmd.Run(); err != nil {
		rc = -1
		if ee, ok := err.(*exec.ExitError); ok {
			rc = ee.ExitCode()
		}
	}
	return out.String(), rc
}

// ------------------------- helpers -----------------------------------------

func RecipeDstMap(rec *Recipe) map[string]string {
	m := make(map[string]string, len(rec.Tensors))
	for i := range rec.Tensors {
		m[rec.Tensors[i].Name] = lowerType(rec.Tensors[i].DstType)
	}
	return m
}

// writePlanTensorTypes writes the tensor-type file that the oracle dry-run and
// the final quantize will consume. Normally only the selected overrides are
// written (the baseline ftype covers the rest). For a float-baseline "down"
// run the file must spell out EVERY tensor — the nominal ftype is quantized,
// so unselected tensors are pinned to the float baseline explicitly.
func writePlanTensorTypes(a *Analysis, plan *OptimizationPlan, path string) {
	if !a.FloatBaselineDown() {
		writeTensorTypeFile(plan, path)
		return
	}
	selected := map[string]string{}
	for _, c := range plan.Selected {
		selected[c.Tensor] = c.ToQtype
	}
	var sb strings.Builder
	n := 0
	for _, t := range a.UpperRecipe.Tensors {
		q := t.DstType
		if v, ok := selected[t.Name]; ok {
			q = v
		}
		if n > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("^")
		sb.WriteString(regexpEscape(t.Name))
		sb.WriteString("$=")
		sb.WriteString(lowerType(q))
		n++
	}
	if sb.Len() > 0 {
		sb.WriteString("\n")
	}
	_ = os.WriteFile(path, []byte(sb.String()), 0o644)
}

func WriteTensorTypeFile(plan *OptimizationPlan, path string) error {
	writeTensorTypeFile(plan, path)
	return nil
}

func writeTensorTypeFile(plan *OptimizationPlan, path string) {
	ov := plan.Overrides()
	// llama_integration.py writes overrides sorted by tensor name.
	sort.SliceStable(ov, func(a, b int) bool {
		if ov[a][0] != ov[b][0] {
			return ov[a][0] < ov[b][0]
		}
		return ov[a][1] < ov[b][1]
	})
	var sb strings.Builder
	for i, pair := range ov {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("^")
		sb.WriteString(regexpEscape(pair[0]))
		sb.WriteString("$=")
		sb.WriteString(pair[1])
	}
	if sb.Len() > 0 {
		sb.WriteString("\n")
	}
	_ = os.WriteFile(path, []byte(sb.String()), 0o644)
}

func regexpEscape(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if strings.ContainsRune(`\.^$|()[]{}*+?`, r) {
			sb.WriteByte('\\')
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

func DeriveImatrixProvenance(imatrixArg string, profile *ImatrixProfile) *gguf.ImatrixProvenance {
	file := truncateKVBytes(imatrixArg)
	var dataset string
	if len(profile.Datasets) > 0 {
		dataset = truncateKVBytes(profile.Datasets[0])
	}
	return &gguf.ImatrixProvenance{
		File:         file,
		Dataset:      dataset,
		EntriesCount: len(profile.Entries),
		ChunksCount:  profile.ChunkCount,
	}
}

func truncateKVBytes(s string) string {
	b := []byte(s)
	if len(b) > kvOverrideStringMaxBytes {
		b = b[:kvOverrideStringMaxBytes]
	}
	return string(b)
}

// countsInDistribution reports whether a recipe tensor participates in the
// qtype parameter distribution. Quantized tensors always count, and so do
// float tensors the plan explicitly keeps as its baseline (the BF16/F16/F32
// upper preset of a "down" run — llama-quantize reports those as unchanged
// when the source already carries the same float type). Only auxiliary float
// tensors that stay F32 untouched are excluded.
func countsInDistribution(t Assignment) bool {
	if t.IsQuantized {
		return true
	}
	if !isFloatPreset(t.DstType) {
		return true
	}
	return lowerType(t.SrcType) != "f32"
}

func DefaultModelName(sourcePath string) string {
	stem := gguf.BaseNameNoSplit(sourcePath)
	if ln := strings.ToUpper(stem); strings.HasSuffix(ln, "-BF16") {
		stem = stem[:len(stem)-5]
	}
	return stem
}

// ApplyOverrides rebuilds the lower-preset recipe with the plan's upgrades.
func ApplyOverrides(lower *Recipe, plan *OptimizationPlan) *Recipe {
	over := map[string]string{}
	for _, c := range plan.Selected {
		over[c.Tensor] = c.ToQtype
	}
	out := &Recipe{TotalTensors: lower.TotalTensors,
		ReportedOrigBytes: lower.ReportedOrigBytes, ReportedNewBytes: lower.ReportedNewBytes}
	out.Tensors = make([]Assignment, len(lower.Tensors))
	copy(out.Tensors, lower.Tensors)
	for i := range out.Tensors {
		if v, ok := over[out.Tensors[i].Name]; ok {
			out.Tensors[i].DstType = v
		}
	}
	return out
}

// QtypeParameterDistribution counts parameter elements per destination qtype
// over the plan-covered tensors of a recipe (see countsInDistribution).
func QtypeParameterDistribution(rec *Recipe) map[string]int {
	dist := map[string]int{}
	for i := range rec.Tensors {
		t := rec.Tensors[i]
		if !countsInDistribution(t) {
			continue
		}
		e := 1
		for _, d := range t.Shape {
			e *= int(d)
		}
		k := lowerType(t.DstType)
		dist[k] += e
	}
	return dist
}

// DominantQtype returns the element-weighted dominant qtype (uppercase) or "".
func DominantQtype(dist map[string]int) string {
	if len(dist) == 0 {
		return ""
	}
	keys := make([]string, 0, len(dist))
	for k := range dist {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool {
		if dist[keys[a]] != dist[keys[b]] {
			return dist[keys[a]] > dist[keys[b]]
		}
		return keys[a] < keys[b]
	})
	return strings.ToUpper(keys[0])
}

// SuggestedFilename implements the release naming convention. direction is the
// mode tag embedded in the name ("UP" or "DOWN").
func SuggestedFilename(modelName string, targetBytes int, dominant, direction string) string {
	gib := float64(targetBytes) / (1 << 30)
	var label string
	switch {
	case abs(gib-float64(int64(gib+0.5))) < 1e-6:
		label = fmt.Sprintf("%dG", maxInt(1, int64(gib+0.5)))
	case abs(gib*2-float64(int64(gib*2+0.5))) < 1e-6:
		label = fmt.Sprintf("%vG", float64(int64(gib*2+0.5))/2)
	default:
		if gib >= 0.5 {
			label = fmt.Sprintf("%dG", int64(gib+0.5))
		} else {
			label = "1G"
		}
	}
	return fmt.Sprintf("%s-FITKIT-%s-%s-%s.gguf", modelName, direction, label, dominant)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func maxInt(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func writeJSON(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	// Match the Python CLI: strings are emitted as-is (no \u003c-style HTML
	// escaping), then a trailing newline (Encoder.Encode appends one).
	enc.SetEscapeHTML(false)
	encRes := enc.Encode(v)
	closeRes := f.Close()
	if encRes != nil {
		return encRes
	}
	return closeRes
}

func makeAnalysisDoc(source, imatrix, imatrixArg, runtimeDir, lower, upper, mode string,
	meta *gguf.QuantizationMetadata, lowerRecipe, upperRecipe *Recipe, cs *CandidateSet,
	profile *ImatrixProfile, lowerSize, upperSize *gguf.Prediction, hashSources bool) map[string]any {

	quantBin := filepath.Join(runtimeDir, "llama-quantize")
	if rb, err := RuntimeBinary(runtimeDir, "llama-quantize"); err == nil {
		quantBin = rb
	}
	return map[string]any{
		"schema_version":   1,
		"fit_gguf_version": "0.2.0",
		"mode":             mode,
		"source":           map[string]any{"path": source, "size_bytes": fileSize(source), "sha256": nil},
		"imatrix":          map[string]any{"path": imatrix, "sha256": nil, "arg": imatrixArg},
		"runtime":          map[string]any{"dir": runtimeDir, "llama_quantize": quantBin},
		"presets": map[string]any{
			"lower": map[string]any{"name": lower, "file_type": PresetFileTypes[lower], "predicted_size_bytes": lowerSize.TotalBytes, "metadata_bytes": lowerSize.MetadataBytes, "tensor_payload_bytes": lowerSize.TensorPayload, "tensor_padding_bytes": lowerSize.TensorPadding},
			"upper": map[string]any{"name": upper, "file_type": PresetFileTypes[upper], "predicted_size_bytes": upperSize.TotalBytes, "metadata_bytes": upperSize.MetadataBytes, "tensor_payload_bytes": upperSize.TensorPayload, "tensor_padding_bytes": upperSize.TensorPadding},
		},
		"metadata": map[string]any{
			"file_type": meta.FileType, "quantization_version": meta.QuantizationVersion,
			"imatrix": map[string]any{"file": meta.Imatrix.File, "dataset": meta.Imatrix.Dataset,
				"entries_count": meta.Imatrix.EntriesCount, "chunks_count": meta.Imatrix.ChunksCount},
		},
		"block_span_auto":      AutoBlockSpan(profile),
		"net_preset_gap_bytes": upperSize.TotalBytes - lowerSize.TotalBytes,
		"candidate_count":      len(cs.Candidates),
		"candidate_tensors":    cs.TensorCount,
		"rejected_count":       len(cs.Rejected),
		"candidates":           cs.Candidates,
		"rejected":             cs.Rejected,
		"lower_recipe":         recipeToJSON(lowerRecipe),
		"upper_recipe":         recipeToJSON(upperRecipe),
	}
}

func recipeToJSON(rec *Recipe) map[string]any {
	tensors := make([]map[string]any, 0, len(rec.Tensors))
	for i := range rec.Tensors {
		t := rec.Tensors[i]
		tensors = append(tensors, map[string]any{
			"ordinal": t.Ordinal, "total_tensors": t.TotalTensors, "name": t.Name,
			"shape": t.Shape, "src_type": t.SrcType, "dst_type": t.DstType,
			"is_quantized": t.IsQuantized, "orig_bytes": t.OrigBytes, "new_bytes": t.NewBytes,
		})
	}
	return map[string]any{
		"total_tensors": rec.TotalTensors, "reported_orig_bytes": rec.ReportedOrigBytes,
		"reported_new_bytes": rec.ReportedNewBytes, "tensors": tensors,
	}
}

func fileSize(p string) int64 {
	if st, err := os.Stat(p); err == nil {
		return st.Size()
	}
	return 0
}

var _ = sort.Ints
