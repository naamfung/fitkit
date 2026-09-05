package fidelity

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"fitgo/gguf"
	"fitgo/pipeline"
)

// Budgets is the search budget gate: Normal <= 8, Precise <= 16.
var Budgets = map[string]int{"normal": 8, "precise": 16}

// ExpandPresetLadder analyzes adjacent preset pairs and returns window
// analysis.json paths. Poison lower presets are skipped (healthy frontier).
func ExpandPresetLadder(ladder []string, source, imatrix, runtime, analysesDir, imatrixArg string, hashSources bool) ([]string, error) {
	if len(ladder) < 2 {
		return nil, fmt.Errorf("preset ladder needs at least two presets")
	}
	if err := os.MkdirAll(analysesDir, 0o755); err != nil {
		return nil, err
	}
	var windows []string
	for i := 0; i < len(ladder)-1; i++ {
		lower := ladder[i]
		upper := ladder[i+1]
		if PoisonPresets[strings.ToUpper(lower)] {
			continue
		}
		outDir := filepath.Join(analysesDir, "analysis-"+lower+"-"+upper)
		analysisFile := filepath.Join(outDir, "analysis.json")
		if !fileExists(analysisFile) {
			if _, err := pipeline.Analyze(source, imatrix, runtime, outDir,
				lower, upper, imatrixArg, hashSources, "up"); err != nil {
				return nil, err
			}
		}
		windows = append(windows, analysisFile)
	}
	return windows, nil
}

// PlanExactSize finds a byte target whose plan delivers exactly targetSize.
func PlanExactSize(analysisPath string, targetSize, upperBound int, modelName string, outPrefix string, maxSteps int) (string, error) {
	if maxSteps <= 0 {
		maxSteps = 40
	}
	low := targetSize
	high := upperBound
	if high <= targetSize {
		high = targetSize + 1
	}
	tryPlan := func(target int) (int, bool) {
		opt, pred, err := pipeline.Plan(analysisPath, outPrefix, target, "balanced", "auto", modelName)
		if err != nil {
			return 0, false
		}
		_ = opt
		return pred.TotalBytes, true
	}
	for i := 0; i < maxSteps; i++ {
		if high-low <= 1 {
			break
		}
		mid := (low + high) / 2
		predicted, ok := tryPlan(mid)
		if !ok {
			high = mid
			continue
		}
		if predicted == targetSize {
			return outPrefix, nil
		}
		if predicted < targetSize {
			low = mid
		} else {
			high = mid
		}
	}
	predicted, ok := tryPlan(high)
	if ok && predicted == targetSize {
		return outPrefix, nil
	}
	return "", fmt.Errorf("exact size %d is not deliverable by this window's candidate ladder", targetSize)
}

func defaultModelName(source string) string {
	return gguf.BaseNameNoSplit(source)
}

// FidelitySearchProduct runs the full product chain for one tier and produces
// the final GGUF. Mirrors fit_gguf.product.fidelity_search_product.
func FidelitySearchProduct(opts ProductOptions) (map[string]any, error) {
	profile := opts.Profile
	if profile == "" {
		profile = "normal"
	}
	budget, ok := Budgets[profile]
	if !ok {
		return nil, fmt.Errorf("profile must be one of [normal precise], got %q", profile)
	}
	tierKey := strings.TrimSpace(strings.ToLower(opts.Tier))
	modelName := opts.ModelName
	if modelName == "" {
		modelName = defaultModelName(opts.Source)
	}

	if opts.FreezePath == "" {
		return nil, &EvalProvenanceError{msg: "freeze_path is required for the product path: pass the frozen eval-v1 FREEZE.json"}
	}
	refManifest := opts.ReferenceManifestPath
	if refManifest == "" {
		var candidates []string
		if matches, err := filepath.Glob(filepath.Join(filepath.Dir(opts.FreezePath), "reference-manifest-*.json")); err == nil {
			candidates = matches
		}
		sort.Strings(candidates)
		if len(candidates) == 1 {
			refManifest = candidates[0]
		} else {
			return nil, &EvalProvenanceError{msg: "reference_manifest_path is required (no unique reference-manifest-*.json found next to the freeze file)"}
		}
	}
	provenance, err := VerifyEvalV1Provenance(opts.RefsDir, opts.EvalDataDir, opts.FreezePath, refManifest, "")
	if err != nil {
		return nil, err
	}

	sourceSHA256 := ""
	contract, err := ResolveContract(modelName, tierKey, opts.GuardRegistry, "")
	if err != nil {
		sourceSHA256, _ = SHA256File(opts.Source)
		contract, err = ResolveContract(modelName, tierKey, opts.GuardRegistry, sourceSHA256)
		if err != nil {
			return nil, err
		}
	}
	if sourceSHA256 != "" && provenance.SourceBF16GGUFSHA256 != "" &&
		!strings.EqualFold(provenance.SourceBF16GGUFSHA256, sourceSHA256) {
		return nil, &EvalProvenanceError{msg: fmt.Sprintf(
			"guard weights binding disagrees with the reference manifest: %s != %s",
			sourceSHA256, provenance.SourceBF16GGUFSHA256)}
	}

	var analysisPaths []string
	if len(opts.AnalysisDirs) > 0 {
		analysisPaths = opts.AnalysisDirs
	} else if len(opts.PresetLadder) > 0 {
		analysisPaths, err = ExpandPresetLadder(
			opts.PresetLadder, opts.Source, opts.Imatrix, opts.Runtime,
			filepath.Join(opts.OutDir, "analyses"), opts.ImatrixArg, opts.HashSources)
		if err != nil {
			return nil, err
		}
	} else {
		return nil, fmt.Errorf("provide either preset_ladder or analysis_dirs")
	}

	windows, err := DiscoverWindows(analysisPaths)
	if err != nil {
		return nil, err
	}
	var healthy []Window
	for _, w := range windows {
		if w.Healthy() {
			healthy = append(healthy, w)
		}
	}
	if len(healthy) == 0 {
		return nil, fmt.Errorf("no healthy windows — tier is NOT REACHABLE")
	}
	if opts.MinSize == 0 {
		opts.MinSize = healthy[0].LowerSize
		for _, w := range healthy {
			if w.LowerSize < opts.MinSize {
				opts.MinSize = w.LowerSize
			}
		}
	}
	if opts.MaxSize == 0 {
		opts.MaxSize = healthy[0].UpperSize
		for _, w := range healthy {
			if w.UpperSize > opts.MaxSize {
				opts.MaxSize = w.UpperSize
			}
		}
	}

	seedPrefix := opts.SeedPrefix
	if seedPrefix == "" {
		seedPrefix = modelName + "-"
	}
	exclude := map[string]bool{}
	for _, n := range opts.ExcludeSeeds {
		exclude[n] = true
	}
	seeds, err := LoadSeeds(opts.ManifestPath, opts.LogsDir, seedPrefix, SeedOptions{
		PoisonNames:             PoisonPresets,
		ExcludeNames:            exclude,
		RequireSeedProvenance:   true,
		ReferenceManifestSHA256: provenance.ReferenceManifestFileSHA256,
	})
	if err != nil {
		return nil, err
	}

	config := RunnerConfig{
		Runtime: opts.Runtime, Imatrix: opts.Imatrix,
		RefsDir: opts.RefsDir, EvalDataDir: opts.EvalDataDir,
		WorkDir: opts.WorkDir, OutDir: opts.OutDir,
		ModelName: modelName, GuardRegistry: opts.GuardRegistry,
		RefineProfile: opts.RefineProfile, Threads: opts.Threads, EvalConcurrency: opts.EvalConcurrency,
		EvalProvenance: provenance, RequireEvalProvenance: true, SourceSHA256: sourceSHA256,
	}

	summary, err := RunTierSearch(contract, config, windows, seeds,
		opts.MinSize, opts.MaxSize, budget, opts.ToleranceBytes, opts.ManifestPath, opts.LogsDir)
	if err != nil {
		return nil, err
	}

	status := fstring(summary["status"])
	best := summary["best"].(map[string]any)
	if status != "verified_pass" || len(best) == 0 {
		summary["artifact"] = nil
		if status == "no_pass" {
			summary["product_status"] = "NOT REACHABLE within the validated healthy frontier"
		}
		summary["guard_binding"] = map[string]any{"model_name": modelName, "source_sha256": strOrNil(sourceSHA256)}
		_ = writeSummaryProduct(opts.OutDir, tierKey, summary)
		return summary, nil
	}

	bestSize := fintMap(best["size_bytes"])
	recipes, _ := summary["artifact_recipes"].(map[string]any)
	analyses, _ := summary["artifact_analyses"].(map[string]any)
	recipe := fstring(recipes[strconv.Itoa(bestSize)])
	analysisDir := fstring(analyses[strconv.Itoa(bestSize)])
	if analysisDir == "" {
		analysisDir = bestAnalysis(bestSize, windows)
	}
	var tensorTypes string
	if recipe != "" {
		tensorTypes = recipe
	} else {
		windowUpper := bestSize
		for _, w := range windows {
			if filepath.Clean(w.AnalysisPath) == filepath.Clean(analysisDir) {
				windowUpper = w.UpperSize
				break
			}
		}
		prefix := filepath.Join(opts.OutDir, "final-plan")
		finalPrefix, err := PlanExactSize(filepath.Join(analysisDir, "analysis.json"),
			bestSize, windowUpper, modelName, prefix, 40)
		if err != nil {
			return nil, err
		}
		tensorTypes = finalPrefix + "-tensor-types.txt"
	}

	outputPath := opts.Output
	if outputPath == "" {
		outputPath = filepath.Join(opts.OutDir,
			fmt.Sprintf("%s-FITKIT-%s-%.2fGiB.gguf", modelName, strings.ToUpper(tierKey), float64(bestSize)/(1<<30)))
	}
	actual, refinalized, err := pipeline.Quantize(
		filepath.Join(analysisDir, "analysis.json"), tensorTypes, outputPath,
		0, opts.Imatrix, "")
	if err != nil {
		return nil, err
	}

	// final artifact must satisfy the contract on its own bytes
	verifier, err := NewSearchExecutor(config, windows, contract, opts.ManifestPath, opts.LogsDir,
		map[int]bool{}, nil, nil, "")
	if err != nil {
		return nil, err
	}
	verifyTag := fmt.Sprintf("%s-final-verify-%s", modelName, tierKey)
	metrics, err := verifier.evalDomains(outputPath, verifyTag)
	if err != nil {
		return nil, fmt.Errorf("final artifact verification eval failed")
	}
	var macroKL, macroTop float64
	for _, d := range Domains {
		macroKL += ffv(metrics[d]["mean_kld"])
		macroTop += ffv(metrics[d]["same_top_pct"])
	}
	macroKL /= float64(len(Domains))
	macroTop /= float64(len(Domains))
	if !contract.Passes(macroKL, macroTop/100.0) {
		return nil, fmt.Errorf("final artifact violates the %s contract: kld %.4f top %.2f", tierKey, macroKL, macroTop)
	}
	if !manifestHas(opts.ManifestPath, verifyTag) {
		digest, _ := SHA256File(outputPath)
		if f, err := os.OpenFile(opts.ManifestPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
			f.WriteString(fmt.Sprintf("%s  %d  %s\n", verifyTag, actual, digest))
			f.Close()
		}
	}

	summary["guard_binding"] = map[string]any{"model_name": modelName, "source_sha256": strOrNil(sourceSHA256)}
	summary["artifact"] = map[string]any{
		"path":                 outputPath,
		"size_bytes":           actual,
		"g2_delta":             actual - refinalized,
		"naming":               fmt.Sprintf("%s-FITKIT-%s-<size>-<primary-qtype>.gguf", modelName, strings.ToUpper(tierKey)),
		"search_tolerance_mib": opts.ToleranceBytes / (1024 * 1024),
		"active_constraint":    summary["active_constraint"],
		"healthy_frontier":     true,
		"verified_metrics":     map[string]any{"macro_kl": round9(macroKL), "same_top_pct": round9(macroTop)},
	}
	_ = writeSummaryProduct(opts.OutDir, tierKey, summary)
	return summary, nil
}

// ProductOptions is the product-path input bundle.
type ProductOptions struct {
	Source                string
	Imatrix               string
	Runtime               string
	RefsDir               string
	EvalDataDir           string
	GuardRegistry         string
	Tier                  string
	OutDir                string
	WorkDir               string
	ManifestPath          string
	LogsDir               string
	ModelName             string
	PresetLadder          []string
	AnalysisDirs          []string
	RefineProfile         string
	Profile               string
	ToleranceBytes        int
	Output                string
	Threads               int
	EvalConcurrency       int
	ImatrixArg            string
	HashSources           bool
	SeedPrefix            string
	ExcludeSeeds          []string
	FreezePath            string
	ReferenceManifestPath string
	MinSize               int
	MaxSize               int
}

func writeSummaryProduct(outDir, tier string, summary map[string]any) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(summary, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outDir, "fidelity-search-"+tier+"-product.json"), append(data, '\n'), 0o644)
}

func bestAnalysis(bestSize int, windows []Window) string {
	for _, w := range windows {
		if w.LowerSize <= bestSize && bestSize <= w.UpperSize {
			return w.AnalysisPath
		}
	}
	return ""
}

func fintMap(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	case json.Number:
		if f, err := t.Float64(); err == nil {
			return int(f)
		}
	}
	return 0
}
