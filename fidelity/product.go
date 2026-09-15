package fidelity

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"fitting/gguf"
	"fitting/pipeline"
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
func PlanExactSize(analysisPath string, targetSize, upperBound int, modelName string, refineProfile string, outPrefix string, maxSteps int) (string, error) {
	if maxSteps <= 0 {
		maxSteps = 40
	}
	low := targetSize
	high := upperBound
	if high <= targetSize {
		high = targetSize + 1
	}
	tryPlan := func(target int) (int, bool) {
		opt, pred, err := pipeline.Plan(analysisPath, outPrefix, target, "balanced", "auto", modelName, refineProfile)
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
		// The manifest belongs to the model, not to the freeze: one freeze
		// covers every model, and each model's bundle carries its own manifest
		// beside its references (the `fit calibrate` layout). Only fall back to
		// the freeze directory for the v0.2 bootstrap layout.
		var err error
		refManifest, err = DiscoverReferenceManifest(opts.RefsDir, opts.FreezePath)
		if err != nil {
			return nil, err
		}
	}
	provenance, err := VerifyEvalV1Provenance(opts.RefsDir, opts.EvalDataDir, opts.FreezePath, refManifest, "")
	if err != nil {
		return nil, err
	}

	// The references were generated from specific BF16 weights. Quantizing
	// different weights against them yields meaningless KL numbers, so this
	// binding is checked unconditionally — since Contract v2 it can no longer
	// ride along on "a Guard pins the weights", because a tier no longer
	// requires a Guard at all.
	sourceSHA256, err := SHA256File(opts.Source)
	if err != nil {
		return nil, err
	}
	if !strings.EqualFold(provenance.SourceBF16GGUFSHA256, sourceSHA256) {
		return nil, &EvalProvenanceError{msg: fmt.Sprintf(
			"the source GGUF and the reference manifest disagree on the BF16 weights: %s != %s — the references belong to different weights",
			sourceSHA256, provenance.SourceBF16GGUFSHA256)}
	}
	contract, err := ResolveContract(modelName, tierKey, opts.GuardRegistry, sourceSHA256)
	if err != nil {
		return nil, err
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

	// Historical manifests may use a shorter naming convention than the guard
	// identifier, so the seed prefix is explicit — but the default is a
	// match-all, because the manifest and log directory handed in here already
	// scope the search to one model's bundle. Cross-model contamination cannot
	// ride in on that: a seed is admitted only when the provenance sidecar
	// attests the live frozen contract AND the exact reference-manifest file
	// being used, and only when its name appears in the given size manifest.
	seedPrefix := opts.SeedPrefix
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
			bestSize, windowUpper, modelName, opts.RefineProfile, prefix, 40)
		if err != nil {
			return nil, err
		}
		tensorTypes = finalPrefix + "-tensor-types.txt"
	}

	// The suffix must name what the file primarily is, and it must be a
	// nameable GGUF preset (see pipeline.PrimaryTypeFromPlan). A deliverable
	// whose primary type cannot be established would ship under a name that
	// does not describe it, so it fails instead.
	primaryType, err := primaryTypeForArtifact(filepath.Join(analysisDir, "analysis.json"), tensorTypes)
	if err != nil {
		return nil, err
	}
	if primaryType == "" {
		return nil, fmt.Errorf("cannot determine the primary type for %s-FITKIT-%s at %d bytes",
			modelName, strings.ToUpper(tierKey), bestSize)
	}

	outputPath := opts.Output
	if outputPath == "" {
		outputPath = filepath.Join(opts.OutDir,
			fmt.Sprintf("%s-FITKIT-%s-%.2fGiB-%s.gguf", modelName, strings.ToUpper(tierKey), float64(bestSize)/(1<<30), primaryType))
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
		"primary_type":         primaryType,
		"naming":               fmt.Sprintf("%s-FITKIT-%s-<size>GiB-<type>.gguf", modelName, strings.ToUpper(tierKey)),
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
	var containing []Window
	for _, w := range windows {
		if w.LowerSize <= bestSize && bestSize <= w.UpperSize {
			containing = append(containing, w)
		}
	}
	if len(containing) == 0 {
		return ""
	}
	// Planning is upgrade-only from a window's lower preset: a size exactly on a
	// shared preset boundary is reproducible from the window where it is the
	// LOWER bound (recipe = that preset, zero upgrades), but as the other
	// window's upper bound it would need every upgrade in the gap to fit, and
	// the tail of a gap usually admits none. Prefer the reproducible side.
	for _, w := range containing {
		if w.LowerSize == bestSize {
			return w.AnalysisPath
		}
	}
	return containing[0].AnalysisPath
}

// parseTensorTypes parses a tensor-types file (one "^tensor$=qtype" line per
// override) into a tensor→qtype map.
func parseTensorTypes(path string) (map[string]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	over := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		idx := strings.Index(line, "$=")
		if idx < 2 || !strings.HasPrefix(line, "^") {
			return nil, fmt.Errorf("malformed tensor-types line: %q", line)
		}
		over[line[1:idx]] = line[idx+2:]
	}
	return over, nil
}

// primaryTypeForArtifact computes the deliverable's primary type from its
// window analysis and tensor-types file — the naming evidence equivalent of
// upstream primary_type_from_plan(plan_record).
func primaryTypeForArtifact(analysisPath, tensorTypesPath string) (string, error) {
	a, err := pipeline.LoadAnalysis(analysisPath)
	if err != nil {
		return "", err
	}
	over, err := parseTensorTypes(tensorTypesPath)
	if err != nil {
		return "", err
	}
	if len(over) == 0 {
		return strings.ToUpper(a.LowerPreset), nil
	}
	plan := &pipeline.OptimizationPlan{}
	for name, q := range over {
		plan.Selected = append(plan.Selected, pipeline.UpgradeCandidate{Tensor: name, ToQtype: q})
	}
	rec := pipeline.ApplyOverrides(a.LowerRecipe, plan)
	dist := pipeline.QtypeParameterDistribution(rec)
	return pipeline.PrimaryTypeFromPlan(a.LowerPreset, len(over), pipeline.DominantQtype(dist)), nil
}

// DiscoverReferenceManifest locates the reference manifest for a model,
// mirroring upstream discover_reference_manifest (0.3.2): the model bundle
// layout first (references/reference-manifest.json, then up one level), with
// the v0.2 freeze-adjacent reference-manifest-*.json glob as a last-resort
// fallback that reports ambiguity instead of silently picking one.
func DiscoverReferenceManifest(refsDir, freezePath string) (string, error) {
	for _, cand := range []string{
		filepath.Join(refsDir, "references", "reference-manifest.json"),
		filepath.Join(refsDir, "reference-manifest.json"),
	} {
		if fileExists(cand) {
			return cand, nil
		}
	}
	matches, err := filepath.Glob(filepath.Join(filepath.Dir(freezePath), "reference-manifest-*.json"))
	if err != nil {
		matches = nil
	}
	sort.Strings(matches)
	if len(matches) == 1 {
		return matches[0], nil
	}
	if len(matches) > 1 {
		return "", &EvalProvenanceError{msg: fmt.Sprintf("ambiguous reference manifest: %v", matches)}
	}
	return "", &EvalProvenanceError{msg: "reference_manifest_path is required (no reference-manifest.json found beside the references or next to the freeze file)"}
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
