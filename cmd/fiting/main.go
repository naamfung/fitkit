// Command fiting is the pure-Go FIT-GGUF driver.  It performs analyze → plan →
// quantize entirely in Go, invoking only llama-quantize (never Python).
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"fitgo/gguf"
	"fitgo/pipeline"
)

func parseSize(s string) (int64, error) {
	t := strings.TrimSpace(s)
	up := strings.ToUpper(t)
	num := t
	mult := 1.0
	for _, u := range []string{"GIB", "GB", "MIB", "MB"} {
		if strings.HasSuffix(up, u) {
			num = strings.TrimSpace(strings.TrimSuffix(up, u))
			switch u {
			case "GIB":
				mult = math.Pow(1024, 3)
			case "GB":
				mult = math.Pow(1000, 3)
			case "MIB":
				mult = math.Pow(1024, 2)
			case "MB":
				mult = math.Pow(1000, 2)
			}
			break
		}
	}
	if strings.HasSuffix(up, "G") && num == t {
		num = strings.TrimSpace(strings.TrimSuffix(t, "G"))
		mult = math.Pow(1024, 3)
	}
	v, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return 0, fmt.Errorf("cannot parse size %q: %v", s, err)
	}
	return int64(math.Floor(v * mult)), nil
}

func mustFile(p string) error {
	if st, err := os.Stat(p); err != nil {
		return fmt.Errorf("cannot access %s: %v", p, err)
	} else if st.IsDir() {
		return fmt.Errorf("%s is a directory, expected a file", p)
	}
	return nil
}

func autoName(model, dominant string, sizeBytes int64, lower, upper, dirTag string) string {
	base := gguf.BaseNameNoSplit(model)
	if base == "" {
		// fallback: use model name or "model"
		if model != "" {
			_, baseName := filepath.Split(model)
			baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))
			if baseName != "" {
				base = baseName
			}
		}
		if base == "" {
			base = "model"
		}
	}
	for _, sfx := range []string{"-BF16", "-F16", "-FP16", "-f16", "-bf16"} {
		base = strings.TrimSuffix(base, sfx)
	}

	// In 'down' mode, dominant should be the upper preset qtype (the starting high-quality qtype).
	// If dominant is empty, matches the lower preset, or in DOWN mode doesn't match upper,
	// fallback to the upper preset qtype.
	dominantUpper := strings.ToUpper(upper)
	if dominant == "" || dominant == strings.ToUpper(lower) || (strings.Contains(dirTag, "DOWN") && dominant != dominantUpper) {
		if dominantUpper != "" {
			dominant = dominantUpper
		} else {
			dominant = "F16"
		}
	}

	giB := float64(sizeBytes) / math.Pow(1024, 3)
	return fmt.Sprintf("%s-FITKIT-%s-%s-%.2fG-lower%s-upper%s.gguf",
		base, dominant, dirTag, giB, lower, upper)
}

// dirTag maps the optimization mode to the direction tag embedded in output
// names: "down" → DOWN, anything else → UP.
func dirTag(mode string) string {
	if strings.EqualFold(mode, "down") {
		return "DOWN"
	}
	return "UP"
}

func main() {
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		fmt.Fprintln(out, "fiting — pure-Go FIT-GGUF driver: analyze → plan → quantize (only calls llama-quantize; no Python)")
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Usage:")
		fmt.Fprintln(out, "  fiting -source <BF16.gguf> -imatrix <im.gguf> -target <size> [options]")
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Flags:")
		flag.PrintDefaults()
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Type names (-lower / -upper; verified against llama-quantize -h):")
		fmt.Fprintln(out, "  float : F32 F16 BF16")
		fmt.Fprintln(out, "  scalar: Q4_0 Q4_1 Q5_0 Q5_1 Q8_0")
		fmt.Fprintln(out, "  new   : Q1_0 Q2_0 MXFP4_MOE")
		fmt.Fprintln(out, "  TQ    : (disabled) TQ1_0 TQ2_0 TQ3_1S TQ4_1S are all disabled")
		fmt.Fprintln(out, "  K     : Q2_K Q2_K_S Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K  (Q3_K/Q4_K/Q5_K = alias->M)")
		fmt.Fprintln(out, "  IQ    : IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_XS IQ3_S IQ3_M IQ4_NL IQ4_XS")
		fmt.Fprintln(out)
		fmt.Fprintln(out, "-target size formats: 5GiB / 5GB / 512MiB / 512MB / 5G / <bytes>")
		fmt.Fprintln(out, "-mode up|down (default up): optimization direction")
		fmt.Fprintln(out, "  up   : start at the lower preset, upgrade the most valuable tensors toward -target")
		fmt.Fprintln(out, "  down : start at the upper preset (high quality), downgrade only the least valuable tensors to fit -target")
		fmt.Fprintln(out, "-no-ladder: disable multi-step ladder candidate generation (binary candidates only)")
		fmt.Fprintln(out, "-allow-requantize: allow using a pre-quantized source (e.g. Q8_0) as the parent")
		fmt.Fprintln(out, "Auto output name: <model>-FITKIT-<dominant>-<UP|DOWN>-<size>.2fG-lower<lower>-upper<upper>.gguf")
		fmt.Fprintln(out, "Output layout: the GGUF and all artifacts go into one subdirectory")
		fmt.Fprintln(out, "  named <stem>/ (stem = output basename without .gguf)")
		fmt.Fprintln(out, "Artifacts inside <stem>/: <stem>-analysis.json <stem>-profile.json")
		fmt.Fprintln(out, "  <stem>-plan.json <stem>-recipe.json <stem>-tensor-types.txt <out>.quantize-record.json")
		fmt.Fprintln(out, "  Disable artifacts with -no-artifacts.")
		fmt.Fprintln(out, "Exit: 0 = G2 byte-exact PASS; 1 = G2 FAIL/error; 2 = bad usage")
	}
	var (
		bf       = flag.String("source", "", "BF16/f16 source GGUF (required)")
		im       = flag.String("imatrix", "", "importance-matrix GGUF (required)")
		targets  = flag.String("target", "", "target size, e.g. 4.8GiB or bytes (required)")
		out      = flag.String("out", "", "output path (optional; auto-named if empty)")
		lower    = flag.String("lower", "Q3_K_M", "lower preset")
		upper    = flag.String("upper", "Q8_0", "upper preset")
		policy   = flag.String("policy", "balanced", "plan policy")
		mode     = flag.String("mode", "up", "optimization direction: up (lower→upper, keep high value) or down (upper→lower, keep high quality)")
		runtime  = flag.String("runtime", `D:\Programs\llama-cpp-repos\laamaafung\build-v17\bin\Release`, "runtime dir")
		keep     = flag.Bool("keep", false, "keep work dir")
		planOnly = flag.Bool("plan-only", false, "plan + print qtype shares without quantizing")
		allowRQ  = flag.Bool("allow-requantize", false, "allow pre-quantized source (e.g. Q8_0) as parent")
		noArt    = flag.Bool("no-artifacts", false, "do not emit plan/recipe/profile/quantize-record artifacts next to the output")
		noLadder = flag.Bool("no-ladder", false, "disable multi-step ladder candidate generation (binary candidates only)")
	)
	flag.Parse()
	pipeline.AllowRequantize = *allowRQ
	pipeline.UseLadder = !*noLadder
	if *bf == "" || *im == "" || *targets == "" {
		fmt.Fprintln(os.Stderr, "fiting: -bf, -imatrix, -target required (-out optional)")
		os.Exit(2)
	}
	bfAbs, _ := filepath.Abs(*bf)
	imAbs, _ := filepath.Abs(*im)
	if err := mustFile(bfAbs); err != nil {
		fatal(err)
	}
	if err := mustFile(imAbs); err != nil {
		fatal(err)
	}
	targetBytes, err := parseSize(*targets)
	if err != nil {
		fatal(err)
	}

	work, err := os.MkdirTemp("", "fitgo-work-")
	if err != nil {
		fatal(err)
	}
	if *keep {
		fmt.Printf("work dir: %s\n", work)
	} else {
		defer os.RemoveAll(work)
	}
	analysisDir := filepath.Join(work, "analysis")
	planPrefix := filepath.Join(work, "plan")

	// Echo the effective parameters up front so any run can be diagnosed from
	// its header alone.
	fmt.Println("parameters:")
	fmt.Printf("  mode: %s\n", *mode)
	fmt.Printf("  source: %s\n", bfAbs)
	fmt.Printf("  imatrix: %s\n", imAbs)
	fmt.Printf("  runtime: %s\n", *runtime)
	fmt.Printf("  target: %s (%d bytes)\n", *targets, targetBytes)
	fmt.Printf("  lower: %s\n", *lower)
	fmt.Printf("  upper: %s\n", *upper)
	fmt.Printf("  policy: %s\n", *policy)
	fmt.Printf("  plan-only: %v\n", *planOnly)
	fmt.Printf("  no-ladder: %v\n", *noLadder)

	fmt.Println("[1/3] analyze")
	if _, err := pipeline.Analyze(bfAbs, imAbs, *runtime, analysisDir, *lower, *upper, imAbs, false, *mode); err != nil {
		fatal(err)
	}
	analysisJSON := filepath.Join(analysisDir, "analysis.json")

	fmt.Println("[2/3] plan")
	optimization, pred, err := pipeline.Plan(analysisJSON, planPrefix, int(targetBytes), *policy, "auto", "")
	if err != nil {
		fatal(err)
	}
	a, _ := pipeline.LoadAnalysis(analysisJSON)
	var baseline *pipeline.Recipe
	if strings.EqualFold(*mode, "down") {
		// In 'down' mode, start from the upper recipe (all tensors at upper qtype)
		baseline = a.UpperRecipe
		if baseline == nil {
			baseline = a.BaselineRecipe()
		}
	} else {
		// In 'up' mode, start from the baseline or lower recipe
		baseline = a.BaselineRecipe()
		if baseline == nil {
			baseline = a.LowerRecipe
		}
	}
	recipe := pipeline.ApplyOverrides(baseline, optimization)
	dist := pipeline.QtypeParameterDistribution(recipe)
	dominant := pipeline.DominantQtype(dist)

	// Fix dominant for 'down' mode: it should be the upper preset qtype.
	// If dominant is empty, matches the lower preset, or in DOWN mode doesn't match upper,
	// fallback to the upper preset qtype.
	dominantUpper := strings.ToUpper(*upper)
	dirT := dirTag(*mode)
	if dominant == "" || dominant == strings.ToUpper(*lower) || (strings.Contains(dirT, "DOWN") && dominant != dominantUpper) {
		if dominantUpper != "" {
			dominant = dominantUpper
		} else {
			dominant = "F16"
		}
	}

	tensorTypes := planPrefix + "-tensor-types.txt"
	fmt.Printf("  predicted=%d selected=%d dominant=%s\n", pred.TotalBytes, len(optimization.Selected), dominant)
	printShares(dist)
	if *planOnly {
		return
	}

	outFile := *out
	if outFile == "" {
		outFile = filepath.Join(filepath.Dir(bfAbs), autoName(bfAbs, dominant, int64(pred.TotalBytes), *lower, *upper, dirTag(*mode)))
	}
	// 输出 GGUF 与全部产物统一收进以输出名（去 .gguf）命名的子目录，避免散落。
	outPath := filepath.Join(filepath.Dir(outFile), strings.TrimSuffix(filepath.Base(outFile), ".gguf"), filepath.Base(outFile))
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		fatal(err)
	}

	recordPath := ""
	if !*noArt {
		if err := emitArtifacts(bfAbs, imAbs, analysisJSON, tensorTypes, a, optimization, pred, dist, targetBytes, outPath, *policy, *lower, *upper, dominant, *mode); err != nil {
			fatal(err)
		}
		recordPath = outPath + ".quantize-record.json"
	}

	fmt.Println("[3/3] quantize")
	actual, expected, err := pipeline.Quantize(analysisJSON, tensorTypes, outPath, 0, imAbs, recordPath)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("done: %s\n  target: %d bytes\n  expected: %d bytes\n  actual: %d bytes\n",
		outPath, targetBytes, expected, actual)
	if int64(actual) != int64(expected) {
		fmt.Fprintf(os.Stderr, "G2 FAIL: actual %d != expected %d\n", actual, expected)
		os.Exit(1)
	}
	fmt.Println("G2 PASS (byte-exact)")
}

func printShares(dist map[string]int) {
	total := 0
	for _, c := range dist {
		total += c
	}
	if total == 0 {
		return
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
	for _, k := range keys {
		fmt.Printf("  share %s: %.1f%% (%d params)\n",
			strings.ToUpper(k), float64(dist[k])/float64(total)*100, dist[k])
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "fiting:", err)
	os.Exit(1)
}

// emitArtifacts writes the structural artifacts next to the output GGUF,
// mirroring the Python fit CLI's default outputs (no raw dry-run logs).
func emitArtifacts(bfAbs, imAbs, analysisJSON, tensorTypes string,
	a *pipeline.Analysis, optimization *pipeline.OptimizationPlan,
	pred *gguf.Prediction, dist map[string]int, targetBytes int64,
	outPath, policy, lower, upper, dominant, mode string) error {

	outDir := filepath.Dir(outPath)
	stem := strings.TrimSuffix(filepath.Base(outPath), ".gguf")

	// <stem>-analysis.json
	analysisPath := filepath.Join(outDir, stem+"-analysis.json")
	if err := copyFile(analysisJSON, analysisPath); err != nil {
		return err
	}
	analysisSHA, _ := pipeline.SHA256File(analysisPath)

	// <stem>-profile.json
	profile, err := pipeline.LoadImatrixProfile(imAbs)
	if err != nil {
		return err
	}
	if err := pipeline.WriteProfileJSON(profile, filepath.Join(outDir, stem+"-profile.json")); err != nil {
		return err
	}

	// -recipe.json
	recipePath := filepath.Join(outDir, stem+"-recipe.json")
	if err := pipeline.WriteFitRecipe(optimization, recipePath, lower, upper); err != nil {
		return err
	}
	recipeSHA, _ := pipeline.SHA256File(recipePath)

	// -tensor-types.txt
	typesPath := filepath.Join(outDir, stem+"-tensor-types.txt")
	if err := copyFile(tensorTypes, typesPath); err != nil {
		return err
	}
	typesSHA, _ := pipeline.SHA256File(typesPath)

	// -plan.json
	mm := pipeline.DefaultModelName(bfAbs)
	record := pipeline.PlanRecord{
		AnalysisPath:       analysisPath,
		AnalysisSHA256:     analysisSHA,
		Policy:             policy,
		TargetBytes:        int(targetBytes),
		LowerPreset:        lower,
		UpperPreset:        upper,
		LowerSizeBytes:     a.LowerSizeBytes,
		UpperSizeBytes:     a.UpperSizeBytes,
		PredictedSizeBytes: pred.TotalBytes,
		MetadataBytes:      pred.MetadataBytes,
		TensorPayload:      pred.TensorPayload,
		TensorPadding:      pred.TensorPadding,
		UnusedBytes:        int(targetBytes) - pred.TotalBytes,
		SelectedCount:      len(optimization.Selected),
		SkippedCount:       optimization.SkippedCount,
		SelectedCostBytes:  optimization.SelectedCostBytes(),
		OracleIterations:   optimization.OracleIterations,
		RecipePath:         recipePath,
		RecipeSHA256:       recipeSHA,
		TensorTypesPath:    typesPath,
		TensorTypesSHA256:  typesSHA,
		ModelName:          mm,
		DominantQtype:      dominant,
		QtypeShares:        qtypeShares(dist),
		SuggestedFilename:  pipeline.SuggestedFilename(mm, int(targetBytes), dominant, dirTag(mode)),
	}
	return pipeline.WritePlanJSON(record, filepath.Join(outDir, stem+"-plan.json"))
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

// qtypeShares returns the fraction of quantized parameter elements per qtype
// (upper-case keys), normalizing the counts in dist.
func qtypeShares(dist map[string]int) map[string]float64 {
	total := 0
	for _, c := range dist {
		total += c
	}
	if total == 0 {
		return map[string]float64{}
	}
	shares := make(map[string]float64, len(dist))
	for k, v := range dist {
		shares[strings.ToUpper(k)] = float64(v) / float64(total)
	}
	return shares
}
