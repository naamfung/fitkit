// Command fitfidelity implements the Fidelity-tier product path: search the
// minimum verified GGUF satisfying KL <= anchor AND Same-top >= Guard floor,
// then emit the exact-byte artifact. Pure-Go port of `fit fidelity-search`.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"fitgo/fidelity"
)

type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

// normalizeEvalParallelArgs rewrites the optional-argument `-eval-parallel`
// flag so the Go flag parser can handle all three forms:
//   - bare `-eval-parallel`        -> `-eval-parallel=2`
//   - `-eval-parallel N` (space)   -> `-eval-parallel=N`
//   - `-eval-parallel=N`           -> unchanged
//
// Absence of the flag leaves the default (1, serial).
func normalizeEvalParallelArgs(args []string) []string {
	const name, bareDefault = "eval-parallel", 2
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "-"+name || a == "--"+name {
			if i+1 < len(args) {
				if _, err := strconv.Atoi(args[i+1]); err == nil {
					out = append(out, a+"="+args[i+1])
					i++
					continue
				}
			}
			out = append(out, fmt.Sprintf("%s=%d", a, bareDefault))
			continue
		}
		out = append(out, a)
	}
	return out
}

func main() {
	var (
		source         = flag.String("source", "", "Source (BF16) GGUF")
		imatrix        = flag.String("imatrix", "", "Importance-matrix GGUF")
		runtime        = flag.String("runtime", "", "Directory containing the pinned llama.cpp binaries")
		refsDir        = flag.String("refs-dir", "", "Directory of bf16-<domain>.kld references")
		evalDataDir    = flag.String("eval-data-dir", "", "Directory of kl-eval domain slices")
		guardRegistry  = flag.String("guard-registry", "", "Guard Profile registry directory")
		tier           = flag.String("tier", "", "Fidelity tier to satisfy (quality|balanced|compact|mini)")
		presetLadder   = flag.String("preset-ladder", "", "Comma-separated presets in ascending-size order")
		refineProfile  = flag.String("refine-profile", "", "Refine Profile JSON (v0.2 band-conditional)")
		profile        = flag.String("profile", "normal", "Search budget: normal <= 8, precise <= 16")
		toleranceMiB   = flag.Int("tolerance-mib", 128, "Search bracket tolerance in MiB (default 128)")
		modelName      = flag.String("model-name", "", "Guard Profile exact-model identifier")
		outDir         = flag.String("out-dir", "", "Records, analyses, final artifact")
		workDir        = flag.String("work-dir", "", "Scratch directory (tmpfs recommended)")
		manifest       = flag.String("manifest", "", "Artifact manifest (appended)")
		logsDir        = flag.String("logs-dir", "", "Eval log directory (seeds are read from here)")
		output         = flag.String("output", "", "Final artifact path")
		threads        = flag.Int("threads", 16, "Evaluator threads")
		evalParallel   = flag.Int("eval-parallel", 1, "Eval domains in parallel: bare -eval-parallel = 2; -eval-parallel N = N; absent = 1 (serial)")
		seedPrefix     = flag.String("seed-prefix", "", "Manifest/log name prefix for prior points")
		skipHash       = flag.Bool("skip-hash", false, "Skip SHA-256 of source during auto-analyze")
		freeze         = flag.String("freeze", "experiments/2026-09-02-eval-v1/FREEZE.json", "Frozen eval-v1 FREEZE.json")
		referenceManif = flag.String("reference-manifest", "", "Reference manifest JSON")
	)
	var analysisDirs, excludeSeeds stringList
	flag.Var(&analysisDirs, "analysis", "Pre-frozen analysis directory (repeatable)")
	flag.Var(&excludeSeeds, "exclude-seed", "Prior point name to exclude from bracket evidence (repeatable)")
	// Normalize the optional-argument eval-parallel flag so that a bare
	// `-eval-parallel` (no value) means 2, `-eval-parallel N`/`-eval-parallel=N`
	// means N, and full absence leaves the default (1, serial).
	flag.CommandLine.Parse(normalizeEvalParallelArgs(os.Args[1:]))

	imatrixArg := *imatrix
	fail := func(name string) {
		fmt.Fprintf(os.Stderr, "fitfidelity: %s is required\n", name)
		os.Exit(2)
	}
	if *source == "" {
		fail("-source")
	}
	if *imatrix == "" {
		fail("-imatrix")
	}
	if *runtime == "" {
		fail("-runtime")
	}
	if *refsDir == "" {
		fail("-refs-dir")
	}
	if *evalDataDir == "" {
		fail("-eval-data-dir")
	}
	if *guardRegistry == "" {
		fail("-guard-registry")
	}
	if *tier == "" {
		fail("-tier")
	}
	if *outDir == "" {
		fail("-out-dir")
	}
	if *workDir == "" {
		fail("-work-dir")
	}
	if *manifest == "" {
		fail("-manifest")
	}
	if *logsDir == "" {
		fail("-logs-dir")
	}
	switch *tier {
	case "quality", "balanced", "compact", "mini":
	default:
		fmt.Fprintf(os.Stderr, "fitfidelity: -tier must be one of quality|balanced|compact|mini\n")
		os.Exit(2)
	}
	var ladder []string
	if *presetLadder != "" {
		ladder = strings.Split(*presetLadder, ",")
	}
	if len(ladder) == 0 && len(analysisDirs) == 0 {
		fmt.Fprintln(os.Stderr, "fitfidelity: provide -preset-ladder or -analysis")
		os.Exit(2)
	}

	summary, err := fidelity.FidelitySearchProduct(fidelity.ProductOptions{
		Source: *source, Imatrix: *imatrix, Runtime: *runtime,
		RefsDir: *refsDir, EvalDataDir: *evalDataDir, GuardRegistry: *guardRegistry,
		Tier: *tier, OutDir: *outDir, WorkDir: *workDir, ManifestPath: *manifest,
		LogsDir: *logsDir, ModelName: *modelName, PresetLadder: ladder,
		AnalysisDirs: analysisDirs, RefineProfile: *refineProfile, Profile: *profile,
		ToleranceBytes: toleranceMB(*toleranceMiB), Output: *output, Threads: *threads,
		EvalConcurrency: *evalParallel,
		ImatrixArg: imatrixArg, HashSources: !*skipHash,
		SeedPrefix: *seedPrefix, ExcludeSeeds: excludeSeeds,
		FreezePath: *freeze, ReferenceManifestPath: *referenceManif,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fitfidelity: error: %v\n", err)
		os.Exit(1)
	}

	status := str(summary["status"])
	best, _ := summary["best"].(map[string]any)
	if status == "verified_pass" && len(best) > 0 {
		size := intVal(best["size_bytes"])
		fmt.Printf("fidelity-search: %s | tier %s | minimum verified PASS %.2fG (kld %.4f, top %.2f%%) | active constraint: %s | %s/%s evals\n",
			status, *tier, float64(size)/(1<<30), f64(best["macro_kl"]), f64(best["same_top"])*100,
			summary["active_constraint"], summary["fresh_evals"], summary["budget"])
		if note := str(summary["note"]); note != "" {
			fmt.Printf("note: %s\n", note)
		}
		artifact, _ := summary["artifact"].(map[string]any)
		if len(artifact) > 0 {
			fmt.Printf("artifact: %s (%d bytes)\n", artifact["path"], intVal(artifact["size_bytes"]))
		}
		exitByStatus(status)
	}
	fmt.Fprintf(os.Stderr, "fidelity-search: %s (tier %s)\n", str(summary["product_status"]), *tier)
	if note := str(summary["note"]); note != "" {
		fmt.Fprintf(os.Stderr, "note: %s\n", note)
	}
	exitByStatus(status)
}

func exitByStatus(status string) {
	switch status {
	case "no_pass":
		os.Exit(3)
	case "noise_inversion":
		os.Exit(4)
	case "budget_exhausted":
		os.Exit(5)
	default:
		os.Exit(2)
	}
}

func toleranceMB(mib int) int { return mib * 1024 * 1024 }

func str(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func intVal(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	case string:
		if i, err := strconv.Atoi(t); err == nil {
			return i
		}
	}
	return 0
}

func f64(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	}
	return 0
}
