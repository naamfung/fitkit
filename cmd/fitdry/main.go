// Command fitdry runs a link-by-link dry-run of the fidelity-search chain
// against real material: analyze (llama-quantize --dry-run) -> eval-v1
// provenance -> guard-profile resolve -> plan size prediction. It never
// quantizes or evaluates; it reports PASS/FAIL per link.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"fitting/fidelity"
	"fitting/pipeline"
)

func step(name string, err error) bool {
	if err != nil {
		fmt.Printf("  [FAIL] %s: %v\n", name, err)
		return false
	}
	fmt.Printf("  [ok ] %s\n", name)
	return true
}

func main() {
	var (
		source     = flag.String("source", "", "Source (BF16) GGUF")
		imatrix    = flag.String("imatrix", "", "Importance-matrix GGUF")
		runtime    = flag.String("runtime", "", "Dir containing llama-quantize/llama-perplexity")
		lower      = flag.String("lower", "Q3_K_M", "Window lower preset")
		upper      = flag.String("upper", "BF16", "Window upper preset")
		outDir     = flag.String("out-dir", "dry/analyses", "Scratch dir for analysis.json")
		refsDir    = flag.String("refs-dir", "", "Dir of bf16-<domain>.kld references")
		evalData   = flag.String("eval-data-dir", "", "Dir of kl-eval domain slices")
		freeze     = flag.String("freeze", "", "Frozen eval-v1 FREEZE.json")
		manifest   = flag.String("reference-manifest", "", "Reference manifest JSON")
		guardReg   = flag.String("guard-registry", "", "Guard Profile registry dir")
		modelName  = flag.String("model-name", "", "Guard identifier (exact-model)")
		sourceSha  = flag.String("source-sha256", "", "Override weights digest for guard binding")
		tier       = flag.String("tier", "balanced", "Fidelity tier")
		skipHash   = flag.Bool("skip-hash", false, "Skip source sha256 for analyze")
	)
	flag.Parse()

	run := func(stepName string, fn func() error) {
		fmt.Printf("[%s]\n", stepName)
		step(stepName, fn())
	}

	// 1) runtime
	abs, err := pipeline.RuntimeBinary(*runtime, "llama-quantize")
	if err != nil || *runtime == "" {
		fmt.Println("  [FAIL] runtime: llama-quantize not found")
		os.Exit(1)
	}
	fmt.Printf("[runtime] llama-quantize=%s\n", abs)
	perm, err := pipeline.RuntimeBinary(*runtime, "llama-perplexity")
	if err != nil {
		fmt.Printf("  [FAIL] runtime: llama-perplexity not found\n")
		os.Exit(1)
	}
	fmt.Printf("          llama-perplexity=%s\n", perm)

	// 2) analyze (llama-quantize --dry-run) with real source+imatrix
	run("analyze", func() error {
		if _, err := pipeline.Analyze(*source, *imatrix, *runtime, *outDir,
			*lower, *upper, filepath.Base(*imatrix), !*skipHash, "up"); err != nil {
			return err
		}
		windows, err := fidelity.DiscoverWindows([]string{*outDir})
		if err != nil {
			return err
		}
		for _, w := range windows {
			fmt.Printf("          window %s->%s  pred %dB .. %dB\n",
				w.LowerPreset, w.UpperPreset, w.LowerSize, w.UpperSize)
		}
		return nil
	})

	// 3) plan size prediction at window midpoint
	run("plan", func() error {
		windows, err := fidelity.DiscoverWindows([]string{*outDir})
		if err != nil {
			return err
		}
		if len(windows) == 0 {
			return fmt.Errorf("no windows")
		}
		w := windows[0]
		mid := (w.LowerSize + w.UpperSize) / 2
		plan, pred, err := pipeline.Plan(filepath.Join(w.AnalysisPath, "analysis.json"),
			filepath.Join(*outDir, "plan"), mid, "balanced", "auto", *modelName, "")
		if err != nil {
			return err
		}
		_ = plan
		fmt.Printf("          target=%d -> predicted=%d\n", mid, pred.TotalBytes)
		return nil
	})

	// 4) eval-v1 provenance (fail-closed: reference .kld must exist)
	sha := *sourceSha
	if *sourceSha == "" && !*skipHash {
		sha, _ = fidelity.SHA256File(*source)
	}
	run("provenance", func() error {
		prov, err := fidelity.VerifyEvalV1Provenance(*refsDir, *evalData, *freeze, *manifest, "")
		if err == nil {
			fmt.Printf("          contract_digest=%s\n", prov.ContractDigest)
			fmt.Printf("          manifest_sha=%s\n", prov.ReferenceManifestFileSHA256)
		}
		return err
	})

	// 5) guard-profile resolve
	run("guard", func() error {
		c, err := fidelity.ResolveContract(*modelName, *tier, *guardReg, sha)
		if err != nil {
			return err
		}
		fmt.Printf("          tier=%s  kl_anchor=%.3f  same_top_reference=%.4f\n",
			c.Tier, c.KLAnchor, c.SameTopReference)
		return nil
	})
}