// Command fitcalibrate implements the fit calibrate production line: turns a
// BF16 GGUF into a Calibration Bundle (guard profile + records + references +
// seed material), mirroring upstream `fit calibrate` (v0.3.3). Only calls
// llama-imatrix / llama-quantize / llama-perplexity; never Python.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"fitting/calibration"
)

func main() {
	var (
		source        = flag.String("source", "", "Source (BF16) GGUF")
		imatrixCorpus = flag.String("imatrix-corpus", "", "Calibration corpus text")
		runtime       = flag.String("runtime", "", "llama.cpp runtime dir (llama-quantize/imatrix/perplexity)")
		evalData      = flag.String("eval-data", "", "Directory with the five frozen eval slices")
		outDir        = flag.String("out-dir", "", "Calibration Bundle output directory")
		modelID       = flag.String("model-id", "", "Exact-model identifier for the guard/registry entry")
		imatrix       = flag.String("imatrix", "", "Reuse an existing imatrix GGUF instead of generating")
		chunks        = flag.Int("chunks", 500, "Imatrix generation chunks")
		extraPresets  = flag.String("extra-presets", "", "Comma-separated extra ladder presets")
		probeBudget   = flag.Int("probe-budget", 4, "Gap probes per tier (contract max 4)")
		nGPULayers    = flag.Int("n-gpu-layers", 99, "Layers offloaded to GPU during evaluation")
		threads       = flag.Int("threads", 16, "Evaluator threads")
		workdir       = flag.String("workdir", "", "Scratch dir (default tmpfs)")
		logDir        = flag.String("log-dir", "", "Subprocess log directory (default: inside the scratch volume)")
		onDisk        = flag.Bool("on-disk", false, "Keep scratch next to the bundle instead of tmpfs")
		contract      = flag.String("contract", "", "Calibration contract JSON (default: embedded fidelity-calibration-v1)")
		replay        = flag.String("replay-existing", "", "Zero-eval mode: re-derive from a recorded observations JSON")
		replayManifest = flag.String("replay-manifest", "", "Artifact manifest for replay dedup keys")
	)
	flag.Parse()

	fail := func(name string) {
		fmt.Fprintf(os.Stderr, "fitcalibrate: %s is required\n", name)
		os.Exit(2)
	}
	if *source == "" {
		fail("-source")
	}
	if *imatrixCorpus == "" {
		fail("-imatrix-corpus")
	}
	if *runtime == "" {
		fail("-runtime")
	}
	if *evalData == "" {
		fail("-eval-data")
	}
	if *outDir == "" {
		fail("-out-dir")
	}
	if *modelID == "" {
		fail("-model-id")
	}

	var extra []string
	if *extraPresets != "" {
		extra = strings.Split(*extraPresets, ",")
	}

	cfg := &calibration.Config{
		Source:         *source,
		ImatrixCorpus:  *imatrixCorpus,
		RuntimeDir:     *runtime,
		EvalDataDir:    *evalData,
		OutDir:         *outDir,
		ModelID:        *modelID,
		ImatrixPath:    *imatrix,
		Chunks:         *chunks,
		ExtraPresets:   extra,
		ProbeBudget:    *probeBudget,
		NGPULayers:     *nGPULayers,
		Threads:        *threads,
		Workdir:        *workdir,
		OnDisk:         *onDisk,
		ContractPath:   *contract,
		LogDir:         *logDir,
		ReplayExisting: *replay,
		ReplayManifest: *replayManifest,
	}

	result, err := cfg.RunCalibrate()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fitcalibrate: error: %v\n", err)
		os.Exit(1)
	}
	if result.Mode == "replay" {
		fmt.Printf("calibrate replay: overall=%s failures=%v\n",
			result.Report["overall_status"], result.Report["open_failures"])
		return
	}
	fmt.Printf("calibrate: bundle=%s overall=%s source_sha256=%s\n",
		result.Bundle, result.Report["overall_status"], result.SourceSHA256)
}
