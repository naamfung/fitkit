package fidelity

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"fitting/pipeline"
)

// PoisonPresets are known poison presets (GPT ruling §12) — never a window
// lower bound, never a bracket seed.
var PoisonPresets = map[string]bool{"Q3_K_S": true, "IQ2_XS": true}

const allowExtendDefault = 768 * 1024 * 1024
const evaluateBumpBytes = 128 * 1024 * 1024

// Window is one preset-pair size window from an analysis directory.
type Window struct {
	AnalysisPath string
	LowerPreset  string
	UpperPreset  string
	LowerSize    int
	UpperSize    int
}

func (w Window) Healthy() bool { return !PoisonPresets[strings.ToUpper(w.LowerPreset)] }

// RunnerConfig binds the executor to the runtime, refs, work and record paths.
type RunnerConfig struct {
	Runtime               string
	Imatrix               string
	RefsDir               string
	EvalDataDir           string
	WorkDir               string
	OutDir                string
	ModelName             string
	GuardRegistry         string
	RefineProfile         string
	Threads               int
	EvalProvenance        *EvalProvenance
	RequireEvalProvenance bool
	SourceSHA256          string
	// EvalConcurrency is the number of eval domains evaluated in parallel by
	// llama-perplexity. Each domain runs its own process that loads the model
	// under test, so it multiplies the evaluator's peak memory. Default 1
	// (serial); >1 is opt-in for high-memory/GPU devices.
	EvalConcurrency int
}

// ResolveContract 构建 KL-only 门限（Fidelity Contract v2）：锚点是全局 KL
// 常量，命名一个 tier 不需要 validated Guard Profile。当给定 registry 中确有
// 覆盖该权重的 validated profile 时，其 same-top floor 作为 SameTopReference
// 附加，仅用于报告。
func ResolveContract(modelName, tier, guardRegistry, sourceSHA256 string) (*TierContract, error) {
	tierKey := strings.TrimSpace(strings.ToLower(tier))
	if !validTier(tierKey) {
		return nil, fmt.Errorf("unknown fidelity tier: %q (expected %v)", tier, tierList)
	}
	contract := &TierContract{Tier: tierKey, KLAnchor: KLAnchors[tierKey]}
	if guardRegistry != "" {
		profile, err := ResolveGuardProfile(modelName, guardRegistry, sourceSHA256)
		if err != nil {
			return nil, err
		}
		if profile != nil {
			contract.SameTopReference = profile.FloorFor(tierKey)
		}
	}
	return contract, nil
}

// DiscoverWindows loads preset-pair windows from analysis directories.
func DiscoverWindows(analysisDirs []string) ([]Window, error) {
	windows := []Window{}
	for _, raw := range analysisDirs {
		analysisFile := raw
		if st, err := os.Stat(raw); err == nil && st.IsDir() {
			analysisFile = filepath.Join(raw, "analysis.json")
		}
		if !fileExists(analysisFile) {
			return nil, fmt.Errorf("analysis not found: %s", analysisFile)
		}
		data, err := os.ReadFile(analysisFile)
		if err != nil {
			return nil, err
		}
		var payload struct {
			Presets struct {
				Lower struct {
					Name               string `json:"name"`
					PredictedSizeBytes int    `json:"predicted_size_bytes"`
				} `json:"lower"`
				Upper struct {
					Name               string `json:"name"`
					PredictedSizeBytes int    `json:"predicted_size_bytes"`
				} `json:"upper"`
			} `json:"presets"`
		}
		if err := json.Unmarshal(data, &payload); err != nil {
			return nil, fmt.Errorf("parse %s: %v", analysisFile, err)
		}
		windows = append(windows, Window{
			AnalysisPath: filepath.Dir(analysisFile),
			LowerPreset:  payload.Presets.Lower.Name,
			UpperPreset:  payload.Presets.Upper.Name,
			LowerSize:    payload.Presets.Lower.PredictedSizeBytes,
			UpperSize:    payload.Presets.Upper.PredictedSizeBytes,
		})
	}
	sort.Slice(windows, func(a, b int) bool { return windows[a].LowerSize < windows[b].LowerSize })
	return windows, nil
}

// SelectWindow picks the narrowest healthy window covering target (with snap).
func SelectWindow(target int, windows []Window, allowExtend int) (Window, int, error) {
	if allowExtend == 0 {
		allowExtend = allowExtendDefault
	}
	var healthy []Window
	for _, w := range windows {
		if w.Healthy() {
			healthy = append(healthy, w)
		}
	}
	if len(healthy) == 0 {
		return Window{}, 0, fmt.Errorf("no healthy windows available (poison presets rejected)")
	}
	var covering []Window
	for _, w := range healthy {
		if w.LowerSize <= target && target <= w.UpperSize {
			covering = append(covering, w)
		}
	}
	if len(covering) > 0 {
		best := covering[0]
		for _, w := range covering[1:] {
			if w.UpperSize-w.LowerSize < best.UpperSize-best.LowerSize {
				best = w
			}
		}
		return best, target, nil
	}
	var best *Window
	bestPlanned := 0
	bestDist := allowExtend + 1
	for i := range healthy {
		for _, edge := range []int{healthy[i].LowerSize, healthy[i].UpperSize} {
			d := absInt(edge - target)
			if d <= allowExtend && d < bestDist {
				bestDist = d
				w := healthy[i]
				best = &w
				bestPlanned = edge
			}
		}
	}
	if best == nil {
		return Window{}, 0, fmt.Errorf("target %d is >%d MiB away from every healthy window edge",
			target, allowExtend/(1024*1024))
	}
	return *best, bestPlanned, nil
}

func absInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// SeedView mirrors the provenance sidecar admission view: tainted (poison
// window), stale (different closure digest), attested (matching closure).
type SeedView struct {
	Tainted     map[string]bool
	Stale       map[string]bool
	Attested    map[string]bool
	ManifestSHA map[string]string
}

func readSeedView(provenancePath string) SeedView {
	view := SeedView{
		Tainted: map[string]bool{}, Stale: map[string]bool{},
		Attested: map[string]bool{}, ManifestSHA: map[string]string{},
	}
	if provenancePath == "" || !fileExists(provenancePath) {
		return view
	}
	raw, err := os.ReadFile(provenancePath)
	if err != nil {
		return view
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var rec map[string]any
		if json.Unmarshal([]byte(line), &rec) != nil {
			continue
		}
		name := fstring(rec["name"])
		if name == "" {
			continue
		}
		lo := strings.ToUpper(fstring(rec["window_lower_preset"]))
		hi := strings.ToUpper(fstring(rec["window_upper_preset"]))
		if PoisonPresets[lo] || PoisonPresets[hi] {
			view.Tainted[name] = true
		}
		if digest, ok := rec["eval_contract_digest"].(string); ok && digest != "" {
			if digest == FrozenContractDigest {
				view.Attested[name] = true
			} else {
				view.Stale[name] = true
			}
		}
		if sha, ok := rec["reference_manifest_sha256"].(string); ok && sha != "" {
			view.ManifestSHA[name] = sha
		}
	}
	return view
}

// LoadSeeds collects budget-free observed points from eval logs + size manifest.
func LoadSeeds(manifestPath, logsDir, modelPrefix string, opt SeedOptions) ([]Seed, error) {
	provenancePath := opt.ProvenancePath
	if provenancePath == "" {
		provenancePath = filepath.Join(filepath.Dir(manifestPath), "seed-provenance.jsonl")
	}
	if opt.RequireSeedProvenance && opt.ReferenceManifestSHA256 == "" {
		return nil, fmt.Errorf("require_seed_provenance needs the active verified reference manifest SHA")
	}
	view := readSeedView(provenancePath)

	domainRe := regexp.MustCompile("-(" + strings.Join(Domains, "|") + ")$")
	byPoint := map[string]map[string]map[string]any{}
	entries, err := os.ReadDir(logsDir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, "eval-") || !strings.HasSuffix(name, ".log") {
			continue
		}
		body := strings.TrimSuffix(strings.TrimPrefix(name, "eval-"), ".log")
		if !strings.HasPrefix(body, modelPrefix) {
			continue
		}
		m := domainRe.FindStringSubmatch(body)
		if m == nil {
			continue
		}
		point := strings.TrimPrefix(strings.TrimSuffix(body, m[0]), modelPrefix)
		if point == "" {
			continue
		}
		text, err := os.ReadFile(filepath.Join(logsDir, name))
		if err != nil {
			continue
		}
		metrics, err := ParseLLaMAKLLog(string(text))
		if err != nil {
			continue
		}
		if byPoint[point] == nil {
			byPoint[point] = map[string]map[string]any{}
		}
		byPoint[point][m[1]] = metrics
	}

	sizes := map[string]int{}
	if raw, err := os.ReadFile(manifestPath); err == nil {
		for _, line := range strings.Split(string(raw), "\n") {
			parts := strings.Fields(line)
			if len(parts) >= 2 && strings.HasPrefix(parts[0], modelPrefix) {
				if v, err := strconv.Atoi(parts[1]); err == nil {
					sizes[strings.TrimPrefix(parts[0], modelPrefix)] = v
				}
			}
		}
	}

	var seeds []Seed
	for point, domains := range byPoint {
		if len(domains) != len(Domains) {
			continue
		}
		size, ok := sizes[point]
		if !ok {
			continue
		}
		skip := false
		for _, poison := range sortedKeys(PoisonPresets) {
			if poi := strings.ToLower(poison); strings.Contains(strings.ToLower(point), poi) {
				skip = true
				break
			}
		}
		if skip || opt.ExcludeNames[point] {
			continue
		}
		fullName := modelPrefix + point
		if view.Tainted[point] || view.Tainted[fullName] {
			continue
		}
		if view.Stale[point] || view.Stale[fullName] {
			continue
		}
		if opt.RequireSeedProvenance {
			if !(view.Attested[point] || view.Attested[fullName]) {
				continue
			}
			got := view.ManifestSHA[point]
			if got == "" {
				got = view.ManifestSHA[fullName]
			}
			if got != opt.ReferenceManifestSHA256 {
				continue
			}
		}
		var klSum, topSum float64
		for _, d := range Domains {
			klSum += ffv(domains[d]["mean_kld"])
			topSum += ffv(domains[d]["same_top_pct"])
		}
		seeds = append(seeds, Seed{
			SizeBytes: size,
			MacroKL:   klSum / float64(len(Domains)),
			SameTop:   topSum / float64(len(Domains)) / 100.0,
		})
	}
	sort.Slice(seeds, func(a, b int) bool { return seeds[a].SizeBytes < seeds[b].SizeBytes })
	return seeds, nil
}

func sortedKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func ffv(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	}
	return 0
}

// SeedOptions controls seed admission from the runner/product path.
type SeedOptions struct {
	PoisonNames             map[string]bool
	ExcludeNames            map[string]bool
	ProvenancePath          string
	RequireSeedProvenance   bool
	ReferenceManifestSHA256 string
}

// SearchExecutor is the plan -> quantize (tmpfs) -> eval-v1 -> record -> release
// executor wired to llama-perplexity.
type SearchExecutor struct {
	Config         RunnerConfig
	Windows        []Window
	Contract       *TierContract
	ManifestPath   string
	LogsOutDir     string
	Counter        int
	delivered      map[int]EvalOutcome
	RecipeBySize   map[int]string
	AnalysisBySize map[int]string
	knownSizes     map[int]bool
	maxKnownFail   *int
	minKnownPass   *int
	ProvenancePath string
	RunID          string
}

// NewSearchExecutor builds an executor and readies the scratch/record dirs.
func NewSearchExecutor(config RunnerConfig, windows []Window, contract *TierContract,
	manifestPath, logsOutDir string, knownSizes map[int]bool,
	maxKnownFail, minKnownPass *int, provenancePath string) (*SearchExecutor, error) {
	if config.RequireEvalProvenance && config.EvalProvenance == nil {
		return nil, fmt.Errorf("require_eval_provenance is set but no verified EvalProvenance was attached")
	}
	if err := os.RemoveAll(config.WorkDir); err != nil {
		return nil, err
	}
	for _, d := range []string{config.WorkDir, config.OutDir, logsOutDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	known := map[int]bool{}
	for k := range knownSizes {
		known[k] = true
	}
	return &SearchExecutor{
		Config: config, Windows: windows, Contract: contract,
		ManifestPath: manifestPath, LogsOutDir: logsOutDir,
		delivered: map[int]EvalOutcome{}, RecipeBySize: map[int]string{},
		AnalysisBySize: map[int]string{}, knownSizes: known,
		maxKnownFail: maxKnownFail, minKnownPass: minKnownPass,
		ProvenancePath: provenancePath,
		RunID:          time.Now().Format("0102-150405"),
	}, nil
}

func (e *SearchExecutor) log(message string) {
	line := time.Now().Format("01-02 15:04:05") + "  " + message
	fmt.Println(line)
	if f, err := os.OpenFile(filepath.Join(e.Config.OutDir, "runner.log"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		f.WriteString(line + "\n")
		f.Close()
	}
}

// planDeliverable plans at target, bumping upward until a fresh deliverable.
func (e *SearchExecutor) planDeliverable(target int, window Window, tag string) (int, int, string, error) {
	bump := evaluateBumpBytes
	current := target
	deliverable := 0
	prefix := filepath.Join(e.Config.OutDir, tag+"-plan")
	lastOpt := (*pipeline.OptimizationPlan)(nil)
	for i := 0; i < 8; i++ {
		opt, pred, err := pipeline.Plan(
			filepath.Join(window.AnalysisPath, "analysis.json"),
			prefix, current, "balanced", "auto", e.Config.ModelName,
			e.Config.RefineProfile,
		)
		if err != nil {
			return 0, 0, "", err
		}
		lastOpt = opt
		_ = lastOpt
		deliverable = pred.TotalBytes
		useless := false
		if _, ok := e.delivered[deliverable]; ok {
			useless = true
		}
		if e.knownSizes[deliverable] {
			useless = true
		}
		if e.maxKnownFail != nil && deliverable <= *e.maxKnownFail {
			useless = true
		}
		if e.minKnownPass != nil && deliverable >= *e.minKnownPass {
			useless = true
		}
		if !useless {
			return deliverable, deliverable, prefix, nil
		}
		current = intMin(window.UpperSize, current+bump)
		e.log(fmt.Sprintf("%s: deliverable %d cannot tighten the bracket; bumping target to %d", tag, deliverable, current))
	}
	return deliverable, deliverable, prefix, nil
}

// evalDomains runs the five eval-v1 domain evaluations; nil on any failure.
// Domains are evaluated concurrently (bounded by Config.EvalConcurrency) so the
// per-domain llama-perplexity model loads overlap instead of loading the model
// under test serially five times. Each subprocess gets the runtime environment
// (own libs + sibling CUDA runtime) so llama.cpp never silently evaluates on
// CPU because a DLL failed to load.
func (e *SearchExecutor) evalDomains(artifact, tag string) (map[string]map[string]any, error) {
	binary, err := pipeline.RuntimeBinary(e.Config.Runtime, "llama-perplexity")
	if err != nil {
		return nil, err
	}
	env := RuntimeEnv(e.Config.Runtime)
	conc := e.Config.EvalConcurrency
	if conc < 1 {
		conc = 1
	}
	sem := make(chan struct{}, conc)

	metrics := map[string]map[string]any{}
	var (
		mu       sync.Mutex
		firstErr error
	)
	var wg sync.WaitGroup
	for _, domain := range Domains {
		domain := domain
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			logPath := filepath.Join(e.LogsOutDir, "eval-"+tag+"-"+domain+".log")
			sliceFile := filepath.Join(e.Config.EvalDataDir, "kl-eval-"+SLICE_SUFFIX[domain]+".txt")
			refFile := filepath.Join(e.Config.RefsDir, "bf16-"+domain+".kld")
			ok := false
			var out string
			var parsed map[string]any
			for attempt := 1; attempt <= 3; attempt++ {
				argv := []string{
					binary, "-m", artifact, "-f", sliceFile,
					"-ngl", "99", "-t", strconv.Itoa(e.Config.Threads),
					"-c", "512", "-b", "512", "--kl-divergence", "--kl-divergence-base", refFile,
				}
				out, rc := execCapture(argv, env)
				p, perr := ParseLLaMAKLLog(out)
				if perr != nil {
					e.log(fmt.Sprintf("%s: eval %s attempt %d failed (rc=%d); retrying", tag, domain, attempt, rc))
					time.Sleep(time.Duration(10*attempt) * time.Second)
					continue
				}
				if rc != 0 {
					e.log(fmt.Sprintf("%s: eval %s attempt %d parsed but exited rc=%d; retrying", tag, domain, attempt, rc))
					time.Sleep(time.Duration(10*attempt) * time.Second)
					continue
				}
				parsed = p
				ok = true
				break
			}
			if !ok {
				mu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("eval %s failed", domain)
				}
				mu.Unlock()
				return
			}
			_ = os.WriteFile(logPath, []byte(out), 0o644)
			mu.Lock()
			metrics[domain] = parsed
			mu.Unlock()
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return metrics, nil
}

func execCapture(argv []string, env []string) (string, int) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	rc := 0
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			rc = ee.ExitCode()
		} else {
			rc = -1
		}
	}
	return out.String(), rc
}

// Evaluate implements the search's Evaluator: plan -> quantize -> eval ->
// record -> release.
func (e *SearchExecutor) Evaluate(targetBytes int) EvalOutcome {
	e.Counter++
	tag := fmt.Sprintf("%s-%s-FS%02d", e.Config.ModelName, e.RunID, e.Counter)
	window, planned, err := SelectWindow(targetBytes, e.Windows, 0)
	if err != nil {
		return EvalOutcome{SizeBytes: targetBytes, HasKL: false, Note: err.Error()}
	}
	if planned != targetBytes {
		e.log(fmt.Sprintf("%s: target %d snapped to %d (%s->%s)", tag, targetBytes, planned, window.LowerPreset, window.UpperPreset))
	}
	deliverable, _, planPrefix, err := e.planDeliverable(planned, window, tag)
	if err != nil {
		return EvalOutcome{SizeBytes: targetBytes, HasKL: false, Note: err.Error()}
	}
	if cached, ok := e.delivered[deliverable]; ok {
		e.log(fmt.Sprintf("%s: window exhausted near target — returning cached outcome for %d", tag, deliverable))
		return cached
	}
	tensorTypes := planPrefix + "-tensor-types.txt"
	artifact := filepath.Join(e.Config.WorkDir, tag+".gguf")
	e.log(fmt.Sprintf("%s: plan deliverable %d (target %d) in %s->%s", tag, deliverable, planned, window.LowerPreset, window.UpperPreset))
	actual, _, err := pipeline.Quantize(
		filepath.Join(window.AnalysisPath, "analysis.json"),
		tensorTypes, artifact, 0, e.Config.Imatrix, "",
	)
	if err != nil {
		return EvalOutcome{SizeBytes: targetBytes, HasKL: false, Note: err.Error()}
	}

	metrics, err := e.evalDomains(artifact, tag)
	if err != nil {
		os.Remove(artifact)
		return EvalOutcome{SizeBytes: actual, HasKL: false, Note: "eval failed"}
	}

	var klSum, topSum float64
	for _, d := range Domains {
		klSum += ffv(metrics[d]["mean_kld"])
		topSum += ffv(metrics[d]["same_top_pct"])
	}
	macroKL := klSum / float64(len(Domains))
	macroTop := topSum / float64(len(Domains))
	passed := e.Contract.Passes(macroKL, macroTop/100.0)
	e.log(fmt.Sprintf("%s: kld %.4f top %.2f %s @ %d", tag, macroKL, macroTop, passWord(passed), actual))

	outcome := EvalOutcome{SizeBytes: actual, MacroKL: macroKL, SameTop: macroTop / 100.0, HasKL: true}
	e.delivered[actual] = outcome
	e.RecipeBySize[actual] = tensorTypes
	e.AnalysisBySize[actual] = window.AnalysisPath
	if e.ProvenancePath != "" {
		recordProvenance(e.ProvenancePath, tag, window, actual, e.Config.EvalProvenance)
	}
	if e.Contract.Passes(outcome.MacroKL, outcome.SameTop) {
		// PASS → tightens min_known_pass lower bound
		if e.minKnownPass == nil || actual < *e.minKnownPass {
			v := actual
			e.minKnownPass = &v
		}
	} else {
		// FAIL → tightens max_known_fail upper bound
		if e.maxKnownFail == nil || actual > *e.maxKnownFail {
			v := actual
			e.maxKnownFail = &v
		}
	}

	if !manifestHas(e.ManifestPath, tag) {
		digest, _ := SHA256File(artifact)
		if f, err := os.OpenFile(e.ManifestPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
			f.WriteString(fmt.Sprintf("%s  %d  %s\n", tag, actual, digest))
			f.Close()
		}
	}
	os.Remove(artifact)
	return outcome
}

func passWord(p bool) string {
	if p {
		return "PASS"
	}
	return "FAIL"
}

func recordProvenance(path, tag string, window Window, sizeBytes int, prov *EvalProvenance) {
	rec := map[string]any{
		"name": tag, "size_bytes": sizeBytes,
		"window_lower_preset": window.LowerPreset,
		"window_upper_preset": window.UpperPreset,
		"attestation":         "unattested",
	}
	if prov != nil {
		rec["attestation"] = "runtime-verified"
		rec["eval_contract_digest"] = prov.ContractDigest
		if sha, err := SHA256File(prov.ReferenceManifestPath); err == nil {
			rec["reference_manifest_sha256"] = sha
		}
	}
	if f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		data, _ := json.Marshal(rec)
		f.WriteString(string(data) + "\n")
		f.Close()
	}
}

func manifestHas(manifestPath, name string) bool {
	if !fileExists(manifestPath) {
		return false
	}
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if parts := strings.Fields(line); len(parts) > 0 && parts[0] == name {
			return true
		}
	}
	return false
}

func intMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// RunTierSearch wires the pure search to the real executor for one tier.
func RunTierSearch(contract *TierContract, config RunnerConfig, windows []Window, seeds []Seed,
	minSize, maxSize, budget, toleranceBytes int, manifestPath, logsOutDir string) (map[string]any, error) {
	known := map[int]bool{}
	for _, s := range seeds {
		known[s.SizeBytes] = true
	}
	var fails, passes []int
	for _, s := range seeds {
		if contract.Passes(s.MacroKL, s.SameTop) {
			passes = append(passes, s.SizeBytes)
		} else {
			fails = append(fails, s.SizeBytes)
		}
	}
	var maxFail, minPass *int
	if len(fails) > 0 {
		best := fails[0]
		for _, v := range fails[1:] {
			if v > best {
				best = v
			}
		}
		maxFail = &best
	}
	if len(passes) > 0 {
		best := passes[0]
		for _, v := range passes[1:] {
			if v < best {
				best = v
			}
		}
		minPass = &best
	}
	provPath := filepath.Join(filepath.Dir(manifestPath), "seed-provenance.jsonl")
	executor, err := NewSearchExecutor(config, windows, contract, manifestPath, logsOutDir,
		known, maxFail, minPass, provPath)
	if err != nil {
		return nil, err
	}
	audit := filepath.Join(config.OutDir, "fidelity-search-"+contract.Tier+".jsonl")
	_ = os.Remove(audit)
	result, err := FidelitySearch(
		*contract, executor.Evaluate, seeds,
		minSize, maxSize, budget, toleranceBytes, 0,
	)
	if err != nil {
		return nil, err
	}
	summary := result.Summary()
	recipes := map[string]string{}
	for size, path := range executor.RecipeBySize {
		recipes[strconv.Itoa(size)] = path
	}
	analyses := map[string]string{}
	for size, path := range executor.AnalysisBySize {
		analyses[strconv.Itoa(size)] = path
	}
	summary["artifact_recipes"] = recipes
	summary["artifact_analyses"] = analyses
	_ = os.RemoveAll(config.WorkDir)
	return summary, nil
}
