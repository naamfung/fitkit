// calibrate.go implements the fit calibrate production line, mirroring
// upstream fit_gguf.calibrate (v0.3.3): pinned inputs -> imatrix coverage
// check -> five-domain references -> standard preset ladder -> boundary windows
// -> gap probes -> floor derivation -> Guard Profile -> Calibration Bundle with
// a candidate registry entry. Fail-closed per the frozen contract.
package calibration

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"fitting/fidelity"
	"fitting/gguf"
	"fitting/pipeline"
	"fitting/registry"
)

// SLICESuffix maps domain -> eval slice filename suffix (eval-v1).
var SLICESuffix = map[string]string{
	"wiki_test":  "64k.txt",
	"wiki_valid": "valid-64k.txt",
	"chinese":    "cn-64k.txt",
	"code":       "code-64k.txt",
	"agent_chat": "agent-64k.txt",
}

// Config carries every input of a calibrate run.
type Config struct {
	Source         string
	ImatrixCorpus  string
	RuntimeDir     string
	EvalDataDir    string
	OutDir         string
	ModelID        string
	ImatrixPath    string
	ImatrixArg     string
	Chunks         int
	NGPULayers     int
	Threads        int
	Workdir        string
	OnDisk         bool
	ExtraPresets   []string
	ProbeBudget    int
	ContractPath   string
	LogDir         string
	ReplayExisting string
	ReplayManifest string
}

// Result is the outcome of a calibrate run.
type Result struct {
	Mode         string
	Bundle       string
	Report       map[string]any
	SourceSHA256 string
}

// ------------------------------------------------------------------ runtime

// runCmd runs argv with output redirected to logPath (the subprocess writes its
// own log, mirroring upstream _run). Returns the exit code.
func runCmd(argv []string, logPath string, env []string) (int, error) {
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return -1, err
	}
	logFile, err := os.Create(logPath)
	if err != nil {
		return -1, err
	}
	defer logFile.Close()
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Env = env
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), nil
		}
		return -1, err
	}
	return 0, nil
}

// refOK is the _logits_ arithmetic completeness check for a reference .kld
// (b10666-era; guards against truncated writes).
func refOK(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, 20)
	n, err := f.Read(head)
	if err != nil || n < 20 {
		return false
	}
	if string(head[:8]) != "_logits_" {
		return false
	}
	nCtx := uint32(head[8]) | uint32(head[9])<<8 | uint32(head[10])<<16 | uint32(head[11])<<24
	nVocab := uint32(head[12]) | uint32(head[13])<<8 | uint32(head[14])<<16 | uint32(head[15])<<24
	nChunk := uint32(head[16]) | uint32(head[17])<<8 | uint32(head[18])<<16 | uint32(head[19])<<24
	nv := 2*((nVocab+1)/2) + 4
	expected := int64(20) + int64(nChunk)*int64(nCtx)*4 + int64(nChunk)*(int64(nCtx)-1-int64(nCtx)/2)*int64(nv)*2
	st, err := os.Stat(path)
	if err != nil {
		return false
	}
	return st.Size() == expected
}

// ------------------------------------------------------------------ evaluate

func sha256HexBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func sha256File(path string) (string, error) {
	return fidelity.SHA256File(path)
}

func fileSize(path string) int64 {
	if st, err := os.Stat(path); err == nil {
		return st.Size()
	}
	return 0
}

func (c *Config) log(message string) {
	fmt.Printf("[calibrate] %s\n", message)
}

// evalArtifact runs the five-domain eval-v1 evaluation of one artifact and
// returns an observation dict.
func (c *Config) evalArtifact(artifact, refsDir, pointID string, env []string) (map[string]any, error) {
	metrics := map[string]any{}
	for _, domain := range fidelity.Domains {
		sliceFile := filepath.Join(c.EvalDataDir, "kl-eval-"+SLICESuffix[domain]+".txt")
		ref := filepath.Join(refsDir, "bf16-"+domain+".kld")
		logPath := filepath.Join(c.LogDir, "eval-"+pointID+"-"+domain+".log")
		var parsed map[string]any
		ok := false
		for attempt := 1; attempt <= 3; attempt++ {
			binary, err := pipeline.RuntimeBinary(c.RuntimeDir, "llama-perplexity")
			if err != nil {
				return nil, err
			}
			rc, err := runCmd([]string{
				binary, "-m", artifact, "-f", sliceFile,
				"-ngl", strconv.Itoa(c.NGPULayers), "-t", strconv.Itoa(c.Threads),
				"-c", "512", "-b", "512",
				"--kl-divergence", "--kl-divergence-base", ref,
			}, logPath, env)
			if err != nil {
				return nil, err
			}
			raw, _ := os.ReadFile(logPath)
			p, perr := fidelity.ParseLLaMAKLLog(string(raw))
			if perr == nil && rc == 0 {
				parsed = p
				ok = true
				break
			}
			c.log(fmt.Sprintf("eval %s/%s attempt %d failed (rc=%d)", pointID, domain, attempt, rc))
			time.Sleep(time.Duration(5*attempt) * time.Second)
		}
		if !ok {
			return nil, calibrationErrf("eval failed: %s/%s", pointID, domain)
		}
		metrics[domain] = parsed
	}
	klSum := 0.0
	topSum := 0.0
	perDomain := map[string]any{}
	for _, domain := range fidelity.Domains {
		m, _ := metrics[domain].(map[string]any)
		klSum += ffloat(m["mean_kld"])
		topSum += ffloat(m["same_top_pct"])
		perDomain[domain] = map[string]any{"mean_kld": ffloat(m["mean_kld"]), "same_top_pct": ffloat(m["same_top_pct"])}
	}
	macroKL := klSum / float64(len(fidelity.Domains))
	macroTop := topSum / float64(len(fidelity.Domains)) / 100.0
	sha, err := sha256File(artifact)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"point_id":        pointID,
		"size_bytes":      int(fileSize(artifact)),
		"artifact_sha256": sha,
		"macro_kl":        macroKL,
		"same_top":        macroTop,
		"per_domain":      perDomain,
	}, nil
}

// ------------------------------------------------------------------- stages

func (c *Config) stageGenerateImatrix(env []string) (string, error) {
	if c.ImatrixPath != "" {
		return c.ImatrixPath, nil
	}
	work := c.Workdir
	if work == "" {
		work = c.OutDir
	}
	imx := filepath.Join(work, "calibration-imatrix.gguf")
	if fileSize(imx) > 0 {
		return imx, nil
	}
	c.log("generating imatrix (corpus, contract default chunks)")
	binary, err := pipeline.RuntimeBinary(c.RuntimeDir, "llama-imatrix")
	if err != nil {
		return "", err
	}
	rc, err := runCmd([]string{
		binary, "-m", c.Source, "-f", c.ImatrixCorpus, "-c", "512",
		"-ngl", strconv.Itoa(c.NGPULayers),
		"--chunks", strconv.Itoa(c.Chunks), "-o", imx,
	}, filepath.Join(c.LogDir, "imatrix.log"), env)
	if err != nil {
		return "", err
	}
	if rc != 0 || fileSize(imx) <= 0 {
		return "", calibrationErrf("REF_GENERATION_FAILED: imatrix generation failed")
	}
	return imx, nil
}

func (c *Config) stageImatrixCoverage(imx string) ([]string, error) {
	profile, err := pipeline.LoadImatrixProfile(imx)
	if err != nil {
		return nil, err
	}
	have := map[string]bool{}
	for _, e := range profile.Entries {
		have[e.Name] = true
	}
	layout, err := gguf.ReadLayout(c.Source)
	if err != nil {
		return nil, err
	}
	var missing []string
	for _, t := range layout.Tensors {
		if have[t.Name] {
			continue
		}
		if strings.Contains(t.Name, "embd") || strings.HasSuffix(t.Name, "norm.weight") {
			continue
		}
		missing = append(missing, t.Name)
	}
	c.log(fmt.Sprintf("imatrix coverage: %d entries, %d missing matrices", len(have), len(missing)))
	return missing, nil
}

func (c *Config) stageReferences(env []string, refsDir string) (map[string]any, error) {
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return nil, err
	}
	domains := map[string]any{}
	for _, domain := range fidelity.Domains {
		sliceFile := filepath.Join(c.EvalDataDir, "kl-eval-"+SLICESuffix[domain]+".txt")
		raw, err := os.ReadFile(sliceFile)
		if err != nil {
			return nil, err
		}
		corpusSHA := sha256HexBytes(raw)
		out := filepath.Join(refsDir, "bf16-"+domain+".kld")
		if fileSize(out) > 0 && refOK(out) {
			domains[domain] = map[string]any{
				"corpus_sha256":       corpusSHA,
				"raw_bytes":           len(raw),
				"reference_kld_sha256": mustSHA(out),
			}
			continue
		}
		logPath := filepath.Join(c.LogDir, "ref-"+domain+".log")
		ok := false
		for attempt := 1; attempt <= 4; attempt++ {
			c.log(fmt.Sprintf("reference %s (attempt %d)", domain, attempt))
			_ = os.Remove(out)
			binary, err := pipeline.RuntimeBinary(c.RuntimeDir, "llama-perplexity")
			if err != nil {
				return nil, err
			}
			rc, err := runCmd([]string{
				binary, "-m", c.Source, "-f", sliceFile,
				"-ngl", strconv.Itoa(c.NGPULayers), "-t", strconv.Itoa(c.Threads),
				"-c", "512", "-b", "512", "--kl-divergence-base", out,
			}, logPath, env)
			if err != nil {
				return nil, err
			}
			if rc == 0 && fileSize(out) > 0 && refOK(out) {
				ok = true
				break
			}
			time.Sleep(10 * time.Second)
		}
		if !ok {
			return nil, calibrationErrf("REF_GENERATION_FAILED: %s", domain)
		}
		domains[domain] = map[string]any{
			"corpus_sha256":       corpusSHA,
			"raw_bytes":           len(raw),
			"reference_kld_sha256": mustSHA(out),
		}
	}
	return domains, nil
}

func mustSHA(path string) string {
	sha, err := sha256File(path)
	if err != nil {
		return ""
	}
	return sha
}

func (c *Config) stageLadder(contract *Contract, env []string, imx, refsDir string, missing []string) ([]map[string]any, error) {
	presets := append([]string{}, contract.LadderStandardPresets()...)
	presets = append(presets, c.ExtraPresets...)
	work := c.Workdir
	if work == "" {
		work = c.OutDir
	}
	observations := loadCurvePoints(c.OutDir)
	known := map[string]bool{}
	for _, o := range observations {
		known[fstring(o["point_id"])] = true
	}
	for _, preset := range presets {
		if known[preset] {
			c.log(fmt.Sprintf("ladder %s: already recorded, skipping", preset))
			continue
		}
		c.log(fmt.Sprintf("ladder %s: quantize", preset))
		artifact := filepath.Join(work, "ladder-"+preset+".gguf")
		binary, err := pipeline.RuntimeBinary(c.RuntimeDir, "llama-quantize")
		if err != nil {
			return nil, err
		}
		rc, err := runCmd([]string{
			binary, "--imatrix", imx, c.Source, artifact, preset,
		}, filepath.Join(c.LogDir, "quantize-"+preset+".log"), env)
		if err != nil {
			return nil, err
		}
		if rc != 0 || fileSize(artifact) <= 0 {
			c.log(fmt.Sprintf("ladder %s: quantize FAILED — point skipped", preset))
			continue
		}
		obs, err := c.evalArtifact(artifact, refsDir, preset, env)
		if err != nil {
			_ = os.Remove(artifact)
			return nil, err
		}
		observations = append(observations, obs)
		appendCurvePoint(c.OutDir, obs)
		_ = os.Remove(artifact)
	}
	return observations, nil
}

func loadCurvePoints(bundle string) []map[string]any {
	curve := filepath.Join(bundle, "curve-points.jsonl")
	raw, err := os.ReadFile(curve)
	if err != nil {
		return nil
	}
	var out []map[string]any
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var obs map[string]any
		if json.Unmarshal([]byte(line), &obs) == nil {
			out = append(out, obs)
		}
	}
	return out
}

func appendCurvePoint(bundle string, obs map[string]any) {
	curve := filepath.Join(bundle, "curve-points.jsonl")
	f, err := os.OpenFile(curve, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	line, _ := json.Marshal(obs)
	f.Write(append(line, '\n'))
}

func largestUncoveredGap(pool []map[string]any, lo, hi float64) (float64, bool) {
	kls := map[float64]bool{}
	for _, o := range pool {
		kls[ffloat(o["macro_kl"])] = true
	}
	kls[lo] = true
	kls[hi] = true
	sortedKL := make([]float64, 0, len(kls))
	for k := range kls {
		sortedKL = append(sortedKL, k)
	}
	sort.Float64s(sortedKL)
	best := 0.0
	bestSpan := 0.0
	for i := 0; i < len(sortedKL)-1; i++ {
		a2 := math.Max(sortedKL[i], lo)
		b2 := math.Min(sortedKL[i+1], hi)
		if b2 > a2 && b2-a2 > bestSpan {
			bestSpan = b2 - a2
			best = (a2 + b2) / 2
		}
	}
	if bestSpan <= 0 {
		return 0, false
	}
	return best, true
}

func (c *Config) stageGapProbes(contract *Contract, env []string, imx, refsDir string, observations []map[string]any) ([]map[string]any, []string) {
	newObs := []map[string]any{}
	var unresolved []string
	imatrixArg := c.ImatrixArg
	if imatrixArg == "" {
		imatrixArg = imx
	}
	budget := c.ProbeBudget
	if budget <= 0 {
		budget = contract.ProbeBudgetPerTier()
	}
	poison := contract.poisonSet()
	for _, tier := range TIERS {
		anchor := contract.TierKLAnchors()[tier]
		lo, hi := WindowBounds(anchor, contract)
		pool := append(append([]map[string]any{}, observations...), newObs...)
		inWindow := windowEvidence(pool, lo, hi, poison)
		if len(inWindow) >= contract.ValidationMinSamples() {
			continue
		}
		if len(inWindow) == 0 {
			below := 0
			above := 0
			for _, o := range pool {
				kl := ffloat(o["macro_kl"])
				if kl < lo {
					below++
				}
				if kl > hi {
					above++
				}
			}
			if below == 0 || above == 0 {
				unresolved = append(unresolved, tier)
				c.log(fmt.Sprintf("gap %s: NOT_REACHABLE (no bracketing observations)", tier))
				continue
			}
		}
		c.log(fmt.Sprintf("gap %s: window [%.4f,%.4f] n=%d — probing", tier, lo, hi, len(inWindow)))
		probes := 0
		for _, o := range pool {
			if probe, ok := o["probe"].(map[string]any); ok && fstring(probe["tier"]) == tier {
				probes++
			}
		}
		for probes < budget {
			pool = append(append([]map[string]any{}, observations...), newObs...)
			inWindow = windowEvidence(pool, lo, hi, poison)
			if len(inWindow) >= contract.ValidationMinSamples() {
				break
			}
			targetKL, ok := largestUncoveredGap(pool, lo, hi)
			if !ok {
				unresolved = append(unresolved, tier)
				break
			}
			byKL := append([]map[string]any(nil), pool...)
			sort.SliceStable(byKL, func(a, b int) bool { return ffloat(byKL[a]["macro_kl"]) < ffloat(byKL[b]["macro_kl"]) })
			var lower, upper map[string]any
			for _, o := range byKL {
				if ffloat(o["macro_kl"]) <= targetKL {
					lower = o
				}
			}
			for _, o := range byKL {
				if ffloat(o["macro_kl"]) >= targetKL {
					upper = o
					break
				}
			}
			if lower == nil || upper == nil {
				unresolved = append(unresolved, tier)
				break
			}
			spanKL := ffloat(upper["macro_kl"]) - ffloat(lower["macro_kl"])
			if spanKL <= 0 {
				unresolved = append(unresolved, tier)
				break
			}
			frac := (targetKL - ffloat(lower["macro_kl"])) / spanKL
			target := int(ffloat(lower["size_bytes"]) + frac*float64(fint(upper["size_bytes"])-fint(lower["size_bytes"])))

			// dry-run anchors must be named presets bracketing the target size
			var presetObs []map[string]any
			for _, o := range pool {
				pid := fstring(o["point_id"])
				if _, ok := pipeline.PresetFileTypes[pid]; ok && !poison[pid] {
					presetObs = append(presetObs, o)
				}
			}
			sort.SliceStable(presetObs, func(a, b int) bool { return fint(presetObs[a]["size_bytes"]) < fint(presetObs[b]["size_bytes"]) })
			var belowP, aboveP map[string]any
			for _, o := range presetObs {
				if fint(o["size_bytes"]) <= target {
					belowP = o
				}
			}
			for _, o := range presetObs {
				if fint(o["size_bytes"]) > target {
					aboveP = o
					break
				}
			}
			if belowP == nil || aboveP == nil {
				unresolved = append(unresolved, tier)
				break
			}
			lowerPreset := fstring(belowP["point_id"])
			upperPreset := fstring(aboveP["point_id"])
			analysisDir := filepath.Join(c.OutDir, "analysis", lowerPreset+"-"+upperPreset)
			analysisJSON := filepath.Join(analysisDir, "analysis.json")
			if !fileExists(analysisJSON) {
				if _, err := pipeline.Analyze(c.Source, imx, c.RuntimeDir, analysisDir,
					lowerPreset, upperPreset, imatrixArg, false, "up"); err != nil {
					unresolved = append(unresolved, tier)
					break
				}
			}
			tag := fmt.Sprintf("probe-%s-%d", tier, probes+1)
			planPrefix := filepath.Join(c.OutDir, "probes", tag)
			opt, _, err := pipeline.Plan(analysisJSON, planPrefix, target, "balanced", "auto", c.ModelID, "")
			if err != nil {
				unresolved = append(unresolved, tier)
				break
			}
			artifact := filepath.Join(workDirOf(c), tag+".gguf")
			if _, _, err := pipeline.Quantize(analysisJSON,
				planPrefix+"-tensor-types.txt", artifact, 0, imatrixArg, ""); err != nil {
				unresolved = append(unresolved, tier)
				break
			}
			probes++
			obs, err := c.evalArtifact(artifact, refsDir, tag, env)
			if err != nil {
				_ = os.Remove(artifact)
				continue
			}
			obs["point_id"] = tag
			planSHA := mustSHA(planPrefix + "-plan.json")
			recipeSHA := mustSHA(planPrefix + "-tensor-types.txt")
			obs["probe"] = map[string]any{"tier": tier, "target_bytes": target, "plan_sha256": planSHA, "recipe_sha256": recipeSHA}
			newObs = append(newObs, obs)
			appendCurvePoint(c.OutDir, obs)
			_ = os.Remove(artifact)
			c.log(fmt.Sprintf("gap %s: probe %s kld=%.4f top=%.4f", tier, tag, ffloat(obs["macro_kl"]), ffloat(obs["same_top"])))
			_ = opt
		}
		pool = append(append([]map[string]any{}, observations...), newObs...)
		inWindow = windowEvidence(pool, lo, hi, poison)
		if len(inWindow) < contract.ValidationMinSamples() {
			unresolved = append(unresolved, tier)
		}
	}
	return newObs, unresolved
}

func workDirOf(c *Config) string {
	if c.Workdir != "" {
		return c.Workdir
	}
	return c.OutDir
}

func windowEvidence(pool []map[string]any, lo, hi float64, poison map[string]bool) []map[string]any {
	var out []map[string]any
	for _, o := range pool {
		kl := ffloat(o["macro_kl"])
		if lo <= kl && kl <= hi && !poison[fstring(o["point_id"])] {
			out = append(out, o)
		}
	}
	return out
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// stageFloors evaluates the contract over the usable observations.
func (c *Config) stageFloors(observations []map[string]any, contract *Contract, openFailures []string) map[string]any {
	poison := contract.poisonSet()
	var usable []map[string]any
	for _, o := range observations {
		if !poison[fstring(o["point_id"])] {
			usable = append(usable, o)
		}
	}
	eval, err := EvaluateObservations(usable, contract, openFailures)
	if err != nil {
		return map[string]any{
			"contract_id": ContractID, "calibration_contract_sha256": contract.SHA256,
			"observation_count": len(usable), "tiers": map[string]any{},
			"overall_status": "candidate", "open_failures": []string{"CONTRACT_ERROR"},
		}
	}
	return eval
}

func writeJSON(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	err = enc.Encode(v)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func writeSeedMaterial(bundle string, observations []map[string]any, presetNames map[string]bool, referenceManifestSHA256, evaluatorContractSHA256 string) error {
	var manifestLines, provenanceLines []string
	for _, obs := range observations {
		point := fstring(obs["point_id"])
		size := fint(obs["size_bytes"])
		sha := fstring(obs["artifact_sha256"])
		manifestLines = append(manifestLines, fmt.Sprintf("%s  %d  %s", point, size, sha))
		anchor := ""
		if presetNames[point] {
			anchor = point
		}
		prov, _ := json.Marshal(map[string]any{
			"name":                     point,
			"size_bytes":               size,
			"window_lower_preset":      anchor,
			"window_upper_preset":      anchor,
			"attestation":              "runtime-verified",
			"eval_contract_digest":     evaluatorContractSHA256,
			"reference_manifest_sha256": referenceManifestSHA256,
		})
		provenanceLines = append(provenanceLines, string(prov))
	}
	manifest := strings.Join(manifestLines, "\n")
	if manifest != "" {
		manifest += "\n"
	}
	prov := strings.Join(provenanceLines, "\n")
	if prov != "" {
		prov += "\n"
	}
	if err := os.WriteFile(filepath.Join(bundle, "state-artifact-manifest.txt"), []byte(manifest), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(bundle, "seed-provenance.jsonl"), []byte(prov), 0o644)
}

// stageEmit writes the Calibration Bundle (the eight-piece layout).
func (c *Config) stageEmit(contract *Contract, evaluation map[string]any, observations []map[string]any, refsDir string, domains map[string]any, missing []string, unresolved []string) (string, error) {
	bundle := c.OutDir
	if err := os.MkdirAll(bundle, 0o755); err != nil {
		return "", err
	}
	sourceSHA, err := sha256File(c.Source)
	if err != nil {
		return "", err
	}

	// guard profile
	guard := map[string]any{
		"calibration_contract":          ContractID,
		"calibration_contract_sha256":   contract.SHA256,
		"evaluator_contract":            "eval-v1",
		"generalization_scope":          "exact_model_only",
		"guard_profile_id":              fmt.Sprintf("guard-%s-exact-v1", c.ModelID),
		"guard_profile_version":         1,
		"scope":                         map[string]any{"identifier": c.ModelID, "type": "exact_model"},
		"source_sha256":                 sourceSHA,
		"status":                        fstring(evaluation["overall_status"]),
		"tiers":                         map[string]any{},
		"validation_basis":              ContractID,
	}
	tiersOut := map[string]any{}
	tiersRaw, _ := evaluation["tiers"].(map[string]any)
	for _, tier := range TIERS {
		tr, _ := tiersRaw[tier].(map[string]any)
		if tr == nil {
			continue
		}
		tiersOut[tier] = map[string]any{
			"kl_anchor":         tr["anchor"],
			"same_top_floor":    tr["floor"],
			"floor_method":      tr["floor_method"],
			"sample_count":      tr["sample_count"],
			"witness":           tr["witness"],
			"validation_status": tr["validation_status"],
		}
	}
	guard["tiers"] = tiersOut
	guard["confidence"] = map[string]any{
		"note":      "fidelity-calibration-v1 onboarding; floor_method/sample_count per tier",
		"window_n":  map[string]any{"quality": countOf(tiersRaw, "quality"), "balanced": countOf(tiersRaw, "balanced"), "compact": countOf(tiersRaw, "compact"), "mini": countOf(tiersRaw, "mini")},
	}
	guard["profile_hash"], _ = fidelity.ProfileHash(guard)
	guardYAML, err := yaml.Marshal(guard)
	if err != nil {
		return "", err
	}
	guardPath := filepath.Join(bundle, "guard-profile.yaml")
	if err := os.WriteFile(guardPath, guardYAML, 0o644); err != nil {
		return "", err
	}

	// reference manifest
	manifest := map[string]any{
		"manifest_schema":         "fit.eval_reference_manifest.v1",
		"evaluator_contract":      "eval-v1",
		"evaluator_contract_hash": fstring(contract.Payload["evaluator_contract_sha256"]),
		"source_bf16_gguf_sha256": sourceSHA,
		"domains":                 domains,
		"runtime_provenance": map[string]any{
			"runtime_dir": c.RuntimeDir,
			"execution":   map[string]any{"n_gpu_layers": c.NGPULayers, "threads": c.Threads, "on_disk": c.OnDisk},
		},
	}
	manifestPath := filepath.Join(bundle, "reference-manifest.json")
	if err := writeJSONIndent(manifestPath, manifest); err != nil {
		return "", err
	}

	// curve points (rewrite in deterministic order)
	curvePath := filepath.Join(bundle, "curve-points.jsonl")
	var curveBytes []byte
	for _, obs := range observations {
		line, _ := json.Marshal(obs)
		curveBytes = append(curveBytes, line...)
		curveBytes = append(curveBytes, '\n')
	}
	if err := os.WriteFile(curvePath, curveBytes, 0o644); err != nil {
		return "", err
	}

	// seed material: the ladder becomes budget-free bracket evidence
	presetNames := map[string]bool{}
	for _, p := range contract.LadderStandardPresets() {
		presetNames[p] = true
	}
	for _, p := range c.ExtraPresets {
		presetNames[p] = true
	}
	manifestSHA, _ := sha256File(manifestPath)
	if err := writeSeedMaterial(bundle, observations, presetNames, manifestSHA, fidelity.FrozenContractDigest); err != nil {
		return "", err
	}

	// calibration record
	derivation := map[string]any{}
	for _, tier := range TIERS {
		tr, _ := tiersRaw[tier].(map[string]any)
		if tr == nil {
			continue
		}
		clean := map[string]any{}
		for k, v := range tr {
			if k != "samples" {
				clean[k] = v
			}
		}
		derivation[tier] = clean
	}
	curveSHA, _ := sha256File(curvePath)
	record := map[string]any{
		"schema":                      "fit.calibration_record.v1",
		"contract_id":                 ContractID,
		"calibration_contract_sha256": contract.SHA256,
		"model_id":                    c.ModelID,
		"created":                     time.Now().Format("2006-01-02"),
		"inputs": map[string]any{
			"source_weights_sha256":    sourceSHA,
			"imatrix_corpus_sha256":    mustSHA(c.ImatrixCorpus),
			"imatrix_chunks":           c.Chunks,
			"evaluator_contract_sha256": fstring(contract.Payload["evaluator_contract_sha256"]),
			"runtime_dir":              c.RuntimeDir,
		},
		"process": map[string]any{
			"imatrix_missing_matrices": missing,
			"unresolved_tiers":         unresolved,
			"curve_points":             len(observations),
		},
		"derivation": derivation,
		"artifacts": map[string]any{
			"guard_profile_sha256":     mustSHA(guardPath),
			"reference_manifest_sha256": manifestSHA,
			"curve_points_sha256":      curveSHA,
		},
		"failures":      evaluation["open_failures"],
		"overall_status": evaluation["overall_status"],
	}
	recordPath := filepath.Join(bundle, "calibration-record.json")
	if err := writeJSONIndent(recordPath, record); err != nil {
		return "", err
	}

	// candidate registry entry
	registryEntry := map[string]any{
		"registry_schema":            registry.RegistrySchema,
		"model_id":                   c.ModelID,
		"source_weights_sha256":      sourceSHA,
		"tokenizer_sha256":           nil,
		"evaluator_contract":         "eval-v1",
		"evaluator_contract_sha256":  fstring(contract.Payload["evaluator_contract_sha256"]),
		"reference_manifest":         map[string]any{"path": "reference-manifest.json", "sha256": manifestSHA},
		"guard_profile":              map[string]any{"path": "guard-profile.yaml", "sha256": mustSHA(guardPath)},
		"calibration": map[string]any{
			"basis":          ContractID,
			"contract":       ContractID,
			"contract_sha256": contract.SHA256,
			"record":         map[string]any{"path": "calibration-record.json", "sha256": mustSHA(recordPath)},
		},
		"scope":      "exact_model",
		"status":     fstring(evaluation["overall_status"]),
		"added_in":   "PENDING",
	}
	entryDigest, err := registry.EntryDigest(registryEntry)
	if err != nil {
		return "", err
	}
	registryEntry["entry_sha256"] = entryDigest
	entryBytes, err := registry.CanonicalJSONBytes(registryEntry)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(bundle, "registry-entry.json"), entryBytes, 0o644); err != nil {
		return "", err
	}

	// profile report (markdown)
	reportLines := []string{
		fmt.Sprintf("# Calibration report — %s", c.ModelID),
		"",
		fmt.Sprintf("- contract: fidelity-calibration-v1 (%s…)…", contract.SHA256[:16]),
		fmt.Sprintf("- source: %s…", sourceSHA[:16]),
		fmt.Sprintf("- overall: **%s**", fstring(evaluation["overall_status"])),
		fmt.Sprintf("- open failures: %v", strOrNone(evaluation["open_failures"])),
		"",
		"| tier | window | n | floor | method | witness | status |",
		"|---|---|---|---|---|---|---|",
	}
	for _, tier := range TIERS {
		tr, _ := tiersRaw[tier].(map[string]any)
		if tr == nil {
			continue
		}
		witness := "✗"
		if tr["witness"] != nil {
			witness = "✓"
		}
		win, _ := tr["window"].([2]float64)
		reportLines = append(reportLines, fmt.Sprintf(
			"| %s | [%v, %v] | %v | %v | %v | %s | %v |",
			tier, win[0], win[1], tr["sample_count"], tr["floor"], tr["floor_method"], witness, tr["validation_status"]))
	}
	if err := os.WriteFile(filepath.Join(bundle, "profile-report.md"), []byte(strings.Join(reportLines, "\n")+"\n"), 0o644); err != nil {
		return "", err
	}

	// SHA256SUMS over every bundle file
	sums := []string{}
	entries, err := os.ReadDir(bundle)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		sha, _ := sha256File(filepath.Join(bundle, e.Name()))
		sums = append(sums, fmt.Sprintf("%s  %s", sha, e.Name()))
	}
	sort.Strings(sums)
	if err := os.WriteFile(filepath.Join(bundle, "SHA256SUMS"), []byte(strings.Join(sums, "\n")+"\n"), 0o644); err != nil {
		return "", err
	}

	admission, err := registry.ValidateBundle(bundle)
	if err != nil {
		return "", err
	}
	c.log(fmt.Sprintf("bundle admissible: %v (%s)", admission["admissible"], fstring(admission["status"])))
	return bundle, nil
}

func writeJSONIndent(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	err = enc.Encode(v)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func countOf(tiers map[string]any, tier string) int {
	if tr, ok := tiers[tier].(map[string]any); ok {
		return fint(tr["sample_count"])
	}
	return 0
}

func strOrNone(v any) string {
	if s, ok := v.([]string); ok && len(s) > 0 {
		return strings.Join(s, ",")
	}
	if v == nil {
		return "none"
	}
	return fmt.Sprintf("%v", v)
}

// -------------------------------------------------------------------- replay

// replayExisting re-derives the calibration from recorded observations with
// zero evaluations (Gate P2-A).
func (c *Config) replayExisting(observationsPath, manifestPath string) (map[string]any, error) {
	contract, err := LoadContract(c.ContractPath)
	if err != nil {
		return nil, err
	}
	payloadRaw, err := os.ReadFile(observationsPath)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(payloadRaw, &payload); err != nil {
		return nil, err
	}
	rawPoints, _ := payload["points"].([]any)
	shas := map[string]string{}
	if raw, err := os.ReadFile(manifestPath); err == nil {
		for _, line := range strings.Split(string(raw), "\n") {
			parts := strings.Fields(line)
			if len(parts) == 3 && strings.HasPrefix(parts[0], c.ModelID+"-") {
				shas[strings.TrimPrefix(parts[0], c.ModelID+"-")] = parts[2]
			}
		}
	}
	observations := make([]map[string]any, 0, len(rawPoints))
	for _, p := range rawPoints {
		pm, _ := p.(map[string]any)
		if pm == nil {
			continue
		}
		point := fstring(pm["point"])
		sha := shas[point]
		if sha == "" {
			sha = strings.Repeat("0", 64)
		}
		observations = append(observations, map[string]any{
			"point_id":        point,
			"size_bytes":      fint(pm["size_bytes"]),
			"artifact_sha256": sha,
			"macro_kl":        ffloat(pm["macro_kl"]),
			"same_top":        ffloat(pm["same_top"]),
		})
	}
	poison := contract.poisonSet()
	var usable []map[string]any
	for _, o := range observations {
		if !poison[fstring(o["point_id"])] {
			usable = append(usable, o)
		}
	}
	return EvaluateObservations(usable, contract, nil)
}

// -------------------------------------------------------------------- main

// defaultScratchRoot returns the scratch volume for the hot loop: the
// FIT_CALIBRATE_TMP override, then a real /dev/shm tmpfs, then the platform
// temp directory.
func defaultScratchRoot() string {
	if override := os.Getenv("FIT_CALIBRATE_TMP"); override != "" {
		return override
	}
	if st, err := os.Stat("/dev/shm"); err == nil && st.IsDir() {
		return "/dev/shm"
	}
	return os.TempDir()
}

// RunCalibrate runs the full calibration production line.
func (c *Config) RunCalibrate() (*Result, error) {
	if err := os.MkdirAll(c.OutDir, 0o755); err != nil {
		return nil, err
	}
	env := fidelity.RuntimeEnv(c.RuntimeDir)

	if c.ReplayExisting != "" {
		manifest := c.ReplayManifest
		if manifest == "" {
			manifest = filepath.Join(c.OutDir, "state-artifact-manifest.txt")
		}
		report, err := c.replayExisting(c.ReplayExisting, manifest)
		if err != nil {
			return nil, err
		}
		if err := writeJSONIndent(filepath.Join(c.OutDir, "replay-report.json"), report); err != nil {
			return nil, err
		}
		c.log(fmt.Sprintf("replay overall=%s failures=%v", fstring(report["overall_status"]), report["open_failures"]))
		return &Result{Mode: "replay", Report: report}, nil
	}

	contract, err := LoadContract(c.ContractPath)
	if err != nil {
		return nil, err
	}
	if contract.EvaluatorContractSHA256() != fidelity.FrozenContractDigest {
		return nil, calibrationErrf("contract pins a different eval-v1 digest than the live closure")
	}

	work := c.Workdir
	if c.OnDisk {
		if work == "" {
			work = filepath.Join(c.OutDir, "work")
		}
	} else {
		if work == "" {
			work = filepath.Join(defaultScratchRoot(), "cal-"+c.ModelID)
		}
	}
	if err := os.MkdirAll(work, 0o755); err != nil {
		return nil, err
	}
	c.Workdir = work
	if c.LogDir == "" {
		c.LogDir = filepath.Join(work, "logs")
	}
	if err := os.MkdirAll(c.LogDir, 0o755); err != nil {
		return nil, err
	}

	scratchRefs := filepath.Join(work, "references")
	publishedRefs := filepath.Join(c.OutDir, "references")

	// reuse verified references from a previous run when published
	if st, err := os.Stat(publishedRefs); err == nil && st.IsDir() {
		entries, _ := os.ReadDir(publishedRefs)
		for _, e := range entries {
			ref := filepath.Join(publishedRefs, e.Name())
			if refOK(ref) {
				publishTree(publishedRefs, scratchRefs, e.Name())
			}
		}
	}

	sourceSHA, err := sha256File(c.Source)
	if err != nil {
		return nil, err
	}
	c.log(fmt.Sprintf("source %s sha=%s…", filepath.Base(c.Source), sourceSHA[:16]))

	imx, err := c.stageGenerateImatrix(env)
	if err != nil {
		return nil, err
	}
	missing, err := c.stageImatrixCoverage(imx)
	if err != nil {
		return nil, err
	}
	domains, err := c.stageReferences(env, scratchRefs)
	if err != nil {
		return nil, err
	}
	observations, err := c.stageLadder(contract, env, imx, scratchRefs, missing)
	if err != nil {
		return nil, err
	}
	newObs, unresolved := c.stageGapProbes(contract, env, imx, scratchRefs, observations)
	evaluation := c.stageFloors(append(append([]map[string]any{}, observations...), newObs...), contract, nil)
	if len(unresolved) > 0 {
		tiersRaw, _ := evaluation["tiers"].(map[string]any)
		for _, tier := range unresolved {
			if tr, ok := tiersRaw[tier].(map[string]any); ok {
				if tr["failure"] == nil {
					tr["failure"] = "INSUFFICIENT_WINDOW"
				}
				tr["validation_status"] = "candidate"
			}
		}
		openFailures := stringSet(fstringList(evaluation["open_failures"]))
		openFailures["INSUFFICIENT_WINDOW"] = true
		failures := make([]string, 0, len(openFailures))
		for f := range openFailures {
			failures = append(failures, f)
		}
		sort.Strings(failures)
		evaluation["open_failures"] = failures
		evaluation["overall_status"] = "candidate"
	}

	// publish scratch -> bundle (bulk copies, safe)
	publishTree(scratchRefs, publishedRefs, "bf16-*.kld")
	if filepath.Clean(c.LogDir) != filepath.Clean(filepath.Join(c.OutDir, "logs")) {
		publishTree(c.LogDir, filepath.Join(c.OutDir, "logs"), "*.log")
	}
	imxCopy := filepath.Join(c.OutDir, "calibration-imatrix.gguf")
	if !fileExists(imxCopy) {
		copyFile(imx, imxCopy)
	}

	allObs := append(append([]map[string]any{}, observations...), newObs...)
	bundle, err := c.stageEmit(contract, evaluation, allObs, scratchRefs, domains, missing, unresolved)
	if err != nil {
		return nil, err
	}
	if filepath.Clean(work) != filepath.Clean(c.OutDir) {
		_ = os.RemoveAll(work)
	}
	return &Result{Mode: "full", Bundle: bundle, Report: evaluation, SourceSHA256: sourceSHA}, nil
}

func fstringList(v any) []string {
	raw, ok := v.([]any)
	if !ok {
		if ss, ok := v.([]string); ok {
			return ss
		}
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func stringSet(xs []string) map[string]bool {
	m := map[string]bool{}
	for _, x := range xs {
		m[x] = true
	}
	return m
}

// publishTree bulk-copies srcDir/pattern into dstDir; only missing or
// size-differing files are copied (resumed runs do not re-copy).
func publishTree(srcDir, dstDir, pattern string) []string {
	matches, err := filepath.Glob(filepath.Join(srcDir, pattern))
	if err != nil {
		return nil
	}
	if len(matches) == 0 {
		return nil
	}
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return nil
	}
	var published []string
	for _, src := range matches {
		if st, err := os.Stat(src); err != nil || st.IsDir() {
			continue
		}
		dst := filepath.Join(dstDir, filepath.Base(src))
		if dstSt, err := os.Stat(dst); err == nil && !dstSt.IsDir() && dstSt.Size() == fileSize(src) {
			continue
		}
		if copyFile(src, dst) == nil {
			published = append(published, filepath.Base(src))
		}
	}
	return published
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}
