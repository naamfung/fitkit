// Package calibration implements the fidelity-calibration-v1 contract and the
// fit calibrate production line, mirroring upstream fit_gguf.calibration /
// fit_gguf.calibrate (v0.3.3). The contract JSON is authoritative — this
// package bends to it, never the reverse.
package calibration

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

//go:embed contracts/fidelity-calibration-v1.json
var contractJSON []byte

// ContractID is the frozen contract identifier.
const ContractID = "fidelity-calibration-v1"

// TIERS is the fixed tier set (mirrors the contract's tier_kl_anchors keys).
var TIERS = []string{"quality", "balanced", "compact", "mini"}

// CalibrationError is raised on contract violations or hard calibration
// failures.
type CalibrationError struct{ Msg string }

func (e *CalibrationError) Error() string { return e.Msg }

func calibrationErrf(format string, a ...any) *CalibrationError {
	return &CalibrationError{Msg: fmt.Sprintf(format, a...)}
}

// Contract is a loaded and self-verified calibration contract.
type Contract struct {
	Payload map[string]any
	SHA256  string
}

func fstring(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func ffloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case uint64:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	}
	return 0
}

func fint(v any) int {
	switch t := v.(type) {
	case int:
		return t
	case int64:
		return int(t)
	case uint64:
		return int(t)
	case float64:
		return int(t)
	case json.Number:
		n, _ := t.Int64()
		return int(n)
	}
	return 0
}

// canonicalJSONBytes serializes v as sorted-key compact JSON without HTML
// escaping — the canonical form the contract digests are taken over.
func canonicalJSONBytes(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// LoadContract loads the contract from path (default: the embedded frozen
// JSON), verifies its self-digest over the canonical payload minus
// contract_sha256, and returns it with the verified digest.
func LoadContract(path string) (*Contract, error) {
	raw := contractJSON
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, calibrationErrf("cannot read calibration contract %s: %v", path, err)
		}
		raw = b
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, calibrationErrf("cannot read calibration contract: %v", err)
	}
	if payload["contract_id"] != ContractID {
		return nil, calibrationErrf("unexpected contract_id")
	}
	digest := fstring(payload["contract_sha256"])
	body := make(map[string]any, len(payload))
	for k, v := range payload {
		if k != "contract_sha256" {
			body[k] = v
		}
	}
	canon, err := canonicalJSONBytes(body)
	if err != nil {
		return nil, err
	}
	if actual := sha256Hex(canon); digest != actual {
		return nil, calibrationErrf("contract fails its own contract_sha256: %s != %s", digest, actual)
	}
	return &Contract{Payload: payload, SHA256: digest}, nil
}

// EvaluatorContractSHA256 returns the eval-v1 digest the contract pins.
func (c *Contract) EvaluatorContractSHA256() string {
	return fstring(c.Payload["evaluator_contract_sha256"])
}

// TierKLAnchors returns the per-tier KL anchors from the contract.
func (c *Contract) TierKLAnchors() map[string]float64 {
	raw, _ := c.Payload["tier_kl_anchors"].(map[string]any)
	anchors := map[string]float64{}
	for k, v := range raw {
		anchors[k] = ffloat(v)
	}
	return anchors
}

// LadderStandardPresets returns the standard ladder preset list.
func (c *Contract) LadderStandardPresets() []string {
	var out []string
	for _, p := range listOfStrings(c.Payload["ladder_standard_presets"]) {
		out = append(out, p)
	}
	return out
}

// PoisonPresets returns the poison preset names (never window evidence).
func (c *Contract) PoisonPresets() []string {
	var out []string
	for _, p := range listOfStrings(c.Payload["poison_presets"]) {
		out = append(out, p)
	}
	return out
}

// ValidationMinSamples is the n >= 3 sample minimum per tier window.
func (c *Contract) ValidationMinSamples() int { return fint(c.Payload["validation_min_samples"]) }

// ProbeBudgetPerTier is the contract's gap-probe budget per tier (max 4).
func (c *Contract) ProbeBudgetPerTier() int { return fint(c.Payload["probe_budget_per_tier"]) }

func listOfStrings(v any) []string {
	raw, ok := v.([]any)
	if !ok {
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

// poisonSet returns the poison presets as a set for O(1) membership.
func (c *Contract) poisonSet() map[string]bool {
	set := map[string]bool{}
	for _, p := range c.PoisonPresets() {
		set[p] = true
	}
	return set
}

// IsPoisonPreset reports whether a point id is a poison preset.
func (c *Contract) IsPoisonPreset(pointID string) bool {
	return c.poisonSet()[pointID]
}

// Trunc4 applies Decimal ROUND_DOWN at 4 decimals to a float, using its
// shortest round-trip decimal representation (never binary-float floor).
func Trunc4(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if i := strings.IndexByte(s, '.'); i >= 0 && len(s) > i+5 {
		s = s[:i+5]
	}
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

// P05 is the linear-interpolated 5th percentile (default method).
func P05(values []float64) (float64, error) {
	if len(values) == 0 {
		return 0, calibrationErrf("p05 of empty sample set")
	}
	xs := append([]float64(nil), values...)
	sort.Float64s(xs)
	if len(xs) == 1 {
		return xs[0], nil
	}
	k := float64(len(xs)-1) * 0.05
	f := int(k)
	c := f + 1
	if c > len(xs)-1 {
		c = len(xs) - 1
	}
	return xs[f] + (xs[c]-xs[f])*(k-float64(f)), nil
}

// WindowBounds returns W(K) = [0.85K, 1.15K], both ends closed.
func WindowBounds(anchor float64, contract *Contract) (float64, float64) {
	rule, _ := contract.Payload["window_rule"].(map[string]any)
	loFactor := 0.85
	hiFactor := 1.15
	if rule != nil {
		if v := ffloat(rule["low_factor"]); v > 0 {
			loFactor = v
		}
		if v := ffloat(rule["high_factor"]); v > 0 {
			hiFactor = v
		}
	}
	return anchor * loFactor, anchor * hiFactor
}

// UniqueObservations dedups observations by artifact_sha256 — repeated evals of
// the same artifact count once; distinct artifacts at the same size are
// distinct observations. Order preserved (first occurrence wins).
func UniqueObservations(observations []map[string]any) ([]map[string]any, error) {
	seen := map[string]bool{}
	out := make([]map[string]any, 0, len(observations))
	for _, obs := range observations {
		sha := fstring(obs["artifact_sha256"])
		if sha == "" {
			return nil, calibrationErrf("observation %v missing artifact_sha256", obs["point_id"])
		}
		if seen[sha] {
			continue
		}
		seen[sha] = true
		out = append(out, obs)
	}
	return out, nil
}

// WindowSamples filters observations inside the tier window [lo, hi].
func WindowSamples(observations []map[string]any, anchor float64, contract *Contract) []map[string]any {
	lo, hi := WindowBounds(anchor, contract)
	var out []map[string]any
	for _, o := range observations {
		kl := ffloat(o["macro_kl"])
		if lo <= kl && kl <= hi {
			out = append(out, o)
		}
	}
	return out
}

// DeriveTier derives the floor, witness, and per-tier validation status for
// one tier (contract §5.4–5.6). Returns a tier-result mapping mirroring
// upstream derive_tier().
func DeriveTier(tier string, anchor float64, observations []map[string]any, contract *Contract, openFailures []string) map[string]any {
	minN := contract.ValidationMinSamples()
	lo, hi := WindowBounds(anchor, contract)
	samples := WindowSamples(observations, anchor, contract)
	n := len(samples)

	result := map[string]any{
		"tier":              tier,
		"anchor":            anchor,
		"window":            [2]float64{lo, hi},
		"sample_count":      n,
		"samples":           []map[string]any{},
		"floor":             nil,
		"floor_method":      nil,
		"witness":           nil,
		"validation_status": "candidate",
		"failure":           nil,
	}
	sorted := append([]map[string]any(nil), samples...)
	sort.SliceStable(sorted, func(a, b int) bool { return ffloat(sorted[a]["macro_kl"]) < ffloat(sorted[b]["macro_kl"]) })
	sampleList := make([]map[string]any, 0, len(sorted))
	for _, s := range sorted {
		sampleList = append(sampleList, map[string]any{
			"point_id":       fstring(s["point_id"]),
			"size_bytes":     fint(s["size_bytes"]),
			"macro_kl":       ffloat(s["macro_kl"]),
			"macro_same_top": ffloat(s["same_top"]),
			"artifact_sha256": fstring(s["artifact_sha256"]),
		})
	}
	result["samples"] = sampleList

	if n == 0 {
		result["failure"] = "INSUFFICIENT_WINDOW"
		return result
	}
	tops := make([]float64, 0, n)
	for _, s := range samples {
		tops = append(tops, ffloat(s["same_top"]))
	}
	var floor float64
	var method string
	if n >= minN {
		p5, err := P05(tops)
		if err != nil {
			result["failure"] = "INSUFFICIENT_WINDOW"
			return result
		}
		floor = Trunc4(p5)
		method = "empirical_p5"
	} else {
		m := tops[0]
		for _, t := range tops[1:] {
			if t < m {
				m = t
			}
		}
		floor = Trunc4(m)
		method = "min_fallback"
	}
	result["floor"] = floor
	result["floor_method"] = method

	// witness: at least one window observation with macro_kl <= anchor AND
	// same_top >= derived floor, the lowest-KL such observation.
	var witness map[string]any
	witnessKL := math.Inf(1)
	for _, s := range samples {
		if ffloat(s["macro_kl"]) <= anchor && ffloat(s["same_top"]) >= floor {
			if ffloat(s["macro_kl"]) < witnessKL {
				witnessKL = ffloat(s["macro_kl"])
				witness = map[string]any{
					"point_id":       fstring(s["point_id"]),
					"macro_kl":       ffloat(s["macro_kl"]),
					"macro_same_top": ffloat(s["same_top"]),
					"pass":           true,
				}
			}
		}
	}
	if witness == nil {
		result["failure"] = "WITNESS_MISSING"
	} else {
		result["witness"] = witness
	}
	if n < minN && result["failure"] == nil {
		result["failure"] = "INSUFFICIENT_WINDOW"
	}
	if len(openFailures) > 0 {
		if result["failure"] == nil {
			result["failure"] = openFailures[0]
		}
	}
	if n >= minN && result["witness"] != nil && len(openFailures) == 0 {
		result["validation_status"] = "validated"
	}
	return result
}

// EvaluateObservations runs the full contract evaluation: dedup -> per-tier
// derivation -> overall status. extraFailures carries session-level open
// failures (e.g. imatrix coverage) that demote every tier.
func EvaluateObservations(observations []map[string]any, contract *Contract, extraFailures []string) (map[string]any, error) {
	unique, err := UniqueObservations(observations)
	if err != nil {
		return nil, err
	}
	tiers := map[string]any{}
	anchors := contract.TierKLAnchors()
	for _, tier := range TIERS {
		tiers[tier] = DeriveTier(tier, anchors[tier], unique, contract, extraFailures)
	}
	if len(extraFailures) > 0 {
		for _, tier := range tiers {
			m, _ := tier.(map[string]any)
			if m != nil {
				if m["failure"] == nil {
					m["failure"] = extraFailures[0]
				}
				m["validation_status"] = "candidate"
			}
		}
	}
	allOK := true
	for _, tier := range TIERS {
		m, _ := tiers[tier].(map[string]any)
		if m == nil || m["validation_status"] != "validated" {
			allOK = false
			break
		}
	}
	failures := map[string]bool{}
	for _, tier := range TIERS {
		if f, ok := tiers[tier].(map[string]any); ok {
			if s := fstring(f["failure"]); s != "" {
				failures[s] = true
			}
		}
	}
	for _, f := range extraFailures {
		failures[f] = true
	}
	failureList := make([]string, 0, len(failures))
	for f := range failures {
		failureList = append(failureList, f)
	}
	sort.Strings(failureList)
	status := "candidate"
	if allOK {
		status = "validated"
	}
	return map[string]any{
		"contract_id":                ContractID,
		"calibration_contract_sha256": contract.SHA256,
		"observation_count":          len(unique),
		"tiers":                      tiers,
		"overall_status":             status,
		"open_failures":              failureList,
	}, nil
}
