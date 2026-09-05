package fidelity

import (
	"encoding/json"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const resultSchema = "fit.eval_result.v1"

var (
	reMeanKLD  = regexp.MustCompile(`Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)`)
	reSameTop  = regexp.MustCompile(`Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)\s+%`)
	reRMSDelta = regexp.MustCompile(`RMS Δp\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*%`)
	klHeader   = "====== KL divergence statistics ======"
)

// ParseLLaMAKLLog extracts contract metrics from one llama-perplexity KL log.
func ParseLLaMAKLLog(text string) (map[string]any, error) {
	if !strings.Contains(text, klHeader) {
		return nil, &EvalLogError{Msg: "not a llama-perplexity KL log (header missing)"}
	}
	mean := reMeanKLD.FindStringSubmatch(text)
	top := reSameTop.FindStringSubmatch(text)
	if mean == nil || top == nil {
		return nil, &EvalLogError{Msg: "KL log missing 'Mean KLD' or 'Same top p' line"}
	}
	rms := reRMSDelta.FindStringSubmatch(text)
	res := map[string]any{
		"mean_kld":            atof(mean[1]),
		"mean_kld_stderr":     atof(mean[2]),
		"same_top_pct":        atof(top[1]),
		"same_top_stderr_pct": atof(top[2]),
		"rms_delta_p_pct":     nil,
	}
	if rms != nil {
		res["rms_delta_p_pct"] = atof(rms[1])
	}
	return res, nil
}

type EvalLogError struct{ Msg string }

func (e *EvalLogError) Error() string { return e.Msg }

type DomainResult struct {
	Domain            string
	Tokens            int
	KL                float64
	SameTop           float64
	PPL               float64
	HasPPL            bool
	RMS               float64
	HasRMS            bool
	ExcludedPositions int
}

func round9(v float64) any {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	return math.Round(v*1e9) / 1e9
}

// BuildEvalResult emits the unified eval-v1 output document (deterministic).
func BuildEvalResult(contractDigest, modelHash, refLogitsHash string, domains []DomainResult, macroKL, macroSameTop float64, haveMacro bool) (map[string]any, error) {
	byDomain := map[string]DomainResult{}
	for _, d := range domains {
		byDomain[d.Domain] = d
	}
	if len(byDomain) != len(Domains) {
		return nil, &evalResultError{"result must cover exactly the frozen domains"}
	}
	weight := 0.2
	if !haveMacro {
		mk := 0.0
		ms := 0.0
		for _, d := range domains {
			mk += weight * d.KL
			ms += weight * d.SameTop
		}
		macroKL, macroSameTop = mk, ms
	}
	worst := domains[0]
	for _, d := range domains[1:] {
		if d.KL > worst.KL {
			worst = d
		}
	}
	totalTokens := 0
	for _, d := range domains {
		totalTokens += d.Tokens
	}
	sort.Slice(domains, func(i, j int) bool { return domains[i].Domain < domains[j].Domain })
	domJSON := map[string]any{}
	for _, d := range domains {
		entry := map[string]any{
			"tokens": d.Tokens, "kl": round9(d.KL), "same_top": round9(d.SameTop),
			"ppl": nil, "rms": nil, "excluded_positions": d.ExcludedPositions,
		}
		if d.HasPPL {
			entry["ppl"] = round9(d.PPL)
		}
		if d.HasRMS {
			entry["rms"] = round9(d.RMS)
		}
		domJSON[d.Domain] = entry
	}
	return map[string]any{
		"result_schema":           resultSchema,
		"evaluator_contract":      ContractID,
		"evaluator_contract_hash": contractDigest,
		"model_hash":              modelHash,
		"reference_logits_hash":   refLogitsHash,
		"reference_manifest_hash": nil,
		"candidate_plan_hash":     nil,
		"total_tokens":            totalTokens,
		"macro_kl":                round9(macroKL),
		"macro_same_top":          round9(macroSameTop),
		"worst_domain_kl":         round9(worst.KL),
		"worst_domain":            worst.Domain,
		"comparison":              nil,
		"domains":                 domJSON,
	}, nil
}

type evalResultError struct{ msg string }

func (e *evalResultError) Error() string { return e.msg }

// SerializeResult deterministically serializes a result map.
func SerializeResult(result map[string]any) string {
	data, _ := json.Marshal(result)
	return string(data)
}

func atof(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return math.NaN()
	}
	return f
}
