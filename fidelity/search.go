package fidelity

import (
	"fmt"
	"sort"
)

// SearchPure state machine for Fidelity Search v1 (coarse→bracket→fine→verify).

const defaultToleranceBytes = 128 * 1024 * 1024

type TierContract struct {
	Tier     string
	KLAnchor float64
	// SameTopReference 是该模型校准的 same-top 参考值（来自 validated Guard
	// Profile，可选）。仅用于报告（"top-1 91.34% vs 校准 floor 93.16%"），
	// 永不改变判定。
	SameTopReference float64
}

// Passes 是 Fidelity Contract v2（v0.3）的 KL-only 硬门限：macroKL <=
// KLAnchor 即通过。same-top 仍被测量、记录并归档，但它只是参考，不参与判定。
func (c TierContract) Passes(macroKL, sameTop float64) bool {
	return macroKL <= c.KLAnchor
}

// Margins 返回 (M_kl, M_top_ref)。M_kl 是唯一约束轴；M_top_ref 仅在已知
// SameTopReference 时才有意义，无参考值时返回 -1 表示不可用。
func (c TierContract) Margins(macroKL, sameTop float64) (float64, float64) {
	mkl := (c.KLAnchor - macroKL) / c.KLAnchor
	if c.SameTopReference <= 0 {
		return mkl, -1
	}
	return mkl, (sameTop - c.SameTopReference) / (1.0 - c.SameTopReference)
}

// ActiveConstraint 自 v0.3 起恒为 "kl"：门限只有一条轴。
func (c TierContract) ActiveConstraint(macroKL, sameTop float64) string {
	return "kl"
}

type SearchPoint struct {
	SizeBytes int
	MacroKL   float64
	SameTop   float64
	Passed    bool
	EvalIndex int
	Source    string
}

type EvalOutcome struct {
	SizeBytes int
	MacroKL   float64
	SameTop   float64
	HasKL     bool // false => failed eval (None fields in Python)
	Note      string
}

type Seed struct {
	SizeBytes int
	MacroKL   float64
	SameTop   float64
}

type SearchResult struct {
	Status           string // verified_pass | budget_exhausted | no_pass | noise_inversion | insufficient_window
	Best             *SearchPoint
	Points           []SearchPoint
	FreshEvals       int
	Budget           int
	BracketFailBytes int // -1 means none
	BracketPassBytes int // -1 means none
	ActiveConstraint string
	ToleranceBytes   int
	Note             string
}

func outcomeToPoint(o EvalOutcome, idx int, source string, c TierContract) *SearchPoint {
	if !o.HasKL {
		return nil
	}
	return &SearchPoint{
		SizeBytes: o.SizeBytes, MacroKL: o.MacroKL, SameTop: o.SameTop,
		Passed: c.Passes(o.MacroKL, o.SameTop), EvalIndex: idx, Source: source,
	}
}

// Summary emits the search-result summary document (mirrors Python summary()).
func (r *SearchResult) Summary() map[string]any {
	var best map[string]any
	if r.Best != nil {
		best = map[string]any{
			"size_bytes": r.Best.SizeBytes,
			"macro_kl":   round9(r.Best.MacroKL),
			"same_top":   round9(r.Best.SameTop),
			"passed":     r.Best.Passed,
			"eval_index": r.Best.EvalIndex,
			"source":     r.Best.Source,
		}
	}
	guarantee := "budget spent before tolerance bracket was reached (observed PASS only)"
	switch r.Status {
	case "verified_pass":
		guarantee = "minimum verified PASS within tolerance bracket"
	case "noise_inversion":
		guarantee = "smallest verified PASS; bracket inverted (curve within eval noise)"
	case "no_pass":
		guarantee = "no evaluated point satisfied the contract"
	}
	points := make([]map[string]any, 0, len(r.Points))
	for _, p := range r.Points {
		points = append(points, map[string]any{
			"size_bytes": p.SizeBytes, "macro_kl": round9(p.MacroKL),
			"same_top": round9(p.SameTop), "passed": p.Passed,
			"eval_index": p.EvalIndex, "source": p.Source,
		})
	}
	return map[string]any{
		"status": r.Status, "guarantee": guarantee, "best": best,
		"fresh_evals": r.FreshEvals, "budget": r.Budget,
		"bracket_fail_bytes": r.bracketOrNil(false),
		"bracket_pass_bytes": r.bracketOrNil(true),
		"active_constraint":  strOrNil(r.ActiveConstraint),
		"tolerance_bytes":    r.ToleranceBytes, "note": strOrNil(r.Note),
		"points": points,
	}
}

func (r *SearchResult) bracketOrNil(pass bool) any {
	v := r.BracketPassBytes
	if !pass {
		v = r.BracketFailBytes
	}
	if v == -1 {
		return nil
	}
	return v
}

func strOrNil(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// Evaluator maps a target byte size to a (possibly failed) evaluation outcome.
type Evaluator func(sizeBytes int) EvalOutcome

// FidelitySearch runs the coarse→bracket→fine→verify search.
// minSize/maxSize bound the searchable range; budget limits fresh evals.
func FidelitySearch(
	contract TierContract,
	evaluate Evaluator,
	seeds []Seed,
	minSize int,
	maxSize int,
	budget int,
	toleranceBytes int,
	coarseStrideBytes int,
) (*SearchResult, error) {
	if minSize >= maxSize {
		return nil, fmt.Errorf("fidelity: min_size %d >= max_size %d", minSize, maxSize)
	}
	if budget < 1 {
		return nil, fmt.Errorf("fidelity: budget must be >= 1")
	}
	if toleranceBytes < 1 {
		return nil, fmt.Errorf("fidelity: tolerance_bytes must be >= 1")
	}
	if toleranceBytes == 0 {
		toleranceBytes = defaultToleranceBytes
	}

	points := make([]SearchPoint, 0, len(seeds)+budget)
	for _, s := range seeds {
		if p := outcomeToPoint(EvalOutcome{SizeBytes: s.SizeBytes, MacroKL: s.MacroKL, SameTop: s.SameTop, HasKL: true}, 0, "observed", contract); p != nil {
			points = append(points, *p)
		}
	}

	fresh := 0
	var submit func(sizeBytes int, source string) *SearchPoint
	submit = func(sizeBytes int, source string) *SearchPoint {
		if fresh >= budget {
			return nil
		}
		fresh++
		p := outcomeToPoint(evaluate(sizeBytes), fresh, source, contract)
		if p != nil {
			points = append(points, *p)
		}
		return p
	}

	bracket := func() (int, int) {
		loF, hiP := -1, -1
		for _, p := range points {
			if !p.Passed {
				if p.SizeBytes > loF {
					loF = p.SizeBytes
				}
			} else {
				if hiP == -1 || p.SizeBytes < hiP {
					hiP = p.SizeBytes
				}
			}
		}
		return loF, hiP
	}
	smallestPass := func() *SearchPoint {
		var best *SearchPoint
		for i := range points {
			if points[i].Passed && (best == nil || points[i].SizeBytes < best.SizeBytes) {
				best = &points[i]
			}
		}
		return best
	}
	finish := func(status string, loF, hiP int, note string) *SearchResult {
		best := smallestPass()
		active := ""
		if best != nil {
			active = contract.ActiveConstraint(best.MacroKL, best.SameTop)
		}
		return &SearchResult{
			Status: status, Best: best,
			Points:     sortPoints(points),
			FreshEvals: fresh, Budget: budget,
			BracketFailBytes: loF, BracketPassBytes: hiP,
			ActiveConstraint: active,
			ToleranceBytes:   toleranceBytes, Note: note,
		}
	}

	// phase 0
	loF, hiP := bracket()
	if hiP >= 0 && loF >= 0 && loF >= hiP {
		return finish("noise_inversion", loF, hiP, ""), nil
	}

	// phase 1 coarse
	stride := coarseStrideBytes
	if stride == 0 {
		stride = maxIntF(1024*1024, (maxSize-minSize)/16)
	}
	walkedToFloor := false
	if hiP == -1 {
		cursor := maxIntF(minSize, minSize)
		if loF >= 0 {
			cursor = maxIntF(minSize, loF+stride)
		}
		for fresh < budget {
			if loF >= 0 && cursor <= loF {
				cursor = loF + stride
			}
			if cursor > maxSize {
				break
			}
			point := submit(cursor, "coarse")
			if point != nil && point.Passed {
				break
			}
			cursor += stride
		}
		loF, hiP = bracket()
		if hiP == -1 {
			return finish("no_pass", loF, hiP, ""), nil
		}
		if loF >= 0 && loF >= hiP {
			return finish("noise_inversion", loF, hiP, ""), nil
		}
	}
	if loF == -1 {
		cursor := maxIntF(minSize, hiP-stride)
		for fresh < budget && cursor >= minSize {
			if hiP >= 0 && cursor >= hiP {
				cursor = hiP - stride
			}
			cursor = maxIntF(minSize, cursor)
			point := submit(cursor, "coarse")
			if point != nil && !point.Passed {
				break
			}
			cursor -= stride
			if cursor < minSize {
				walkedToFloor = true
				break
			}
		}
		loF, hiP = bracket()
	}

	// phase 2 bisect
	failedProbes := map[int]bool{}
	for fresh < budget {
		loF, hiP = bracket()
		if hiP == -1 || loF == -1 {
			break
		}
		if hiP-loF <= toleranceBytes {
			break
		}
		mid := (loF + hiP) / 2
		shift := maxIntF(1, toleranceBytes/8)
		for failedProbes[mid] && mid+shift < hiP {
			mid += shift
		}
		if failedProbes[mid] {
			break
		}
		point := submit(mid, "bracket")
		if point == nil {
			failedProbes[mid] = true
			continue
		}
		loF, hiP = bracket()
		if loF >= 0 && hiP >= 0 && loF >= hiP {
			return finish("noise_inversion", loF, hiP, ""), nil
		}
	}

	// phase 3 verify
	loF, hiP = bracket()
	if hiP == -1 {
		return finish("no_pass", loF, hiP, ""), nil
	}
	if loF == -1 {
		if walkedToFloor {
			return finish("verified_pass", loF, hiP,
				"no FAIL found within [min_size, hi_pass]; true crossing may lie below the searchable floor"), nil
		}
		return finish("budget_exhausted", loF, hiP, ""), nil
	}
	if hiP-loF <= toleranceBytes {
		return finish("verified_pass", loF, hiP, ""), nil
	}
	return finish("budget_exhausted", loF, hiP, ""), nil
}

func sortPoints(points []SearchPoint) []SearchPoint {
	out := append([]SearchPoint(nil), points...)
	sort.Slice(out, func(i, j int) bool { return out[i].SizeBytes < out[j].SizeBytes })
	return out
}

func maxIntF(a, b int) int {
	if a > b {
		return a
	}
	return b
}
