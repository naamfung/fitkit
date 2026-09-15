package pipeline

import (
	"fmt"
	"sort"

	gguf "fitting/gguf"
)

// UpgradeCandidate is one positive-size lower→upper tensor transition.
type UpgradeCandidate struct {
	Tensor         string  `json:"tensor"`
	FromQtype      string  `json:"from_qtype"`
	ToQtype        string  `json:"to_qtype"`
	DeltaBytes     int     `json:"delta_bytes"`
	Importance     float64 `json:"importance"`
	RawImportance  float64 `json:"raw_importance"`
	ExpectedGain   float64 `json:"expected_gain"`
	UtilityPerByte float64 `json:"utility_per_byte"`
	Profiled       bool    `json:"profiled"`
	Block          int     `json:"block"`
	Role           string  `json:"role"`
	// Step is the 1-based position of this ladder step within the tensor's
	// multi-step transition (1 = shallowest). TotalSteps is the number of steps
	// the tensor's ladder has. Binary (non-ladder) candidates carry 0/0.
	Step       int `json:"step,omitempty"`
	TotalSteps int `json:"total_steps,omitempty"`
}

// RejectedTransition is a rejected candidate and its reason.
type RejectedTransition struct {
	Tensor     string `json:"tensor"`
	FromQtype  string `json:"from_qtype"`
	ToQtype    string `json:"to_qtype"`
	DeltaBytes int    `json:"delta_bytes"`
	Reason     string `json:"reason"`
}

// CandidateSet bundles candidates + bounds.
type CandidateSet struct {
	Candidates     []UpgradeCandidate
	Rejected       []RejectedTransition
	LowerSizeBytes int
	UpperSizeBytes int
	// TensorCount is the number of distinct tensors with at least one candidate.
	// SkippedCount (plan) is defined against this, not len(Candidates), because
	// ladder candidates repeat tensor names across steps.
	TensorCount int
	// Direction is the optimization direction behind the candidates: "up"
	// (lower→upper, selected tensors are upgraded) or "down" (upper→lower,
	// selected tensors are downgraded to shrink toward the target).
	Direction string
}

func (c *CandidateSet) CandidateBudgetBytes() int {
	s := 0
	for _, c := range c.Candidates {
		s += c.DeltaBytes
	}
	return s
}

func qtypeBPW(qtype string) (float64, error) {
	traits, ok := gguf.GGMLTypeTraits[qtype]
	if !ok {
		return 0, fmt.Errorf("pipeline: unsupported qtype in candidate transition: %s", qtype)
	}
	return float64(traits[1]) * 8.0 / float64(traits[0]), nil
}

// GenerateUpgradeCandidates builds the per-tensor transitions between the lower
// and upper presets in the requested direction.
//
//   - mode "up": transitions are lower→upper (upgrades that add bytes),
//     utility = importance×bits-gained / bytes; the optimizer selects tensors to
//     keep the most-valuable upgrades within a target starting from the lower.
//   - mode "down": transitions are upper→lower (downgrades that save bytes),
//     utility = importance×bits-lost / bytes-saved; the optimizer selects tensors
//     to sacrifice the least-valuable few, starting from the upper (high quality)
//     and shrinking to the target.
//
// DeltaBytes is always a positive magnitude.
func GenerateUpgradeCandidates(
	lowerRecipe *Recipe,
	upperRecipe *Recipe,
	lowerSize *gguf.Prediction,
	upperSize *gguf.Prediction,
	profile *ImatrixProfile,
	mode string,
) (*CandidateSet, error) {
	lowerAssn := lowerRecipe.TensorMap()
	upperAssn := upperRecipe.TensorMap()
	if len(lowerAssn) != len(upperAssn) {
		return nil, fmt.Errorf("pipeline: lower and upper recipe tensor sets differ")
	}
	for n := range lowerAssn {
		if _, ok := upperAssn[n]; !ok {
			return nil, fmt.Errorf("pipeline: tensor %s only in lower recipe", n)
		}
	}
	lowerSizes := map[string]int{}
	upperSizes := map[string]int{}
	for _, t := range lowerSize.Tensors {
		lowerSizes[t.Name] = t.PaddedBytes
	}
	for _, t := range upperSize.Tensors {
		upperSizes[t.Name] = t.PaddedBytes
	}
	for n := range lowerAssn {
		if _, ok := lowerSizes[n]; !ok {
			return nil, fmt.Errorf("pipeline: tensor %s missing lower size", n)
		}
		if _, ok := upperSizes[n]; !ok {
			return nil, fmt.Errorf("pipeline: tensor %s missing upper size", n)
		}
	}

	if mode != "down" {
		mode = "up"
	}
	down := mode == "down"

	rejectReason := "not_a_strict_encoded_precision_promotion"
	qtypeBPWof := func(q string) (float64, error) { return qtypeBPW(q) }
	if down {
		rejectReason = "not_a_strict_encoded_precision_reduction"
	}

	profiles := profile.EntryMap()
	var candidates []UpgradeCandidate
	var rejected []RejectedTransition

	for i := range lowerRecipe.Tensors {
		tensor := lowerRecipe.Tensors[i]
		upper := upperAssn[tensor.Name]
		lowerQ := lowerType(tensor.DstType)
		upperQ := lowerType(upper.DstType)
		if lowerQ == upperQ {
			continue
		}
		var fromQ, toQ string
		if down {
			fromQ, toQ = upperQ, lowerQ
		} else {
			fromQ, toQ = lowerQ, upperQ
		}
		sizeGap := upperSizes[tensor.Name] - lowerSizes[tensor.Name]

		toBPW, err := qtypeBPWof(toQ)
		if err != nil {
			return nil, err
		}
		fromBPW, err := qtypeBPWof(fromQ)
		if err != nil {
			return nil, err
		}
		// bits carried by the transition, always reported as a positive magnitude
		// (gained in "up", lost in "down").
		bpwMag := toBPW - fromBPW
		if toBPW < fromBPW {
			bpwMag = fromBPW - toBPW
		}
		if sizeGap <= 0 || bpwMag <= 0 {
			rejected = append(rejected, RejectedTransition{
				Tensor: tensor.Name, FromQtype: fromQ, ToQtype: toQ,
				DeltaBytes: sizeGap, Reason: rejectReason,
			})
			continue
		}

		tp := profiles[tensor.Name]
		profiled := tp != nil
		var importance, rawImportance float64
		if tp != nil {
			importance = tp.RoleRelativeMean
			rawImportance = tp.Mean
		}
		// expectedGain is the quality magnitude (importance × bits moved). In
		// "down" this is the harm removed; utility = harm per byte saved.
		expectedGain := importance * bpwMag
		block := -1
		role := "unprofiled"
		if tp != nil {
			block = tp.Block
			role = tp.Role
		}
		candidates = append(candidates, UpgradeCandidate{
			Tensor: tensor.Name, FromQtype: fromQ, ToQtype: toQ,
			DeltaBytes: sizeGap, Importance: importance, RawImportance: rawImportance,
			ExpectedGain: expectedGain, UtilityPerByte: expectedGain / float64(sizeGap),
			Profiled: profiled, Block: block, Role: role,
		})
	}

	sort.SliceStable(candidates, func(a, b int) bool { return candidates[a].Tensor < candidates[b].Tensor })
	sort.SliceStable(rejected, func(a, b int) bool { return rejected[a].Tensor < rejected[b].Tensor })

	return &CandidateSet{
		Candidates: candidates, Rejected: rejected,
		LowerSizeBytes: lowerSize.TotalBytes, UpperSizeBytes: upperSize.TotalBytes,
		TensorCount: len(candidates), Direction: mode,
	}, nil
}

// LadderQtypes is the curated bpw-descending ladder of intermediate qtypes used
// for multi-step (graduated) transitions between the lower and upper presets.
// Same-bpw neighbours (e.g. Q4_K vs IQ4_NL) collapse at generation time because
// they differ by zero bytes per tensor.
var LadderQtypes = []string{
	"f32", "f16", "bf16",
	"q8_0", "q6_k", "q5_k", "q4_k", "iq4_nl", "iq4_xs",
	"q3_k", "iq3_m", "iq3_s", "iq3_xs", "iq3_xxs",
	"q2_k", "iq2_m", "iq2_s", "iq2_xs", "iq2_xxs",
	"iq1_m", "iq1_s",
}

// ladderSegment returns the bpw-descending qtype levels between upperQ and
// lowerQ (upper first, lower last). Types absent from the curated ladder fall
// back to a single direct transition [upperQ, lowerQ]; equal types return nil.
func ladderSegment(ladder []string, upperQ, lowerQ string) []string {
	if upperQ == lowerQ {
		return nil
	}
	idxU, idxL := -1, -1
	for i, q := range ladder {
		if q == upperQ {
			idxU = i
		}
		if q == lowerQ {
			idxL = i
		}
	}
	if idxU >= 0 && idxL >= 0 && idxU < idxL {
		return ladder[idxU : idxL+1]
	}
	return []string{upperQ, lowerQ}
}

// GenerateLadderCandidates builds the per-tensor multi-step transitions between
// the lower and upper presets along the curated ladder. Each adjacent level
// pair within a tensor's segment becomes one Step candidate (see
// UpgradeCandidate.Step). The optimizer selects a prefix of each tensor's steps
// so a tensor can be downgraded/upgraded only part of the way toward the lower
// preset. Binary (-no-ladder) generation stays in GenerateUpgradeCandidates.
func GenerateLadderCandidates(
	lowerRecipe *Recipe,
	upperRecipe *Recipe,
	lowerSize *gguf.Prediction,
	upperSize *gguf.Prediction,
	layout *gguf.Layout,
	profile *ImatrixProfile,
	mode string,
) (*CandidateSet, error) {
	lowerAssn := lowerRecipe.TensorMap()
	upperAssn := upperRecipe.TensorMap()
	if len(lowerAssn) != len(upperAssn) {
		return nil, fmt.Errorf("pipeline: lower and upper recipe tensor sets differ")
	}
	for n := range lowerAssn {
		if _, ok := upperAssn[n]; !ok {
			return nil, fmt.Errorf("pipeline: tensor %s only in lower recipe", n)
		}
	}

	if mode != "down" {
		mode = "up"
	}
	down := mode == "down"
	rejectReason := "not_a_strict_encoded_precision_promotion"
	if down {
		rejectReason = "not_a_strict_encoded_precision_reduction"
	}

	profiles := profile.EntryMap()
	var candidates []UpgradeCandidate
	var rejected []RejectedTransition
	tensorsWithSteps := 0

	for i := range lowerRecipe.Tensors {
		tensor := lowerRecipe.Tensors[i]
		upper := upperAssn[tensor.Name]
		lowerQ := lowerType(tensor.DstType)
		upperQ := lowerType(upper.DstType)
		if lowerQ == upperQ {
			continue
		}

		seg := ladderSegment(LadderQtypes, upperQ, lowerQ)
		if len(seg) < 2 {
			continue
		}

		// Per-level padded sizes; skip levels the tensor cannot be encoded as
		// (unknown qtype or ne0 not divisible by the block size).
		type level struct {
			q    string
			size int
		}
		var levels []level
		for _, q := range seg {
			_, padded, ok := gguf.TensorPaddedSize(layout, tensor.Name, q)
			if !ok {
				continue
			}
			levels = append(levels, level{q: q, size: padded})
		}
		if len(levels) < 2 {
			rejected = append(rejected, RejectedTransition{
				Tensor: tensor.Name, FromQtype: upperQ, ToQtype: lowerQ,
				DeltaBytes: 0, Reason: rejectReason,
			})
			continue
		}

		tp := profiles[tensor.Name]
		profiled := tp != nil
		var importance, rawImportance float64
		if tp != nil {
			importance = tp.RoleRelativeMean
			rawImportance = tp.Mean
		}
		block := -1
		role := "unprofiled"
		if tp != nil {
			block = tp.Block
			role = tp.Role
		}

		var tensorSteps []UpgradeCandidate
		for k := 1; k < len(levels); k++ {
			hi, lo := levels[k-1], levels[k]
			delta := hi.size - lo.size
			if delta <= 0 {
				continue
			}
			hiBPW, err := qtypeBPW(hi.q)
			if err != nil {
				return nil, err
			}
			loBPW, err := qtypeBPW(lo.q)
			if err != nil {
				return nil, err
			}
			bits := hiBPW - loBPW
			if bits <= 0 {
				continue
			}
			fromQ, toQ := hi.q, lo.q
			if !down {
				fromQ, toQ = lo.q, hi.q
			}
			expectedGain := importance * bits
			tensorSteps = append(tensorSteps, UpgradeCandidate{
				Tensor: tensor.Name, FromQtype: fromQ, ToQtype: toQ,
				DeltaBytes: delta, Importance: importance, RawImportance: rawImportance,
				ExpectedGain: expectedGain, UtilityPerByte: expectedGain / float64(delta),
				Profiled: profiled, Block: block, Role: role,
			})
		}
		if len(tensorSteps) == 0 {
			rejected = append(rejected, RejectedTransition{
				Tensor: tensor.Name, FromQtype: upperQ, ToQtype: lowerQ,
				DeltaBytes: 0, Reason: rejectReason,
			})
			continue
		}
		tensorsWithSteps++
		for j := range tensorSteps {
			tensorSteps[j].Step = j + 1
			tensorSteps[j].TotalSteps = len(tensorSteps)
		}
		candidates = append(candidates, tensorSteps...)
	}

	sort.SliceStable(candidates, func(a, b int) bool { return candidates[a].Tensor < candidates[b].Tensor })
	sort.SliceStable(rejected, func(a, b int) bool { return rejected[a].Tensor < rejected[b].Tensor })

	return &CandidateSet{
		Candidates: candidates, Rejected: rejected,
		LowerSizeBytes: lowerSize.TotalBytes, UpperSizeBytes: upperSize.TotalBytes,
		TensorCount: tensorsWithSteps, Direction: mode,
	}, nil
}

func lowerType(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 32
		}
	}
	return string(b)
}
