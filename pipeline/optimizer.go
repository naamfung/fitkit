package pipeline

import (
	"crypto/sha256"
	"fmt"
)

type OptimizationPlan struct {
	SchemaVersion      int
	TargetBytes        int
	LowerSizeBytes     int
	PredictedSizeBytes int
	UnusedBytes        int
	Selected           []UpgradeCandidate
	SkippedCount       int
	OracleIterations   int
	// RefineNote carries the refine_profile record written to plan.json when a
	// refine profile was applied; nil when none.
	RefineNote map[string]any
}

func (p *OptimizationPlan) SelectedCostBytes() int { return p.PredictedSizeBytes - p.LowerSizeBytes }

func (p *OptimizationPlan) Overrides() [][2]string {
	out := make([][2]string, 0, len(p.Selected))
	for _, c := range p.Selected {
		out = append(out, [2]string{c.Tensor, c.ToQtype})
	}
	return out
}

func validateCandidates(cands []UpgradeCandidate, lowerSize, target int) error {
	if target < lowerSize {
		return fmt.Errorf("pipeline: target %d below lower baseline %d", target, lowerSize)
	}
	for _, c := range cands {
		if c.DeltaBytes <= 0 {
			return fmt.Errorf("pipeline: all candidate costs must be positive")
		}
	}
	// Ladder steps of one tensor must form the contiguous prefix 1..TotalSteps
	// so the optimizer can enforce a prefix-constrained selection.
	steps := map[string]map[int]bool{}
	for _, c := range cands {
		if c.TotalSteps <= 1 {
			continue
		}
		if steps[c.Tensor] == nil {
			steps[c.Tensor] = map[int]bool{}
		}
		steps[c.Tensor][c.Step] = true
	}
	for tensor, set := range steps {
		for i := 1; i <= len(set); i++ {
			if !set[i] {
				return fmt.Errorf("pipeline: tensor %s ladder steps must be contiguous", tensor)
			}
		}
	}
	return nil
}

// greedy sort: (-utility, -expected_gain, delta_bytes, tensor, to_qtype)
func lessUtilexists(a, b UpgradeCandidate) bool {
	if a.UtilityPerByte != b.UtilityPerByte {
		return a.UtilityPerByte > b.UtilityPerByte
	}
	if a.ExpectedGain != b.ExpectedGain {
		return a.ExpectedGain > b.ExpectedGain
	}
	if a.DeltaBytes != b.DeltaBytes {
		return a.DeltaBytes < b.DeltaBytes
	}
	if a.Tensor != b.Tensor {
		return a.Tensor < b.Tensor
	}
	return a.ToQtype < b.ToQtype
}

// selectionOrder sorts candidates for the direction's selection sweep: "up"
// keeps the most-valuable upgrades first (descending utility); "down" sacrifices
// the least-harmful tensors first (ascending utility).
func selectionOrder(cands []UpgradeCandidate, down bool) []UpgradeCandidate {
	if down {
		return sortedCandidates(cands, func(a, b UpgradeCandidate) bool { return lessUtilexists(b, a) })
	}
	return sortedCandidates(cands, lessUtilexists)
}

func sortedCandidates(cands []UpgradeCandidate, less func(a, b UpgradeCandidate) bool) []UpgradeCandidate {
	out := append([]UpgradeCandidate(nil), cands...)
	sortUpgrade(out, less)
	return out
}

// normalizeSteps rewrites binary (non-ladder) candidates as single-step ladder
// candidates so every selection path can treat candidates uniformly.
func normalizeSteps(cands []UpgradeCandidate) {
	for i := range cands {
		if cands[i].TotalSteps <= 1 {
			cands[i].Step = 1
			cands[i].TotalSteps = 1
		}
	}
}

// stepEligible reports whether ladder step c may be taken when depth steps of
// its tensor are already taken (a prefix constraint: step k needs steps 1..k-1).
func stepEligible(c UpgradeCandidate, depth int) bool {
	return depth == c.Step-1
}

// collapseSteps merges each tensor's selected ladder steps into one final-state
// candidate: the tensor's deepest reached qtype with cumulative delta, expected
// gain and utility. Downstream (overrides, recipe, tensor-types, shares) only
// sees the final per-tensor assignment.
func collapseSteps(selected []UpgradeCandidate) []UpgradeCandidate {
	first := map[string]UpgradeCandidate{}
	deepest := map[string]UpgradeCandidate{}
	delta := map[string]int{}
	for _, c := range selected {
		if _, ok := first[c.Tensor]; !ok {
			first[c.Tensor] = c
		}
		if d, ok := deepest[c.Tensor]; !ok || c.Step > d.Step {
			deepest[c.Tensor] = c
		}
		delta[c.Tensor] += c.DeltaBytes
	}
	out := make([]UpgradeCandidate, 0, len(first))
	for tensor, f := range first {
		// Single-step (binary) transitions collapse to themselves exactly,
		// preserving the generated expected gain and utility.
		if f.TotalSteps == 1 {
			out = append(out, f)
			continue
		}
		d := deepest[tensor]
		bits := 0.0
		if fromBPW, err := qtypeBPW(f.FromQtype); err == nil {
			if toBPW, err := qtypeBPW(d.ToQtype); err == nil {
				bits = fromBPW - toBPW
				if bits < 0 {
					bits = -bits
				}
			}
		}
		eg := f.Importance * bits
		util := 0.0
		if delta[tensor] > 0 {
			util = eg / float64(delta[tensor])
		}
		out = append(out, UpgradeCandidate{
			Tensor: tensor, FromQtype: f.FromQtype, ToQtype: d.ToQtype,
			DeltaBytes: delta[tensor], Importance: f.Importance, RawImportance: f.RawImportance,
			ExpectedGain: eg, UtilityPerByte: util,
			Profiled: f.Profiled, Block: f.Block, Role: f.Role,
			Step: d.Step, TotalSteps: f.TotalSteps,
		})
	}
	sortUpgrade(out, func(a, b UpgradeCandidate) bool { return a.Tensor < b.Tensor })
	return out
}

// selectPlan builds the size-exact plan for the given selection sweep order.
// In "up" mode the lower baseline grows by adding upgrades up to target; in
// "down" mode the upper baseline (high quality) is shrunk by downgrading the
// least-harmful selected tensors until the target fits. Ladder steps are taken
// prefix-wise (a tensor may only be moved part-way along its ladder), then
// collapsed to one final assignment per tensor.
func selectPlan(cs *CandidateSet, ordered []UpgradeCandidate, target int) *OptimizationPlan {
	normalizeSteps(cs.Candidates)
	normalizeSteps(ordered)
	if cs.Direction == "down" {
		budget := cs.UpperSizeBytes - target
		depth := map[string]int{}
		var selected []UpgradeCandidate
		saved := 0
		for saved < budget {
			progressed := false
			for _, c := range ordered {
				if saved >= budget {
					break
				}
				if depth[c.Tensor] >= c.TotalSteps || !stepEligible(c, depth[c.Tensor]) {
					continue
				}
				depth[c.Tensor]++
				selected = append(selected, c)
				saved += c.DeltaBytes
				progressed = true
			}
			if !progressed {
				break
			}
		}
		selected, saved = trimBackSteps(selected, budget)
		collapsed := collapseSteps(selected)
		return &OptimizationPlan{
			SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
			PredictedSizeBytes: cs.UpperSizeBytes - saved, UnusedBytes: budget - saved,
			Selected: collapsed, SkippedCount: cs.TensorCount - len(collapsed),
		}
	}

	remaining := target - cs.LowerSizeBytes
	depth := map[string]int{}
	var selected []UpgradeCandidate
	for remaining > 0 {
		progressed := false
		for _, c := range ordered {
			if depth[c.Tensor] >= c.TotalSteps || !stepEligible(c, depth[c.Tensor]) {
				continue
			}
			if c.DeltaBytes <= remaining {
				depth[c.Tensor]++
				remaining -= c.DeltaBytes
				selected = append(selected, c)
				progressed = true
			}
		}
		if !progressed {
			break
		}
	}
	collapsed := collapseSteps(selected)
	return &OptimizationPlan{
		SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
		PredictedSizeBytes: target - remaining, UnusedBytes: remaining,
		Selected: collapsed, SkippedCount: cs.TensorCount - len(collapsed),
	}
}

// trimBackSteps restores the most valuable selected ladder steps while the
// remaining savings still cover the budget (predicted size never drops below
// the target). Unlike whole-tensor rollback it operates on the step sequence
// BEFORE collapsing, so a tensor can be rolled back one ladder step at a time —
// the finest granularity the candidate ladder offers. This closes the leftover
// budget gap (e.g. a 40+ MiB over-degrade) that whole-tensor rollback cannot
// fill when every remaining tensor's delta exceeds the gap. It recomputes and
// returns the total bytes saved by the (possibly reduced) selection. Used by
// every down-mode path so overshoot handling is consistent.
func trimBackSteps(selected []UpgradeCandidate, budget int) ([]UpgradeCandidate, int) {
	byTensor := map[string][]UpgradeCandidate{}
	for _, c := range selected {
		byTensor[c.Tensor] = append(byTensor[c.Tensor], c)
	}
	saved := 0
	for _, steps := range byTensor {
		for _, c := range steps {
			saved += c.DeltaBytes
		}
	}
	if saved <= budget {
		return selected, saved
	}
	for {
		// candidates: each tensor's deepest taken step (rolling a tensor back
		// starts at its deepest step)
		var candidates []UpgradeCandidate
		for _, steps := range byTensor {
			if len(steps) == 0 {
				continue
			}
			candidates = append(candidates, steps[len(steps)-1])
		}
		if len(candidates) == 0 {
			break
		}
		sortUpgrade(candidates, lessUtilexists)
		progressed := false
		for _, cand := range candidates {
			if saved-cand.DeltaBytes < budget {
				// rolling this step back would push the predicted size below the
				// target; a smaller step elsewhere may still fit the gap
				continue
			}
			saved -= cand.DeltaBytes
			steps := byTensor[cand.Tensor]
			byTensor[cand.Tensor] = steps[:len(steps)-1]
			progressed = true
		}
		if !progressed {
			break
		}
	}
	out := make([]UpgradeCandidate, 0, len(selected))
	for _, steps := range byTensor {
		out = append(out, steps...)
	}
	return out, saved
}

func optimizeGreedy(target int, cs *CandidateSet) (*OptimizationPlan, error) {
	if err := validateCandidates(cs.Candidates, cs.LowerSizeBytes, target); err != nil {
		return nil, err
	}
	ordered := selectionOrder(cs.Candidates, cs.Direction == "down")
	return selectPlan(cs, ordered, target), nil
}

func optimizeRandom(target int, cs *CandidateSet, seed string) (*OptimizationPlan, error) {
	if err := validateCandidates(cs.Candidates, cs.LowerSizeBytes, target); err != nil {
		return nil, err
	}
	hashKey := func(c UpgradeCandidate) []byte {
		h := sha256.Sum256([]byte(seed + "\x00" + c.Tensor + "\x00" + c.ToQtype))
		return h[:]
	}
	ordered := append([]UpgradeCandidate(nil), cs.Candidates...)
	sortSHA(ordered, hashKey)
	return selectPlan(cs, ordered, target), nil
}

func optimizeBlockBalanced(target int, cs *CandidateSet, blockSpan int) (*OptimizationPlan, error) {
	if blockSpan <= 0 {
		return nil, fmt.Errorf("pipeline: block_span must be positive")
	}
	if err := validateCandidates(cs.Candidates, cs.LowerSizeBytes, target); err != nil {
		return nil, err
	}
	normalizeSteps(cs.Candidates)
	groups := map[int][]UpgradeCandidate{}
	for _, c := range cs.Candidates {
		if c.Block >= 0 {
			g := c.Block / blockSpan
			groups[g] = append(groups[g], c)
		}
	}
	if len(groups) == 0 {
		return nil, fmt.Errorf("pipeline: block-balanced optimization requires profiled block candidates")
	}
	groupIDs := make([]int, 0, len(groups))
	for g := range groups {
		groupIDs = append(groupIDs, g)
	}
	sortInts(groupIDs)

	// Within a block group, "up" upgrades in descending utility; "down"
	// sacrifices in ascending utility (least harm first).
	down := cs.Direction == "down"
	order := func(cands []UpgradeCandidate) []UpgradeCandidate {
		return selectionOrder(cands, down)
	}

	depth := map[string]int{}
	var selected []UpgradeCandidate
	if down {
		// Remove bytes: each group downgrades its least-harmful tensors to hit
		// its share of the total budget, then a leftover pass catches the rest.
		budget := cs.UpperSizeBytes - target
		quota := budget / len(groupIDs)
		remQuota := budget % len(groupIDs)
		for pos, gid := range groupIDs {
			groupNeed := quota
			if pos == len(groupIDs)-1 {
				groupNeed += remQuota
			}
			ordered := order(groups[gid])
			for groupNeed > 0 {
				progressed := false
				for _, c := range ordered {
					if groupNeed <= 0 {
						break
					}
					if depth[c.Tensor] >= c.TotalSteps || !stepEligible(c, depth[c.Tensor]) {
						continue
					}
					depth[c.Tensor]++
					selected = append(selected, c)
					groupNeed -= c.DeltaBytes
					progressed = true
				}
				if !progressed {
					break
				}
			}
		}
		saved := 0
		for _, c := range selected {
			saved += c.DeltaBytes
		}
		remaining := budget - saved
		var leftovers []UpgradeCandidate
		for _, c := range cs.Candidates {
			if depth[c.Tensor] < c.Step {
				leftovers = append(leftovers, c)
			}
		}
		for remaining > 0 {
			progressed := false
			for _, c := range order(leftovers) {
				if remaining <= 0 {
					break
				}
				if depth[c.Tensor] >= c.TotalSteps || !stepEligible(c, depth[c.Tensor]) {
					continue
				}
				depth[c.Tensor]++
				selected = append(selected, c)
				remaining -= c.DeltaBytes
				progressed = true
			}
			if !progressed {
				break
			}
		}
		// Step-level rollback: the per-group quotas can collectively overshoot
		// the budget (when a group's total savings barely exceed its quota it
		// consumes every candidate). Restore the most valuable selected ladder
		// steps while the remaining savings still cover the budget, so only what
		// the target strictly requires is sacrificed.
		selected, saved = trimBackSteps(selected, budget)
		collapsed := collapseSteps(selected)
		remaining = budget - saved
		return &OptimizationPlan{
			SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
			PredictedSizeBytes: cs.UpperSizeBytes - (budget - remaining), UnusedBytes: remaining,
			Selected: collapsed, SkippedCount: cs.TensorCount - len(collapsed),
		}, nil
	}

	budget := target - cs.LowerSizeBytes
	quota := budget / len(groupIDs)
	remQuota := budget % len(groupIDs)
	for pos, gid := range groupIDs {
		groupRemaining := quota
		if pos == len(groupIDs)-1 {
			groupRemaining += remQuota
		}
		ordered := order(groups[gid])
		for groupRemaining > 0 {
			progressed := false
			for _, c := range ordered {
				if groupRemaining <= 0 {
					break
				}
				if depth[c.Tensor] >= c.TotalSteps || !stepEligible(c, depth[c.Tensor]) {
					continue
				}
				if c.DeltaBytes <= groupRemaining {
					depth[c.Tensor]++
					selected = append(selected, c)
					groupRemaining -= c.DeltaBytes
					progressed = true
				}
			}
			if !progressed {
				break
			}
		}
	}

	spent := 0
	for _, c := range selected {
		spent += c.DeltaBytes
	}
	remaining := budget - spent
	var leftovers []UpgradeCandidate
	for _, c := range cs.Candidates {
		if depth[c.Tensor] < c.Step {
			leftovers = append(leftovers, c)
		}
	}
	for remaining > 0 {
		progressed := false
		for _, c := range order(leftovers) {
			if remaining <= 0 {
				break
			}
			if depth[c.Tensor] >= c.TotalSteps || !stepEligible(c, depth[c.Tensor]) {
				continue
			}
			if c.DeltaBytes <= remaining {
				depth[c.Tensor]++
				selected = append(selected, c)
				remaining -= c.DeltaBytes
				progressed = true
			}
		}
		if !progressed {
			break
		}
	}
	collapsed := collapseSteps(selected)
	return &OptimizationPlan{
		SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
		PredictedSizeBytes: target - remaining, UnusedBytes: remaining,
		Selected: collapsed, SkippedCount: cs.TensorCount - len(collapsed),
	}, nil
}
