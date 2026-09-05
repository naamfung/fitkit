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
	names := map[string]bool{}
	for _, c := range cands {
		if names[c.Tensor] {
			return fmt.Errorf("pipeline: candidate tensor names must be unique")
		}
		names[c.Tensor] = true
		if c.DeltaBytes <= 0 {
			return fmt.Errorf("pipeline: all candidate costs must be positive")
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

// selectPlan builds the size-exact plan for the given selection sweep order.
// In "up" mode the lower baseline grows by adding upgrades up to target; in
// "down" mode the upper baseline (high quality) is shrunk by downgrading the
// least-harmful selected tensors until the target fits.
func selectPlan(cs *CandidateSet, ordered []UpgradeCandidate, target int) *OptimizationPlan {
	if cs.Direction == "down" {
		budget := cs.UpperSizeBytes - target
		var selected []UpgradeCandidate
		saved := 0
		for _, c := range ordered {
			if saved >= budget {
				break
			}
			selected = append(selected, c)
			saved += c.DeltaBytes
		}
		return &OptimizationPlan{
			SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
			PredictedSizeBytes: cs.UpperSizeBytes - saved, UnusedBytes: budget - saved,
			Selected: selected, SkippedCount: len(cs.Candidates) - len(selected),
		}
	}

	remaining := target - cs.LowerSizeBytes
	var selected []UpgradeCandidate
	for _, c := range ordered {
		if c.DeltaBytes <= remaining {
			selected = append(selected, c)
			remaining -= c.DeltaBytes
		}
	}
	return &OptimizationPlan{
		SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
		PredictedSizeBytes: target - remaining, UnusedBytes: remaining,
		Selected: selected, SkippedCount: len(cs.Candidates) - len(selected),
	}
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

	selectedMap := map[string]bool{}
	var selected []UpgradeCandidate
	if down {
		// Remove bytes: each group downgrades its least-harmful tensors to hit
		// its share of the total budget, then a leftover pass catches remainder.
		budget := cs.UpperSizeBytes - target
		quota := budget / len(groupIDs)
		remQuota := budget % len(groupIDs)
		for pos, gid := range groupIDs {
			groupNeed := quota
			if pos == len(groupIDs)-1 {
				groupNeed += remQuota
			}
			for _, c := range order(groups[gid]) {
				if groupNeed <= 0 {
					break
				}
				selected = append(selected, c)
				selectedMap[c.Tensor] = true
				groupNeed -= c.DeltaBytes
			}
		}
		saved := 0
		for _, c := range selected {
			saved += c.DeltaBytes
		}
		remaining := budget - saved
		var leftovers []UpgradeCandidate
		for _, c := range cs.Candidates {
			if !selectedMap[c.Tensor] {
				leftovers = append(leftovers, c)
			}
		}
		for _, c := range order(leftovers) {
			if remaining <= 0 {
				break
			}
			selected = append(selected, c)
			selectedMap[c.Tensor] = true
			remaining -= c.DeltaBytes
		}
		// Trim back: the per-group quotas can collectively overshoot the budget
		// (when a group's total savings barely exceed its quota it consumes every
		// candidate, leaving nothing at the upper preset). Restore the most
		// valuable selected tensors to the upper preset while the remaining
		// savings still cover the budget, so only what the target strictly
		// requires is sacrificed.
		saved = 0
		for _, c := range selected {
			saved += c.DeltaBytes
		}
		if saved > budget {
			for _, c := range sortedCandidates(selected, lessUtilexists) {
				if saved-c.DeltaBytes < budget {
					break
				}
				saved -= c.DeltaBytes
				delete(selectedMap, c.Tensor)
			}
			selected = nil
			for _, c := range cs.Candidates {
				if selectedMap[c.Tensor] {
					selected = append(selected, c)
				}
			}
		}
		remaining = budget - saved
		return &OptimizationPlan{
			SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
			PredictedSizeBytes: cs.UpperSizeBytes - (budget - remaining), UnusedBytes: remaining,
			Selected: selected, SkippedCount: len(cs.Candidates) - len(selected),
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
		for _, c := range order(groups[gid]) {
			if c.DeltaBytes <= groupRemaining {
				selected = append(selected, c)
				selectedMap[c.Tensor] = true
				groupRemaining -= c.DeltaBytes
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
		if !selectedMap[c.Tensor] {
			leftovers = append(leftovers, c)
		}
	}
	for _, c := range order(leftovers) {
		if c.DeltaBytes <= remaining {
			selected = append(selected, c)
			remaining -= c.DeltaBytes
		}
	}
	return &OptimizationPlan{
		SchemaVersion: 1, TargetBytes: target, LowerSizeBytes: cs.LowerSizeBytes,
		PredictedSizeBytes: target - remaining, UnusedBytes: remaining,
		Selected: selected, SkippedCount: len(cs.Candidates) - len(selected),
	}, nil
}
