package pipeline

import (
	"path/filepath"
	"testing"

	gguf "fitting/gguf"
)

// Tolerated "same type" tensor exercises the no-transition path: it must not
// produce a candidate regardless of direction.
var modeTestTensors = []string{"blk.0.ffn_up.weight", "blk.0.attn_v.weight", "blk.0.attn_norm.weight"}

func modeTestLower(upper bool) *Recipe {
	dst := "Q4_K"
	if upper {
		dst = "Q8_0"
	}
	// the norm tensor stays F16 in both presets (no transition)
	tName := map[string]string{
		"blk.0.ffn_up.weight":   dst,
		"blk.0.attn_v.weight":   dst,
		"blk.0.attn_norm.weight": "F16",
	}
	rec := &Recipe{TotalTensors: len(modeTestTensors)}
	for i, n := range modeTestTensors {
		rec.Tensors = append(rec.Tensors, Assignment{
			Name: n, DstType: tName[n], IsQuantized: n != "blk.0.attn_norm.weight",
			Ordinal: i, TotalTensors: len(modeTestTensors),
		})
	}
	return rec
}

func modeTestSizes() (lower, upper *gguf.Prediction) {
	lower = &gguf.Prediction{TotalBytes: 310, Tensors: []gguf.TensorSize{
		{Name: "blk.0.ffn_up.weight", PaddedBytes: 100},
		{Name: "blk.0.attn_v.weight", PaddedBytes: 200},
		{Name: "blk.0.attn_norm.weight", PaddedBytes: 10},
	}}
	upper = &gguf.Prediction{TotalBytes: 510, Tensors: []gguf.TensorSize{
		{Name: "blk.0.ffn_up.weight", PaddedBytes: 200},
		{Name: "blk.0.attn_v.weight", PaddedBytes: 300},
		{Name: "blk.0.attn_norm.weight", PaddedBytes: 10},
	}}
	return lower, upper
}

// modeTestProfile returns a profile with block-indexed entries so
// block-balanced selection has candidate groups to work with.
func modeTestProfile() *ImatrixProfile {
	return &ImatrixProfile{Entries: []ImatrixTensorProfile{
		{Name: "blk.0.ffn_up.weight", Block: 0, Role: "ffn_up", RoleRelativeMean: 1.5, Mean: 2.0},
		{Name: "blk.0.attn_v.weight", Block: 0, Role: "attn_v", RoleRelativeMean: 2.5, Mean: 3.0},
	}}
}

func modeCandidateSet(t *testing.T, mode string) *CandidateSet {
	t.Helper()
	lower, upper := modeTestSizes()
	cs, err := GenerateUpgradeCandidates(modeTestLower(false), modeTestLower(true), lower, upper, modeTestProfile(), mode)
	if err != nil {
		t.Fatalf("GenerateUpgradeCandidates(%s): %v", mode, err)
	}
	return cs
}

func assertDirection(t *testing.T, cs *CandidateSet, wantFrom, wantTo, wantDir string) {
	t.Helper()
	if cs.Direction != wantDir {
		t.Fatalf("Direction=%s want %s", cs.Direction, wantDir)
	}
	// the norm tensor has no transition; exactly two candidates remain
	if len(cs.Candidates) != 2 {
		t.Fatalf("candidate count=%d want 2", len(cs.Candidates))
	}
	for _, c := range cs.Candidates {
		if c.FromQtype != wantFrom || c.ToQtype != wantTo {
			t.Errorf("transition %s->%s want %s->%s (%s)", c.FromQtype, c.ToQtype, wantFrom, wantTo, c.Tensor)
		}
		if c.DeltaBytes <= 0 {
			t.Errorf("delta_bytes=%d must be positive (%s)", c.DeltaBytes, c.Tensor)
		}
	}
}

func TestGenerateUpgradeCandidatesUpDirection(t *testing.T) {
	assertDirection(t, modeCandidateSet(t, "up"), "q4_k", "q8_0", "up")
}

func TestGenerateUpgradeCandidatesDownDirection(t *testing.T) {
	assertDirection(t, modeCandidateSet(t, "down"), "q8_0", "q4_k", "down")
}

func TestCandidateSizeGapEqualAcrossDirections(t *testing.T) {
	up := modeCandidateSet(t, "up")
	down := modeCandidateSet(t, "down")
	by := map[string]map[string]int{} // tensor -> {fromdelta}
	for _, c := range up.Candidates {
		by[c.Tensor] = map[string]int{"up": c.DeltaBytes}
	}
	for _, c := range down.Candidates {
		if by[c.Tensor]["up"] != c.DeltaBytes {
			t.Errorf("tensor %s delta differs across directions: up=%d down=%d", c.Tensor, by[c.Tensor]["up"], c.DeltaBytes)
		}
	}
}

// TestSelectPlanBoundsVerifies that in both directions the chosen size stays
// within [lower, target] (down) / [lower, target] (up).
func TestSelectPlanBounds(t *testing.T) {
	down := modeCandidateSet(t, "down")
	up := modeCandidateSet(t, "up")

	// down: mid target must land within [lower, target]
	p := selectPlan(down, selectionOrder(down.Candidates, true), 410)
	if p.PredictedSizeBytes < 310 || p.PredictedSizeBytes > 410 {
		t.Errorf("down mid pred=%d want in [310,410]", p.PredictedSizeBytes)
	}
	// down: target == lower removes everything
	if p := selectPlan(down, selectionOrder(down.Candidates, true), 310); p.PredictedSizeBytes != 310 {
		t.Errorf("down min pred=%d want 310", p.PredictedSizeBytes)
	}
	// up: target == upper keeps everything
	if p := selectPlan(up, selectionOrder(up.Candidates, false), 510); p.PredictedSizeBytes != 510 {
		t.Errorf("up max pred=%d want 510", p.PredictedSizeBytes)
	}
	// up: mid target cannot shrink below lower
	if p := selectPlan(up, selectionOrder(up.Candidates, false), 400); p.PredictedSizeBytes < 310 || p.PredictedSizeBytes > 400 {
		t.Errorf("up mid pred=%d want in [310,400]", p.PredictedSizeBytes)
	}
}

func TestGreedyDownTargetsBounds(t *testing.T) {
	down := modeCandidateSet(t, "down")
	for _, target := range []int{410, 360, 510, 310} {
		p, err := optimizeGreedy(target, down)
		if err != nil {
			t.Fatalf("optimizeGreedy(%d): %v", target, err)
		}
		if p.PredictedSizeBytes < 310 || p.PredictedSizeBytes > target {
			t.Errorf("greedy down target=%d pred=%d out of [310,target]", target, p.PredictedSizeBytes)
		}
	}
}

func TestGreedyUpTargetsBounds(t *testing.T) {
	up := modeCandidateSet(t, "up")
	for _, target := range []int{410, 510, 310} {
		p, err := optimizeGreedy(target, up)
		if err != nil {
			t.Fatalf("optimizeGreedy(%d): %v", target, err)
		}
		if p.PredictedSizeBytes < 310 || p.PredictedSizeBytes > target {
			t.Errorf("greedy up target=%d pred=%d out of [310,target]", target, p.PredictedSizeBytes)
		}
	}
}

func TestRandomAndBlockBalancedDown(t *testing.T) {
	down := modeCandidateSet(t, "down")
	for _, target := range []int{410, 360, 510, 310} {
		p, err := optimizeRandom(target, down, "seed")
		if err != nil {
			t.Fatalf("random (%d): %v", target, err)
		}
		if p.PredictedSizeBytes < 310 || p.PredictedSizeBytes > target {
			t.Errorf("random down target=%d pred=%d", target, p.PredictedSizeBytes)
		}
		pb, err := optimizeBlockBalanced(target, down, 1)
		if err != nil {
			t.Fatalf("balanced (%d): %v", target, err)
		}
		if pb.PredictedSizeBytes < 310 || pb.PredictedSizeBytes > target {
			t.Errorf("balanced down target=%d pred=%d", target, pb.PredictedSizeBytes)
		}
	}
}

func TestAnalysisDirectionHelpers(t *testing.T) {
	lower := &Recipe{}
	upper := &Recipe{}
	a := &Analysis{Mode: "down", LowerPreset: "Q4_K", UpperPreset: "Q8_0", LowerRecipe: lower, UpperRecipe: upper}
	if a.Direction() != "down" || a.BaselineRecipe() != upper || a.RecommendedPreset() != "Q8_0" {
		t.Errorf("down helpers wrong: dir=%s base=%p preset=%s", a.Direction(), a.BaselineRecipe(), a.RecommendedPreset())
	}
	b := &Analysis{Mode: "", LowerPreset: "Q4_K", UpperPreset: "Q8_0", LowerRecipe: lower, UpperRecipe: upper}
	if b.Direction() != "up" || b.BaselineRecipe() != lower || b.RecommendedPreset() != "Q4_K" {
		t.Errorf("up/default helpers wrong: dir=%s base=%p preset=%s", b.Direction(), b.BaselineRecipe(), b.RecommendedPreset())
	}
}

// TestAnalysisRoundTripModeAndUpper validates that analysis.json persists mode
// and the upper recipe and that LoadAnalysis restores them.
func TestAnalysisRoundTripModeAndUpper(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "analysis.json")

	doc := map[string]any{
		"schema_version": 1,
		"mode":           "down",
		"source":         map[string]any{"path": "source.gguf", "size_bytes": 1, "sha256": nil},
		"imatrix":        map[string]any{"path": "im.gguf", "sha256": nil, "arg": "im.gguf"},
		"runtime":        map[string]any{"dir": "runtime", "llama_quantize": "llama-quantize"},
		"presets": map[string]any{
			"lower": map[string]any{"name": "Q4_K", "file_type": 10, "predicted_size_bytes": int64(310)},
			"upper": map[string]any{"name": "Q8_0", "file_type": 11, "predicted_size_bytes": int64(510)},
		},
		"metadata": map[string]any{"file_type": 10, "quantization_version": 2},
		"lower_recipe": map[string]any{
			"total_tensors": 1, "reported_orig_bytes": 0, "reported_new_bytes": 0,
			"tensors": []map[string]any{{"ordinal": 0, "total_tensors": 1, "name": "x", "shape": []int64{1}, "src_type": "F16", "dst_type": "Q4_K", "is_quantized": true, "orig_bytes": 0, "new_bytes": 0}},
		},
		"upper_recipe": map[string]any{
			"total_tensors": 1, "reported_orig_bytes": 0, "reported_new_bytes": 0,
			"tensors": []map[string]any{{"ordinal": 0, "total_tensors": 1, "name": "x", "shape": []int64{1}, "src_type": "F16", "dst_type": "Q8_0", "is_quantized": true, "orig_bytes": 0, "new_bytes": 0}},
		},
	}
	if err := writeJSON(path, doc); err != nil {
		t.Fatalf("writeJSON: %v", err)
	}
	a, err := LoadAnalysis(path)
	if err != nil {
		t.Fatalf("LoadAnalysis: %v", err)
	}
	if a.Mode != "down" {
		t.Errorf("Mode=%q want down", a.Mode)
	}
	if a.UpperRecipe == nil || len(a.UpperRecipe.Tensors) != 1 || a.UpperRecipe.Tensors[0].DstType != "Q8_0" {
		t.Errorf("upper recipe not restored: %+v", a.UpperRecipe)
	}
	if a.Direction() != "down" || a.RecommendedPreset() != "Q8_0" {
		t.Errorf("restored analysis direction wrong: dir=%s preset=%s", a.Direction(), a.RecommendedPreset())
	}
}

// TestBlockBalancedDownTrimsOvershoot verifies that when the per-group quotas
// collectively overshoot the budget (target close to the lower preset), the
// most valuable tensors are restored to the upper preset instead of being
// downgraded as well: the plan must land at the target and spare the
// highest-utility candidate.
func TestBlockBalancedDownTrimsOvershoot(t *testing.T) {
	mk := func(name string, block, delta int, util float64) UpgradeCandidate {
		return UpgradeCandidate{
			Tensor: name, FromQtype: "q8_0", ToQtype: "iq4_xs",
			DeltaBytes: delta, UtilityPerByte: util,
			ExpectedGain: util * float64(delta), Block: block,
		}
	}
	// lower=400, upper=1000, target=700 => budget 300, quota 150 per group.
	// Each group's two candidates sum to 200 (> quota), so the naive loop
	// consumes every candidate and overshoots to 400 bytes saved (predicted
	// 600). The trim pass must restore the highest-utility candidate D and land
	// on exactly the target.
	cs := &CandidateSet{
		Candidates: []UpgradeCandidate{
			mk("A", 0, 100, 1e-5),
			mk("B", 0, 100, 2e-5),
			mk("C", 1, 100, 1.5e-5),
			mk("D", 1, 100, 3e-5),
		},
		LowerSizeBytes: 400, UpperSizeBytes: 1000,
		Direction: "down",
	}
	p, err := optimizeBlockBalanced(700, cs, 1)
	if err != nil {
		t.Fatalf("optimizeBlockBalanced: %v", err)
	}
	if p.PredictedSizeBytes != 700 {
		t.Errorf("predicted=%d want 700 (overshoot must be trimmed)", p.PredictedSizeBytes)
	}
	for _, c := range p.Selected {
		if c.Tensor == "D" {
			t.Errorf("highest-utility tensor D must stay at the upper preset, but was selected")
		}
	}
	if len(p.Selected) != 3 {
		t.Errorf("selected=%d want 3", len(p.Selected))
	}
}

// TestQtypeParameterDistributionFloatBaseline verifies that a "down" plan from
// a float upper preset reports the kept float baseline in the qtype
// distribution. llama-quantize marks BF16->BF16 as "unchanged", so those kept
// tensors must still count; the untouched F32 aux tensors stay excluded.
func TestQtypeParameterDistributionFloatBaseline(t *testing.T) {
	rec := &Recipe{Tensors: []Assignment{
		{Name: "blk.0.ffn_up.weight", Shape: []int64{4096, 12288}, SrcType: "BF16", DstType: "IQ1_S", IsQuantized: false}, // selected override
		{Name: "blk.0.attn_v.weight", Shape: []int64{4096, 2048}, SrcType: "BF16", DstType: "BF16", IsQuantized: false},  // kept baseline
		{Name: "blk.0.attn_norm.weight", Shape: []int64{4096}, SrcType: "F32", DstType: "F32", IsQuantized: false},       // aux float
	}}
	dist := QtypeParameterDistribution(rec)
	if len(dist) != 2 {
		t.Fatalf("dist=%v want 2 covered qtypes (IQ1_S, BF16)", dist)
	}
	if dist["iq1_s"] != 4096*12288 {
		t.Errorf("iq1_s elements=%d want %d", dist["iq1_s"], 4096*12288)
	}
	if dist["bf16"] != 4096*2048 {
		t.Errorf("bf16 elements=%d want %d", dist["bf16"], 4096*2048)
	}
	if _, ok := dist["f32"]; ok {
		t.Errorf("unchanged F32 aux tensor must be excluded, dist=%v", dist)
	}
}

// ladderTestRecipes builds a two-tensor recipe pair: a big weight that moves
// BF16→Q4_K along the ladder, and a norm that stays F16 in both presets.
func ladderTestRecipes() (lower, upper *Recipe) {
	lower = &Recipe{Tensors: []Assignment{
		{Name: "blk.0.ffn_up.weight", Shape: []int64{4096, 4096}, DstType: "Q4_K", IsQuantized: true},
		{Name: "blk.0.attn_norm.weight", Shape: []int64{4096}, DstType: "F16", IsQuantized: true},
	}}
	upper = &Recipe{Tensors: []Assignment{
		{Name: "blk.0.ffn_up.weight", Shape: []int64{4096, 4096}, DstType: "BF16", IsQuantized: false},
		{Name: "blk.0.attn_norm.weight", Shape: []int64{4096}, DstType: "F16", IsQuantized: false},
	}}
	return lower, upper
}

// TestGenerateLadderCandidatesSteps verifies the ladder segment between BF16 and
// Q4_K yields the expected contiguous steps (same-bpw neighbours collapse, only
// strict size decreases survive) with cumulative per-step deltas.
func TestGenerateLadderCandidatesSteps(t *testing.T) {
	layout := &gguf.Layout{
		Alignment: 32,
		TensorMap: map[string]gguf.TensorInfo{
			"blk.0.ffn_up.weight":   {Name: "blk.0.ffn_up.weight", Shape: []int64{4096, 4096}},
			"blk.0.attn_norm.weight": {Name: "blk.0.attn_norm.weight", Shape: []int64{4096}},
		},
	}
	lower, upper := ladderTestRecipes()
	profile := &ImatrixProfile{}
	for _, mode := range []string{"down", "up"} {
		cs, err := GenerateLadderCandidates(lower, upper,
			&gguf.Prediction{TotalBytes: 100}, &gguf.Prediction{TotalBytes: 200},
			layout, profile, mode)
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		if cs.TensorCount != 1 || len(cs.Candidates) != 4 {
			t.Fatalf("%s: TensorCount=%d steps=%d want 1/4", mode, cs.TensorCount, len(cs.Candidates))
		}
		wantFrom := []string{"bf16", "q8_0", "q6_k", "q5_k"}
		wantTo := []string{"q8_0", "q6_k", "q5_k", "q4_k"}
		wantDelta := []int{15728640, 4063232, 2228224, 2097152}
		for i, c := range cs.Candidates {
			if c.Tensor != "blk.0.ffn_up.weight" || c.Step != i+1 || c.TotalSteps != 4 {
				t.Errorf("%s step %d: tensor=%s Step=%d TotalSteps=%d", mode, i, c.Tensor, c.Step, c.TotalSteps)
			}
			if c.DeltaBytes != wantDelta[i] {
				t.Errorf("%s step %d delta=%d want %d", mode, i, c.DeltaBytes, wantDelta[i])
			}
			if mode == "down" {
				if c.FromQtype != wantFrom[i] || c.ToQtype != wantTo[i] {
					t.Errorf("%s step %d: %s->%s want %s->%s", mode, i, c.FromQtype, c.ToQtype, wantFrom[i], wantTo[i])
				}
			} else if c.FromQtype != wantTo[i] || c.ToQtype != wantFrom[i] {
				t.Errorf("%s step %d: %s->%s want %s->%s", mode, i, c.FromQtype, c.ToQtype, wantTo[i], wantFrom[i])
			}
		}
	}
}

// TestLadderPrefixConstraint verifies the optimizer never takes a deeper ladder
// step before its shallower predecessor (the deeper step sorts first here via
// the delta tie-break), and that multi-step downgrades collapse to one final
// assignment with the cumulative delta.
func TestLadderPrefixConstraint(t *testing.T) {
	mk := func(name string, step, total int, from, to string, delta int) UpgradeCandidate {
		return UpgradeCandidate{
			Tensor: name, FromQtype: from, ToQtype: to, DeltaBytes: delta,
			Importance: 1.0, ExpectedGain: 1.0, UtilityPerByte: 1e-5,
			Step: step, TotalSteps: total,
		}
	}
	cs := &CandidateSet{
		Candidates: []UpgradeCandidate{
			mk("X", 2, 2, "q8_0", "q6_k", 20), // deeper, smaller delta → sorts first
			mk("X", 1, 2, "bf16", "q8_0", 50),
		},
		LowerSizeBytes: 300, UpperSizeBytes: 500, TensorCount: 1,
		Direction: "down",
	}
	// budget = 500-440 = 60: only a prefix (both steps, 70 bytes) reaches it.
	p := selectPlan(cs, selectionOrder(cs.Candidates, true), 440)
	if len(p.Selected) != 1 {
		t.Fatalf("selected=%d want 1 collapsed", len(p.Selected))
	}
	c := p.Selected[0]
	if c.ToQtype != "q6_k" || c.DeltaBytes != 70 {
		t.Errorf("collapsed to=%s delta=%d want q6_k/70", c.ToQtype, c.DeltaBytes)
	}
	if p.PredictedSizeBytes != 430 {
		t.Errorf("predicted=%d want 430", p.PredictedSizeBytes)
	}
	if p.SkippedCount != 0 {
		t.Errorf("skipped=%d want 0", p.SkippedCount)
	}
}