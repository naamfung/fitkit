package pipeline

import (
	"math"
	"testing"
)

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) < 1e-12
}

func candidateSetForRefine() *CandidateSet {
	return &CandidateSet{
		Candidates: []UpgradeCandidate{
			{Tensor: "blk.0.attn_q.weight", ToQtype: "q6_k", DeltaBytes: 100, Importance: 1, ExpectedGain: 2, UtilityPerByte: 0.02, Role: "attn_q", Block: 0},
			{Tensor: "blk.16.ffn_up.weight", ToQtype: "q6_k", DeltaBytes: 100, Importance: 1, ExpectedGain: 2, UtilityPerByte: 0.02, Role: "ffn_up", Block: 16},
			{Tensor: "output_norm.weight", ToQtype: "q6_k", DeltaBytes: 100, Importance: 1, ExpectedGain: 2, UtilityPerByte: 0.02, Role: "output_norm", Block: -1},
		},
	}
}

// TestApplyRefineRoleOnly pins the role-only (bootstrap-v0) form: every
// candidate's utility is multiplied by its role's C_role (default 1.0).
func TestApplyRefineRoleOnly(t *testing.T) {
	cs := candidateSetForRefine()
	profile := map[string]any{
		"schema":           "fit.refine_profile.v1",
		"profile_id":       "test",
		"split":            "dev",
		"source":           map[string]any{},
		"role_correction":  map[string]any{"attn_q": 1.25, "ffn_up": 0.75},
		"calibration_status": map[string]any{"c_role_form": "proposal-calibration-bootstrap-v0"},
	}
	usage, err := ApplyRefineCorrections(cs, profile)
	if err != nil {
		t.Fatal(err)
	}
	if len(usage) != 0 {
		t.Fatalf("role-only form must not record band usage, got %v", usage)
	}
	byRole := map[string]float64{}
	for _, c := range cs.Candidates {
		byRole[c.Role] = c.UtilityPerByte
	}
	if !closeEnough(byRole["attn_q"], 0.02*1.25) {
		t.Fatalf("attn_q utility = %v, want %v", byRole["attn_q"], 0.02*1.25)
	}
	if !closeEnough(byRole["ffn_up"], 0.02*0.75) {
		t.Fatalf("ffn_up utility = %v, want %v", byRole["ffn_up"], 0.02*0.75)
	}
	if !closeEnough(byRole["output_norm"], 0.02) {
		t.Fatalf("unprofiled role must keep utility, got %v", byRole["output_norm"])
	}
}

// TestApplyRefineBandCells pins the band-conditional form: the narrowest
// matching upgrade cell reweights the candidate and records usage; tensors
// outside every cell fall back to C_role.
func TestApplyRefineBandCells(t *testing.T) {
	cs := candidateSetForRefine()
	profile := map[string]any{
		"schema":           "fit.refine_profile.v1",
		"profile_id":       "test",
		"split":            "dev",
		"source":           map[string]any{},
		"role_correction":  map[string]any{"attn_q": 1.0, "ffn_up": 1.0, "output_norm": 1.0},
		"band_correction": map[string]any{
			"cells": []any{
				map[string]any{"cell_id": "attn_q:early:q6_k->q8_0", "role": "attn_q", "direction": "upgrade", "min_block": 0, "max_block": 15, "c": 1.4},
			},
		},
		"calibration_status": map[string]any{"c_role_form": "band-conditional-bootstrap-v1"},
	}
	usage, err := ApplyRefineCorrections(cs, profile)
	if err != nil {
		t.Fatal(err)
	}
	if usage["attn_q:early:q6_k->q8_0"] != 1 {
		t.Fatalf("band usage = %v, want 1 cell hit", usage)
	}
	byTensor := map[string]float64{}
	for _, c := range cs.Candidates {
		byTensor[c.Tensor] = c.UtilityPerByte
	}
	if !closeEnough(byTensor["blk.0.attn_q.weight"], 0.02*1.4) {
		t.Fatalf("blk.0.attn_q utility = %v, want %v", byTensor["blk.0.attn_q.weight"], 0.02*1.4)
	}
	if !closeEnough(byTensor["blk.16.ffn_up.weight"], 0.02) {
		t.Fatalf("ffn_up outside cells must keep role fallback, got %v", byTensor["blk.16.ffn_up.weight"])
	}
	if !closeEnough(byTensor["output_norm.weight"], 0.02) {
		t.Fatalf("non-block tensor must keep role fallback, got %v", byTensor["output_norm.weight"])
	}
}
