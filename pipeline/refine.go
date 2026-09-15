// refine.go applies a Refine Profile to a candidate set, mirroring the
// upstream fit_gguf.refine.profile application path used by pipeline.plan():
// reweights each candidate's utility_per_byte by C_role, or by the narrowest
// matching band cell when the profile carries band-conditional cells.
package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
)

// Refine profile constants (mirror upstream refine/profile.py).
const (
	refineSchema = "fit.refine_profile.v1"
	refineCMin   = 0.5
	refineCMax   = 1.5
)

// tensorBlock (in support.go) extracts the layer index from a "blk.N." tensor
// name; non-block tensors fall back to the role-level correction.

func refineFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	}
	return 0
}

// validateRefineProfile enforces the refine profile structural contract
// (mirror upstream validate_profile).
func validateRefineProfile(profile map[string]any) error {
	if profile["schema"] != refineSchema {
		return fmt.Errorf("refine profile: schema %v != %s", profile["schema"], refineSchema)
	}
	for _, key := range []string{"profile_id", "split", "source", "role_correction"} {
		if _, ok := profile[key]; !ok {
			return fmt.Errorf("refine profile: missing field %s", key)
		}
	}
	roles, _ := profile["role_correction"].(map[string]any)
	for role, value := range roles {
		c := refineFloat(value)
		if c < refineCMin || c > refineCMax {
			return fmt.Errorf("refine profile: role_correction[%s] out of range: %v", role, value)
		}
	}
	if band, ok := profile["band_correction"].(map[string]any); ok {
		cells, _ := band["cells"].([]any)
		for _, item := range cells {
			cell, _ := item.(map[string]any)
			if cell == nil {
				continue
			}
			for _, key := range []string{"role", "cell_id", "min_block", "max_block", "c", "direction"} {
				if cell[key] == nil {
					return fmt.Errorf("refine profile: band cell missing field %s", key)
				}
			}
			c := refineFloat(cell["c"])
			if c < refineCMin || c > refineCMax {
				return fmt.Errorf("refine profile: band cell %v c out of range", cell["cell_id"])
			}
			if int(refineFloat(cell["min_block"])) > int(refineFloat(cell["max_block"])) {
				return fmt.Errorf("refine profile: band cell %v inverted block range", cell["cell_id"])
			}
		}
	}
	return nil
}

// LoadRefineProfile loads and validates a refine profile JSON.
func LoadRefineProfile(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("refine profile %s: %v", path, err)
	}
	var profile map[string]any
	if err := json.Unmarshal(raw, &profile); err != nil {
		return nil, fmt.Errorf("refine profile %s: %v", path, err)
	}
	if err := validateRefineProfile(profile); err != nil {
		return nil, err
	}
	return profile, nil
}

// resolveBandCorrection returns the correction factor for one candidate: the
// narrowest matching upgrade utility cell when the profile carries band cells,
// else the role-level C_role (default 1.0). cellID is "" when only the role
// fallback applied.
func resolveBandCorrection(profile map[string]any, role string, block int) (factor float64, cellID string) {
	if band, ok := profile["band_correction"].(map[string]any); ok {
		cells, _ := band["cells"].([]any)
		if block >= 0 && len(cells) > 0 {
			bestC := 0.0
			bestID := ""
			bestSpan := -1
			for _, item := range cells {
				cell, _ := item.(map[string]any)
				if cell == nil {
					continue
				}
				if cell["role"] != role || cell["direction"] != "upgrade" {
					continue
				}
				lo := int(refineFloat(cell["min_block"]))
				hi := int(refineFloat(cell["max_block"]))
				if lo <= block && block <= hi {
					span := hi - lo
					if bestSpan < 0 || span < bestSpan {
						bestSpan = span
						bestC = refineFloat(cell["c"])
						bestID = fmt.Sprintf("%v", cell["cell_id"])
					}
				}
			}
			if bestSpan >= 0 {
				return bestC, bestID
			}
		}
	}
	roles, _ := profile["role_correction"].(map[string]any)
	if v, ok := roles[role]; ok {
		return refineFloat(v), ""
	}
	return 1.0, ""
}

// ApplyRefineCorrections reweights the candidate set in place by the refine
// profile and returns the per-cell usage counts for the plan record. Size
// deltas and the candidate budget are unchanged — only the ranking feature
// moves.
func ApplyRefineCorrections(cs *CandidateSet, profile map[string]any) (map[string]int, error) {
	usage := map[string]int{}
	band, _ := profile["band_correction"].(map[string]any)
	cells, _ := band["cells"].([]any)
	if len(cells) == 0 {
		// role-only (bootstrap-v0) form: C_role applied to every candidate
		roles, _ := profile["role_correction"].(map[string]any)
		for i := range cs.Candidates {
			factor := 1.0
			if v, ok := roles[cs.Candidates[i].Role]; ok {
				factor = refineFloat(v)
			}
			cs.Candidates[i].UtilityPerByte *= factor
		}
		return usage, nil
	}
	// band-conditional form: narrowest matching utility cell per candidate
	for i := range cs.Candidates {
		c := &cs.Candidates[i]
		block, _ := tensorBlock(c.Tensor)
		factor, cellID := resolveBandCorrection(profile, c.Role, block)
		if cellID != "" {
			usage[cellID]++
		}
		c.UtilityPerByte *= factor
	}
	return usage, nil
}

// BuildRefineNote builds the refine_profile note recorded in the plan record
// (mirror upstream pipeline.plan's refine_note).
func BuildRefineNote(profile map[string]any, usage map[string]int) map[string]any {
	roles := map[string]any{}
	if rc, ok := profile["role_correction"].(map[string]any); ok {
		for r, c := range rc {
			roles[r] = refineFloat(c)
		}
	}
	note := map[string]any{
		"profile_id":       fmt.Sprintf("%v", profile["profile_id"]),
		"c_role_form":      fmt.Sprintf("%v", mapPath(profile, "calibration_status", "c_role_form")),
		"role_corrections": roles,
	}
	if len(usage) > 0 {
		note["band_cells_applied"] = usage
		if band, ok := profile["band_correction"].(map[string]any); ok {
			if cells, ok := band["cells"].([]any); ok {
				note["band_cells_available"] = len(cells)
			}
		}
	}
	return note
}

func mapPath(profile map[string]any, outer, inner string) any {
	if m, ok := profile[outer].(map[string]any); ok {
		return m[inner]
	}
	return nil
}
