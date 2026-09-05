package fidelity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Tiere thresholds (global KL Core) and guard-profile scopes/statuses.
var KLAnchors = map[string]float64{"quality": 0.05, "balanced": 0.10, "compact": 0.15, "mini": 0.20}
var tierList = []string{"quality", "balanced", "compact", "mini"}
var validScopes = map[string]bool{"exact_model": true, "family": true, "architecture": true}
var validStatuses = map[string]bool{"candidate": true, "validated": true}

const refuseMessage = "Fidelity validation unavailable: No validated Same-top Guard Profile for this model/family. Options: --calibrate-fidelity --experimental-fidelity"

// RequireGuardProfile resolves a validated profile covering modelName for tier,
// raising the contract refusal (error) when none exists.
func RequireGuardProfile(modelName, tier, registryDir, sourceSHA256 string) (*GuardProfile, error) {
	tierKey := strings.TrimSpace(strings.ToLower(tier))
	if !validTier(tierKey) {
		return nil, fmt.Errorf("unknown fidelity tier: %q (expected %v)", tier, tierList)
	}
	profile, err := ResolveGuardProfile(modelName, registryDir, sourceSHA256)
	if err != nil {
		return nil, err
	}
	if profile == nil {
		return nil, fmt.Errorf("%s", refuseMessage)
	}
	return profile, nil
}

func validTier(t string) bool {
	for _, v := range tierList {
		if v == t {
			return true
		}
	}
	return false
}

type GuardProfile struct {
	ProfileID       string
	Version         int
	ScopeType       string
	ScopeIdentifier string
	Status          string
	KLAnchors       map[string]float64
	Floors          map[string]float64
	SourceSHA256    string
	Raw             map[string]any
}

func (g *GuardProfile) FloorFor(tier string) float64 { return g.Floors[tier] }

// canonicalContent serializes profile except profile_hash, sorted keys.
func canonicalContent(profile map[string]any) (string, error) {
	cp := make(map[string]any, len(profile))
	for k, v := range profile {
		if k == "profile_hash" {
			continue
		}
		cp[k] = v
	}
	data, err := json.Marshal(cp)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ProfileHash is sha256 of the canonical content.
func ProfileHash(profile map[string]any) (string, error) {
	cc, err := canonicalContent(profile)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(cc))
	return hex.EncodeToString(sum[:]), nil
}

func smallFloatEqual(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func hex64(x string) bool {
	if len(x) != 64 {
		return false
	}
	for _, c := range []byte(x) {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// ValidateGuardProfile enforces the guard-profile contract.
func ValidateGuardProfile(profile map[string]any) error {
	if profile["evaluator_contract"] != "eval-v1" {
		return fmt.Errorf("guard profile must pin evaluator_contract: eval-v1")
	}
	scope, _ := profile["scope"].(map[string]any)
	if scope == nil || !validScopes[fstring(scope["type"])] {
		return fmt.Errorf("scope.type must be one of {exact_model, family, architecture}")
	}
	if fstring(scope["identifier"]) == "" {
		return fmt.Errorf("scope.identifier is required")
	}
	if sha, ok := profile["source_sha256"].(string); ok && sha != "" && !hex64(sha) {
		return fmt.Errorf("source_sha256 must be a 64-hex lowercase digest")
	}
	if !validStatuses[fstring(profile["status"])] {
		return fmt.Errorf("status must be one of {candidate, validated}")
	}
	tiersRaw, ok := profile["tiers"].(map[string]any)
	if !ok {
		return fmt.Errorf("tiers must be a mapping")
	}
	for _, t := range tierList {
		if _, ok := tiersRaw[t]; !ok {
			return fmt.Errorf("tiers must define exactly %v", tierList)
		}
	}
	if len(tiersRaw) != len(tierList) {
		return fmt.Errorf("tiers must define exactly %v", tierList)
	}
	for _, t := range tierList {
		spec, _ := tiersRaw[t].(map[string]any)
		if spec == nil {
			return fmt.Errorf("tiers.%s must be a mapping", t)
		}
		anchor := ffloat(spec["kl_anchor"])
		if !smallFloatEqual(anchor, KLAnchors[t]) {
			return fmt.Errorf("tiers.%s.kl_anchor must be %v", t, KLAnchors[t])
		}
		floor := ffloat(spec["same_top_floor"])
		if !(floor > 0 && floor <= 1.0) {
			return fmt.Errorf("tiers.%s.same_top_floor out of range", t)
		}
	}
	if fstring(profile["guard_profile_id"]) == "" {
		return fmt.Errorf("guard_profile_id is required")
	}
	return nil
}

// LoadGuardProfile loads, validates, and returns a GuardProfile.
func LoadGuardProfile(path string) (*GuardProfile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var profile map[string]any
	if err := yaml.Unmarshal(raw, &profile); err != nil {
		return nil, fmt.Errorf("guard profile %s yaml: %v", path, err)
	}
	if err := ValidateGuardProfile(profile); err != nil {
		return nil, fmt.Errorf("guard profile %s: %v", path, err)
	}
	if expected, ok := profile["profile_hash"].(string); ok && expected != "" {
		got, _ := ProfileHash(profile)
		if got != expected {
			return nil, fmt.Errorf("guard profile %s fails its own profile_hash", path)
		}
	}
	scope, _ := profile["scope"].(map[string]any)
	tiersRaw, _ := profile["tiers"].(map[string]any)
	anchors := map[string]float64{}
	floors := map[string]float64{}
	for _, t := range tierList {
		spec, _ := tiersRaw[t].(map[string]any)
		anchors[t] = ffloat(spec["kl_anchor"])
		floors[t] = ffloat(spec["same_top_floor"])
	}
	g := &GuardProfile{
		ProfileID:       fstring(profile["guard_profile_id"]),
		Version:         1,
		ScopeType:       fstring(scope["type"]),
		ScopeIdentifier: fstring(scope["identifier"]),
		Status:          fstring(profile["status"]),
		KLAnchors:       anchors,
		Floors:          floors,
		SourceSHA256:    fstring(profile["source_sha256"]),
		Raw:             profile,
	}
	if ver, ok := profile["guard_profile_version"]; ok {
		g.Version = fint(ver)
	}
	if g.SourceSHA256 == "" {
		g.SourceSHA256 = ""
	}
	return g, nil
}

// ResolveGuardProfile finds the best validated profile covering modelName.
func ResolveGuardProfile(modelName, registryDir, sourceSHA256 string) (*GuardProfile, error) {
	entries, err := os.ReadDir(registryDir)
	if err != nil {
		return nil, nil // not a dir → no profile
	}
	var matches []*GuardProfile
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !(filepath.Ext(name) == ".yaml" || filepath.Ext(name) == ".yml") {
			continue
		}
		g, err := LoadGuardProfile(filepath.Join(registryDir, name))
		if err != nil {
			continue
		}
		if g.Status != "validated" {
			continue
		}
		if g.SourceSHA256 != "" {
			if sourceSHA256 == "" || g.SourceSHA256 != sourceSHA256 {
				continue
			}
		}
		switch g.ScopeType {
		case "exact_model":
			if g.ScopeIdentifier == modelName {
				matches = append(matches, g)
			}
		case "family":
			if len(modelName) >= len(g.ScopeIdentifier) && modelName[:len(g.ScopeIdentifier)] == g.ScopeIdentifier {
				matches = append(matches, g)
			}
		}
	}
	if len(matches) == 0 {
		return nil, nil
	}
	rank := map[string]int{"exact_model": 0, "family": 1, "architecture": 2}
	sort.SliceStable(matches, func(a, b int) bool {
		if rank[matches[a].ScopeType] != rank[matches[b].ScopeType] {
			return rank[matches[a].ScopeType] < rank[matches[b].ScopeType]
		}
		if matches[a].Version != matches[b].Version {
			return matches[a].Version > matches[b].Version
		}
		return matches[a].ProfileID < matches[b].ProfileID
	})
	return matches[0], nil
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
	}
	return 1
}
