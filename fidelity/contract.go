package fidelity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
)

// eval-v1 constants.
const (
	ContractID     = "eval-v1"
	ContractSchema = "fit.evaluator_contract.v1"
	DomainWeight   = 0.2
)

// Domains is the frozen five-domain corpus (eval-v1).
var Domains = []string{"wiki_test", "wiki_valid", "chinese", "code", "agent_chat"}

// SLICE_SUFFIX maps domain -> eval slice filename suffix.
var SLICE_SUFFIX = map[string]string{
	"wiki_test": "64k", "wiki_valid": "valid-64k", "chinese": "cn-64k",
	"code": "code-64k", "agent_chat": "agent-64k",
}

// LoadContract loads a contract dictionary from a canonical JSON file.
func LoadContract(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("contract: %v", err)
	}
	return doc, nil
}

// CanonicalJSON serializes deterministic JSON (sorted keys, no extra space).
func CanonicalJSON(obj any) string {
	data, _ := json.Marshal(obj) // map[string]any marshals sorted keys already
	return string(data)
}

// ContractDigest hashes the canonical JSON of a contract.
func ContractDigest(contract map[string]any) string {
	sum := sha256.Sum256([]byte(CanonicalJSON(contract)))
	return hex.EncodeToString(sum[:])
}

// ValidateContract enforces the contract invariants.
func ValidateContract(c map[string]any) error {
	if c["schema"] != ContractSchema {
		return fmt.Errorf("contract: schema %v != %s", c["schema"], ContractSchema)
	}
	if c["contract_id"] != ContractID {
		return fmt.Errorf("contract: contract_id %v != %s", c["contract_id"], ContractID)
	}
	weights, _ := c["metrics"].(map[string]any)["domain_weights"].(map[string]any)
	if weights == nil {
		return fmt.Errorf("contract: missing domain_weights")
	}
	if len(weights) != len(Domains) {
		return fmt.Errorf("contract: domain weights must cover %v", Domains)
	}
	total := 0.0
	for _, w := range weights {
		total += toF(w)
	}
	if total < 1.0-1e-9 || total > 1.0+1e-9 {
		return fmt.Errorf("contract: domain weights must sum to 1.0, got %v", total)
	}
	return nil
}

func toF(v any) float64 {
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
