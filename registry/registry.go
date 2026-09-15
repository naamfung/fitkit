// Package registry implements the Fidelity Registry v1 trust layer, mirroring
// upstream fit_gguf.registry (v0.3.3): which model-specific Calibration Bundles
// are trusted, keyed by source-weights SHA-256 plus the live eval-v1 digest —
// never by model name. verify here is structural/content integrity verification,
// NOT an official-authenticity attestation.
package registry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"fitting/fidelity"
)

// RegistrySchema is the registry index schema identifier.
const RegistrySchema = "fit.fidelity_registry.v1"

const evalContract = "eval-v1"

// LegacyBasis is the immutable grandfather basis for the two bootstrap models.
const LegacyBasis = "legacy_bootstrap_v0"

// CalibrationBasis is the fidelity-calibration-v1 basis for new entries.
const CalibrationBasis = "fidelity-calibration-v1"

// IndexName is the registry index file name inside the registry directory.
const IndexName = "fidelity-registry-v1.json"

// GrandfatheredSourceSHAs is the closed grandfather set: the only source
// digests that may carry calibration.basis = legacy_bootstrap_v0. Admission of
// NEW legacy entries is refused forever.
var GrandfatheredSourceSHAs = map[string]bool{
	"f95456457fededfaf9f51cd4739a00aada0795be372882844071cc19146634cd": true, // orcarouter
	"aa73aeb45870f7ebb9e5d523323b88468a2b3b613918e941bda439d8c1b59d42": true, // spark-x25-4b-abliterated
}

// RegistryError is raised when the registry fails structural, hash, or semantic
// checks.
type RegistryError struct{ Msg string }

func (e *RegistryError) Error() string { return e.Msg }

func registryErrf(format string, a ...any) *RegistryError {
	return &RegistryError{Msg: fmt.Sprintf(format, a...)}
}

func is64hex(v any) bool {
	s, ok := v.(string)
	if !ok || len(s) != 64 {
		return false
	}
	for _, c := range []byte(s) {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

func fstring(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// CanonicalJSONBytes serializes v as sorted-key compact JSON without HTML
// escaping — the canonical form every registry digest is taken over.
func CanonicalJSONBytes(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

func sha256Bytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// SHA256File computes the lowercase hex SHA-256 of a file's bytes.
func SHA256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// EntryDigest computes the entry digest over the canonical bytes of the entry
// minus its own entry_sha256 field.
func EntryDigest(entry map[string]any) (string, error) {
	payload := make(map[string]any, len(entry))
	for k, v := range entry {
		if k != "entry_sha256" {
			payload[k] = v
		}
	}
	canon, err := CanonicalJSONBytes(payload)
	if err != nil {
		return "", err
	}
	return sha256Bytes(canon), nil
}

func safeRelative(root, rel string) (string, error) {
	if rel == "" || strings.ContainsRune(rel, '\\') {
		// registry pins are written with forward slashes on every platform
		return "", registryErrf("invalid registry path: %q", rel)
	}
	candidate := filepath.FromSlash(rel)
	if filepath.IsAbs(candidate) {
		return "", registryErrf("registry paths must be relative, got: %q", rel)
	}
	resolved := filepath.Clean(filepath.Join(root, candidate))
	rootResolved := filepath.Clean(root)
	if resolved != rootResolved && !strings.HasPrefix(resolved, rootResolved+string(os.PathSeparator)) {
		return "", registryErrf("registry path escapes the registry root: %q", rel)
	}
	return resolved, nil
}

func readJSON(path string, what string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, registryErrf("cannot read %s %s: %v", what, path, err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, registryErrf("cannot read %s %s: %v", what, path, err)
	}
	return payload, nil
}

// Index is the loaded and validated registry index.
type Index struct {
	Raw     map[string]any
	Entries []map[string]any // each row: source_weights_sha256 / model_id / status / scope / entry_sha256
	Root    string
}

// LoadIndex loads and validates the registry index under root.
func LoadIndex(root string) (*Index, error) {
	indexPath := filepath.Join(root, "registry", IndexName)
	index, err := readJSON(indexPath, "registry index")
	if err != nil {
		return nil, err
	}
	if index["registry_schema"] != RegistrySchema {
		return nil, registryErrf("%s: unexpected registry_schema", indexPath)
	}
	if index["evaluator_contract"] != evalContract {
		return nil, registryErrf("%s: evaluator_contract must be %s", indexPath, evalContract)
	}
	if !is64hex(index["evaluator_contract_sha256"]) {
		return nil, registryErrf("%s: evaluator_contract_sha256 must be 64-hex", indexPath)
	}
	if index["evaluator_contract_sha256"] != fidelity.FrozenContractDigest {
		return nil, registryErrf("%s: evaluator_contract_sha256 does not match the live eval-v1 digest — the registry targets a different evaluator closure", indexPath)
	}
	rawEntries, ok := index["entries"].([]any)
	if !ok {
		return nil, registryErrf("%s: entries must be a list", indexPath)
	}
	seen := map[string]bool{}
	keys := make([][4]string, 0, len(rawEntries))
	var entries []map[string]any
	for _, item := range rawEntries {
		row, ok := item.(map[string]any)
		if !ok {
			return nil, registryErrf("%s: entry row must be an object", indexPath)
		}
		sha := fstring(row["source_weights_sha256"])
		if !is64hex(sha) {
			return nil, registryErrf("%s: entry with invalid source sha: %v", indexPath, sha)
		}
		if seen[sha] {
			return nil, registryErrf("%s: duplicate source sha in index: %s", indexPath, sha)
		}
		seen[sha] = true
		if !is64hex(row["entry_sha256"]) {
			return nil, registryErrf("%s: entry %s missing entry_sha256", indexPath, sha)
		}
		entries = append(entries, row)
		keys = append(keys, [4]string{sha, fstring(row["model_id"]), fstring(row["status"]), fstring(row["scope"])})
	}
	if !sort.SliceIsSorted(keys, func(a, b int) bool {
		for i := 0; i < 4; i++ {
			if keys[a][i] != keys[b][i] {
				return keys[a][i] < keys[b][i]
			}
		}
		return false
	}) {
		return nil, registryErrf("%s: entries must be deterministically sorted", indexPath)
	}
	return &Index{Raw: index, Entries: entries, Root: root}, nil
}

// LoadEntry loads and verifies one registry entry by source-weights digest.
func LoadEntry(root string, sourceSHA string, index *Index) (map[string]any, error) {
	if !is64hex(sourceSHA) {
		return nil, registryErrf("invalid source sha: %v", sourceSHA)
	}
	idx := index
	if idx == nil {
		var err error
		idx, err = LoadIndex(root)
		if err != nil {
			return nil, err
		}
	}
	var row map[string]any
	for _, r := range idx.Entries {
		if fstring(r["source_weights_sha256"]) == sourceSHA {
			row = r
			break
		}
	}
	if row == nil {
		return nil, registryErrf("Fidelity validation unavailable: no registry entry for these source weights")
	}
	entryPath, err := safeRelative(root, "registry/entries/"+sourceSHA+".json")
	if err != nil {
		return nil, err
	}
	entry, err := readJSON(entryPath, "registry entry")
	if err != nil {
		return nil, err
	}
	if entry["registry_schema"] != RegistrySchema {
		return nil, registryErrf("%s: unexpected registry_schema", entryPath)
	}
	if fstring(entry["source_weights_sha256"]) != sourceSHA {
		return nil, registryErrf("%s: entry source_weights_sha256 does not match its file name", entryPath)
	}
	if fstring(row["entry_sha256"]) != fstring(entry["entry_sha256"]) {
		return nil, registryErrf("%s: index entry_sha256 does not match the entry", entryPath)
	}
	digest, err := EntryDigest(entry)
	if err != nil {
		return nil, err
	}
	if digest != fstring(entry["entry_sha256"]) {
		return nil, registryErrf("%s: entry fails its own entry_sha256", entryPath)
	}
	return entry, nil
}

func loadPinnedJSON(root string, pin map[string]any, what string) (map[string]any, error) {
	path, err := safeRelative(root, fstring(pin["path"]))
	if err != nil {
		return nil, err
	}
	if !fileExists(path) {
		return nil, registryErrf("%s file missing: %v", what, pin["path"])
	}
	actual, err := SHA256File(path)
	if err != nil {
		return nil, err
	}
	if actual != fstring(pin["sha256"]) {
		return nil, registryErrf("%s sha256 mismatch: %v", what, pin["path"])
	}
	return readJSON(path, what)
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func checkCalibration(entry map[string]any, root string) error {
	cal, ok := entry["calibration"].(map[string]any)
	if !ok {
		return registryErrf("entry.calibration must be an object")
	}
	basis := fstring(cal["basis"])
	recordPin, _ := cal["record"].(map[string]any)
	if recordPin == nil || !is64hex(recordPin["sha256"]) {
		return registryErrf("entry.calibration.record must pin a sha256")
	}
	if _, err := loadPinnedJSON(root, recordPin, "calibration record"); err != nil {
		return err
	}
	switch basis {
	case LegacyBasis:
		if !GrandfatheredSourceSHAs[fstring(entry["source_weights_sha256"])] {
			return registryErrf("legacy_bootstrap_v0 is a closed grandfather set: new entries may not declare it")
		}
		if cal["contract"] != nil || cal["contract_sha256"] != nil {
			return registryErrf("legacy bootstrap entries must leave contract null")
		}
	case CalibrationBasis:
		if !is64hex(cal["contract_sha256"]) {
			return registryErrf("fidelity-calibration-v1 entries must pin contract_sha256")
		}
	default:
		return registryErrf("unknown calibration basis: %v", basis)
	}
	return nil
}

// VerifiedEntry is one successfully verified registry entry.
type VerifiedEntry struct {
	SourceWeightsSHA256 string
	ModelID             string
	Status              string
	Floors              map[string]float64
}

// VerifyRegistry runs full structural + hash + cross-object verification of the
// registry under root. Returns a report; any failure is returned as error.
func VerifyRegistry(root string) (map[string]any, error) {
	index, err := LoadIndex(root)
	if err != nil {
		return nil, err
	}
	liveDigest := fidelity.FrozenContractDigest
	var verified []VerifiedEntry
	for _, row := range index.Entries {
		sha := fstring(row["source_weights_sha256"])
		entry, err := LoadEntry(root, sha, index)
		if err != nil {
			return nil, err
		}
		if fstring(entry["entry_sha256"]) != fstring(row["entry_sha256"]) {
			return nil, registryErrf("entries/%s: index/entry digest drift", sha)
		}

		guardPin, _ := entry["guard_profile"].(map[string]any)
		if guardPin == nil {
			return nil, registryErrf("entries/%s: guard_profile must be a pin object", sha)
		}
		guardPath, err := safeRelative(root, fstring(guardPin["path"]))
		if err != nil {
			return nil, err
		}
		guardActual, err := SHA256File(guardPath)
		if err != nil {
			return nil, err
		}
		if guardActual != fstring(guardPin["sha256"]) {
			return nil, registryErrf("entries/%s: guard_profile sha256 mismatch", sha)
		}
		guard, err := fidelity.LoadGuardProfile(guardPath)
		if err != nil {
			return nil, err
		}

		manifestPin, _ := entry["reference_manifest"].(map[string]any)
		if manifestPin == nil {
			return nil, registryErrf("entries/%s: reference_manifest must be a pin object", sha)
		}
		manifest, err := loadPinnedJSON(root, manifestPin, "reference manifest")
		if err != nil {
			return nil, err
		}

		// cross-object bundle closure
		if guard.SourceSHA256 != sha {
			return nil, registryErrf("entries/%s: guard source_sha256 %s does not bind these source weights", sha, guard.SourceSHA256)
		}
		if fstring(manifest["source_bf16_gguf_sha256"]) != sha {
			return nil, registryErrf("entries/%s: manifest source_bf16_gguf_sha256 does not bind these source weights", sha)
		}
		manifestTokenizer := fstring(manifest["tokenizer_sha256"])
		entryTokenizer := fstring(entry["tokenizer_sha256"])
		if is64hex(manifestTokenizer) && manifestTokenizer != entryTokenizer {
			return nil, registryErrf("entries/%s: tokenizer_sha256 disagrees between entry and manifest", sha)
		}
		if fstring(entry["evaluator_contract_sha256"]) != liveDigest {
			return nil, registryErrf("entries/%s: evaluator_contract_sha256 does not match the live eval-v1 digest", sha)
		}
		manifestHash := fstring(manifest["evaluator_contract_hash"])
		if manifestHash == "" {
			return nil, registryErrf("entries/%s: manifest must pin evaluator_contract_hash", sha)
		}
		if manifestHash != liveDigest {
			return nil, registryErrf("entries/%s: manifest evaluator_contract_hash does not match the live eval-v1 digest", sha)
		}
		if fstring(guardPin["sha256"]) != "" && !is64hex(guardPin["sha256"]) {
			return nil, registryErrf("entries/%s: guard pin sha must be 64-hex", sha)
		}
		if err := checkCalibration(entry, root); err != nil {
			return nil, err
		}
		if fstring(entry["scope"]) != "exact_model" {
			return nil, registryErrf("entries/%s: scope is not resolvable in registry v1 (family/architecture are reserved)", sha)
		}
		status := fstring(entry["status"])
		if status != "validated" && status != "candidate" {
			return nil, registryErrf("entries/%s: invalid status %q", sha, status)
		}
		verified = append(verified, VerifiedEntry{
			SourceWeightsSHA256: sha,
			ModelID:             fstring(entry["model_id"]),
			Status:              status,
			Floors:              guard.Floors,
		})
	}
	return map[string]any{
		"registry_schema":           RegistrySchema,
		"evaluator_contract_digest": liveDigest,
		"entries_verified":          verified,
		"note": "structural/content integrity only — NOT an official-authenticity attestation (official trust root = the FIT release)",
	}, nil
}

// ListEntries returns the index rows for the CLI list command.
func ListEntries(root string) ([]map[string]any, error) {
	index, err := LoadIndex(root)
	if err != nil {
		return nil, err
	}
	return index.Entries, nil
}

// ValidateBundle performs the read-only admission check for a Calibration
// Bundle: draft registry entry structure, digest, calibration-basis admission
// rules, and that the referenced guard/manifest/record files exist and
// hash-match. Does NOT write to the official registry.
func ValidateBundle(bundleDir string) (map[string]any, error) {
	entryPath := filepath.Join(bundleDir, "registry-entry.json")
	entry, err := readJSON(entryPath, "registry entry")
	if err != nil {
		return nil, err
	}
	if entry["registry_schema"] != RegistrySchema {
		return nil, registryErrf("%s: unexpected registry_schema", entryPath)
	}
	var problems []string
	sha := fstring(entry["source_weights_sha256"])
	if !is64hex(sha) {
		problems = append(problems, "source_weights_sha256 must be 64-hex")
	}
	digest, err := EntryDigest(entry)
	if err != nil {
		return nil, err
	}
	if digest != fstring(entry["entry_sha256"]) {
		problems = append(problems, "entry fails its own entry_sha256")
	}
	cal, _ := entry["calibration"].(map[string]any)
	basis := fstring(cal["basis"])
	if basis == LegacyBasis && !GrandfatheredSourceSHAs[sha] {
		problems = append(problems, "legacy_bootstrap_v0 admission is closed to new sources")
	}
	if basis == CalibrationBasis && !is64hex(cal["contract_sha256"]) {
		problems = append(problems, "calibration contract_sha256 must be 64-hex")
	}
	for _, key := range []string{"guard_profile", "reference_manifest"} {
		pin, _ := entry[key].(map[string]any)
		if pin == nil {
			continue
		}
		target := filepath.Join(bundleDir, filepath.FromSlash(fstring(pin["path"])))
		actual, err := SHA256File(target)
		if err != nil || actual != fstring(pin["sha256"]) {
			problems = append(problems, key+": referenced file missing or hash mismatch")
		}
	}
	if cal != nil {
		recordPin, _ := cal["record"].(map[string]any)
		if recordPin != nil {
			target := filepath.Join(bundleDir, filepath.FromSlash(fstring(recordPin["path"])))
			actual, err := SHA256File(target)
			if err != nil || actual != fstring(recordPin["sha256"]) {
				problems = append(problems, "calibration: referenced file missing or hash mismatch")
			}
		}
	}
	if len(problems) > 0 {
		return nil, registryErrf("bundle rejected: %s", strings.Join(problems, "; "))
	}
	return map[string]any{
		"admissible":           true,
		"model_id":             fstring(entry["model_id"]),
		"source_weights_sha256": sha,
		"status":               fstring(entry["status"]),
		"note": "bundle is admissible — official promotion is a maintainer/release workflow, not a CLI flag",
	}, nil
}
