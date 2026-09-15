package fidelity

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// FrozenContractDigest is the frozen eval-v1 contract digest (rc2) pinned by
// experiments/2026-09-02-eval-v1/FREEZE.json. The live contract embedded in
// fit_gguf.eval.EVAL_V1 hashes to exactly this value.
const FrozenContractDigest = "5ce78dee9d11e6dfe83416628d0459d462719c0ecddf194c53ea9629db243d7c"

type EvalProvenanceError struct{ msg string }

func (e *EvalProvenanceError) Error() string { return e.msg }
func evalProvenanceErrf(format string, a ...any) *EvalProvenanceError {
	return &EvalProvenanceError{msg: fmt.Sprintf(format, a...)}
}

// EvalProvenance is a verified binding between local eval inputs and the
// frozen eval-v1 closure.
type EvalProvenance struct {
	FreezePath                  string
	ContractDigest              string
	ReferenceManifestPath       string
	ReferenceKldSHA256          map[string]string
	CorpusSHA256                map[string]string
	SourceBF16GGUFSHA256        string
	ReferenceManifestFileSHA256 string
}

// SHA256File computes the sha256 hex digest of a file.
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

// VerifyEvalV1Provenance verifies local evaluation inputs against the frozen
// eval-v1 closure, fail-closed. Returns the verified binding.
func VerifyEvalV1Provenance(refsDir, evalDataDir, freezePath, referenceManifestPath, sourceSHA256 string) (*EvalProvenance, error) {
	freezeRaw, err := os.ReadFile(freezePath)
	if err != nil {
		return nil, evalProvenanceErrf("cannot read eval-v1 freeze file %s: %v", freezePath, err)
	}
	var freeze map[string]any
	if err := json.Unmarshal(freezeRaw, &freeze); err != nil {
		return nil, evalProvenanceErrf("cannot read eval-v1 freeze file %s: %v", freezePath, err)
	}
	if freeze["status"] != "FROZEN" {
		return nil, evalProvenanceErrf("%s: eval contract is not FROZEN", freezePath)
	}
	frozenDigest := fstring(freeze["final_contract_digest"])
	if frozenDigest != FrozenContractDigest {
		return nil, evalProvenanceErrf("evaluator contract drift: live digest %s != frozen %s (%s)", FrozenContractDigest, frozenDigest, freezePath)
	}

	manifestRaw, err := os.ReadFile(referenceManifestPath)
	if err != nil {
		return nil, evalProvenanceErrf("cannot read reference manifest %s: %v", referenceManifestPath, err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(manifestRaw, &manifest); err != nil {
		return nil, evalProvenanceErrf("cannot read reference manifest %s: %v", referenceManifestPath, err)
	}
	if manifest["manifest_schema"] != "fit.eval_reference_manifest.v1" {
		return nil, evalProvenanceErrf("%s: unexpected reference manifest schema", referenceManifestPath)
	}
	domains, _ := manifest["domains"].(map[string]any)
	if domains == nil || len(domains) != len(Domains) {
		return nil, evalProvenanceErrf("%s: reference manifest must pin exactly %v", referenceManifestPath, Domains)
	}
	if fstring(manifest["evaluator_contract_hash"]) != frozenDigest {
		return nil, evalProvenanceErrf("%s: evaluator_contract_hash is not pinned to the frozen contract %s", referenceManifestPath, frozenDigest)
	}
	manifestSHA, err := SHA256File(referenceManifestPath)
	if err != nil {
		return nil, err
	}
	// freeze_conditions.reference_regeneration.manifest_sha256_prefix is a
	// historical v0.2 bootstrap record and is NOT enforced (0.3.2): a freeze
	// states how to measure and must cover any model, while per-model trust is
	// the Fidelity Registry's job (it pins each released manifest by full
	// SHA-256, keyed by source weights).
	sourcePin := fstring(manifest["source_bf16_gguf_sha256"])
	if sourcePin == "" || len(sourcePin) != 64 {
		return nil, evalProvenanceErrf("%s: missing/invalid source_bf16_gguf_sha256 — references must be bound to the generating weights", referenceManifestPath)
	}
	if sourceSHA256 != "" && !strings.EqualFold(sourcePin, sourceSHA256) {
		return nil, evalProvenanceErrf("%s: source_bf16_gguf_sha256 %s != supplied weights digest %s — the references belong to different weights", referenceManifestPath, sourcePin, sourceSHA256)
	}

	kldShas := map[string]string{}
	corpusShas := map[string]string{}
	for _, domain := range sortedCopy(Domains) {
		pin, _ := domains[domain].(map[string]any)
		kldFile := filepath.Join(refsDir, "bf16-"+domain+".kld")
		if !fileExists(kldFile) {
			return nil, evalProvenanceErrf("missing reference file: %s", kldFile)
		}
		actual, err := SHA256File(kldFile)
		if err != nil {
			return nil, err
		}
		if actual != fstring(pin["reference_kld_sha256"]) {
			return nil, evalProvenanceErrf("%s: sha256 %s != frozen reference %s", kldFile, actual, fstring(pin["reference_kld_sha256"]))
		}
		kldShas[domain] = actual

		sliceFile := filepath.Join(evalDataDir, "kl-eval-"+SLICE_SUFFIX[domain]+".txt")
		if !fileExists(sliceFile) {
			return nil, evalProvenanceErrf("missing evaluation slice: %s", sliceFile)
		}
		actual, err = SHA256File(sliceFile)
		if err != nil {
			return nil, err
		}
		if actual != fstring(pin["corpus_sha256"]) {
			return nil, evalProvenanceErrf("%s: sha256 %s != frozen corpus %s", sliceFile, actual, fstring(pin["corpus_sha256"]))
		}
		corpusShas[domain] = actual
	}

	return &EvalProvenance{
		FreezePath:                  freezePath,
		ContractDigest:              FrozenContractDigest,
		ReferenceManifestPath:       referenceManifestPath,
		ReferenceKldSHA256:          kldShas,
		CorpusSHA256:                corpusShas,
		SourceBF16GGUFSHA256:        sourcePin,
		ReferenceManifestFileSHA256: manifestSHA,
	}, nil
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func sortedCopy(a []string) []string {
	out := append([]string(nil), a...)
	sort.Strings(out)
	return out
}
