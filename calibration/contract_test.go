package calibration

import (
	"math"
	"testing"
)

// TestLoadContractFrozenSHA pins the frozen contract: LoadContract must verify
// its own contract_sha256 over the canonical payload (the same canonical bytes
// upstream hashes), proving the Go canonical serialization matches Python.
func TestLoadContractFrozenSHA(t *testing.T) {
	contract, err := LoadContract("")
	if err != nil {
		t.Fatalf("LoadContract: %v", err)
	}
	if contract.SHA256 != "455338c52de7aa1b4d826ae759a69f8837476ffdb1c16e2d007a289a03c54d47" {
		t.Fatalf("contract_sha256 = %s, want the frozen digest", contract.SHA256)
	}
	if contract.EvaluatorContractSHA256() != "5ce78dee9d11e6dfe83416628d0459d462719c0ecddf194c53ea9629db243d7c" {
		t.Fatalf("evaluator_contract_sha256 mismatch")
	}
	if got := len(contract.LadderStandardPresets()); got != 12 {
		t.Fatalf("ladder presets = %d, want 12", got)
	}
}

func TestTrunc4(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{0.12345, 0.1234},
		{0.123456789, 0.1234},
		{0.99999, 0.9999},
		{0.5, 0.5},
		{0.9999999999999999, 0.9999},
	}
	for _, c := range cases {
		if got := Trunc4(c.in); got != c.want {
			t.Errorf("Trunc4(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestP05(t *testing.T) {
	// Linear-interpolated 5th percentile over 100 samples: index 4.95.
	values := make([]float64, 100)
	for i := range values {
		values[i] = float64(i)
	}
	got, err := P05(values)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-4.95) > 1e-12 {
		t.Fatalf("P05 = %v, want 4.95", got)
	}
	// single element returns it
	if got, err := P05([]float64{0.42}); err != nil || got != 0.42 {
		t.Fatalf("P05 single = %v, %v", got, err)
	}
}

// TestEvaluateObservationsValidated pins the promotion rule: n >= 3 window
// samples + a witness + no open failures -> validated on every tier.
func TestEvaluateObservationsValidated(t *testing.T) {
	contract, err := LoadContract("")
	if err != nil {
		t.Fatal(err)
	}
	// three observations per tier window, lowest-KL one below the anchor
	// (witness), all same_top well above any derived floor
	type row struct {
		id  string
		kl  float64
		top float64
	}
	rows := []row{
		{"q-1", 0.045, 0.99}, {"q-2", 0.049, 0.98}, {"q-3", 0.055, 0.97},
		{"b-1", 0.090, 0.96}, {"b-2", 0.099, 0.95}, {"b-3", 0.110, 0.94},
		{"c-1", 0.130, 0.93}, {"c-2", 0.149, 0.92}, {"c-3", 0.170, 0.91},
		{"m-1", 0.175, 0.90}, {"m-2", 0.199, 0.89}, {"m-3", 0.225, 0.88},
	}
	obs := make([]map[string]any, 0, len(rows))
	for i, r := range rows {
		obs = append(obs, map[string]any{
			"point_id": r.id, "size_bytes": i, "artifact_sha256": r.id,
			"macro_kl": r.kl, "same_top": r.top,
		})
	}
	eval, err := EvaluateObservations(obs, contract, nil)
	if err != nil {
		t.Fatal(err)
	}
	if eval["overall_status"] != "validated" {
		t.Fatalf("overall_status = %v, want validated (failures: %v)", eval["overall_status"], eval["open_failures"])
	}
}

func TestPoisonExcluded(t *testing.T) {
	contract, err := LoadContract("")
	if err != nil {
		t.Fatal(err)
	}
	if !contract.IsPoisonPreset("IQ2_XS") || !contract.IsPoisonPreset("Q3_K_S") {
		t.Fatal("poison presets not recognized")
	}
	if contract.IsPoisonPreset("Q4_K_M") {
		t.Fatal("Q4_K_M is not a poison preset")
	}
}
