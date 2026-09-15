// plancheck validates the Go Plan() against a frozen Python analysis.json and
// its recorded plan-plan.json (predicted_size_bytes).  It does NOT call
// Python — only llama-quantize for the oracle dry-run.
//
// Usage:
//
//	plancheck <analysis.json> <plan-plan.json> <target-bytes> [policy]
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"fitting/pipeline"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintln(os.Stderr, "usage: plancheck <analysis.json> <plan-plan.json> <target-bytes> [policy]")
		os.Exit(2)
	}
	analysisPath := os.Args[1]
	planJSON := os.Args[2]
	target, err := strconv.Atoi(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "bad target:", err)
		os.Exit(2)
	}
	policy := "balanced"
	if len(os.Args) >= 5 {
		policy = os.Args[4]
	}

	outPrefix := os.TempDir() + string(os.PathSeparator) + "fitgo-plancheck"
	plan, pred, err := pipeline.Plan(analysisPath, outPrefix, target, policy, "auto", "", "")
	if err != nil {
		fmt.Fprintln(os.Stderr, "plan error:", err)
		os.Exit(1)
	}

	// read recorded expected
	raw, _ := os.ReadFile(planJSON)
	var rec struct {
		Predicted int `json:"predicted_size_bytes"`
		Lower     int `json:"lower_size_bytes"`
		Recipe    string `json:"recipe_path"`
	}
	_ = json.Unmarshal(raw, &rec)

	fmt.Printf("go plan   : predicted=%d metadata=%d payload=%d padding=%d selected=%d\n",
		pred.TotalBytes, pred.MetadataBytes, pred.TensorPayload, pred.TensorPadding, len(plan.Selected))
	fmt.Printf("py record : predicted=%d\n", rec.Predicted)

	if rec.Predicted == 0 || pred.TotalBytes == rec.Predicted {
		fmt.Println("RESULT   : MATCH")
	} else {
		fmt.Printf("RESULT   : MISMATCH (delta %d)\n", pred.TotalBytes-rec.Predicted)
		os.Exit(1)
	}
}