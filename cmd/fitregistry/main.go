// Command fitregistry implements the Fidelity Registry v1 read-only product
// CLI: list / show / verify / validate, mirroring upstream `fit registry`
// (v0.3.3). Official registry changes are a maintainer/release workflow.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"fitting/registry"
)

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "fitregistry: "+format+"\n", a...)
	os.Exit(2)
}

func main() {
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		fmt.Fprintln(out, "fitregistry — Fidelity Registry v1 (read-only): list | show | verify | validate")
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Usage:")
		fmt.Fprintln(out, "  fitregistry -root <dir> list")
		fmt.Fprintln(out, "  fitregistry -root <dir> show <64-hex source sha | model id>")
		fmt.Fprintln(out, "  fitregistry -root <dir> verify")
		fmt.Fprintln(out, "  fitregistry validate <bundle-dir>")
		fmt.Fprintln(out)
		fmt.Fprintln(out, "-root is the package directory containing registry/ and profiles/guard/.")
		flag.PrintDefaults()
	}
	root := flag.String("root", ".", "Package directory containing registry/ and profiles/guard/")
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(2)
	}
	cmd := args[0]

	switch cmd {
	case "list":
		entries, err := registry.ListEntries(*root)
		if err != nil {
			fail("list: %v", err)
		}
		fmt.Printf("%-64s  %-24s  %-12s  %s\n", "source_weights_sha256", "model_id", "status", "scope")
		for _, e := range entries {
			fmt.Printf("%-64s  %-24s  %-12s  %s\n",
				rawStr(e["source_weights_sha256"]), rawStr(e["model_id"]),
				rawStr(e["status"]), rawStr(e["scope"]))
		}
	case "show":
		if len(args) < 2 {
			fail("show needs a target: <64-hex source sha | model id>")
		}
		target := strings.ToLower(args[1])
		index, err := registry.LoadIndex(*root)
		if err != nil {
			fail("show: %v", err)
		}
		var sha string
		for _, e := range index.Entries {
			if strings.EqualFold(rawStr(e["source_weights_sha256"]), target) ||
				strings.EqualFold(rawStr(e["model_id"]), target) {
				sha = rawStr(e["source_weights_sha256"])
				break
			}
		}
		if sha == "" {
			fail("show: no registry entry for %q", target)
		}
		entry, err := registry.LoadEntry(*root, sha, index)
		if err != nil {
			fail("show: %v", err)
		}
		fmt.Printf("source_weights_sha256: %s\n", rawStr(entry["source_weights_sha256"]))
		fmt.Printf("model_id:              %s\n", rawStr(entry["model_id"]))
		fmt.Printf("status:                %s\n", rawStr(entry["status"]))
		fmt.Printf("scope:                 %s\n", rawStr(entry["scope"]))
		fmt.Printf("entry_sha256:          %s\n", rawStr(entry["entry_sha256"]))
		if pin, ok := entry["guard_profile"].(map[string]any); ok {
			fmt.Printf("guard_profile:         %s (%s)\n", rawStr(pin["path"]), rawStr(pin["sha256"]))
		}
		if pin, ok := entry["reference_manifest"].(map[string]any); ok {
			fmt.Printf("reference_manifest:    %s (%s)\n", rawStr(pin["path"]), rawStr(pin["sha256"]))
		}
	case "verify":
		report, err := registry.VerifyRegistry(*root)
		if err != nil {
			fail("verify: %v", err)
		}
		verified, _ := report["entries_verified"].([]registry.VerifiedEntry)
		fmt.Printf("registry_schema:          %s\n", report["registry_schema"])
		fmt.Printf("evaluator_contract_digest: %s\n", report["evaluator_contract_digest"])
		fmt.Printf("entries verified: %d\n", len(verified))
		for _, v := range verified {
			fmt.Printf("  %-64s  %-24s  %s\n", v.SourceWeightsSHA256, v.ModelID, v.Status)
		}
	case "validate":
		if len(args) < 2 {
			fail("validate needs a bundle directory")
		}
		admission, err := registry.ValidateBundle(args[1])
		if err != nil {
			fail("validate: %v", err)
		}
		fmt.Printf("admissible: %v\n", admission["admissible"])
		fmt.Printf("model_id:   %s\n", rawStr(admission["model_id"]))
		fmt.Printf("status:     %s\n", rawStr(admission["status"]))
		fmt.Printf("note:       %s\n", rawStr(admission["note"]))
	default:
		fail("unknown command %q (expected list|show|verify|validate)", cmd)
	}
}

func rawStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}
