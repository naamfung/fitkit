package gguf

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Parser for llama.cpp dry-run quantization logs.  We only need the
// per-tensor destination qtype for size prediction; the many validation
// checks of the Python parser guard reproducibility, but do not change the
// predicted bytes, so the Go port keeps the parse lean but faithful.

var (
	reAnsi      = regexp.MustCompile(`\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])`)
	reTensor    = regexp.MustCompile(`^(?:.*?\s+)?\[\s*(\d+)\s*/\s*(\d+)\s*\]\s+(\S+)\s+-\s+\[([\d\s,]+)\],\s+type\s*=\s*([A-Za-z0-9_]+),\s+size\s*=\s*(.+)$`)
	reQuantized = regexp.MustCompile(`^\s*(\d+(?:\.\d+)?)\s*MiB\s*->\s*(\d+(?:\.\d+)?)\s*MiB\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*$`)
	reUnchanged = regexp.MustCompile(`^\s*(\d+(?:\.\d+)?)\s*MiB\s*$`)
)

// DstTypeMap parses a dry-run / oracle log and returns tensor name → dst qtype
// (lowercased).  Tensors that are unchanged keep their source type.
func DstTypeMap(logText string) (map[string]string, error) {
	result := make(map[string]string)
	total := -1

	for _, rawLine := range strings.Split(logText, "\n") {
		line := strings.TrimSpace(reAnsi.ReplaceAllString(rawLine, ""))
		if line == "" {
			continue
		}
		if !reCandidateMatcher(reTensor, line) {
			// model/quant size summary or noise lines; skip
			continue
		}
		m := reTensor.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		// m[1]=ordinal, m[2]=total, m[3]=name, m[4]=shape, m[5]=src_type, m[6]=size_part
		curTotal, _ := strconv.Atoi(m[2])
		if total == -1 {
			total = curTotal
		} else if curTotal != total {
			return nil, fmt.Errorf("dryrun: inconsistent total tensor count %d vs %d", curTotal, total)
		}

		name := m[3]
		srcType := m[5]
		sizePart := m[6]

		dst := srcType
		if qm := reQuantized.FindStringSubmatch(sizePart); qm != nil {
			dst = qm[3]
		} else if !reUnchanged.MatchString(sizePart) {
			return nil, fmt.Errorf("dryrun: malformed tensor size part %q", sizePart)
		}
		result[name] = toLower(dst)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("dryrun: no tensor lines found in log")
	}
	if total != -1 && len(result) != total {
		return nil, fmt.Errorf("dryrun: parsed %d tensors, expected %d", len(result), total)
	}
	return result, nil
}

// reCandidateMatcher reports whether the stripped line looks like a candidate
// tensor line, mirroring Python's `CANDIDATE_LINE_RE.search(line)` followed by
// a strict full match.  We simply attempt the full match.
func reCandidateMatcher(_ *regexp.Regexp, line string) bool {
	return reTensor.MatchString(line)
}

// ParseDstTypeMapFromFile is a convenience for reading a log file.
func ParseDstTypeMapFromFile(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("dryrun: read log %s: %v", path, err)
	}
	return DstTypeMap(string(data))
}
