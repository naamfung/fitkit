package pipeline

import (
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

const bytesPerMiB = 1024 * 1024

// Assignment is one per-tensor quantize assignment parsed from a dry-run log.
type Assignment struct {
	Ordinal      int
	TotalTensors int
	Name         string
	Shape        []int64
	SrcType      string
	DstType      string
	IsQuantized  bool
	OrigBytes    int
	NewBytes     int
}

// Recipe is a parsed dry-run result.
type Recipe struct {
	Tensors           []Assignment
	TotalTensors      int
	ReportedOrigBytes int
	ReportedNewBytes  int
}

func (r *Recipe) TensorMap() map[string]*Assignment {
	m := make(map[string]*Assignment, len(r.Tensors))
	for i := range r.Tensors {
		m[r.Tensors[i].Name] = &r.Tensors[i]
	}
	return m
}

// mibToBytes converts a printed MiB value (e.g. "96.000") to bytes using
// Decimal rounding half-up, matching Python's mib_to_bytes.
func mibToBytes(mib string) int64 {
	r, ok := new(big.Rat).SetString(strings.TrimSpace(mib))
	if !ok {
		// fallback: parse as float
		f, _ := strconv.ParseFloat(strings.TrimSpace(mib), 64)
		r = new(big.Rat).SetFloat64(f)
	}
	r.Mul(r, big.NewRat(bytesPerMiB, 1))
	num := new(big.Int).Set(r.Num())
	den := r.Denom()
	q, rem := num.QuoRem(num, den, new(big.Int))
	// roll half up (positive values)
	if rem.Lsh(absBigInt(rem), 1).Cmp(den) >= 0 {
		q.Add(q, big.NewInt(1))
	}
	return q.Int64()
}

func absBigInt(x *big.Int) *big.Int {
	return new(big.Int).Abs(x)
}

// stripANSI mirrors Python's ANSI_ESCAPE_RE.sub("", line).
func stripANSI(s string) string {
	b := make([]byte, 0, len(s))
	i := 0
	for i < len(s) {
		if s[i] == 0x1b {
			i++
			if i < len(s) && s[i] == '[' {
				i++
				for i < len(s) && !(s[i] >= 0x40 && s[i] <= 0x7e) {
					i++
				}
				if i < len(s) {
					i++
				}
			}
			continue
		}
		b = append(b, s[i])
		i++
	}
	return string(b)
}

// ParseRecipe parses llama.cpp dry-run output into a Recipe.
func ParseRecipe(logText string) (*Recipe, error) {
	rec := &Recipe{}
	seen := map[int]bool{}
	expectedTotal := -1

	for _, rawLine := range strings.Split(logText, "\n") {
		line := strings.TrimSpace(stripANSI(rawLine))
		if line == "" {
			continue
		}
		m := tensorLineRE.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		ord, err1 := strconv.Atoi(m[1])
		total, err2 := strconv.Atoi(m[2])
		if err1 != nil || err2 != nil {
			return nil, fmt.Errorf("dryrun: bad ordinal/total: %s", line)
		}
		if expectedTotal == -1 {
			expectedTotal = total
		} else if total != expectedTotal {
			return nil, fmt.Errorf("dryrun: inconsistent total %d vs %d", total, expectedTotal)
		}
		if seen[ord] {
			return nil, fmt.Errorf("dryrun: duplicate ordinal %d", ord)
		}
		seen[ord] = true

		name := m[3]
		srcType := m[5]
		sizePart := m[6]

		var shapeStr string
		// m[4] = shape like "  4096,  12288,      1,      1"
		shapeStr = m[4]
		parts := strings.Split(shapeStr, ",")
		shape := make([]int64, 0, len(parts))
		for _, p := range parts {
			d, err := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
			if err != nil || d <= 0 {
				return nil, fmt.Errorf("dryrun: bad shape %q for %s", shapeStr, name)
			}
			shape = append(shape, d)
		}

		dst := srcType
		isQuantized := false
		var origMib, newMib string
		if qm := quantizedSizeRE.FindStringSubmatch(sizePart); qm != nil {
			dst = qm[3]
			origMib, newMib = qm[1], qm[2]
			isQuantized = true
		} else if um := unchangedSizeRE.FindStringSubmatch(sizePart); um != nil {
			origMib, newMib = um[1], um[1]
		} else {
			return nil, fmt.Errorf("dryrun: malformed size part %q", sizePart)
		}

		rec.Tensors = append(rec.Tensors, Assignment{
			Ordinal:      ord,
			TotalTensors: total,
			Name:         name,
			Shape:        shape,
			SrcType:      srcType,
			DstType:      dst,
			IsQuantized:  isQuantized,
			OrigBytes:    int(mibToBytes(origMib)),
			NewBytes:     int(mibToBytes(newMib)),
		})
	}

	if len(rec.Tensors) == 0 {
		return nil, fmt.Errorf("dryrun: no tensor lines found")
	}
	if expectedTotal != -1 && len(rec.Tensors) != expectedTotal {
		return nil, fmt.Errorf("dryrun: parsed %d tensors, expected %d", len(rec.Tensors), expectedTotal)
	}

	// sort by ordinal
	sortAssignments(rec.Tensors)
	for ord := 1; ord <= len(rec.Tensors); ord++ {
		if rec.Tensors[ord-1].Ordinal != ord {
			return nil, fmt.Errorf("dryrun: missing ordinal %d", ord)
		}
	}
	rec.TotalTensors = expectedTotal
	rec.ReportedOrigBytes = 0
	rec.ReportedNewBytes = 0
	for _, t := range rec.Tensors {
		rec.ReportedOrigBytes += t.OrigBytes
		rec.ReportedNewBytes += t.NewBytes
	}
	return rec, nil
}
