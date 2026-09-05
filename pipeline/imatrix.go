package pipeline

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	gguf "fitgo/gguf"
)

const (
	inSum2Suffix = ".in_sum2"
	countsSuffix = ".counts"
)

type ImatrixTensorProfile struct {
	Name               string
	Block              int
	Role               string
	Width              int
	CountValues        int
	CountMin           int
	CountMax           int
	CountSum           int
	Mean               float64
	RMS                float64
	Stddev             float64
	Minimum            float64
	P50                float64
	P95                float64
	P99                float64
	Maximum            float64
	NonzeroFraction    float64
	GlobalRelativeMean float64
	RoleRelativeMean   float64
	GlobalPercentile   float64
	RolePercentile     float64
}

type ImatrixProfile struct {
	SchemaVersion int
	SourceFile    string
	Datasets      []string
	ChunkCount    int
	ChunkSize     int
	Entries       []ImatrixTensorProfile
}

func (p *ImatrixProfile) EntryMap() map[string]*ImatrixTensorProfile {
	m := make(map[string]*ImatrixTensorProfile, len(p.Entries))
	for i := range p.Entries {
		m[p.Entries[i].Name] = &p.Entries[i]
	}
	return m
}

func neumaierSum(v []float64) float64 {
	var sum, comp float64
	for _, x := range v {
		t := sum + x
		if math.Abs(sum) >= math.Abs(x) {
			comp += (sum - t) + x
		} else {
			comp += (x - t) + sum
		}
		sum = t
	}
	return sum + comp
}

func percentile(sorted []float64, fraction float64) float64 {
	pos := float64(len(sorted)-1) * fraction
	lower := math.Floor(pos)
	upper := math.Ceil(pos)
	if lower == upper {
		return sorted[int(lower)]
	}
	w := pos - lower
	return sorted[int(lower)]*(1-w) + sorted[int(upper)]*w
}

func median(values []float64) float64 {
	s := append([]float64(nil), values...)
	sort.Float64s(s)
	return percentile(s, 0.5)
}

func rankPercentiles(values []float64) []float64 {
	n := len(values)
	res := make([]float64, n)
	if n == 1 {
		res[0] = 1.0
		return res
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		if values[order[a]] == values[order[b]] {
			return a < b
		}
		return values[order[a]] < values[order[b]]
	})
	processed := make([]bool, n)
	start := 0
	for start < n {
		end := start + 1
		for end < n && values[order[end]] == values[order[start]] {
			end++
		}
		avgRank := (float64(start) + float64(end) - 1) / 2
		p := avgRank / float64(n-1)
		for pos := start; pos < end; pos++ {
			idx := order[pos]
			res[idx] = p
			processed[idx] = true
		}
		start = end
	}
	return res
}

func (p *ImatrixProfile) maxBlock() int {
	mx := 0
	for _, e := range p.Entries {
		if e.Block > mx {
			mx = e.Block
		}
	}
	return mx
}

func AutoBlockSpan(profile *ImatrixProfile) int {
	return (profile.maxBlock() + 1 + 3) / 4
}

// LoadImatrixProfile loads and summarizes a canonical GGUF imatrix.
func LoadImatrixProfile(path string) (*ImatrixProfile, error) {
	layout, err := gguf.ReadLayout(path)
	if err != nil {
		return nil, err
	}
	fields := map[string]gguf.Field{}
	for _, f := range layout.Fields {
		fields[f.Key] = f
	}
	generalType, ok := fields["general.type"]
	if !ok || generalType.ScalarValue != "imatrix" {
		return nil, fmt.Errorf("pipeline: %s is not marked as an imatrix", path)
	}

	var datasets []string
	if dv, ok := fields["imatrix.datasets"]; ok {
		if sa, isArr := dv.ScalarValue.([]string); isArr {
			datasets = sa
		}
	}
	ccf, ok1 := fields["imatrix.chunk_count"]
	csf, ok2 := fields["imatrix.chunk_size"]
	if !ok1 || !ok2 || len(datasets) == 0 {
		return nil, fmt.Errorf("pipeline: imatrix metadata is incomplete")
	}
	chunkCount := toInt(ccf.ScalarValue)
	chunkSize := toInt(csf.ScalarValue)
	if chunkCount <= 0 || chunkSize <= 0 {
		return nil, fmt.Errorf("pipeline: imatrix chunk metadata must be positive")
	}

	// group tensors into sums/counts pairs by base name
	type pair struct{ sums, counts *gguf.TensorInfo }
	pairs := map[string]*pair{}
	for i := range layout.Tensors {
		t := &layout.Tensors[i]
		switch {
		case strings.HasSuffix(t.Name, inSum2Suffix):
			base := strings.TrimSuffix(t.Name, inSum2Suffix)
			pr, ok := pairs[base]
			if !ok {
				pr = &pair{}
				pairs[base] = pr
			}
			pr.sums = t
		case strings.HasSuffix(t.Name, countsSuffix):
			base := strings.TrimSuffix(t.Name, countsSuffix)
			pr, ok := pairs[base]
			if !ok {
				pr = &pair{}
				pairs[base] = pr
			}
			pr.counts = t
		default:
			return nil, fmt.Errorf("pipeline: unexpected imatrix tensor suffix: %s", t.Name)
		}
	}

	keys := make([]string, 0, len(pairs))
	for k := range pairs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []ImatrixTensorProfile
	for _, name := range keys {
		pr := pairs[name]
		if pr.sums == nil || pr.counts == nil {
			return nil, fmt.Errorf("pipeline: mismatched imatrix sums/counts pair: %s", name)
		}
		sums, err := readF32Tensor(f, layout, pr.sums)
		if err != nil {
			return nil, err
		}
		counts, err := readF32Tensor(f, layout, pr.counts)
		if err != nil {
			return nil, err
		}
		entry, err := profileEntry(name, sums, counts)
		if err != nil {
			return nil, err
		}
		entries = append(entries, *entry)
	}

	// global + role normalization
	means := make([]float64, len(entries))
	for i := range entries {
		means[i] = entries[i].Mean
	}
	globalMedian := median(means)
	globalP := rankPercentiles(means)

	roleIndices := map[string][]int{}
	for i := range entries {
		roleIndices[entries[i].Role] = append(roleIndices[entries[i].Role], i)
	}
	roleMedians := map[string]float64{}
	rolePercentiles := map[int]float64{}
	for role, idx := range roleIndices {
		vals := make([]float64, len(idx))
		for j, i := range idx {
			vals[j] = means[i]
		}
		roleMedians[role] = median(vals)
		rp := rankPercentiles(vals)
		for j, i := range idx {
			rolePercentiles[i] = rp[j]
		}
	}

	for i := range entries {
		entries[i].GlobalRelativeMean = divOrZero(entries[i].Mean, globalMedian)
		entries[i].RoleRelativeMean = divOrZero(entries[i].Mean, roleMedians[entries[i].Role])
		entries[i].GlobalPercentile = globalP[i]
		entries[i].RolePercentile = rolePercentiles[i]
	}

	return &ImatrixProfile{
		SchemaVersion: 1, SourceFile: baseName(path),
		Datasets: datasets, ChunkCount: chunkCount, ChunkSize: chunkSize, Entries: entries,
	}, nil
}

func divOrZero(a, b float64) float64 {
	if b > 0 {
		return a / b
	}
	return 0
}

func baseName(p string) string {
	if idx := strings.LastIndex(p, "\\"); idx >= 0 {
		return p[idx+1:]
	}
	if idx := strings.LastIndex(p, "/"); idx >= 0 {
		return p[idx+1:]
	}
	return p
}

func toInt(v any) int {
	switch t := v.(type) {
	case int:
		return t
	case int64:
		return int(t)
	case uint32:
		return int(t)
	case uint64:
		return int(t)
	case float64:
		return int(t)
	}
	return 0
}

func readF32Tensor(f *os.File, layout *gguf.Layout, t *gguf.TensorInfo) ([]float64, error) {
	if t.TypeID != 0 {
		return nil, fmt.Errorf("pipeline: imatrix tensor %s must be F32, got type id %d", t.Name, t.TypeID)
	}
	count := 1
	for _, d := range t.Shape {
		count *= int(d)
	}
	pos := int64(layout.DataOffset) + int64(t.RelativeOffset)
	if _, err := f.Seek(pos, 0); err != nil {
		return nil, err
	}
	buf := make([]byte, count*4)
	if _, err := f.Read(buf); err != nil {
		return nil, err
	}
	out := make([]float64, count)
	for i := 0; i < count; i++ {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:])))
	}
	return out, nil
}

func profileEntry(name string, sums, counts []float64) (*ImatrixTensorProfile, error) {
	m := blockRoleRE.FindStringSubmatch(name)
	if m == nil {
		return nil, fmt.Errorf("pipeline: unsupported imatrix tensor name: %s", name)
	}
	if len(counts) == 0 || len(sums)%len(counts) != 0 {
		return nil, fmt.Errorf("pipeline: imatrix sums/counts shape mismatch for %s", name)
	}

	countInts := make([]int, 0, len(counts))
	for _, v := range counts {
		if (math.IsInf(v, 0) || math.IsNaN(v)) || v < 0 {
			return nil, fmt.Errorf("pipeline: invalid imatrix count for %s: %v", name, v)
		}
		rounded := math.Floor(v + 0.5)
		if math.Abs(v-rounded) > 1e-4 {
			return nil, fmt.Errorf("pipeline: non-integral imatrix count for %s: %v", name, v)
		}
		countInts = append(countInts, int(rounded))
	}

	perWidth := len(sums) / len(countInts)
	normalized := make([]float64, 0, len(sums))
	for ci, c := range countInts {
		start := ci * perWidth
		for _, raw := range sums[start : start+perWidth] {
			var v float64
			if c > 0 {
				v = raw / float64(c)
			} else {
				v = 1.0
			}
			if (math.IsInf(v, 0) || math.IsNaN(v)) || v < 0 {
				return nil, fmt.Errorf("pipeline: invalid normalized imatrix value for %s", name)
			}
			normalized = append(normalized, v)
		}
	}

	sortedValues := append([]float64(nil), normalized...)
	sort.Float64s(sortedValues)
	mean := neumaierSum(normalized) / float64(len(normalized))
	var msq float64
	{
		ss := make([]float64, len(normalized))
		for i, v := range normalized {
			ss[i] = v * v
		}
		msq = neumaierSum(ss) / float64(len(normalized))
	}
	variance := msq - mean*mean
	if variance < 0 {
		variance = 0
	}
	nz := 0
	for _, v := range normalized {
		if v > 0 {
			nz++
		}
	}

	block := atoiGroup(m[1])
	role := m[2]
	return &ImatrixTensorProfile{
		Name: name, Block: block, Role: role,
		Width: len(normalized), CountValues: len(countInts),
		CountMin: minArr(countInts), CountMax: maxArr(countInts), CountSum: sumArr(countInts),
		Mean: mean, RMS: math.Sqrt(msq), Stddev: math.Sqrt(variance),
		Minimum: sortedValues[0], P50: percentile(sortedValues, 0.5),
		P95: percentile(sortedValues, 0.95), P99: percentile(sortedValues, 0.99),
		Maximum:         sortedValues[len(sortedValues)-1],
		NonzeroFraction: float64(nz) / float64(len(normalized)),
	}, nil
}

func atoiGroup(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0
		}
		n = n*10 + int(s[i]-'0')
	}
	return n
}

func minArr(a []int) int {
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
func maxArr(a []int) int {
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}
func sumArr(a []int) int {
	s := 0
	for _, v := range a {
		s += v
	}
	return s
}
