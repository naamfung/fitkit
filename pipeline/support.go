package pipeline

import (
	"crypto/sha256"
	"regexp"
	"sort"
	"strconv"
)

var (
	tensorLineRE    = regexp.MustCompile(`^(?:.*?\s+)?\[\s*(\d+)\s*/\s*(\d+)\s*\]\s+(\S+)\s+-\s+\[([\d\s,]+)\],\s+type\s*=\s*([A-Za-z0-9_]+),\s+size\s*=\s*(.+)$`)
	quantizedSizeRE = regexp.MustCompile(`^\s*(\d+(?:\.\d+)?)\s*MiB\s*->\s*(\d+(?:\.\d+)?)\s*MiB\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*$`)
	unchangedSizeRE = regexp.MustCompile(`^\s*(\d+(?:\.\d+)?)\s*MiB\s*$`)
	blockRoleRE     = regexp.MustCompile(`^blk\.(\d+)\.(.+)\.weight$`)
	tensorBlockRE   = regexp.MustCompile(`^blk\.(\d+)\.`)
)

func sortAssignments(t []Assignment) {
	sort.SliceStable(t, func(i, j int) bool { return t[i].Ordinal < t[j].Ordinal })
}

func tensorBlock(name string) (int, bool) {
	m := tensorBlockRE.FindStringSubmatch(name)
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return n, true
}

// sortUpgrade sorts by a less function using sort.SliceStable.
func sortUpgrade(c []UpgradeCandidate, less func(a, b UpgradeCandidate) bool) {
	// We need a stable sort; use a wrapper.
	sw := &upgradeSlice{data: c, less: less}
	sort.Stable(sw)
}

type upgradeSlice struct {
	data []UpgradeCandidate
	less func(a, b UpgradeCandidate) bool
}

func (s *upgradeSlice) Len() int           { return len(s.data) }
func (s *upgradeSlice) Less(i, j int) bool { return s.less(s.data[i], s.data[j]) }
func (s *upgradeSlice) Swap(i, j int)      { s.data[i], s.data[j] = s.data[j], s.data[i] }

// sortSHA sorts candidates by SHA-256 hash of seed\x00tensor\x00to_qtype.
func sortSHA(c []UpgradeCandidate, hashKey func(UpgradeCandidate) []byte) {
	sort.SliceStable(c, func(i, j int) bool {
		hi := hashKey(c[i])
		hj := hashKey(c[j])
		for k := 0; k < len(hi) && k < len(hj); k++ {
			if hi[k] != hj[k] {
				return hi[k] < hj[k]
			}
		}
		return len(hi) < len(hj)
	})
}

// sortInts sorts a slice of ints.
var sortInts = sort.Ints

// ensure unused import is satisfied
var _ = sha256.Sum256
