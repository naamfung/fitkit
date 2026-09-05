package fidelity

import "math"

// eval-v1 metric kernels (transcribed from fit_gguf.eval.metrics).

const exclusionBudget = 0.001
const refLogprobCutoff = -16.0

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// neumaierSum is an accurate (compensated) float64 summation.
func neumaierSum(xs []float64) float64 {
	var s, comp float64
	for _, x := range xs {
		t := s + x
		if math.Abs(s) >= math.Abs(x) {
			comp += (s - t) + x
		} else {
			comp += (x - t) + s
		}
		s = t
	}
	return s + comp
}

// LogSoftmax is a stable log-softmax in float64 (max-subtracted).
func LogSoftmax(logits []float64) []float64 {
	top := logits[0]
	for _, x := range logits[1:] {
		if x > top {
			top = x
		}
	}
	exps := make([]float64, len(logits))
	for i, x := range logits {
		exps[i] = math.Exp(x - top)
	}
	logTotal := top + math.Log(neumaierSum(exps))
	out := make([]float64, len(logits))
	for i, x := range logits {
		out[i] = x - logTotal
	}
	return out
}

// KLDivergence computes D_KL(P_ref || P_quant) at one position (nats).
func KLDivergence(refLogits, candLogits []float64) float64 {
	if len(refLogits) != len(candLogits) {
		return math.NaN()
	}
	lp := LogSoftmax(refLogits)
	lq := LogSoftmax(candLogits)
	var terms []float64
	for i := range lp {
		if lp[i] <= refLogprobCutoff {
			continue
		}
		if lq[i] == math.Inf(-1) {
			return math.Inf(1)
		}
		terms = append(terms, math.Exp(lp[i])*(lp[i]-lq[i]))
	}
	return neumaierSum(terms)
}

// ArgmaxLowestTie returns argmax with lowest-index tie resolution.
func ArgmaxLowestTie(values []float64) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

// SameTop reports whether reference and candidate argmax agree.
func SameTop(refLogits, candLogits []float64) bool {
	return ArgmaxLowestTie(refLogits) == ArgmaxLowestTie(candLogits)
}

// DomainAggregate is the per-domain two-level aggregation result.
type DomainAggregate struct {
	Domain            string
	NPositions        int
	ExcludedPositions int
	KLMean            float64
	SameTopFrac       float64
	PPL               float64
	HasPPL            bool
}

func (d *DomainAggregate) ExclusionRate() float64 {
	total := d.NPositions + d.ExcludedPositions
	if total == 0 {
		return 0
	}
	return float64(d.ExcludedPositions) / float64(total)
}

// AggregateDomain computes the per-domain mean over counted positions.
func AggregateDomain(domain string, klValues []float64, sameTop []bool, nll []float64, hasNLL bool) DomainAggregate {
	n := len(klValues)
	var klVec, nlls []float64
	topCount := 0
	for i, k := range klValues {
		if !finite(k) {
			continue
		}
		klVec = append(klVec, k)
		if sameTop[i] {
			topCount++
		}
		if hasNLL {
			nlls = append(nlls, nll[i])
		}
	}
	agg := DomainAggregate{
		Domain: domain, NPositions: len(klVec),
		ExcludedPositions: n - len(klVec),
		KLMean:            math.Inf(1), SameTopFrac: 0,
	}
	if len(klVec) > 0 {
		agg.KLMean = neumaierSum(klVec) / float64(len(klVec))
		agg.SameTopFrac = float64(topCount) / float64(len(klVec))
	}
	if hasNLL && len(nlls) > 0 {
		agg.PPL = math.Exp(neumaierSum(nlls) / float64(len(nlls)))
		agg.HasPPL = true
	}
	return agg
}

// MacroKL is the equal-weight mean of domain means.
func MacroKL(results []DomainAggregate) float64 {
	s := 0.0
	for _, r := range results {
		s += DomainWeight * r.KLMean
	}
	return s
}
