package pipeline

import "testing"

// TestResolveTarget pins the upstream resolve_target semantics: lower + a
// rational fraction of the preset gap with integer truncation.
func TestResolveTarget(t *testing.T) {
	cases := []struct {
		fit    string
		lower  int
		upper  int
		target int
	}{
		{"0.5", 1000, 2000, 1500},
		{"1/3", 1000, 2000, 1333}, // 1000 + (1000*1)//3 = 1333
		{"0.1", 1000, 2000, 1100},
		{"3/4", 1000, 2000, 1750},
	}
	for _, c := range cases {
		got, err := ResolveTarget(c.lower, c.upper, c.fit)
		if err != nil {
			t.Fatalf("ResolveTarget(%s) error: %v", c.fit, err)
		}
		if got != c.target {
			t.Errorf("ResolveTarget(%s, %d, %d) = %d, want %d", c.fit, c.lower, c.upper, got, c.target)
		}
	}
	for _, bad := range []string{"0", "1", "2", "-0.5", "abc"} {
		if _, err := ResolveTarget(1000, 2000, bad); err == nil {
			t.Errorf("ResolveTarget(%q) must reject out-of-range fit", bad)
		}
	}
}
