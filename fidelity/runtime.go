// runtime.go builds the process environment for llama.cpp runtime binaries so
// their shared libraries resolve, mirroring upstream llama_integration.runtime_env
// (0.3.2). A Windows CUDA release ships ggml-cuda.dll inside the binary
// directory but keeps the CUDA runtime it links against (cudart64_*.dll,
// cublas*_*.dll) in a sibling directory; without it llama.cpp silently falls
// back to CPU — measured ~78x slower.
package fidelity

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// cudaRuntimeSiblings finds sibling directories beside the runtime that carry
// the CUDA runtime the GPU backend links against.
func cudaRuntimeSiblings(runtimeDir string) []string {
	parent := filepath.Dir(filepath.Clean(runtimeDir))
	entries, err := os.ReadDir(parent)
	if err != nil {
		return nil
	}
	var found []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(parent, e.Name())
		if filepath.Clean(dir) == filepath.Clean(runtimeDir) {
			continue
		}
		if hasDLL(dir, "cudart64_*.dll") || hasDLL(dir, "cublas64_*.dll") {
			found = append(found, dir)
		}
	}
	return found
}

func hasDLL(dir, pattern string) bool {
	matches, err := filepath.Glob(filepath.Join(dir, pattern))
	return err == nil && len(matches) > 0
}

// RuntimeEnv builds the environment for a runtime binary with its libraries
// discoverable: PATH on Windows, LD_LIBRARY_PATH elsewhere, plus any sibling
// CUDA runtime directory.
func RuntimeEnv(runtimeDir string) []string {
	ordered := append([]string{runtimeDir}, cudaRuntimeSiblings(runtimeDir)...)
	prefix := strings.Join(ordered, string(os.PathListSeparator))
	env := os.Environ()
	key := "LD_LIBRARY_PATH"
	if runtime.GOOS == "windows" {
		key = "PATH"
	}
	for i, kv := range env {
		if strings.HasPrefix(kv, key+"=") {
			env[i] = key + "=" + prefix + string(os.PathListSeparator) + strings.TrimPrefix(kv, key+"=")
			return env
		}
	}
	return append(env, key+"="+prefix)
}
