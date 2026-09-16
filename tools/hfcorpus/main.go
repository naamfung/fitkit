// Command hfcorpus converts a HuggingFace dataset repo's parquet files into a
// plain-text corpus for llama-imatrix / fitkit `fitcalibrate --imatrix-corpus`.
//
// Usage:
//
//	fitkit/tools/hfcorpus -dataset openbmb/UltraData-Code -out corpus-code.txt
//	    [-file data/train-00000-of-00001.parquet]  # pick one file (default: first)
//	    [-column text]                             # text column (default: auto-detect)
//	    [-max-rows 5000]                           # stop after N sampled rows (default: 5000)
//	    [-max-bytes 8388608]                       # stop after N bytes of text (default: 8 MiB)
//	    [-samples 1]                               # rows to join per block (1 = verbatim)
//	    [-token <hf-token>]                        # optional, for gated datasets
//	    [-schema]                                  # print the parquet schema and exit
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/parquet-go/parquet-go"
)

type hfFile struct {
	RFilename string `json:"rfilename"`
}

type hfRepo struct {
	ID       string   `json:"id"`
	Siblings []hfFile `json:"siblings"`
}

func get(url, token string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return http.DefaultClient.Do(req)
}

func fetchRepo(dataset, token string) (*hfRepo, error) {
	resp, err := get("https://huggingface.co/api/datasets/"+dataset, token)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HF API %s: %s", resp.Status, dataset)
	}
	var repo hfRepo
	if err := json.NewDecoder(resp.Body).Decode(&repo); err != nil {
		return nil, err
	}
	return &repo, nil
}

// cacheRoot is the local cache directory for downloaded parquet files,
// next to the program itself for visibility: <exe dir>/hfcorpus-cache.
// HFCORPUS_CACHE overrides it when set.
func cacheRoot() string {
	if env := os.Getenv("HFCORPUS_CACHE"); env != "" {
		return env
	}
	exe, err := os.Executable()
	if err != nil {
		return filepath.Join(".", "hfcorpus-cache")
	}
	return filepath.Join(filepath.Dir(exe), "hfcorpus-cache")
}

// cachePath maps a dataset + repo file to a stable local cache path keyed by
// the SHA-256 of the pair, so repeated runs reuse the download and never
// hammer the HF endpoint.
func cachePath(dataset, file string) string {
	sum := sha256.Sum256([]byte(dataset + "/" + file))
	dir := filepath.Join(cacheRoot(), hex.EncodeToString(sum[:8]))
	return filepath.Join(dir, filepath.Base(file))
}

// download returns the parquet file locally: the cache hit when present (and
// -no-cache not set), otherwise a fresh download stored into the cache.
func download(dataset, file, token string, noCache bool) (string, error) {
	cached := cachePath(dataset, file)
	if !noCache {
		if st, err := os.Stat(cached); err == nil && !st.IsDir() && st.Size() > 0 {
			fmt.Printf("       cache hit: %s\n", cached)
			return cached, nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(cached), 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(filepath.Dir(cached), "hfcorpus-*.parquet")
	if err != nil {
		return "", err
	}
	tmp.Close()
	url := "https://huggingface.co/datasets/" + dataset + "/resolve/main/" + filepath.ToSlash(file)
	resp, err := get(url, token)
	if err != nil {
		os.Remove(tmp.Name())
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("download %s: %s", url, resp.Status)
	}
	out, err := os.OpenFile(tmp.Name(), os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		os.Remove(tmp.Name())
		return "", err
	}
	out.Close()
	_ = os.Remove(cached) // Windows rename fails if the target exists
	if err := os.Rename(tmp.Name(), cached); err != nil {
		return "", err
	}
	return cached, nil
}

// pickTextColumn chooses a leaf column by preferred name (case-insensitive),
// else the first leaf column.
func pickTextColumn(schema *parquet.Schema) string {
	preferred := []string{"text", "content", "code", "messages", "input", "snippet"}
	var leaves []string
	for _, f := range schema.Fields() {
		leaves = append(leaves, f.Name())
	}
	for _, p := range preferred {
		for _, n := range leaves {
			if strings.EqualFold(n, p) || strings.HasSuffix(strings.ToLower(n), "."+p) {
				return n
			}
		}
	}
	if len(leaves) > 0 {
		return leaves[0]
	}
	return ""
}

func main() {
	dataset := flag.String("dataset", "", "HF dataset id, e.g. openbmb/UltraData-Code")
	outPath := flag.String("out", "", "output .txt corpus path (optional when -imatrix-out is set)")
	specFile := flag.String("file", "", "repo path of one parquet file (default: first)")
	specCol := flag.String("column", "", "text column (default: auto-detect)")
	maxRows := flag.Int("max-rows", 5000, "max sampled rows")
	maxBytes := flag.Int("max-bytes", 8*1024*1024, "max text bytes")
	samples := flag.Int("samples", 1, "rows to join per block with blank lines (1 = verbatim)")
	token := flag.String("token", "", "optional HF token for gated datasets")
	schemaOnly := flag.Bool("schema", false, "print the parquet schema and exit")
	noCache := flag.Bool("no-cache", false, "force re-download, ignoring the local cache")
	model := flag.String("model", "", "BF16 source GGUF to generate an imatrix for (requires -imatrix-out + -runtime)")
	imatrixOut := flag.String("imatrix-out", "", "output imatrix .gguf path; when set, runs llama-imatrix on the corpus")
	runtimeDir := flag.String("runtime", "", "llama.cpp runtime dir containing llama-imatrix (required with -imatrix-out)")
	chunks := flag.Int("chunks", 500, "llama-imatrix --chunks")
	ngl := flag.Int("ngl", 99, "llama-imatrix -ngl (GPU layers)")
	threads := flag.Int("threads", 16, "llama-imatrix -t threads")
	flag.Parse()
	if *dataset == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *outPath == "" && *imatrixOut == "" {
		fmt.Fprintln(os.Stderr, "error: one of -out or -imatrix-out is required")
		os.Exit(2)
	}
	if *imatrixOut != "" && (*model == "" || *runtimeDir == "") {
		fmt.Fprintln(os.Stderr, "error: -imatrix-out requires -model (BF16 source) and -runtime (llama.cpp dir)")
		os.Exit(2)
	}

	repo, err := fetchRepo(*dataset, *token)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	var parquets []string
	for _, f := range repo.Siblings {
		if strings.HasSuffix(f.RFilename, ".parquet") {
			parquets = append(parquets, f.RFilename)
		}
	}
	if len(parquets) == 0 {
		fmt.Fprintf(os.Stderr, "error: no parquet files in %s\n", *dataset)
		os.Exit(1)
	}
	target := *specFile
	if target == "" {
		target = parquets[0]
	}
	found := false
	for _, p := range parquets {
		if p == target {
			found = true
			break
		}
	}
	if !found {
		n := 5
		if len(parquets) < n {
			n = len(parquets)
		}
		fmt.Fprintf(os.Stderr, "error: -file %s not in dataset (have %v ...)\n", target, parquets[:n])
		os.Exit(1)
	}

	fmt.Printf("[1/3] downloading %s: %s\n", *dataset, target)
	// download returns the local file (a cache hit or a fresh download stored
	// into the cache); the cached parquet is kept for future runs.
	localPath, err := download(*dataset, target, *token, *noCache)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	f, err := os.Open(localPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	pf, err := parquet.OpenFile(f, st.Size())
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: parquet:", err)
		os.Exit(1)
	}
	schema := pf.Schema()

	if *schemaOnly {
		for _, fl := range schema.Fields() {
			fmt.Printf("  %s\n", fl.Name())
		}
		return
	}

	column := *specCol
	if column == "" {
		column = pickTextColumn(schema)
	}
	if column == "" {
		fmt.Fprintln(os.Stderr, "error: no leaf column found")
		os.Exit(1)
	}
	leaf, ok := schema.Lookup(strings.Split(column, ".")...)
	if !ok {
		fmt.Fprintf(os.Stderr, "error: column %q not in schema\n", column)
		os.Exit(1)
	}
	colIndex := leaf.ColumnIndex
	fmt.Printf("[2/3] column: %s (index %d)\n", column, colIndex)

	corpusPath := *outPath
	keepCorpus := *outPath != ""
	if corpusPath == "" {
		tmp, err := os.CreateTemp("", "hfcorpus-*.txt")
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		corpusPath = tmp.Name()
		tmp.Close()
		_ = os.Remove(corpusPath) // CreateTemp already created it; os.Create below re-creates
	}
	out, err := os.Create(corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer out.Close()

	reader := parquet.NewReader(pf)
	numLeaf := len(schema.Columns())
	row := make(parquet.Row, numLeaf)
	written := 0
	rows := 0
	var block []string
	flush := func() {
		if len(block) == 0 {
			return
		}
		chunk := strings.Join(block, "\n\n") + "\n\n"
		out.WriteString(chunk)
		written += len(chunk)
		block = block[:0]
	}

	for rows < *maxRows && written < *maxBytes {
		for i := range row {
			row[i] = parquet.Value{}
		}
		n, err := reader.ReadRows([]parquet.Row{row})
		if err == io.EOF || n == 0 {
			break
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "error: read:", err)
			os.Exit(1)
		}
		// ReadRows fills the pre-allocated row; extract the text column
		var text string
		for _, v := range row {
			if v.Column() == colIndex {
				text = v.String()
				break
			}
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		rows++
		if *samples > 1 {
			block = append(block, text)
			if len(block) >= *samples {
				flush()
			}
		} else {
			chunk := text + "\n\n"
			out.WriteString(chunk)
			written += len(chunk)
		}
	}
	flush()
	fmt.Printf("[3/3] done: corpus %s (%d bytes from %d rows)\n", corpusPath, written, rows)

	if *imatrixOut != "" {
		if err := runImatrix(*runtimeDir, *model, corpusPath, *imatrixOut, *chunks, *ngl, *threads); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		fmt.Printf("[4/4] imatrix: %s\n", *imatrixOut)
	}
	if !keepCorpus {
		_ = os.Remove(corpusPath)
	}
}

// resolveRuntimeBinary finds a llama.cpp tool in the runtime dir, mirroring the
// platform-aware order used across fitkit (.exe/.cmd/.bat on Windows first).
func resolveRuntimeBinary(dir, name string) string {
	candidates := []string{name, name + ".exe", name + ".cmd", name + ".bat"}
	if runtime.GOOS == "windows" {
		candidates = []string{name + ".exe", name + ".cmd", name + ".bat", name}
	}
	for _, c := range candidates {
		p := filepath.Join(dir, c)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}
	return filepath.Join(dir, candidates[0])
}

// runImatrix invokes llama-imatrix to derive an importance matrix from the
// corpus against the given BF16 source model.
func runImatrix(runtimeDir, model, corpus, imatrixOut string, chunks, ngl, threads int) error {
	binary := resolveRuntimeBinary(runtimeDir, "llama-imatrix")
	if _, err := os.Stat(binary); err != nil {
		return fmt.Errorf("llama-imatrix not found in %s", runtimeDir)
	}
	argv := []string{
		binary, "-m", model, "-f", corpus,
		"-c", "512", "-ngl", strconv.Itoa(ngl), "-t", strconv.Itoa(threads),
		"--chunks", strconv.Itoa(chunks), "-o", imatrixOut,
	}
	fmt.Printf("[4/4] llama-imatrix %s ...\n", filepath.Base(model))
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("llama-imatrix failed: %w", err)
	}
	if st, err := os.Stat(imatrixOut); err != nil || st.Size() == 0 {
		return fmt.Errorf("llama-imatrix produced no output at %s", imatrixOut)
	}
	return nil
}
