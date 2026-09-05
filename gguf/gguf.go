// Package gguf reads GGUF metadata and tensor layouts, and predicts the exact
// output size of a pinned llama-quantize build.  This is a byte-exact
// reimplementation of the Python fit_gguf.gguf module.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// ---- constants (pinned to llama.cpp build 10666 / 4e97ac86e) ---------------

const (
	Magic            = "GGUF"
	Version          = 3
	DefaultAlignment = 32
	KVStringMaxBytes = 127
)

// GGUF value types (from gguf.h).
const (
	UINT8   = 0
	INT8    = 1
	UINT16  = 2
	INT16   = 3
	UINT32  = 4
	INT32   = 5
	FLOAT32 = 6
	BOOL    = 7
	STRING  = 8
	ARRAY   = 9
	UINT64  = 10
	INT64   = 11
	FLOAT64 = 12
)

var fixedValueSizes = map[uint8]int{
	UINT8: 1, INT8: 1, UINT16: 2, INT16: 2,
	UINT32: 4, INT32: 4, FLOAT32: 4, BOOL: 1,
	UINT64: 8, INT64: 8, FLOAT64: 8,
}

// GGMLTypeTraits maps qtype name → (blockElements, blockBytes).
// Pinned to ggml-common.h block structs / ggml.c type_traits blob sizes of the
// pinned runtime (laamaafung).  typeSize == sizeof(block_*).
var GGMLTypeTraits = map[string][2]int{
	"f32":     {1, 4},
	"bf16":    {1, 2},
	"f16":     {1, 2},
	"q1_0":    {128, 18}, // half + QK1_0/8
	"q2_0":    {64, 18},  // half + QK2_0/4
	"q4_0":    {32, 18},  // half + QK4_0/2
	"q4_1":    {32, 20},  // 2*half + QK4_1/2
	"mxfp4":   {32, 17},  // uint8 e + QK_MXFP4/2
	"iq1_s":   {256, 50},
	"iq1_m":   {256, 56},
	"iq2_xxs": {256, 66},
	"iq2_xs":  {256, 74},
	"iq2_s":   {256, 82},
	"iq2_m":   {256, 82}, // same layout as iq2_s
	"iq3_xxs": {256, 98},
	"iq3_s":   {256, 110},
	"iq3_m":   {256, 110}, // same layout as iq3_s
	"iq4_xs":  {256, 136},
	"iq4_nl":  {32, 18},
	"q2_k":    {256, 84},
	"q3_k":    {256, 110},
	"q4_k":    {256, 144},
	"q5_0":    {32, 22},
	"q5_1":    {32, 24},
	"q5_k":    {256, 176},
	"q6_k":    {256, 210},
	"q8_0":    {32, 34},
	// TQ types (tq1_0/tq2_0/tq3_1s/tq4_1s) disabled via the whitelist.
}

// ---- errors ---------------------------------------------------------------

type Error struct{ msg string }

func (e *Error) Error() string { return "gguf: " + e.msg }

func newError(format string, args ...any) *Error {
	return &Error{msg: fmt.Sprintf(format, args...)}
}

// ---- data types -----------------------------------------------------------

type Field struct {
	Key         string
	ValueType   uint8
	EncodedSize int
	ScalarValue any // int | float64 | bool | string | []string | nil
}

type TensorInfo struct {
	Name           string
	Shape          []int64
	TypeID         uint32
	RelativeOffset uint64
}

type Layout struct {
	Version          uint32
	Alignment        uint32
	MetadataCount    uint64
	Fields           []Field
	Tensors          []TensorInfo
	TensorInfoBytes  int
	RawMetadataBytes int
	DataOffset       int64
	TensorMap        map[string]TensorInfo
}

type ImatrixProvenance struct {
	File         string `json:"file"`
	Dataset      string `json:"dataset,omitempty"`
	EntriesCount int    `json:"entries_count"`
	ChunksCount  int    `json:"chunks_count,omitempty"`
}

type QuantizationMetadata struct {
	FileType            int                `json:"file_type"`
	QuantizationVersion int                `json:"quantization_version"`
	Imatrix             *ImatrixProvenance `json:"imatrix,omitempty"`
}

type TensorSize struct {
	Name         string
	Qtype        string
	PayloadBytes int
	PaddedBytes  int
}

type Prediction struct {
	MetadataBytes int
	TensorPayload int
	TensorPadding int
	TotalBytes    int
	Tensors       []TensorSize
}

// ---- ANSI escape stripping (needed by dry-run log parser) -----------------

var ansiReplacer = newAnsiStripper()

func newAnsiStripper() *ansiStripper { return &ansiStripper{} }

type ansiStripper struct{}

func (*ansiStripper) Strip(s string) string {
	// We need the same behavior as Python's re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "").
	// A simple rune-by-rune pass is sufficient for the ANSI escape sequences
	// that llama.cpp emits (CSI sequences: \x1b[<params>m).
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
					i++ // skip the final byte
				}
			}
			continue
		}
		b = append(b, s[i])
		i++
	}
	return string(b)
}

// ---- GGUF binary reader ---------------------------------------------------

type reader struct {
	r   io.ReadSeeker
	buf [8]byte
}

func newReader(r io.ReadSeeker) *reader { return &reader{r: r} }

func (rd *reader) readExact(n int) ([]byte, error) {
	b := make([]byte, n)
	if _, err := io.ReadFull(rd.r, b); err != nil {
		return nil, newError("read error: %v", err)
	}
	return b, nil
}

func (rd *reader) unpackU8() (uint8, error) {
	b, err := rd.readExact(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (rd *reader) unpackU32() (uint32, error) {
	b, err := rd.readExact(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (rd *reader) unpackU64() (uint64, error) {
	b, err := rd.readExact(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (rd *reader) unpackI64() (int64, error) {
	b, err := rd.readExact(8)
	if err != nil {
		return 0, err
	}
	return int64(binary.LittleEndian.Uint64(b)), nil
}

func (rd *reader) unpackF32() (float32, error) {
	b, err := rd.readExact(4)
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(b)), nil
}

func (rd *reader) unpackF64() (float64, error) {
	b, err := rd.readExact(8)
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(b)), nil
}

func (rd *reader) readString() (string, error) {
	length, err := rd.unpackU64()
	if err != nil {
		return "", err
	}
	b, err := rd.readExact(int(length))
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) {
		return "", newError("GGUF string is not valid UTF-8")
	}
	return string(b), nil
}

func (rd *reader) skip(n int) error {
	if n < 0 {
		return newError("Cannot skip a negative GGUF value size")
	}
	_, err := rd.readExact(n)
	return err
}

// readOrSkipValue reads a GGUF value; when captureStringArray is true and the
// value is an ARRAY of STRING, it returns the parsed strings.  Returns nil for
// skipped values.
func (rd *reader) readOrSkipValue(vt uint8, captureStringArray bool) (any, error) {
	if sz, ok := fixedValueSizes[vt]; ok {
		b, err := rd.readExact(sz)
		if err != nil {
			return nil, err
		}
		switch vt {
		case UINT8:
			return int(b[0]), nil
		case INT8:
			return int(int8(b[0])), nil
		case UINT16:
			return int(binary.LittleEndian.Uint16(b)), nil
		case INT16:
			return int(int16(binary.LittleEndian.Uint16(b))), nil
		case UINT32:
			return int(binary.LittleEndian.Uint32(b)), nil
		case INT32:
			return int(int32(binary.LittleEndian.Uint32(b))), nil
		case FLOAT32:
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(b))), nil
		case BOOL:
			return b[0] != 0, nil
		case UINT64:
			return int(binary.LittleEndian.Uint64(b)), nil
		case INT64:
			return int(int64(binary.LittleEndian.Uint64(b))), nil
		case FLOAT64:
			return math.Float64frombits(binary.LittleEndian.Uint64(b)), nil
		}
		return nil, nil
	}
	if vt == STRING {
		return rd.readString()
	}
	if vt == ARRAY {
		etRaw, err := rd.unpackU32()
		if err != nil {
			return nil, err
		}
		et := uint8(etRaw)
		count, err := rd.unpackU64()
		if err != nil {
			return nil, err
		}
		if et == ARRAY {
			return nil, newError("Nested GGUF arrays are unsupported")
		}
		if fsz, ok := fixedValueSizes[et]; ok {
			if err := rd.skip(int(count) * fsz); err != nil {
				return nil, err
			}
			return nil, nil
		}
		if et == STRING {
			if captureStringArray {
				strs := make([]string, int(count))
				for i := 0; i < int(count); i++ {
					s, err := rd.readString()
					if err != nil {
						return nil, err
					}
					strs[i] = s
				}
				return strs, nil
			}
			for i := 0; i < int(count); i++ {
				length, err := rd.unpackU64()
				if err != nil {
					return nil, err
				}
				if err := rd.skip(int(length)); err != nil {
					return nil, err
				}
			}
			return nil, nil
		}
		return nil, newError("Unknown GGUF array element type: %d", et)
	}
	return nil, newError("Unknown GGUF value type: %d", vt)
}

// ---- public API -----------------------------------------------------------

// readLayoutFile reads the GGUF metadata and tensor descriptors from a single
// file. It does not expand llama.cpp split shards; see ReadLayout.
func readLayoutFile(path string) (*Layout, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, newError("cannot open %s: %v", path, err)
	}
	defer f.Close()

	rd := newReader(f)
	version, tensorCount, metadataCount, err := readPreamble(rd)
	if err != nil {
		return nil, err
	}

	fields := make([]Field, 0, metadataCount)
	seenFields := make(map[string]bool)
	for i := 0; i < int(metadataCount); i++ {
		start, _ := f.Seek(0, io.SeekCurrent)
		key, err := rd.readString()
		if err != nil {
			return nil, err
		}
		if seenFields[key] {
			return nil, newError("Duplicate GGUF metadata key: %s", key)
		}
		vt, err := rd.unpackU32()
		if err != nil {
			return nil, err
		}
		sv, err := rd.readOrSkipValue(uint8(vt), key == "imatrix.datasets")
		if err != nil {
			return nil, err
		}
		end, _ := f.Seek(0, io.SeekCurrent)
		fields = append(fields, Field{
			Key:         key,
			ValueType:   uint8(vt),
			EncodedSize: int(end - start),
			ScalarValue: sv,
		})
		seenFields[key] = true
	}

	tensorInfoStart, _ := f.Seek(0, io.SeekCurrent)
	tensors := make([]TensorInfo, 0, tensorCount)
	seenTensors := make(map[string]bool)
	for i := 0; i < int(tensorCount); i++ {
		t, err := readTensor(rd)
		if err != nil {
			return nil, err
		}
		if seenTensors[t.Name] {
			return nil, newError("Duplicate GGUF tensor name: %s", t.Name)
		}
		seenTensors[t.Name] = true
		tensors = append(tensors, *t)
	}

	rawMetadataEnd, _ := f.Seek(0, io.SeekCurrent)
	rawMetadataBytes := int(rawMetadataEnd)

	// alignment
	alignment := uint32(DefaultAlignment)
	for _, fld := range fields {
		if fld.Key == "general.alignment" {
			if v, ok := fld.ScalarValue.(int); ok {
				alignment = uint32(v)
			}
		}
	}
	dataOffset := align(rawMetadataBytes, int(alignment))

	tm := make(map[string]TensorInfo)
	for _, t := range tensors {
		tm[t.Name] = t
	}

	return &Layout{
		Version:          version,
		Alignment:        alignment,
		MetadataCount:    metadataCount,
		Fields:           fields,
		Tensors:          tensors,
		TensorInfoBytes:  rawMetadataBytes - int(tensorInfoStart),
		RawMetadataBytes: rawMetadataBytes,
		DataOffset:       int64(dataOffset),
		TensorMap:        tm,
	}, nil
}

// readTensor reads a single tensor descriptor from a GGUF stream.
func readTensor(rd *reader) (*TensorInfo, error) {
	name, err := rd.readString()
	if err != nil {
		return nil, err
	}
	dims, err := rd.unpackU32()
	if err != nil {
		return nil, err
	}
	if dims < 1 || dims > 4 {
		return nil, newError("Unsupported dimension count %d for tensor %s", dims, name)
	}
	shape := make([]int64, dims)
	for j := 0; j < int(dims); j++ {
		d, err := rd.unpackU64()
		if err != nil {
			return nil, err
		}
		if d <= 0 {
			return nil, newError("Non-positive dimension for tensor %s: %d", name, d)
		}
		shape[j] = int64(d)
	}
	typeID, err := rd.unpackU32()
	if err != nil {
		return nil, err
	}
	relOff, err := rd.unpackU64()
	if err != nil {
		return nil, err
	}
	return &TensorInfo{Name: name, Shape: shape, TypeID: typeID, RelativeOffset: relOff}, nil
}

// readPreamble reads the GGUF magic, version, tensor count and metadata count.
func readPreamble(rd *reader) (version uint32, tensorCount, metadataCount uint64, err error) {
	magic, err := rd.readExact(4)
	if err != nil {
		return 0, 0, 0, err
	}
	if string(magic) != Magic {
		return 0, 0, 0, newError("Not a little-endian GGUF file")
	}
	version, err = rd.unpackU32()
	if err != nil {
		return 0, 0, 0, err
	}
	if version != Version {
		return 0, 0, 0, newError("Only GGUF v%d is supported, got v%d", Version, version)
	}
	tensorCount, err = rd.unpackU64()
	if err != nil {
		return 0, 0, 0, err
	}
	metadataCount, err = rd.unpackU64()
	if err != nil {
		return 0, 0, 0, err
	}
	return version, tensorCount, metadataCount, nil
}

// readTensorsFile reads only the tensor descriptors from a single GGUF file,
// skipping its metadata keys. It is used to collect the tensors of subsidiary
// llama.cpp split shards.
func readTensorsFile(path string) ([]TensorInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, newError("cannot open %s: %v", path, err)
	}
	defer f.Close()

	rd := newReader(f)
	_, tensorCount, metadataCount, err := readPreamble(rd)
	if err != nil {
		return nil, err
	}
	for i := 0; i < int(metadataCount); i++ {
		if _, err := rd.readString(); err != nil {
			return nil, err
		}
		vt, err := rd.unpackU32()
		if err != nil {
			return nil, err
		}
		if _, err := rd.readOrSkipValue(uint8(vt), false); err != nil {
			return nil, err
		}
	}
	tensors := make([]TensorInfo, 0, tensorCount)
	for i := 0; i < int(tensorCount); i++ {
		t, err := readTensor(rd)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, *t)
	}
	return tensors, nil
}

func scalarInt(l *Layout, key string) (int, bool) {
	for _, f := range l.Fields {
		if f.Key == key {
			if v, ok := f.ScalarValue.(int); ok {
				return v, true
			}
			return 0, false
		}
	}
	return 0, false
}

// splitStem detects a llama.cpp shard postfix ("-00001-of-00003") on a GGUF
// stem and returns the shared stem (the prefix every shard is named from).
func splitStem(stem string) (string, bool) {
	i := strings.LastIndex(stem, "-of-")
	if i < 0 {
		return "", false
	}
	countStr := stem[i+len("-of-"):]
	if len(countStr) != 5 || !digitsOnly(countStr) {
		return "", false
	}
	before := stem[:i]
	j := strings.LastIndex(before, "-")
	if j < 0 {
		return "", false
	}
	noStr := before[j+1:]
	if len(noStr) != 5 || !digitsOnly(noStr) {
		return "", false
	}
	return before[:j], true
}

func shardStem(base string) (string, bool) {
	return splitStem(strings.TrimSuffix(base, ".gguf"))
}

func digitsOnly(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// IsSplit reports whether path names a llama.cpp split shard (e.g.
// "model-00001-of-00003.gguf").
func IsSplit(path string) bool {
	_, ok := shardStem(filepath.Base(path))
	return ok
}

// BaseNameNoSplit returns the base filename with ".gguf" removed and, when the
// path names a llama.cpp split shard, the shard postfix removed too. For both
// "model-00001-of-00003.gguf" and "model.gguf" it returns "model".
func BaseNameNoSplit(path string) string {
	stem, _ := shardStem(filepath.Base(path))
	return stem
}

// ReadLayout reads the GGUF metadata and tensor descriptors for a model path.
// When the path is the first shard of a llama.cpp split model, it also reads
// every subsidiary shard and merges their tensors so the returned layout is the
// full tensor set, matching what llama.cpp / llama-quantize load.
func ReadLayout(path string) (*Layout, error) {
	layout, err := readLayoutFile(path)
	if err != nil {
		return nil, err
	}

	splitNo, hasNo := scalarInt(layout, "split.no")
	splitCount, hasCount := scalarInt(layout, "split.count")
	if !hasCount || splitCount <= 1 {
		return layout, nil
	}
	if hasNo && splitNo != 0 {
		return nil, newError("illegal split file idx %d; model must be loaded with the first split", splitNo)
	}
	stem, ok := shardStem(filepath.Base(path))
	if !ok {
		return nil, newError("cannot determine the split prefix of %s", path)
	}
	dir := filepath.Dir(path)
	for i := 1; i < splitCount; i++ {
		shard := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", stem, i+1, splitCount))
		tensors, err := readTensorsFile(shard)
		if err != nil {
			return nil, err
		}
		for _, t := range tensors {
			if _, dup := layout.TensorMap[t.Name]; dup {
				return nil, newError("invalid model: tensor '%s' is duplicated", t.Name)
			}
			layout.TensorMap[t.Name] = t
			layout.Tensors = append(layout.Tensors, t)
		}
	}
	return layout, nil
}

// CanonicalShape trims trailing-1 dimensions (minimum 1 dim).
func CanonicalShape(shape []int64) []int64 {
	end := len(shape)
	for end > 1 && shape[end-1] == 1 {
		end--
	}
	return shape[:end]
}

func EncodedFieldSize(key string, valueType uint8, value interface{}) int {
	keySize := 8 + len(key) + 4
	switch valueType {
	case UINT32:
		return keySize + 4
	case STRING:
		s := fmt.Sprint(value)
		return keySize + 8 + len(s)
	default:
		panic(fmt.Sprintf("gguf: unsupported synthesized value type %d", valueType))
	}
}

// OutputTensorInfoBytes estimates the tensor-info section size as the pinned
// quantizer re-emits it (trailing-singleton-dims normalization).
func OutputTensorInfoBytes(layout *Layout) int {
	total := 0
	for _, tensor := range layout.Tensors {
		dims := len(CanonicalShape(tensor.Shape))
		total += 8 + len(tensor.Name) + 4 + 8*dims + 4 + 8
	}
	return total
}

// PredictOutputMetadataSize predicts the metadata size of the quantized output.
func PredictOutputMetadataSize(layout *Layout, meta *QuantizationMetadata) (int, error) {
	if layout.Alignment != DefaultAlignment {
		return 0, newError("Pinned quantizer writes with alignment %d; source alignment is %d", DefaultAlignment, layout.Alignment)
	}

	// Build field size map from source layout.
	fs := make(map[string]int)
	for _, f := range layout.Fields {
		fs[f.Key] = f.EncodedSize
	}

	// Remove split fields.
	delete(fs, "split.no")
	delete(fs, "split.count")
	delete(fs, "split.tensors.count")

	// Add the two always-present synthesized fields.
	fs["general.quantization_version"] = EncodedFieldSize("general.quantization_version", UINT32, meta.QuantizationVersion)
	fs["general.file_type"] = EncodedFieldSize("general.file_type", UINT32, meta.FileType)

	// Add imatrix fields if present.
	if meta.Imatrix != nil {
		im := meta.Imatrix
		fs["quantize.imatrix.file"] = EncodedFieldSize("quantize.imatrix.file", STRING, im.File)
		fs["quantize.imatrix.entries_count"] = EncodedFieldSize("quantize.imatrix.entries_count", UINT32, im.EntriesCount)
		if im.Dataset != "" {
			fs["quantize.imatrix.dataset"] = EncodedFieldSize("quantize.imatrix.dataset", STRING, im.Dataset)
		}
		if im.ChunksCount > 0 {
			fs["quantize.imatrix.chunks_count"] = EncodedFieldSize("quantize.imatrix.chunks_count", UINT32, im.ChunksCount)
		}
	}

	// Sum fields
	sum := 0
	for _, sz := range fs {
		sum += sz
	}

	raw := 24 + sum + OutputTensorInfoBytes(layout)
	return align(raw, DefaultAlignment), nil
}

// TensorPaddedSize computes the payload and padded byte size of a single source
// tensor when encoded as dstType. ok is false when the tensor or the qtype is
// unknown, or when ne0 is not divisible by the qtype's block size (the type
// cannot encode that tensor).
func TensorPaddedSize(layout *Layout, name string, dstType string) (payload, padded int, ok bool) {
	src, found := layout.TensorMap[name]
	if !found {
		return 0, 0, false
	}
	dstLower := toLower(dstType)
	traits, known := GGMLTypeTraits[dstLower]
	if !known {
		return 0, 0, false
	}
	blockSize := traits[0]
	typeSize := traits[1]
	ne0 := int(CanonicalShape(src.Shape)[0])
	if ne0%blockSize != 0 {
		return 0, 0, false
	}
	rows := 1
	for _, d := range CanonicalShape(src.Shape)[1:] {
		rows *= int(d)
	}
	payload = (ne0 / blockSize) * typeSize * rows
	padded = align(payload, DefaultAlignment)
	return payload, padded, true
}

// PredictQuantizedSize predicts the exact output GGUF size for a recipe.
// recipe maps tensor name → dst_type (lowercase, e.g. "q4_k").
func PredictQuantizedSize(layout *Layout, recipe map[string]string, meta *QuantizationMetadata) (*Prediction, error) {
	sourceTensors := layout.TensorMap
	if len(sourceTensors) != len(recipe) {
		missing := make([]string, 0)
		extra := make([]string, 0)
		for n := range sourceTensors {
			if _, ok := recipe[n]; !ok {
				missing = append(missing, n)
			}
		}
		for n := range recipe {
			if _, ok := sourceTensors[n]; !ok {
				extra = append(extra, n)
			}
		}
		if len(missing) > 0 || len(extra) > 0 {
			return nil, newError("Recipe/source tensor mismatch; missing=%v, extra=%v", missing, extra)
		}
	}

	// Build sorted tensor names (stable order).
	sortedNames := make([]string, 0, len(recipe))
	for n := range recipe {
		sortedNames = append(sortedNames, n)
	}
	sort.Strings(sortedNames)

	sizes := make([]TensorSize, 0, len(recipe))
	for _, name := range sortedNames {
		dstType := recipe[name]
		payload, padded, ok := TensorPaddedSize(layout, name, dstType)
		if !ok {
			dstLower := toLower(dstType)
			traits, known := GGMLTypeTraits[dstLower]
			if !known {
				return nil, newError("Unsupported destination qtype: %s", dstType)
			}
			ne0 := int(CanonicalShape(sourceTensors[name].Shape)[0])
			return nil, newError("Tensor %s ne0=%d is not divisible by %s block size %d", name, ne0, dstLower, traits[0])
		}
		sizes = append(sizes, TensorSize{
			Name: name, Qtype: toLower(dstType), PayloadBytes: payload, PaddedBytes: padded,
		})
	}

	metaBytes, err := PredictOutputMetadataSize(layout, meta)
	if err != nil {
		return nil, err
	}

	tensorPayload := 0
	tensorPadded := 0
	for _, s := range sizes {
		tensorPayload += s.PayloadBytes
		tensorPadded += s.PaddedBytes
	}

	return &Prediction{
		MetadataBytes: metaBytes,
		TensorPayload: tensorPayload,
		TensorPadding: tensorPadded - tensorPayload,
		TotalBytes:    metaBytes + tensorPadded,
		Tensors:       sizes,
	}, nil
}

// ---- helpers --------------------------------------------------------------

func align(value, alignment int) int {
	if alignment <= 0 || alignment&(alignment-1) != 0 {
		panic(fmt.Sprintf("gguf: alignment must be a positive power of two, got %d", alignment))
	}
	return (value + alignment - 1) & -alignment
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		b[i] = c
	}
	return string(b)
}

// Ensure io.ReadSeeker is used (silence unused import).
var _ = io.ReadSeeker(nil)
