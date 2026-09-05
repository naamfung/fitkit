package pipeline

// PresetFileTypes is the whitelist of quantize output types accepted by
// -lower / -upper. It mirrors llama-quantize's allowed types (name -> ftype
// value), verified against `llama-quantize -h` and the ggml.c type_traits
// blob sizes of the pinned runtime (laamaafung).
//
// To enable/disable a type, add/remove its name here — nothing else needs to
// change. A name that is not in this map is rejected as "unknown preset".
var PresetFileTypes = map[string]int{
	// float
	"F16": 1, "F32": 0, "BF16": 32,
	// scalar quantization
	"Q8_0": 7, "Q5_0": 8, "Q5_1": 9, "Q4_0": 2, "Q4_1": 3,
	// new types
	"Q1_0": 40, "Q2_0": 41, "MXFP4_MOE": 38,
	// K-family
	"Q2_K": 10, "Q2_K_S": 21,
	"Q3_K": 12, "Q3_K_S": 11, "Q3_K_M": 12, "Q3_K_L": 13,
	"Q4_K": 15, "Q4_K_S": 14, "Q4_K_M": 15,
	"Q5_K": 17, "Q5_K_S": 16, "Q5_K_M": 17, "Q6_K": 18,
	// IQ-family
	"IQ1_S": 24, "IQ1_M": 31,
	"IQ2_XXS": 19, "IQ2_XS": 20, "IQ2_S": 28, "IQ2_M": 29,
	"IQ3_XXS": 23, "IQ3_XS": 22, "IQ3_S": 26, "IQ3_M": 27,
	"IQ4_NL": 25, "IQ4_XS": 30,
	// TQ/ternary — the whole TQ family is disabled: TQ4_1S measured garbage
	// output, and the encoding path proved unstable under local disk loads.
}
