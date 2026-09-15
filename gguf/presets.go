// Package gguf provides GGUF metadata reading and prediction logic.
package gguf

import (
	"strings"
)

// PresetFileTypes is the list of nameable GGUF quantization presets recognized by llama.cpp and Hugging Face.
var PresetFileTypes = []string{
	"f32", "f16", "bf16",
	"q8_0",
	"q4_0", "q4_1",
	"q5_0", "q5_1",
	"q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
	"q4_k_s", "q4_k_m", "q4_k_l",
	"q5_k_s", "q5_k_m", "q5_k_l",
	"q6_k",
	"iq1_s", "iq1_m",
	"iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
	"iq3_xxs", "iq3_s", "iq3_m",
	"iq4_xs", "iq4_nl",
}

// IsBareTensorType checks if the given type is a bare tensor type like Q4_K, Q3_K, Q5_K.
func IsBareTensorType(ty string) bool {
	lower := strings.ToLower(ty)
	switch lower {
	case "q4_k", "q3_k", "q5_k":
		return true
	default:
		return false
	}
}

// isNameablePreset checks if the given type string is a nameable preset in PresetFileTypes.
func isNameablePreset(p string) bool {
	lower := strings.ToLower(p)
	for _, pt := range PresetFileTypes {
		if strings.EqualFold(pt, lower) {
			return true
		}
	}
	return false
}

// formatPreset converts a lowercase preset name to the standard uppercase llama.cpp preset name.
func formatPreset(p string) string {
	lower := strings.ToLower(strings.TrimSpace(p))
	switch lower {
	case "q4_k_m":
		return "Q4_K_M"
	case "q4_k_s":
		return "Q4_K_S"
	case "q4_k_l":
		return "Q4_K_L"
	case "q3_k_m":
		return "Q3_K_M"
	case "q3_k_s":
		return "Q3_K_S"
	case "q3_k_l":
		return "Q3_K_L"
	case "q5_k_m":
		return "Q5_K_M"
	case "q5_k_s":
		return "Q5_K_S"
	case "q5_k_l":
		return "Q5_K_L"
	case "q6_k":
		return "Q6_K"
	case "iq3_m":
		return "IQ3_M"
	case "iq3_s":
		return "IQ3_S"
	case "iq3_xs":
		return "IQ3_XS"
	case "iq4_xs":
		return "IQ4_XS"
	case "iq4_nl":
		return "IQ4_NL"
	case "iq2_xxs":
		return "IQ2_XXS"
	case "iq2_xs":
		return "IQ2_XS"
	case "iq2_s":
		return "IQ2_S"
	case "iq2_m":
		return "IQ2_M"
	case "iq1_s":
		return "IQ1_S"
	case "iq1_m":
		return "IQ1_M"
	case "q2_k":
		return "Q2_K"
	case "q3_k":
		return "Q3_K"
	case "q4_k":
		return "Q4_K"
	case "q5_k":
		return "Q5_K"
	case "q8_0":
		return "Q8_0"
	case "q4_0":
		return "Q4_0"
	case "q4_1":
		return "Q4_1"
	case "q5_0":
		return "Q5_0"
	case "q5_1":
		return "Q5_1"
	case "f32":
		return "F32"
	case "f16":
		return "F16"
	case "bf16":
		return "BF16"
	default:
		// Fallback: return uppercased original or normalized form
		return strings.ToUpper(p)
	}
}

// PrimaryTypeFromPlan determines the nameable preset suffix for GGUF file naming.
// recipeBasePreset is the base preset from the recipe (e.g., "Q4_K_M").
// dominantType is the dominant quantization type from the plan overrides.
// hasOverrides indicates whether there are tensor overrides in the plan.
func PrimaryTypeFromPlan(recipeBasePreset, dominantType string, hasOverrides bool) string {
	if !hasOverrides {
		// overrides no tensor -> return the window's lower preset (assume recipeBasePreset is already a nameable preset)
		return formatPreset(recipeBasePreset)
	}

	dominantLower := strings.ToLower(dominantType)
	// Check if dominantType is a nameable preset name
	if isNameablePreset(dominantLower) {
		return formatPreset(dominantType)
	}

	// overridden, dominant type is a bare tensor type -> return the recipe's base preset
	if IsBareTensorType(dominantType) {
		return formatPreset(recipeBasePreset)
	}

	// Fallback: try to format the dominantType as a preset
	if isNameablePreset(dominantLower) {
		return formatPreset(dominantType)
	}

	// Final fallback to recipeBasePreset
	return formatPreset(recipeBasePreset)
}
