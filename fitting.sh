export PATH="D:/Programs/llama-cpp-repos/fitkit/fitting/bin":$PATH
Runtime="D:/Programs/llama-cpp-repos/laamaafung/build-v17/bin/Release"
############################################################################
ORIGINAL="G:\WorkModels\Qwen3.5-9B\LuffyTheFox\Qwen3.5-9B-Uncensored-Genesis-BF16.gguf"
############################################################################
IMatrix="C:\WorkModels\Qwen3.5-9B\imatrix-genesis.gguf"
############################################################################
# ./fitting -source <母本.gguf> -imatrix <im.gguf> -target <size> -lower <下界> -upper <上界> <-allow-requantize>

# fitting -source $ORIGINAL -imatrix $IMatrix -runtime $Runtime -target 4.88GiB -lower IQ4_XS -upper BF16 -mode down -plan-only

#"quality, balanced, compact, mini"
fitting -source $ORIGINAL -imatrix $IMatrix -runtime $Runtime -target 4.88GiB -lower IQ4_XS -upper BF16 -mode up -out "C:\WorkModels\Qwen3.5-9B\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix" -policy quality
