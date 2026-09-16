hfcorpus -dataset openbmb/UltraData-Code `
         -file data/UltraData-Code-L2/go/UltraData-Code-L2-go-part-00001-of-00006.parquet `
         -model <你的BF16母本.gguf> `
         -imatrix-out <输出>.gguf `
         -runtime <llama.cpp目录> -chunks 500
