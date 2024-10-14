Network ResNets {
Layer block_0_1 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 24, C: 3, R: 3, S: 3, Y:128, X:128 }
}

Layer block_1_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 48, C: 24, R: 3, S: 3, Y:64, X:64 }
}

Layer block_0_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 48, R: 1, S: 1, Y:64, X:64 }
}

Layer block_0_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 32, R: 3, S: 3, Y:64, X:64 }
}

Layer block_0_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 168, C: 32, R: 1, S: 1, Y:64, X:64 }
}

Layer block_0_4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 168, C: 48, R: 1, S: 1, Y:64, X:64 }
}

Layer block_0_5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 168, R: 1, S: 1, Y:64, X:64 }
}

Layer block_0_6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 40, R: 3, S: 3, Y:64, X:64 }
}

Layer block_0_7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 168, C: 40, R: 1, S: 1, Y:64, X:64 }
}

Layer block_3_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 168, R: 1, S: 1, Y:64, X:64 }
}

Layer block_3_2 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 80, C: 80, R: 3, S: 3, Y:64, X:64 }
}

Layer block_3_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 80, R: 1, S: 1, Y:32, X:32 }
}

Layer block_3_4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 168, R: 1, S: 1, Y:32, X:32 }
}

Layer block_4_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 104, C: 408, R: 1, S: 1, Y:32, X:32 }
}

Layer block_4_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 104, C: 104, R: 3, S: 3, Y:32, X:32 }
}

Layer block_4_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 104, R: 1, S: 1, Y:32, X:32 }
}

Layer block_5_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 408, R: 1, S: 1, Y:32, X:32 }
}

Layer block_5_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 80, R: 3, S: 3, Y:32, X:32 }
}

Layer block_5_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 80, R: 1, S: 1, Y:32, X:32 }
}

Layer block_6_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 208, C: 408, R: 1, S: 1, Y:32, X:32 }
}

Layer block_6_2 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 208, C: 208, R: 3, S: 3, Y:32, X:32 }
}

Layer block_6_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 816, C: 208, R: 1, S: 1, Y:16, X:16 }
}

Layer block_6_4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 816, C: 408, R: 1, S: 1, Y:16, X:16 }
}

Layer block_7_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 208, C: 816, R: 1, S: 1, Y:16, X:16 }
}

Layer block_7_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 208, C: 208, R: 3, S: 3, Y:16, X:16 }
}

Layer block_7_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 816, C: 208, R: 1, S: 1, Y:16, X:16 }
}

Layer block_8_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 816, R: 1, S: 1, Y:16, X:16 }
}

Layer block_8_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 160, R: 3, S: 3, Y:16, X:16 }
}

Layer block_8_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 816, C: 160, R: 1, S: 1, Y:16, X:16 }
}

Layer block_8_4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 208, C: 816, R: 1, S: 1, Y:16, X:16 }
}

Layer block_8_5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 208, C: 208, R: 3, S: 3, Y:16, X:16 }
}

Layer block_8_6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 816, C: 208, R: 1, S: 1, Y:16, X:16 }
}

Layer block_11_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 816, R: 1, S: 1, Y:16, X:16 }
}

Layer block_11_2 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 408, C: 408, R: 3, S: 3, Y:16, X:16 }
}

Layer block_11_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1640, C: 408, R: 1, S: 1, Y:8, X:8 }
}

Layer block_11_4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1640, C: 816, R: 1, S: 1, Y:8, X:8 }
}

Layer block_12_1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 1640, R: 1, S: 1, Y:8, X:8 }
}

Layer block_12_2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 408, C: 408, R: 3, S: 3, Y:8, X:8 }
}

Layer block_12_3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1640, C: 408, R: 1, S: 1, Y:8, X:8 }
}

}
