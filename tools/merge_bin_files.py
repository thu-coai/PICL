import sys
import numpy as np
from data_utils.indexed_dataset import make_builder, index_file_path, data_file_path

input_path_1 = sys.argv[1]
input_path_2 = sys.argv[2]
output_path = sys.argv[3]

output_builder = make_builder(data_file_path(output_path), impl="mmap", dtype=np.uint16)

output_builder.merge_file_(input_path_1)
output_builder.merge_file_(input_path_2)

output_builder.finalize(index_file_path(output_path))