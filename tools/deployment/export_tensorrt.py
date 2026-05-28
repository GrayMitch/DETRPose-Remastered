import os 
import shutil

input_folder = 'onnx_engines'
input_files = [f for f in os.listdir(input_folder)]

output_folder = 'trt_engines'
output_files = [f.replace('onnx', 'engine') for f in input_files]

os.makedirs(output_folder, exist_ok=True)

trtexec="/usr/src/tensorrt/bin/trtexec"

for f_in, f_out in zip(input_files, output_files):
	# Export TensorRT engine
	cmd = f'{trtexec} --onnx="{input_folder}/{f_in}" --saveEngine="{output_folder}/{f_out}" --fp16'
	print(f'running:\t{cmd}')
	os.system(cmd)
	
	# Copy class_mappings.json if it exists
	class_mappings_src = f'{input_folder}/{f_in.replace(".onnx", "_class_mappings.json")}'
	if os.path.exists(class_mappings_src):
		class_mappings_dst = f'{output_folder}/{f_out.replace(".engine", "_class_mappings.json")}'
		shutil.copy2(class_mappings_src, class_mappings_dst)
		print(f'Copied class mappings: {class_mappings_dst}')