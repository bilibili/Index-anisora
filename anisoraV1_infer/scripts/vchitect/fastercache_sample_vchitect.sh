save_dir=output
ckpt_path=/your/path/to/Vchitect-2.0/Vchitect-2.0-2B
python scripts/vchitect/fastercache_sample_vchitect.py --test_file /your/path/to/prompt.txt --save_dir "${save_dir}" --ckpt_path "${ckpt_path}"
