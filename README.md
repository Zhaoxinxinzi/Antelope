# Antelope
# Official Implementation for Paper "Antelope: Potent and Concealed Jailbreak Attack Strategy" Accepted by CIKM2025

## Attack
```bash
# nudity
CUDA_VISIBLE_DEVICES=1 python attack.py --ori_sentence "two persons" --target_word 'naked'
#violence
CUDA_VISIBLE_DEVICES=1 python attack.py --ori_sentence "two monster" --target_word 'violence'
```

## Generate images
```bash
python generate.py -f prompt_path -o save_path -n number_per_prompt
```


