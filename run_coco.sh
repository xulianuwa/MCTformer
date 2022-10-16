######### train MCTformer V2 ##########
python main.py --model deit_small_MCTformerV2_patch16_224 \
                --batch-size 64 \
                --data-set COCO \
                --img-list coco \
                --data-path /Datasets/MSCOCO \
                --layer-index 12 \
                --output_dir /MCTformer_results/MCTformer_v2/coco \
                --label-file-path COCO_cls_label.npy \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

######### Generating class-specific localization maps ##########
python main.py --model deit_small_MCTformerV2_patch16_224 \
                --data-set COCOMS \
                --scales 1.0 \
                --img-list coco \
                --data-path /Datasets/MSCOCO \
                --output_dir /MCTformer_results/MCTformer_v2/coco \
                --resume /MCTformer_results/MCTformer_v2/coco/MCTformerV2_coco.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --label-file-path COCO_cls_labels.npy \
                --attention-dir /MCTformer_results/MCTformer_v2/coco/fused-patchrefine \
                --cam-npy-dir /MCTformer_results/MCTformer_v2/coco/fused-patchrefine-npy