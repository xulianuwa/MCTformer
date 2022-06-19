######### train MCTformer V1 ##########
python main.py --model deit_small_MCTformerV1_patch16_224 \
                --batch-size 64 \
                --data-set VOC12 \
                --img-list voc12 \
                --data-path /Datasets/VOCdevkit/VOC2012 \
                --layer-index 12 \
                --output_dir /MCTformer_results/MCTformer_v1 \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

######### train MCTformer V2 ##########
python main.py --model deit_small_MCTformerV2_patch16_224 \
                --batch-size 64 \
                --data-set VOC12 \
                --img-list voc12 \
                --data-path /Datasets/VOCdevkit/VOC2012 \
                --layer-index 12 \
                --output_dir /MCTformer_results/MCTformer_v2 \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

######### Generating class-specific localization maps ##########
python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list Datasets/VOCdevkit/VOC2012/ImageSets/Segmentation \
                --data-path /Datasets/VOCdevkit/VOC2012 \
                --output_dir /MCTformer_results/MCTformer_v1 \
                --resume /MCTformer_results/MCTformer_v1/checkpoint.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --attention-dir /MCTformer_results/MCTformer_v1/attn-patchrefine \
                --cam-npy-dir /MCTformer_results/MCTformer_v1/attn-patchrefine-npy \


######### Evaluating the generated class-specific localization maps ##########
python evaluation.py --list /Datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_id.txt \
                     --gt_dir /Datasets/VOCdevkit/VOC2012/SegmentationClassAug \
                     --logfile /MCTformer_results/MCTformer_v1/attn-patchrefine-npy/evallog.txt \
                     --type npy \
                     --curve True \
                     --predict_dir /MCTformer_results/MCTformer_v1/attn-patchrefine-npy \
                     --comment "train1464"