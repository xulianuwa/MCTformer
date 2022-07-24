######### Generating class-specific localization maps ##########
python main.py --model deit_small_MCTformerV2_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list voc12 \
                --data-path /Datasets/VOCdevkit/VOC2012 \
                --resume MCTformerV2.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --attention-dir /MCTformer_results/MCTformer_v2/attn-patchrefine \
                --cam-npy-dir /MCTformer_results/MCTformer_v2/attn-patchrefine-npy \
                --out-crf /MCTformer_results/MCTformer_v2/attn-patchrefine-npy-crf \

python psa/train_aff.py --weights res38_cls.pth \
                        --voc12_root /Datasets/VOCdevkit/VOC2012 \
                        --la_crf_dir /MCTformer_results/MCTformer_v2/attn-patchrefine-npy-crf_1 \
                        --ha_crf_dir /MCTformer_results/MCTformer_v2/attn-patchrefine-npy-crf_12 \


python psa/infer_aff.py --weights resnet38_aff.pth \
                    --infer_list voc12/train_id.txt \
                    --cam_dir /MCTformer_results/MCTformer_v2/attn-patchrefine-npy \
                    --voc12_root /Datasets/VOCdevkit/VOC2012 \
                    --out_rw /MCTformer_results/MCTformer_v2/pgt-psa-rw \

python evaluation.py --list voc12/train_id.txt \
                     --gt_dir /Desktop/Datasets/VOCdevkit/VOC2012/SegmentationClassAug \
                     --logfile /MCTformer_results/MCTformer_v2/pgt-psa-rw/evallog.txt \
                     --type png \
                     --predict_dir /MCTformer_results/MCTformer_v2/pgt-psa-rw \
                     --comment "train 1464"
