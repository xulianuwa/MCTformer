
python seg/train_seg.py --network resnet38_seg \
                    --num_epochs 30 \
                    --seg_pgt_path \
                    --init_weights \
                    --save_path  \
                    --list_path voc12/train_aug_id.txt \
                    --img_path \
                    --num_classes 21 \
                    --batch_size 4

python seg/infer_seg.py --weights \
                      --network resnet38_seg \
                      --list_path voc12/val_id.txt \
                      --gt_path \
                      --img_path \
                      --save_path val_ms_crf \
                      --save_path_c val_ms_crf_c \
                      --scales 0.5 0.75 1.0 1.25 1.5 \
                      --use_crf True