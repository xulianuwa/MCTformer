######### train MCTformerPlus ##########
python main.py  --data-path VOCdevkit/VOC2012 \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \

######## Generating class-specific localization maps ##########
python main.py --data-set VOC12MS \
                --img-list VOCdevkit/VOC2012/ImageSets/Segmentation \
                --data-path VOCdevkit/VOC2012 \
                --gen_attention_maps \
                --resume saved_model/checkpoint.pth \

######### Evaluating the generated class-specific localization maps ##########
python evaluation.py --list voc12/train_id.txt \
                     --gt_dir VOCdevkit/VOC2012/SegmentationClassAug \
                     --logfile cam-npy/evallog.txt \
                     --type npy \
                     --curve True \
                     --predict_dir cam-npy \
                     --comment "train1464"

