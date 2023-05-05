python demo.py \
	  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml   \
	   --input ../../../data/crop_test_frame/images/*.jpg  \
	    --output ../../../data/detectron2_result \
	     --opts MODEL.WEIGHTS ../../../weights/model_final_2d9806.pkl
