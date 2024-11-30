## YOLO

This program uses 2 pretrained models to detect person and clothes,`yolo11` and `deepfashion2_yolov8s-seg` which is trained on dataset `deepfashion2`.

I use `yolo11` to detect person in images to distinguish between `person wearing clothes` and `clothes-only`, and judge that if the main object take most of the image and no overlap.Then use `deepfashion2_yolov8s-seg` to get boxes of clothes ,especially clothes worn by people.

The output form is introduced.Specially, under each folder named after the kind of clothes, there is a file named `parameters.pkl` to store all the paramters of boxes of clothes for SAM to segment clothes.
