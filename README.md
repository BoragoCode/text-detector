<h1>Text Detector</h1>

Model obtains 92% accuracy. Most of the models loss is due to the datasets
including text that is bigger than the kernel size. An increase in accuracy
would be expected if these images where excluded from the training and testing datasets.

If you would like to train this model you will need to Keras with a Tensorflow backend.

You will also need to download the training and testing datasets, these can be found here: https://drive.google.com/file/d/1pCttVLUqRgdfCBYqlbyowjjXEF04RTT7/view?usp=sharing

To generate model use:
python text_detector_generator.py

If you want to run the existing scripts to utilize this model you will need opencv.

To use existing model to make a prediction use (image must be 60x100px):
python detect_text.py path/to/image

If you would like to use any image use:
python detect_draw_text.py path/to/image
