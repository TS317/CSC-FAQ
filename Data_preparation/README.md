Note:
- 'generate_frm_h5.py' is used for generating .h5 files from your datasets, in a parrallel mode;
For example, original input files are video frames and has data structure similar to **/subjects/videos/frames**;
And the output are subject-based. If you have 100 subjects in your dataset, the code will generate 100 .h5 file, and each contains all the video data of one subject;

If the dataset is too large (contains many subjects' data), you can split it into 2 parts and run each part individually. For example, in the code, the data are split into two parts with arguments 'part'.

- 'generate_frm_h5.py' is used for generating .h5 files from your datasets, where the original input data are not frames, but videos

- 'data_loader_h5.py' is the code used for load .h5 file from disk for training.
