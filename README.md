# CS3244-NIS
Applying Neural Networks to Detect Hacking Attempts

Attacks from hackers can be a serious security threat especially with the prevalence of the internet. Our group has attempted to apply machine learning in building a Network Intrusion Detection System (NIDS).

Our project aims to explore temporal network traffic data to extract relevant features that differentiate network intrusion from normal traffic.

For a practical evaluation of the project, we will attempt to detect and block network traffic while an attack is performed.

Training data is not included in this repository.

# Running the code

To perform training and testing, run `classifier_train_and_test.py`. Test and training data folders are to be listed in `normal_dirs` (for normal traffic) and `breach_dirs` for attack traffic. There are other variables that should be defined too, such as `b_size` (batch size), `num_of_test_files`, `num_of_val_files` and `limit` (training size = limit - test files - val files).

After training, several other files are created in the same directory. These files are images of the change in loss and accuracy over each epoch, encoded data to be used for PCA analysis and further testing as well as the trained models.

# Running PCA analysis

Code located in the PCA_Analysis folder. The script `create_view.py` should be pretty self-explanatory.

# Zero-day detection

Run `classifier_test.py`, which is basically the training script except it reads the data from the saved files and instead only tests on folders within `normal_dirs` and `breach_dirs`. The final accuracy for the test files will be output.

# Realtime detection

Code is in `realtime_IDS.py`, using the same readings as those in zero-day detection classifier_test, except this time it keeps watch on a folder. Any time a packet appears in the folder, it will then read the packet and detect if it is malicious or not. If it is, it will ssh into a server and drop the IP using iptables to stop the attack.