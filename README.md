# LSTM-Car-Following-Model
Time-series LSTM model trained on NGSIM data for car-following speed prediction (predict next speed)

Wanted to share this project as it will lead into other projects I am working on in the future. Please feel free to connect and reach out to me on [LinkedIn](www.linkedin.com/in/rohanbuch)

## Structure:
main.py - Training and evaluation script
preprocess.py - Preprocessing the NGSIM data used to train and test the model
RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv - Full original dataset used from NGSIM I80 vehicle trajectory data (reconstructed and without motorcycles)
processed_ngsim.csv - Preprocessed dataset (from NGSIM I80 data)

## Notes
1. Dependencies:
   - pandas
   - numpy
   - scikit-learn
   - keras / tensorflow
   - matplotlib
     
2. Dataset: NGSIM Vehicle Trajectory Data
   - From: "Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data"
   - Under attachments, download I-80-Emeryville-CA.zip
   - Navigate through i-80-vehicle-trajectory-data.zip --> vehicle-trajectory-data --> 0400pm-0415pm --> select the reconstructed csv file without motorcycles
   - Other time frames are available (may not be reconstructed data)
   
3. You can choose amonngst the vehicle_id's manually to see the model predict for a specific follower against their actual next speed. The training includes the whole dataset excluding the selected follower.
   - Note that not all vehicle IDs will work. You may have to filter through to find them. In trials I have found success with vehicles: 102, 103, 222, 267...
