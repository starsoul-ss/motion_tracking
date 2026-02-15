mkdir -p dataset
# Define dataset root
DATASET_ROOT=/home/axell/Desktop/dataset_new/retarget_g1 # change to your dataset root path

python scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/AMASS --mem-path dataset/amass_all
python scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/LAFAN --mem-path dataset/lafan_all