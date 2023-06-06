import os
import pandas as pd
import logging
from glob import glob
from transformers import pipeline
from tenacity import retry, stop_after_attempt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
import shutil
import chardet
import json

class DataTransformer:
    def __init__(self, data_directory='data', csv_directory='csv_data', backup_directory='backup', attempts=3, chunksize=1000):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.data_directory = Path(data_directory)
        self.csv_directory = Path(csv_directory)
        self.backup_directory = Path(backup_directory)
        self.attempts = attempts
        self.chunksize = chunksize

        # Create directories if not exists
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.csv_directory.mkdir(parents=True, exist_ok=True)
        self.backup_directory.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(filename='data_transformation.log', level=logging.INFO)

    def detect_delimiter(self, data_file):
        with open(data_file, 'rb') as file:
            result = chardet.detect(file.read())
        return result['encoding']

    @retry(stop=stop_after_attempt(3))
    def convert_to_csv(self, data_file):
        try:
            file_format = data_file.suffix.lstrip('.')
            if file_format == 'csv':
                shutil.copy(data_file, self.csv_directory)
            elif file_format == 'json':
                df = pd.read_json(data_file)
                csv_file_path = self.csv_directory / data_file.with_suffix('.csv').name
                df.to_csv(csv_file_path, index=False)
            elif file_format == 'xlsx':
                df = pd.read_excel(data_file)
                csv_file_path = self.csv_directory / data_file.with_suffix('.csv').name
                df.to_csv(csv_file_path, index=False)
            else:
                delimiter = self.detect_delimiter(data_file)
                df = pd.read_csv(data_file, delimiter=delimiter)
                csv_file_path = self.csv_directory / data_file.with_suffix('.csv').name
                df.to_csv(csv_file_path, index=False)
            logging.info(f"File converted: {csv_file_path}")
        except Exception as e:
            logging.error(f"An error occurred during file conversion: {e}")

    def transform(self, data, transformation_function=None):
        if transformation_function is None:
            data["sentiment"] = self.sentiment_analyzer(data["text"])
        else:
            data = transformation_function(data)
        return data

    def worker(self, data_file):
        if data_file.is_file():
            self.convert_to_csv(data_file)
def transformation_worker(self, csv_file):
    try:
        # Backup the original data
        shutil.copy(csv_file, self.backup_directory / csv_file.name)

        # Initialize an empty DataFrame to hold transformed data
        transformed_df = pd.DataFrame()

        # Read and transform data in chunks
        chunk_iter = pd.read_csv(csv_file, chunksize=self.chunksize)
        for chunk in chunk_iter:
            try:
                transformed_chunk = self.transform(chunk)
                transformed_df = pd.concat([transformed_df, transformed_chunk], ignore_index=True)
            except Exception as e:
                logging.error(f"An error occurred during transformation of a chunk: {e}")
                continue

        # Write transformed data to csv
        transformed_df.to_csv(csv_file, index=False)

        logging.info(f"Data transformed: {csv_file}")
    except Exception as e:
        logging.error(f"An error occurred during data transformation: {e}")


    def run(self):
        data_files = list(self.data_directory.iterdir())
        if not data_files:
            logging.error("No data files found.")
            with open(self.data_directory / 'placeholder.data', 'w') as f:
                f.write('placeholder')
            return

        with Pool(processes=cpu_count()) as p:
            list(tqdm(p.imap(self.worker, data_files), total=len(data_files)))

        csv_files = list(self.csv_directory.iterdir())
        if not csv_files:
            logging.error("No CSV files found.")
            return

        with Pool(processes=cpu_count()) as p:
            list(tqdm(p.imap(self.transformation_worker, csv_files), total=len(csv_files)))

