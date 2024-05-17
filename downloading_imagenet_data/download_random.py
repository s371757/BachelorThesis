#!/usr/bin/env python3
import os
import numpy as np
import requests
import argparse
import json
import time
import logging
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL, HTTPError

# Setup argument parser
parser = argparse.ArgumentParser(description='ImageNet random image downloader')
parser.add_argument('-number_of_folders', default=50, type=int)
parser.add_argument('-images_per_folder', default=50, type=int)
parser.add_argument('-output_dir', default='output', type=str, help='Directory where the folders will be created')
parser.add_argument('-scrape_only_flickr', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-debug', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

# Setup logging
logging.basicConfig(filename='imagenet_downloader.log', level=logging.DEBUG if args.debug else logging.INFO)
logging.info("Starting ImageNet random image downloader")
print("Starting ImageNet random image downloader")

# Check and create output directory if it doesn't exist
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
    logging.info(f'Created output directory {args.output_dir}')
print(f"Output directory '{args.output_dir}' exists.")

IMAGENET_API_WNID_TO_URLS = lambda wnid: f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'
current_folder = os.path.dirname(os.path.realpath(__file__))
class_info_json_filepath = os.path.join(current_folder, 'imagenet_class_info.json')

print(f"Loading class info from '{class_info_json_filepath}'")
logging.info(f"Loading class info from '{class_info_json_filepath}'")
with open(class_info_json_filepath) as class_info_json_f:
    class_info_dict = json.load(class_info_json_f)

print(f"Loaded class info for {len(class_info_dict)} classes")
logging.info(f"Loaded class info for {len(class_info_dict)} classes")

total_images_needed = args.number_of_folders * args.images_per_folder
print(f"Total images needed: {total_images_needed}")
logging.info(f"Total images needed: {total_images_needed}")

scraping_stats = {
    'all': {'tried': 0, 'success': 0, 'time_spent': 0}
}

def add_debug_csv_row(row):
    with open('stats.csv', "a") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",")
        csv_writer.writerow(row)

if args.debug:
    add_debug_csv_row([
        "all_tried", "all_success", "all_time_spent"
    ])

def add_stats_to_debug_csv():
    add_debug_csv_row([
        scraping_stats['all']['tried'],
        scraping_stats['all']['success'],
        scraping_stats['all']['time_spent']
    ])

def print_stats():
    actual_all_time_spent = time.time() - scraping_t_start
    processes_all_time_spent = scraping_stats['all']['time_spent']
    actual_processes_ratio = actual_all_time_spent / processes_all_time_spent if processes_all_time_spent else 1.0

    logging.info(f'STATS:')
    logging.info(f' tried {scraping_stats["all"]["tried"]} urls with {scraping_stats["all"]["success"]} successes')
    if scraping_stats["all"]["tried"] > 0:
        logging.info(f'{100.0 * scraping_stats["all"]["success"] / scraping_stats["all"]["tried"]}% success rate')
    if scraping_stats["all"]["success"] > 0:
        logging.info(f'{scraping_stats["all"]["time_spent"] * actual_processes_ratio / scraping_stats["all"]["success"]} seconds spent per successful image download')
    print(f'STATS:')
    print(f' tried {scraping_stats["all"]["tried"]} urls with {scraping_stats["all"]["success"]} successes')
    if scraping_stats["all"]["tried"] > 0:
        print(f'{100.0 * scraping_stats["all"]["success"] / scraping_stats["all"]["tried"]}% success rate')
    if scraping_stats["all"]["success"] > 0:
        print(f'{scraping_stats["all"]["time_spent"] * actual_processes_ratio / scraping_stats["all"]["success"]} seconds spent per successful image download')

scraping_t_start = time.time()

def download_image(img_url, folder_path):
    if not img_url:
        return False

    print(f"Processing {img_url}")

    t_start = time.time()

    def finish(status):
        t_spent = time.time() - t_start
        scraping_stats['all']['time_spent'] += t_spent
        scraping_stats['all']['tried'] += 1
        if status == 'success':
            scraping_stats['all']['success'] += 1
            return True
        elif status == 'failure':
            return False
        else:
            logging.error(f'Invalid status {status}')
            exit(1)

    try:
        img_resp = requests.get(img_url, timeout=10)
        img_resp.raise_for_status()
    except HTTPError as e:
        logging.debug(f"HTTP error {e.response.status_code} for URL {img_url}: {e}")
        print(f"HTTP error {e.response.status_code} for URL {img_url}: {e}")
        return finish('failure')
    except (ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL) as e:
        logging.debug(f"Request failed for {img_url}: {e}")
        print(f"Request failed for {img_url}: {e}")
        return finish('failure')

    if 'image' not in img_resp.headers.get('content-type', '') or len(img_resp.content) < 1000:
        print(f"Invalid content type or too small content, skipping URL: {img_url}")
        return finish('failure')

    if not img_resp.headers['content-type'].lower().startswith('image/jpeg'):
        print(f"Non-JPEG image, skipping URL: {img_url}")
        return finish('failure')

    img_name = img_url.split('/')[-1].split("?")[0]
    if not img_name:
        print(f"Empty image name, skipping URL: {img_url}")
        return finish('failure')

    img_file_path = os.path.join(folder_path, img_name)
    print(f'Saving image in {img_file_path}')
    logging.debug(f'Saving image in {img_file_path}')

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)

    return finish('success')

for i in range(args.number_of_folders):
    folder_name = f'random500_{i}'
    folder_path = os.path.join(args.output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    images_downloaded = 0
    potential_class_pool = [
        key for key, val in class_info_dict.items()
        if (args.scrape_only_flickr and int(val['flickr_img_url_count']) > 0) or
        (not args.scrape_only_flickr and int(val['img_url_count']) > 0)
    ]

    while images_downloaded < args.images_per_folder:
        classes_to_scrape = np.random.choice(potential_class_pool, min(len(potential_class_pool), args.images_per_folder - images_downloaded), replace=False).tolist()

        print(f"Creating folder {folder_name} with images from {len(classes_to_scrape)} classes")
        logging.info(f"Creating folder {folder_name} with images from {len(classes_to_scrape)} classes")

        for class_wnid in classes_to_scrape:
            class_name = class_info_dict[class_wnid]["class_name"]
            logging.info(f'Scraping images for class "{class_name}"')
            url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

            time.sleep(0.05)
            resp = requests.get(url_urls)
            urls = [url for url in resp.content.decode('utf-8').split('\r\n') if url]
            logging.info(f"Total URLs to process for class '{class_name}': {len(urls)}")

            for url in urls:
                if download_image(url, folder_path):
                    images_downloaded += 1
                    break

            if images_downloaded >= args.images_per_folder:
                break

        # Remove scraped classes from the potential pool
        potential_class_pool = [cls for cls in potential_class_pool if cls not in classes_to_scrape]

        if not potential_class_pool:
            logging.error(f"Not enough classes to fulfill request. Needed: {args.images_per_folder}, available: {len(potential_class_pool)}.")
            print(f"Not enough classes to fulfill request. Needed: {args.images_per_folder}, available: {len(potential_class_pool)}.")
            break

print_stats()
logging.info('Image downloading completed')
print('Image downloading completed')
