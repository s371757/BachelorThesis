#!/usr/bin/env python3
import os
import numpy as np
import requests
import argparse
import json
import time
import logging
import csv
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL, HTTPError

# Setup argument parser
parser = argparse.ArgumentParser(description='ImageNet image scraper')
parser.add_argument('-scrape_only_flickr', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-number_of_classes', default=10, type=int)
parser.add_argument('-images_per_class', default=50, type=int)
parser.add_argument('-data_root', default='imagenet', type=str)
parser.add_argument('-use_class_list', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-class_list', default=[], nargs='*')
parser.add_argument('-debug', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

# Setup logging
logging.basicConfig(filename='imagenet_scraper.log', level=logging.DEBUG if args.debug else logging.INFO)
logging.info("Starting ImageNet scraper")

if not os.path.isdir(args.data_root):
    logging.error(f'Folder {args.data_root} does not exist! Please provide an existing folder in -data_root arg!')
    exit(1)

IMAGENET_API_WNID_TO_URLS = lambda wnid: f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'
current_folder = os.path.dirname(os.path.realpath(__file__))
class_info_json_filepath = os.path.join(current_folder, 'imagenet_class_info.json')

with open(class_info_json_filepath) as class_info_json_f:
    class_info_dict = json.load(class_info_json_f)

# Select classes to scrape
classes_to_scrape = []
if args.use_class_list:
    for item in args.class_list:
        if item in class_info_dict:
            classes_to_scrape.append(item)
        else:
            logging.error(f'Class {item} not found in ImageNet')
            exit(1)
else:
    potential_class_pool = [
        key for key, val in class_info_dict.items()
        if (args.scrape_only_flickr and int(val['flickr_img_url_count']) * 0.9 > args.images_per_class) or
        (not args.scrape_only_flickr and int(val['img_url_count']) * 0.8 > args.images_per_class)
    ]
    if len(potential_class_pool) < args.number_of_classes:
        logging.error(f"With {args.images_per_class} images per class there are only {len(potential_class_pool)} classes to choose from. Decrease number of classes or decrease images per class.")
        exit(1)
    classes_to_scrape = np.random.choice(potential_class_pool, args.number_of_classes, replace=False).tolist()

logging.info(f"Picked classes: {[class_info_dict[class_wnid]['class_name'] for class_wnid in classes_to_scrape]}")

imagenet_images_folder = os.path.join(args.data_root, 'imagenet_images')
os.makedirs(imagenet_images_folder, exist_ok=True)

scraping_stats = {
    'all': {'tried': 0, 'success': 0, 'time_spent': 0},
    'is_flickr': {'tried': 0, 'success': 0, 'time_spent': 0},
    'not_flickr': {'tried': 0, 'success': 0, 'time_spent': 0}
}

def add_debug_csv_row(row):
    with open('stats.csv', "a") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",")
        csv_writer.writerow(row)

if args.debug:
    add_debug_csv_row([
        "all_tried", "all_success", "all_time_spent",
        "is_flickr_tried", "is_flickr_success", "is_flickr_time_spent",
        "not_flickr_tried", "not_flickr_success", "not_flickr_time_spent"
    ])

def add_stats_to_debug_csv():
    add_debug_csv_row([
        scraping_stats['all']['tried'],
        scraping_stats['all']['success'],
        scraping_stats['all']['time_spent'],
        scraping_stats['is_flickr']['tried'],
        scraping_stats['is_flickr']['success'],
        scraping_stats['is_flickr']['time_spent'],
        scraping_stats['not_flickr']['tried'],
        scraping_stats['not_flickr']['success'],
        scraping_stats['not_flickr']['time_spent']
    ])

def print_stats(cls):
    actual_all_time_spent = time.time() - scraping_t_start
    processes_all_time_spent = scraping_stats['all']['time_spent']
    actual_processes_ratio = actual_all_time_spent / processes_all_time_spent if processes_all_time_spent else 1.0

    logging.info(f'STATS For class {cls}:')
    logging.info(f' tried {scraping_stats[cls]["tried"]} urls with {scraping_stats[cls]["success"]} successes')
    if scraping_stats[cls]["tried"] > 0:
        logging.info(f'{100.0 * scraping_stats[cls]["success"] / scraping_stats[cls]["tried"]}% success rate for {cls} urls')
    if scraping_stats[cls]["success"] > 0:
        logging.info(f'{scraping_stats[cls]["time_spent"] * actual_processes_ratio / scraping_stats[cls]["success"]} seconds spent per successful {cls} image download')

scraping_t_start = time.time()

def get_image(img_url):
    if not img_url:
        return

    logging.debug(f"Processing {img_url}")

    cls = 'is_flickr' if 'flickr' in img_url else 'not_flickr'
    if cls == 'not_flickr' and args.scrape_only_flickr:
        return

    t_start = time.time()

    def finish(status):
        t_spent = time.time() - t_start
        scraping_stats[cls]['time_spent'] += t_spent
        scraping_stats['all']['time_spent'] += t_spent
        scraping_stats[cls]['tried'] += 1
        scraping_stats['all']['tried'] += 1
        if status == 'success':
            scraping_stats[cls]['success'] += 1
            scraping_stats['all']['success'] += 1
        elif status == 'failure':
            pass
        else:
            logging.error(f'Invalid status {status}')
            exit(1)
        return

    try:
        img_resp = requests.get(img_url, timeout=10)
        img_resp.raise_for_status()
    except HTTPError as e:
        logging.debug(f"HTTP error {e.response.status_code} for URL {img_url}: {e}")
        return finish('failure')
    except (ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL) as e:
        logging.debug(f"Request failed for {img_url}: {e}")
        return finish('failure')

    if 'image' not in img_resp.headers.get('content-type', '') or len(img_resp.content) < 1000:
        return finish('failure')

    img_name = img_url.split('/')[-1].split("?")[0]
    if not img_name:
        return finish('failure')

    img_file_path = os.path.join(class_folder, img_name)
    logging.debug(f'Saving image in {img_file_path}')

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)

    return finish('success')

for class_wnid in classes_to_scrape:
    class_name = class_info_dict[class_wnid]["class_name"]
    logging.info(f'Scraping images for class "{class_name}"')
    url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

    time.sleep(0.05)
    resp = requests.get(url_urls)
    class_folder = os.path.join(imagenet_images_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)
    scraping_stats['all']['tried'] = 0
    scraping_stats['all']['success'] = 0

    urls = [url.decode('utf-8') for url in resp.content.splitlines()]
    logging.info(f"Total URLs to process: {len(urls)}")

    while scraping_stats['all']['success'] < args.images_per_class and urls:
        for url in urls:
            get_image(url)
            if scraping_stats['all']['success'] >= args.images_per_class:
                break
        if scraping_stats['all']['success'] < args.images_per_class:
            urls = urls[scraping_stats['all']['tried']:]  # Continue with remaining URLs

logging.info('Image scraping completed')
