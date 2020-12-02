#!/bin/sh
set -e

LG=$1
WIKI_DUMP_NAME=${LG}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/${LG}wiki/latest/$WIKI_DUMP_NAME

# download latest Wikipedia dump in chosen language
echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c -N $WIKI_DUMP_DOWNLOAD_URL -P data/wikidump 
echo "Succesfully downloaded the latest $LG-language Wikipedia dump to data/wikidump"

# extract downloaded dump file
echo "Extracting the latest $LG-language Wikipedia dump to data/wikiextracted/$LG/raw..."
python3 -m wikiextractor.WikiExtractor --processes 4 -o data/wikiextracted/$LG/raw data/wikidump/$WIKI_DUMP_NAME
echo "Succesfully extracted the latest $LG-language Wikipedia dump to data/wikiextracted/$LG/raw"

# clean downloaded dump file
echo "Clean the extracted $LG-language Wikipedia files..."
python3 clean_extracted_wiki_dump.py --language $LG
echo "Succesfully cleaned the extracted $LG-language Wikipedia files and save to data/wikiextracted/$LG/cleaned"


