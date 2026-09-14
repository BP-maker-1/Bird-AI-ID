import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import zipfile
import requests

COMMON_NAMES = [
    "New Zealand Fantail",
    "Tui",
    "Bellbird",
    "Kereru",
    "Silvereye",
    "Welcome Swallow",
    "Grey Warbler",
    "New Zealand Kingfisher",
    "Australasian Harrier",
    "Morepork",
    "North Island Robin",
    "South Island Robin",
    "North Island Tomtit",
    "South Island Tomtit",
    "North Island Rifleman",
    "Whitehead",
    "Yellowhead",
    "Stitchbird",
    "Fernbird",
    "South Island Saddleback",
    "North Island Saddleback",
    "Weka",
    "Takahe",
    "New Zealand Buff-banded Rail",
    "North Island Brown Kiwi",
    "Southern Brown Kiwi",
    "Great Spotted Kiwi",
    "Little Spotted Kiwi",
    "Kea",
    "Kaka",
    "Kakapo",
    "Yellow-crowned Parakeet",
    "Red-crowned Parakeet",
    "Paradise Shelduck",
    "Pukeko",
    "Australasian Swamphen",
    "Black Swan",
    "Mallard",
    "Pacific Black Duck",
    "New Zealand Scaup",
    "New Zealand Shoveler",
    "Australasian Grebe",
    "New Zealand Grebe",
    "Eurasian Coot",
    "Blue Duck",
    "Variable Oystercatcher",
    "South Island Pied Oystercatcher",
    "Pied Stilt",
    "Bar-tailed Godwit",
    "Wrybill",
    "New Zealand Plover",
    "Spur-winged Plover",
    "Band-dotted Dotterel",
    "Ruddy Turnstone",
    "Red-billed Gull",
    "Black-backed Gull",
    "Black-fronted Tern",
    "Caspian Tern",
    "White-fronted Tern",
    "Australasian Gannet",
    "Pied Cormorant",
    "Little Black Cormorant",
    "Little Pied Cormorant",
    "Great Cormorant",
    "Spotted Shag",
    "Chatham Shag",
    "Fiordland Penguin",
    "Little Blue Penguin",
    "Yellow-eyed Penguin",
    "Southern Royal Albatross",
    "Salvin's Albatross",
    "Shy Albatross",
    "Northern Giant Petrel",
    "Fairy Prion",
    "Fluttering Shearwater",
    "Sooty Shearwater",
    "Brown Skua",
    "White-faced Heron",
    "Reef Heron",
    "Australasian Bittern",
    "Royal Spoonbill",
    "Australian Magpie",
    "Common Myna",
    "European Starling",
    "European House Sparrow",
    "Common Chaffinch",
    "European Greenfinch",
    "European Goldfinch",
    "Common Redpoll",
    "Yellowhammer",
    "Dunnock",
    "Eurasian Blackbird",
    "Song Thrush",
    "Skylark",
    "Barbary Dove",
    "Feral Pigeon",
    "California Quail",
    "Brown Quail",
    "Ring-necked Pheasant",
    "Domestic Turkey",
    "Little Owl",
]

DATASET_DIR = Path("./nz_birds_100_species")
ZIP_FILENAME = "nz_birds_100_species_20k.zip"
IMAGES_PER_SPECIES = 200
MAX_WORKERS = 16  # Downloads up to 16 images concurrently


def download_single_image(session, img_url, file_path):
    """Worker function to fetch and write individual image files."""
    try:
        resp = session.get(img_url, timeout=10)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def process_species(common_name, session):
    """Fetches image URLs for a species and downloads them using a worker pool."""
    folder_name = re.sub(r'[\\/*?:"<>|]', "", common_name).strip().replace(" ", "_")
    species_folder = DATASET_DIR / folder_name
    species_folder.mkdir(exist_ok=True)

    image_urls = []
    page = 1

    # Fetch image metadata from API
    while len(image_urls) < IMAGES_PER_SPECIES:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "q": common_name,
            "place_id": 6803,
            "photos": "true",
            "per_page": 100,
            "page": page,
            "quality_grade": "research",
            "iconic_taxa": "Aves",
        }
        try:
            res = session.get(url, params=params, timeout=15).json()
            results = res.get("results", [])
            if not results:
                break

            for obs in results:
                for photo in obs.get("photos", []):
                    if len(image_urls) >= IMAGES_PER_SPECIES:
                        break
                    img_url = photo["url"].replace("square", "medium")
                    image_urls.append((photo["id"], img_url))

            page += 1
            if len(results) < 100:
                break
        except Exception:
            break

    # Download image files concurrently
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, (photo_id, img_url) in enumerate(image_urls, 1):
            file_path = species_folder / f"{idx:03d}_{photo_id}.jpg"
            futures.append(
                executor.submit(
                    download_single_image, session, img_url, file_path
                )
            )

        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"[✓] {common_name}: {success_count} images downloaded.")
    return success_count


def main():
    DATASET_DIR.mkdir(exist_ok=True)

    # Establish connection pooling to reuse HTTP connections
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS
    )
    session.mount("https://", adapter)

    total_images = 0
    print(f"Starting multithreaded download across {len(COMMON_NAMES)} species...")

    for idx, common_name in enumerate(COMMON_NAMES, 1):
        print(f"[{idx}/100] Querying '{common_name}'...")
        count = process_species(common_name, session)
        total_images += count

    print("\nCompressing downloaded images into ZIP archive...")
    with zipfile.ZipFile(ZIP_FILENAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, DATASET_DIR)
                zipf.write(full_path, arcname)

    print(f"\nDone! {total_images} images saved to '{ZIP_FILENAME}'.")


if __name__ == "__main__":
    main()
