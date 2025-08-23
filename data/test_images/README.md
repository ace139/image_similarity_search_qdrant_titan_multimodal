# Test Images Folder Structure

This directory contains organized test images for the food similarity search system.

## Directory Structure

```
test_images/
├── user1/          # Alice Chen - Health-conscious professional
│   ├── monday/
│   │   ├── breakfast/  # Steel-cut oats with berries
│   │   ├── lunch/      # Quinoa salad with vegetables  
│   │   └── dinner/     # Teriyaki tofu stir-fry
│   ├── tuesday/ ... friday/
│   ├── saturday/
│   └── sunday/
├── user2/          # Marcus Johnson - Athletic trainer
│   ├── monday/
│   │   ├── breakfast/  # Eggs with turkey bacon
│   │   ├── lunch/      # Grilled chicken with sweet potato
│   │   └── dinner/     # Pan-seared salmon
│   ├── tuesday/ ... sunday/
├── user3/          # Sofia Martinez - Busy mom
│   ├── monday/
│   │   ├── breakfast/  # Cereal with milk
│   │   ├── lunch/      # PB&J sandwich
│   │   └── dinner/     # Spaghetti and meatballs
│   ├── tuesday/ ... sunday/
├── user4/          # David Kim - College student
│   ├── monday/
│   │   ├── breakfast/  # Instant ramen with egg
│   │   ├── lunch/      # Cafeteria pizza
│   │   └── dinner/     # Korean bibimbap
│   ├── tuesday/ ... sunday/
└── user5/          # Emma Thompson - Foodie
    ├── monday/
    │   ├── breakfast/  # Avocado toast with poached egg
    │   ├── lunch/      # Buddha bowl with tahini
    │   └── dinner/     # Pan-seared duck breast
    ├── tuesday/ ... sunday/
```

## User Profiles

Each user has distinct dietary preferences and meal patterns across 7 days:

- **user1** (Alice Chen): Health-conscious professional - vegetarian/flexitarian meals
- **user2** (Marcus Johnson): Athletic trainer - high-protein, occasional fast food
- **user3** (Sofia Martinez): Busy mom - family-friendly, quick fixes, takeout
- **user4** (David Kim): College student - budget meals, dorm food, occasional splurges
- **user5** (Emma Thompson): Foodie - gourmet experiments, seasonal ingredients

## How to Add Images

1. **Choose the right user and day** based on the meal profile:
   - Each user has Monday through Sunday folders
   - Each day has breakfast, lunch, and dinner subfolders
   - Total: 5 users × 7 days × 3 meals = 105 image slots

2. **Choose the right meal folder**:
   - `breakfast/`: Morning meals
   - `lunch/`: Midday meals  
   - `dinner/`: Evening meals

3. **Image requirements**:
   - Formats: JPG, PNG, WEBP, GIF, BMP, TIFF
   - Resolution: Minimum 800x600, preferred 1200x900
   - Content: Food should be the main subject
   - Lighting: Good natural lighting preferred
   - Background: Clean, uncluttered

4. **Naming convention**:
   - Use descriptive names: `oatmeal_berries.jpg`, `grilled_chicken.png`
   - Avoid spaces, use underscores or hyphens

## Running the Upload Script

After adding images to the folders, run:

```bash
# Test what would be uploaded (dry run)
python data/upload_test_dataset.py --dry-run

# Actually upload the images
python data/upload_test_dataset.py
```

The script will:
1. Find images in each user/meal folder
2. Generate detailed descriptions using Claude Vision
3. Create multi-modal embeddings (image + description)
4. Upload images and metadata to S3
5. Index embeddings in S3 Vectors for similarity search
