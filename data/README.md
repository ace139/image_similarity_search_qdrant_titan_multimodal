# 🍽️ Test Dataset for Image Similarity Search

This directory contains a comprehensive test dataset for your image similarity search application.

## 📁 Directory Structure

```
data/
├── README.md                    # This file
├── test_dataset.json           # Dataset configuration and metadata
├── upload_test_dataset.py      # Batch upload script
└── test_images/                 # Image files organized by user and meal
    ├── user1/
    │   ├── breakfast/
    │   ├── lunch/
    │   └── dinner/
    ├── user2/
    │   ├── breakfast/
    │   ├── lunch/
    │   └── dinner/
    ├── user3/
    │   ├── breakfast/
    │   ├── lunch/
    │   └── dinner/
    ├── user4/
    │   ├── breakfast/
    │   ├── lunch/
    │   └── dinner/
    └── user5/
        ├── breakfast/
        ├── lunch/
        └── dinner/
```

## 👥 Test Users & Meals

### 🥗 User 1 - Alice Chen (Health-conscious vegetarian)
- **Breakfast**: Healthy oatmeal bowl with berries
- **Lunch**: Quinoa salad with roasted vegetables  
- **Dinner**: Grilled tofu stir-fry

### 💪 User 2 - Marcus Johnson (Fitness enthusiast)
- **Breakfast**: High-protein scrambled eggs with bacon
- **Lunch**: Grilled chicken breast with sweet potato
- **Dinner**: Pan-seared salmon with roasted vegetables

### 🌿 User 3 - Sofia Martinez (Mediterranean family meals)
- **Breakfast**: Greek yogurt with granola and honey
- **Lunch**: Mediterranean wrap with hummus
- **Dinner**: Pasta primavera with vegetables

### 🍜 User 4 - David Kim (Asian cuisine explorer)
- **Breakfast**: Traditional congee (rice porridge)
- **Lunch**: Homemade ramen bowl with egg
- **Dinner**: Korean bibimbap rice bowl

### 🍾 User 5 - Emma Thompson (Gourmet foodie)
- **Breakfast**: Artisanal avocado toast with poached egg
- **Lunch**: Gourmet burrata salad
- **Dinner**: Fine dining duck breast with gastrique

## 📸 Adding Images

Manually find and add food images for each meal:

**Step 1: Find Images**
For each meal, search for food images using the suggested search terms in `test_dataset.json`:

**Example searches:**
- "healthy breakfast bowl oatmeal berries"
- "scrambled eggs bacon protein breakfast"  
- "quinoa salad vegetables colorful bowl"
- "grilled chicken breast sweet potato"
- "korean bibimbap rice bowl colorful"

**Step 2: Image Requirements**
- **Format**: JPG, PNG, WebP
- **Resolution**: Minimum 800x600, preferred 1200x900
- **Quality**: Good lighting, food as main subject
- **Background**: Clean, uncluttered
- **Style**: Attractively plated food photos

**Step 3: Organize Images**
Place one image per meal type per user:
```
data/test_images/user1/breakfast/oatmeal_bowl.jpg
data/test_images/user1/lunch/quinoa_salad.jpg
data/test_images/user1/dinner/tofu_stirfry.jpg
data/test_images/user2/breakfast/eggs_bacon.jpg
... and so on
```

## 🚀 Uploading the Dataset

### Dry Run (Test without uploading)
```bash
python data/upload_test_dataset.py --dry-run
```

### Actual Upload
```bash
python data/upload_test_dataset.py
```

This will:
1. ✅ Read all images from `test_images/` directory
2. ✅ Generate embeddings using Bedrock Titan model
3. ✅ Upload images to S3 (`images/` prefix)
4. ✅ Upload metadata + embeddings to S3 (`embeddings/` prefix)
5. ✅ Index vectors in S3 Vectors with optimized metadata
6. ✅ Apply all meal-specific metadata (tags, calories, protein, etc.)

## 🔍 Testing Similarity Search

Once uploaded, you can test various search scenarios:

### User-Specific Searches
- Search for "Alice" (user1) meals only
- Search for "Marcus" (user2) high-protein meals
- Search for "Sofia" (user3) Mediterranean dishes

### Meal Type Filtering  
- Find all breakfast images
- Find all dinner images
- Find lunch meals with >20g protein

### Tag-Based Filtering
- Search for "vegetarian" meals
- Find "asian" cuisine
- Look for "healthy" options

### Nutritional Filtering
- High protein meals (>30g)
- Lower calorie options (<400 cal)
- Specific dietary restrictions

### Similarity Searches
- Upload a new breakfast image and find similar breakfasts
- Search for "grilled chicken" and see what matches
- Upload an Asian dish and find similar Asian meals

## 📊 Expected Results

With this diverse dataset, you should see:

- **Good clustering**: Similar meal types group together
- **User preference patterns**: Each user's dietary style is distinct
- **Cross-meal similarities**: Asian dishes across users should cluster
- **Nutritional correlations**: High-protein meals should be similar
- **Visual similarity**: Well-plated dishes should match regardless of cuisine

## 🎯 Perfect for Testing

This dataset is ideal for demonstrating:
- Multi-user image similarity search
- Metadata filtering and search
- Visual similarity across different cuisines  
- User-specific recommendation patterns
- Nutritional content-based filtering

---

**Ready to test your image similarity search! 🚀**
