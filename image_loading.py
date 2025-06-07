# Step 1: Open the file
with open('C:/Users/abish/Desktop/Guvi_Files/Project_Final/archive/Text/captions.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Step 2: Create a dictionary to store image and its captions
captions = {}

for line in lines[1:]:  # skip the header line "image,caption"
    image, caption = line.strip().split(',', 1)  # split only at the first comma
    if image not in captions:
        captions[image] = []
    captions[image].append(caption)

# Step 3: Print the number of images and sample captions
print("Total images:", len(captions))

# Print captions for the first image
first_image = list(captions.keys())[0]
print("\nImage name:", first_image)
print("Captions:")
for cap in captions[first_image]:
    print("-", cap)

