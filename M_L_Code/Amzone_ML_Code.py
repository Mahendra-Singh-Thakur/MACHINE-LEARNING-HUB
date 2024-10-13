# # import cv2
# # import easyocr
# # import matplotlib.pyplot as plt
# # import re
# # from matplotlib import rcParams

# # # Define unit categories
# # unit_categories = {
# #     'length': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM'},
# #     'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM'},
# #     'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM'},
# #     'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM'},
# #     'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton', 'g', 'kg', 'µg', 'mg', 'oz', 'lb', 'ton'},
# #     'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton', 'g', 'kg', 'µg', 'mg', 'oz', 'lb', 'ton'},
# #     'voltage': {'kilovolt', 'millivolt', 'volt', 'kv', 'mv', 'v'},
# #     'wattage': {'kilowatt', 'watt', 'w', 'KW', 'W', 'w'},
# #     'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart', 'cl', 'cf', 'in³', 'cup', 'dl', 'floz', 'gal', 'imp gal', 'l', 'µl', 'ml', 'pt', 'qt'}
# # }

# # def contains_units(term, units):
# #     return any(unit in term for unit in units)

# # def filter_terms_by_category(terms, category):
# #     units = unit_categories.get(category, set())
# #     return [term for term in terms if contains_units(term, units)]

# # def process_image_for_units(img, category):
# #     try:
# #         # Set figure size for matplotlib
# #         rcParams['figure.figsize'] = 8, 16

# #         if img is None:
# #             raise ValueError("Image is None")

# #         # Convert image to grayscale if not already
# #         if len(img.shape) == 3 and img.shape[2] == 3:
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #         # Preprocess the image
# #         img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
# #         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #         img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
# #         # cv2.imshow("Enhanced Black and White Image",img)
# #         # cv2.waitKey(0)

# #         # Initialize EasyOCR reader
# #         reader = easyocr.Reader(['en'])
# #         # Read text from the image
# #         output = reader.readtext(img, detail=0)  # Using detail=0 to get text only

# #         # Combine all text segments into a single paragraph
# #         paragraph = ' '.join(output)
# #         paragraph = paragraph.lower()

# #         # Display the extracted paragraph
# #         print("Extracted Paragraph:")
# #         print(paragraph)

# #         # Function to extract words containing numeric characters and units
# #         def extract_words_with_numbers_and_units(text):
# #             # Regular expression to match numeric values and units
# #             pattern = re.compile(r'(\d+(\.\d+)?|\.\d+)\s*([a-zA-Z]+)')
# #             matches = pattern.findall(text)
# #             return [f'{number} {unit.lower()}' for number, _, unit in matches]

# #         # Extract words containing numeric characters and units from the paragraph
# #         words_with_numbers_and_units = extract_words_with_numbers_and_units(paragraph)

# #         # Display extracted words containing numeric characters and units
# #         print("\nWords Containing Numeric Characters and Units:")
# #         for word in words_with_numbers_and_units:
# #             print(word)

# #         # Display the preprocessed image (optional)
# #         plt.imshow(img, cmap='gray')
# #         # plt.title('Preprocessed Image')
# #         # plt.axis('off')
# #         plt.show()

# #         # Filter terms based on the selected category
# #         filtered_terms = filter_terms_by_category(words_with_numbers_and_units, category)
# #         print("\nFiltered Terms Based on Category '{}':".format(category))
# #         print(filtered_terms)

# #         return filtered_terms  # Return the filtered terms for DataFrame

# #     except Exception as e:
# #         print(f"Error processing image: {e}")
# #         return []  # Return an empty list in case of error

# # # Example usage
# # img_path = 'img/image_9.jpg'  # Path to your image
# # category = 'length'  # Category to filter
# # img = cv2.imread(img_path)  # Load the image
# # process_image_for_units(img, category)


# import cv2
# import easyocr
# import matplotlib.pyplot as plt
# import re
# from matplotlib import rcParams  # Fixed this import
# unit_categories = {
#     'length': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM' ,'"'},
#     'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM','"'},
#     'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM','"'},
#     'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM','"'},
#     'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton', 'g', 'kg', 'µg', 'mg', 'oz', 'lb', 'ton'},
#     'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton', 'g', 'kg', 'µg', 'mg', 'oz', 'lb', 'ton'},
#     'voltage': {'kilovolt', 'millivolt', 'volt', 'kv', 'mv', 'v'},
#     'wattage': {'kilowatt', 'watt', 'w', 'KW', 'W', 'w'},
#     'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart', 'cl', 'cf', 'in³', 'cup', 'dl', 'floz', 'gal', 'imp gal', 'l', 'µl', 'ml', 'pt', 'qt'}
# }
# def contains_units(term, units):
#     return any(unit in term for unit in units)
# def convert_units(unit_list):
#     # Define the mapping from abbreviations to full names
#     unit_mapping = {
#         'cmi' : 'centimetre',
#         'centimeter': 'centimetre',
#         'cm': 'centimetre',
#         'ocm':'centimetre',
#         'inches': 'inch',
#         'inche':'inch',
#         'in': 'inch',
#         '"': 'inch',
#         'cm': 'centimetre',
#         'centimetre': 'centimetre',
#         'foot': 'feet',
#         'metre': 'metre',
#         'millimetre': 'millimetre',
#         'yard': 'yards',
#         'millimetre': 'millimetre',
#         'cm': 'centimetre',
#         'centimeter': 'centimetre',
#         'm': 'metre',
#         'meter': 'metre',
#         'ft': 'feet',
#         'feet': 'feet',
#         'yd': 'yards',
#         'yards': 'yard',
#         'in': 'inch',
#         'inch': 'inch',
#         'yard': 'yards',
#         'cm': 'centimetre',
#         'centimeter': 'centimetre',
#         'm': 'metre',
#         'meter': 'metre',
#         'ft': 'feet',
#         'feet': 'feet',
#         'yd': 'yard',
#         'yards': 'yard',
#         'in': 'inch',
#         'inch': 'inch',
#         'g': 'grams',
#         'gram': 'grams',
#         'kg': 'kilogram',
#         'kilogram': 'kilogram',
#         'µg': 'microgram',
#         'microgram': 'microgram',
#         'mg': 'milligram',
#         'milligram': 'milligram',
#         'oz': 'ounce',
#         'ounce': 'ounce',
#         'lb': 'pound',
#         'lbs':'pound',
#         'pound': 'pound',
#         'ton': 'ton',
#         'kw': 'kilowatt',
#         'kw': 'kilowatt',
#         'w': 'watt',
#         'volt': 'volt',
#         'kilovolt': 'kilovolt',
#         'millivolt': 'millivolt',
#         'v': 'volt',
#         'kv': 'kilovolt',
#         'mv': 'millivolt',
#         'liter': 'litre',
#         'litre': 'litre',
#         'l': 'litre',
#         'cl': 'centilitre',
#         'µl': 'microlitre',
#         'centilitre': 'centilitre',
#         'cubic foot': 'cubic feet',
#         'cubic inch': 'cubic inche',
#         'cf': 'cubic feet',
#         'ml': 'millilitre',
#         'millilitre': 'millilitre',
#         'in³': 'cubic inche',
#         'cubic inch': 'cubic inche',
#         'cubic foot': 'cubic feet',
#         'imp gal': 'imperial gallon',
#         'imperial gallon': 'imperial gallon',
#         'cup': 'cups',
#         'dl': 'decilitres',
#         'decilitre': 'decilitres',
#         'floz': 'fluid ounce',
#         'gal': 'gallon',
#         'pt': 'pint',
#         'qt': 'quart',
#         'fl oz': 'fluid ounce',
#         'gallon': 'gallon',
#         'pint': 'pint',
#         'quart': 'quart',
#         'fluid ounce': 'fluid ounce'
#     }
#     def convert_word(word):
#         # Split the word into numeric and unit parts
#         parts = word.split()
#         if len(parts) == 2:
#             numeric_part, unit_part = parts
#             numeric_part = float(numeric_part)  # Convert numeric part to float
#             # Convert unit part to its full name
#             full_unit = unit_mapping.get(unit_part, unit_part)
#             return f"{numeric_part} {full_unit}"
#         return word  # In case the format is unexpected

#     # Apply conversion to each item in the list
#     converted_list = [convert_word(item) for item in unit_list]
#     return converted_list
# def filter_terms_by_category(terms, category):
#     units = unit_categories.get(category, set())
#     return [term for term in terms if contains_units(term, units)]
# import re

# # Conversion factors for different categories
# conversion_factors = {
#     'length': {
#         'millimetre': 1,
#         'centimetre': 10,
#         'inch': 25.4,
#         'foot': 304.8,
#         'yard': 914.4,
#         'metre': 1000,
#         'kilometre': 1000000,
#         'cm': 10,
#         'CM': 10
#     },
#     'item_weight': {
#         'gram': 1,
#         'kilogram': 1000,
#         'microgram': 0.001,
#         'milligram': 0.001,
#         'ounce': 28.3495,
#         'pound': 453.592,
#         'ton': 1000000,
#         'g': 1,
#         'kg': 1000,
#         'µg': 0.001,
#         'mg': 0.001,
#         'oz': 28.3495,
#         'lb': 453.592,
#         'ton': 1000000
#     },
#     'voltage': {
#         'kilovolt': 1000000,
#         'millivolt': 0.001,
#         'volt': 1,
#         'kv': 1000000,
#         'mv': 0.001,
#         'v': 1
#     },
#     'wattage': {
#         'kilowatt': 1000,
#         'watt': 1,
#         'w': 1,
#         'KW': 1000,
#         'W': 1
#     },
#     'item_volume': {
#         'centilitre': 10,
#         'cubic foot': 28316.8466,
#         'cubic inch': 16.387,
#         'cup': 236.588,
#         'decilitre': 100,
#         'fluid ounce': 29.5735,
#         'gallon': 3785.41,
#         'imperial gallon': 4546.09,
#         'litre': 1000,
#         'microlitre': 0.001,
#         'millilitre': 1,
#         'pint': 473.176,
#         'quart': 946.353,
#         'cl': 10,
#         'cf': 28316.8466,
#         'in³': 16.387,
#         'dl': 100,
#         'floz': 29.5735,
#         'gal': 3785.41,
#         'imp gal': 4546.09,
#         'l': 1000,
#         'µl': 0.001,
#         'ml': 1,
#         'pt': 473.176,
#         'qt': 946.353
#     }
# }

# def parse_value_unit(value_unit_str):
#     # Extract numerical value and unit from the string
#     match = re.match(r'(\d+\.?\d*)\s*([a-zA-Z\s]+)', value_unit_str, re.IGNORECASE)
#     if match:
#         value = float(match.group(1))
#         unit = match.group(2).strip().lower()
#         return value, unit
#     return None

# def format_value(value, unit):
#     # Format the output as "value unit"
#     return f"{value} {unit}"

# def convert_to_standard(value, unit, category):
#     # Convert value to the base unit within the category
#     if unit in conversion_factors[category]:
#         base_value = value * conversion_factors[category][unit]
#         return base_value
#     return None

# def format_range(min_value, max_value, unit):
#     # Format the range in "[min-max] unit"
#     return f"[{min_value}-{max_value}] {unit}"

# def format_unit(value_unit_str, category):
#     # Handle range and single value cases
#     if '-' in value_unit_str:
#         min_str, max_str = value_unit_str.split('-')
#         min_parsed = parse_value_unit(min_str)
#         max_parsed = parse_value_unit(max_str)
#         if min_parsed and max_parsed:
#             min_value, min_unit = min_parsed
#             max_value, max_unit = max_parsed
#             if min_unit == max_unit:
#                 min_base = convert_to_standard(min_value, min_unit, category)
#                 max_base = convert_to_standard(max_value, max_unit, category)
#                 if min_base is not None and max_base is not None:
#                     min_formatted = min_base / conversion_factors[category][min_unit]
#                     max_formatted = max_base / conversion_factors[category][max_unit]
#                     return format_range(min_formatted, max_formatted, min_unit)
#     else:
#         parsed = parse_value_unit(value_unit_str)
#         if parsed:
#             value, unit = parsed
#             base_value = convert_to_standard(value, unit, category)
#             if base_value is not None:
#                 return format_value(base_value / conversion_factors[category][unit], unit)
#     return None

# def process_list(value_list, category):
#     results = []
#     for value_str in value_list:
#         formatted_output = format_unit(value_str, category)
#         if formatted_output:
#             results.append(formatted_output)
#     return results
# def convert_weight_to_kg(weight_str):
#     # Split the value and unit from the string
#     value, unit = weight_str.split()
#     value = float(value)

#     # Conversion factors to kilograms
#     conversion_factors = {
#         'g': 1e-3, 'gram': 1e-3, 'grams': 1e-3,
#         'kg': 1, 'kilogram': 1, 'kilograms': 1,
#         'µg': 1e-9, 'microgram': 1e-9, 'micrograms': 1e-9,
#         'mg': 1e-6, 'milligram': 1e-6, 'milligrams': 1e-6,
#         'oz': 0.0283495, 'ounce': 0.0283495, 'ounces': 0.0283495,
#         'lb': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
#         'ton': 1000, 'tons': 1000
#     }

#     # Normalize the unit
#     unit = unit.lower()
#     if unit not in conversion_factors:
#         raise ValueError(f"Unsupported unit: {unit}")

#     return value * conversion_factors[unit]

# def maximum_weight_cal(weight_list):
#     # Create a list to store tuples (original_weight, converted_to_kg)
#     weights_in_kg = [(weight_str, convert_weight_to_kg(weight_str)) for weight_str in weight_list]

#     # Find the maximum weight based on the kg values
#     max_weight_tuple = max(weights_in_kg, key=lambda x: x[1])

#     # Return the original format of the maximum weight
#     return max_weight_tuple[0]
# def all_same(lst):
#     # Check if the list is not empty and if all elements in the list are the same as the first element
#     return len(lst) > 0 and all(element == lst[0] for element in lst)
# reader = easyocr.Reader(['en'])
# a=0
# def update_var():
#     global a
#     a=a+1
#     print(a)
# def process_image_for_units(img, category):
#     try:
#         # Set figure size for matplotlib
#         rcParams['figure.figsize'] = 8, 16

#         if img is None:
#             raise ValueError("Image is None")

#         # Convert image to grayscale if not already
#         if len(img.shape) == 3 and img.shape[2] == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Preprocess the image
#         img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
#         _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

#         # Initialize EasyOCR reader


#         # Read text from the image
#         output = reader.readtext(img, detail=0)  # Using detail=0 to get text only

#         # Combine all text segments into a single paragraph
#         paragraph = ' '.join(output)
#         paragraph = paragraph.lower()

#         # Display the extracted paragraph
#         #print("Extracted Paragraph:")
#         #print(paragraph)

#         # Function to extract words containing numeric characters and units
#         def extract_words_with_numbers_and_units(text):
#             # Regular expression to match numeric values and units
#             pattern = re.compile(r'(\d+(\.\d+)?|\.\d+)\s*([a-zA-Z]+)')
#             matches = pattern.findall(text)
#             return [f'{number} {unit.lower()}' for number, _, unit in matches]

#         # Extract words containing numeric characters and units from the paragraph
#         words_with_numbers_and_units = extract_words_with_numbers_and_units(paragraph)

#         # Display extracted words containing numeric characters and units
#         #print("\nWords Containing Numeric Characters and Units:")
#             #print(word)

#         # Display the preprocessed image (optional)

#         # Filter terms based on the selected category
#         filtered_terms = filter_terms_by_category(words_with_numbers_and_units, category)
#         #print("\nFiltered Terms Based on Category '{}':".format(category))
#         #print(filtered_terms)

#         converted= convert_units(filtered_terms)
#         #print("\nConverted Units:")
#         #print(converted);
#         update_var()
#         string_="done+"
#         if(len(converted)==1):
#           print(string_)
#           return converted[0]
#         if(len(converted)==0):
#           print(string_)
#           return ""
#         if(all_same(converted)):
#           print(string_)
#           return converted[0]
#         if(category=="maximum_weight_recommendation"):
#           print(string_)
#           return maximum_weight_cal(converted)
#         if category == "length":
#           if len(converted) == 1:
#               print(string_)
#               return converted[0]
#           print(string_)
#           return max(converted)


#         elif category == "width":
#             print(string_)
#             return min(converted)

#         elif category in ["height", "depth"]:
#             print(string_)
#             return max(converted)

#         return "nhi"


#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return []  # Return an empty list in case of error
# """# Example usage
# img_path = '/content/81R8CrWXthL.jpg'  # Path to your image
# category = 'item_weight'  # Category to filter
# img = cv2.imread(img_path)  # Load the image
# process_image_for_units(img, category)"""
# import requests
# import numpy as np
# import cv2
# import requests
# # import pillow as PIL
# from PIL import Image
def fun():
    print("hello")
# def download_image(url):
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
#     }
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()  # Raise HTTPError for bad responses
#         return response.content  # Return the image content if successful
#     except requests.exceptions.RequestException as e:
#         raise ValueError(f"Failed to download image from URL: {url}, error: {e}")


# def process_row(row):
#     try:
#         # img = download_image(row.image_link)

# # Load the image
#         img = Image.open("preprocessed_image.jpg")
#         # Process the image (e.g., prediction logic)
#         return img  # Return the proce ssed image or prediction
#     except ValueError as e:
#         print(e)
#         return ""  # Return a default value for rows where image download fails

# from concurrent.futures import ThreadPoolExecutor, as_completed
# # Use ThreadPoolExecutor for parallel processing
# def process_dataframe_parallel(df, max_workers=10000):
#     # Default max_workers is optimized based on the system's available resources
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_row, row) for row in df.itertuples(index=False)]
        
#         predictions = []
#         for future in as_completed(futures):
#           predictions.append(future.result()) # Append empty string for rows with processing errors
#     return predictions

# # Load the CSV file
# import pandas as pd
# df = pd.read_csv('Data/Data copy.csv')

# # Process the DataFrame in parallel (without specifying max_workers lets ThreadPoolExecutor optimize the number of threads)
# df['prediction'] = process_dataframe_parallel(df)

# # Save the DataFrame with predictions to a new CSV file
# df.to_csv('result.csv', index=False)



import cv2
#pip install easyocr
import easyocr
import matplotlib.pyplot as plt
import re
from matplotlib import rcParams  # Fixed this import
unit_categories = {
    'length': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM' ,'"'},
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM','"'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM','"'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard', 'cm', 'CM','"'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton', 'g', 'kg', 'µg', 'mg', 'oz', 'lb', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton', 'g', 'kg', 'µg', 'mg', 'oz', 'lb', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt', 'kv', 'mv', 'v'},
    'wattage': {'kilowatt', 'watt', 'w', 'KW', 'W', 'w'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart', 'cl', 'cf', 'in³', 'cup', 'dl', 'floz', 'gal', 'imp gal', 'l', 'µl', 'ml', 'pt', 'qt'}
}
def contains_units(term, units):
    return any(unit in term for unit in units)
def convert_units(unit_list):
    # Define the mapping from abbreviations to full names
    unit_mapping = {
        'cmi' : 'centimetre',
        'centimeter': 'centimetre',
        'cm': 'centimetre',
        'ocm':'centimetre',
        'inches': 'inch',
        'inche':'inch',
        'in': 'inch',
        '"': 'inch',
        'cm': 'centimetre',
        'centimetre': 'centimetre',
        'foot': 'feet',
        'metre': 'metre',
        'millimetre': 'millimetre',
        'yard': 'yards',
        'millimetre': 'millimetre',
        'cm': 'centimetre',
        'centimeter': 'centimetre',
        'm': 'metre',
        'meter': 'metre',
        'ft': 'feet',
        'feet': 'feet',
        'yd': 'yards',
        'yards': 'yard',
        'in': 'inch',
        'inch': 'inch',
        'yard': 'yards',
        'cm': 'centimetre',
        'centimeter': 'centimetre',
        'm': 'metre',
        'meter': 'metre',
        'ft': 'feet',
        'feet': 'feet',
        'yd': 'yard',
        'yards': 'yard',
        'in': 'inch',
        'inch': 'inch',
        'g': 'grams',
        'gram': 'grams',
        'kg': 'kilogram',
        'kilogram': 'kilogram',
        'µg': 'microgram',
        'microgram': 'microgram',
        'mg': 'milligram',
        'milligram': 'milligram',
        'oz': 'ounce',
        'ounce': 'ounce',
        'lb': 'pound',
        'lbs':'pound',
        'pound': 'pound',
        'ton': 'ton',
        'kw': 'kilowatt',
        'kw': 'kilowatt',
        'w': 'watt',
        'volt': 'volt',
        'kilovolt': 'kilovolt',
        'millivolt': 'millivolt',
        'v': 'volt',
        'kv': 'kilovolt',
        'mv': 'millivolt',
        'liter': 'litre',
        'litre': 'litre',
        'l': 'litre',
        'cl': 'centilitre',
        'µl': 'microlitre',
        'centilitre': 'centilitre',
        'cubic foot': 'cubic feet',
        'cubic inch': 'cubic inche',
        'cf': 'cubic feet',
        'ml': 'millilitre',
        'millilitre': 'millilitre',
        'in³': 'cubic inche',
        'cubic inch': 'cubic inche',
        'cubic foot': 'cubic feet',
        'imp gal': 'imperial gallon',
        'imperial gallon': 'imperial gallon',
        'cup': 'cups',
        'dl': 'decilitres',
        'decilitre': 'decilitres',
        'floz': 'fluid ounce',
        'gal': 'gallon',
        'pt': 'pint',
        'qt': 'quart',
        'fl oz': 'fluid ounce',
        'gallon': 'gallon',
        'pint': 'pint',
        'quart': 'quart',
        'fluid ounce': 'fluid ounce'
    }
    def convert_word(word):
        # Split the word into numeric and unit parts
        parts = word.split()
        if len(parts) == 2:
            numeric_part, unit_part = parts
            numeric_part = float(numeric_part)  # Convert numeric part to float
            # Convert unit part to its full name
            full_unit = unit_mapping.get(unit_part, unit_part)
            return f"{numeric_part} {full_unit}"
        return word  # In case the format is unexpected

    # Apply conversion to each item in the list
    converted_list = [convert_word(item) for item in unit_list]
    return converted_list
def filter_terms_by_category(terms, category):
    units = unit_categories.get(category, set())
    return [term for term in terms if contains_units(term, units)]
import re

# Conversion factors for different categories
conversion_factors = {
    'length': {
        'millimetre': 1,
        'centimetre': 10,
        'inch': 25.4,
        'foot': 304.8,
        'yard': 914.4,
        'metre': 1000,
        'kilometre': 1000000,
        'cm': 10,
        'CM': 10
    },
    'item_weight': {
        'gram': 1,
        'kilogram': 1000,
        'microgram': 0.001,
        'milligram': 0.001,
        'ounce': 28.3495,
        'pound': 453.592,
        'ton': 1000000,
        'g': 1,
        'kg': 1000,
        'µg': 0.001,
        'mg': 0.001,
        'oz': 28.3495,
        'lb': 453.592,
        'ton': 1000000
    },
    'voltage': {
        'kilovolt': 1000000,
        'millivolt': 0.001,
        'volt': 1,
        'kv': 1000000,
        'mv': 0.001,
        'v': 1
    },
    'wattage': {
        'kilowatt': 1000,
        'watt': 1,
        'w': 1,
        'KW': 1000,
        'W': 1
    },
    'item_volume': {
        'centilitre': 10,
        'cubic foot': 28316.8466,
        'cubic inch': 16.387,
        'cup': 236.588,
        'decilitre': 100,
        'fluid ounce': 29.5735,
        'gallon': 3785.41,
        'imperial gallon': 4546.09,
        'litre': 1000,
        'microlitre': 0.001,
        'millilitre': 1,
        'pint': 473.176,
        'quart': 946.353,
        'cl': 10,
        'cf': 28316.8466,
        'in³': 16.387,
        'dl': 100,
        'floz': 29.5735,
        'gal': 3785.41,
        'imp gal': 4546.09,
        'l': 1000,
        'µl': 0.001,
        'ml': 1,
        'pt': 473.176,
        'qt': 946.353
    }
}

def parse_value_unit(value_unit_str):
    # Extract numerical value and unit from the string
    match = re.match(r'(\d+\.?\d*)\s*([a-zA-Z\s]+)', value_unit_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip().lower()
        return value, unit
    return None

def format_value(value, unit):
    # Format the output as "value unit"
    return f"{value} {unit}"

def convert_to_standard(value, unit, category):
    # Convert value to the base unit within the category
    if unit in conversion_factors[category]:
        base_value = value * conversion_factors[category][unit]
        return base_value
    return None

def format_range(min_value, max_value, unit):
    # Format the range in "[min-max] unit"
    return f"[{min_value}-{max_value}] {unit}"

def format_unit(value_unit_str, category):
    # Handle range and single value cases
    if '-' in value_unit_str:
        min_str, max_str = value_unit_str.split('-')
        min_parsed = parse_value_unit(min_str)
        max_parsed = parse_value_unit(max_str)
        if min_parsed and max_parsed:
            min_value, min_unit = min_parsed
            max_value, max_unit = max_parsed
            if min_unit == max_unit:
                min_base = convert_to_standard(min_value, min_unit, category)
                max_base = convert_to_standard(max_value, max_unit, category)
                if min_base is not None and max_base is not None:
                    min_formatted = min_base / conversion_factors[category][min_unit]
                    max_formatted = max_base / conversion_factors[category][max_unit]
                    return format_range(min_formatted, max_formatted, min_unit)
    else:
        parsed = parse_value_unit(value_unit_str)
        if parsed:
            value, unit = parsed
            base_value = convert_to_standard(value, unit, category)
            if base_value is not None:
                return format_value(base_value / conversion_factors[category][unit], unit)
    return None

def process_list(value_list, category):
    results = []
    for value_str in value_list:
        formatted_output = format_unit(value_str, category)
        if formatted_output:
            results.append(formatted_output)
    return results
def convert_weight_to_kg(weight_str):
    # Split the value and unit from the string
    value, unit = weight_str.split()
    value = float(value)

    # Conversion factors to kilograms
    conversion_factors = {
        'g': 1e-3, 'gram': 1e-3, 'grams': 1e-3,
        'kg': 1, 'kilogram': 1, 'kilograms': 1,
        'µg': 1e-9, 'microgram': 1e-9, 'micrograms': 1e-9,
        'mg': 1e-6, 'milligram': 1e-6, 'milligrams': 1e-6,
        'oz': 0.0283495, 'ounce': 0.0283495, 'ounces': 0.0283495,
        'lb': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
        'ton': 1000, 'tons': 1000
    }

    # Normalize the unit
    unit = unit.lower()
    if unit not in conversion_factors:
        raise ValueError(f"Unsupported unit: {unit}")

    return value * conversion_factors[unit]

def maximum_weight_cal(weight_list):
    # Create a list to store tuples (original_weight, converted_to_kg)
    weights_in_kg = [(weight_str, convert_weight_to_kg(weight_str)) for weight_str in weight_list]

    # Find the maximum weight based on the kg values
    max_weight_tuple = max(weights_in_kg, key=lambda x: x[1])

    # Return the original format of the maximum weight
    return max_weight_tuple[0]
def all_same(lst):
    # Check if the list is not empty and if all elements in the list are the same as the first element
    return len(lst) > 0 and all(element == lst[0] for element in lst)
reader = easyocr.Reader(['en'])
a=0
def update_var():
    global a
    a=a+1
    print(a)
def process_image_for_units(img, category):
    try:
        # Set figure size for matplotlib
        rcParams['figure.figsize'] = 8, 16

        if img is None:
            raise ValueError("Image is None")

        # Convert image to grayscale if not already
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocess the image
        img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        # Initialize EasyOCR reader


        # Read text from the image
        output = reader.readtext(img, detail=0)  # Using detail=0 to get text only

        # Combine all text segments into a single paragraph
        paragraph = ' '.join(output)
        paragraph = paragraph.lower()

        # Display the extracted paragraph
        #print("Extracted Paragraph:")
        #print(paragraph)

        # Function to extract words containing numeric characters and units
        def extract_words_with_numbers_and_units(text):
            # Regular expression to match numeric values and units
            pattern = re.compile(r'(\d+(\.\d+)?|\.\d+)\s*([a-zA-Z]+)')
            matches = pattern.findall(text)
            return [f'{number} {unit.lower()}' for number, _, unit in matches]

        # Extract words containing numeric characters and units from the paragraph
        words_with_numbers_and_units = extract_words_with_numbers_and_units(paragraph)

        # Display extracted words containing numeric characters and units
        #print("\nWords Containing Numeric Characters and Units:")
            #print(word)

        # Display the preprocessed image (optional)

        # Filter terms based on the selected category
        filtered_terms = filter_terms_by_category(words_with_numbers_and_units, category)
        #print("\nFiltered Terms Based on Category '{}':".format(category))
        #print(filtered_terms)

        converted= convert_units(filtered_terms)
        #print("\nConverted Units:")
        #print(converted);
        update_var()
        string_="done+"
        if(len(converted)==1):
          print(string_)
          return converted[0]
        if(len(converted)==0):
          print(string_)
          return ""
        if(all_same(converted)):
          print(string_)
          return converted[0]
        if(category=="maximum_weight_recommendation"):
          print(string_)
          return maximum_weight_cal(converted)
        if category == "length":
          if len(converted) == 1:
              print(string_)
              return converted[0]
          print(string_)
          return max(converted)


        elif category == "width":
            print(string_)
            return min(converted)

        elif category in ["height", "depth"]:
            print(string_)
            return max(converted)

        return "nhi"


    except Exception as e:
        print(f"Error processing image: {e}")
        return []  # Return an empty list in case of error
"""# Example usage
img_path = '/content/81R8CrWXthL.jpg'  # Path to your image
category = 'item_weight'  # Category to filter
img = cv2.imread(img_path)  # Load the image
process_image_for_units(img, category)"""
import requests
import numpy as np
import cv2
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        return image
    else:
        raise ValueError(f"Failed to download image from URL: {url}")
def process_row(row):
  img = download_image(row.image_link)
  return process_image_for_units(img, row.entity_name) # Return empty string if there's an error

from concurrent.futures import ThreadPoolExecutor, as_completed
# Use ThreadPoolExecutor for parallel processing
def process_dataframe_parallel(df, max_workers=10000):
    # Default max_workers is optimized based on the system's available resources
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for row in df.itertuples(index=False)]
        
        predictions = []
        for future in as_completed(futures):
          predictions.append(future.result()) # Append empty string for rows with processing errors
    return predictions

# Load the CSV file
import pandas as pd
df = pd.read_csv('test.csv')

# Process the DataFrame in parallel (without specifying max_workers lets ThreadPoolExecutor optimize the number of threads)
df['prediction'] = process_dataframe_parallel(df)

# Save the DataFrame with predictions to a new CSV file
df.to_csv('result.csv', index=False)
